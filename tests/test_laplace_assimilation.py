from types import SimpleNamespace

import numpy as np
import pytest
import torch

import wf_demo.laplace_assimilation as laplace
from wf_demo.laplace_assimilation import (
    AnalyticPixelPrior,
    PriorRegularizer,
    _gain,
    _inverse_model_step,
    active_column_indices,
    map_estimate,
    map_solution,
    one_d_prediction,
    one_d_prediction_and_jacobian,
    sample_inverse_hessian,
    update_reduction,
    zero_d_prediction_and_jacobian,
)
from wf_demo.pixel_model import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PIXEL_COUNT,
    behind_bit_indices,
)


TOOL = ("6kHz", "83ft")


def prior_config(model, horizontal, vertical, std=1.2, mean=0.3):
    return {
        "covariance_model": model,
        "horizontal_correlation": horizontal,
        "vertical_correlation": vertical,
        "prior_mean": mean,
        "prior_standard_deviation": std,
    }


def test_point_forecast_serialization_uses_two_components():
    predictions = np.arange(10 * 3, dtype=float).reshape(10, 3)

    forecast = laplace._forecast_dictionary(predictions, ["point", TOOL])

    assert forecast["point"].shape == (2, 3)
    assert forecast[TOOL].shape == (8, 3)
    np.testing.assert_array_equal(forecast["point"], predictions[:2])
    np.testing.assert_array_equal(forecast[TOOL], predictions[2:])


class LinearZeroDSimulator:
    all_data_types = [TOOL]

    def point_prediction(self, state, position, smoothed=True):
        return np.repeat(3.0 * np.asarray(state)[self.state_index] + 2.0, 8)

    def point_prediction_and_jacobian(self, state, position):
        prediction = self.point_prediction(state, position)
        jacobian = np.zeros((8, GRID_HEIGHT))
        jacobian[:, position[0]] = 3.0
        return prediction, jacobian

    def run_fwd_sim(self, state, member_i):
        values = np.asarray(state["x"])
        predictions = []
        for member in values:
            response = np.repeat(3.0 * member[self.state_index] + 2.0, 8)
            predictions.append([{TOOL: response}])
        return predictions


class ExponentialZeroDSimulator:
    all_data_types = [TOOL]

    def point_prediction(self, state, position, smoothed=True):
        return np.repeat(np.exp(np.asarray(state)[self.state_index]), 8)

    def point_prediction_and_jacobian(self, state, position):
        prediction = self.point_prediction(state, position)
        jacobian = np.zeros((8, GRID_HEIGHT))
        jacobian[:, position[0]] = prediction[0]
        return prediction, jacobian

    def run_fwd_sim(self, state, member_i):
        values = np.asarray(state["x"])
        predictions = []
        for member in values:
            predictions.append([{TOOL: np.repeat(np.exp(member[self.state_index]), 8)}])
        return predictions


def test_zero_d_uses_exact_sensitivity_at_bit_cell_only():
    position = (5, 9)
    simulator = LinearZeroDSimulator()
    simulator.state_index = position[0] * GRID_WIDTH + position[1]

    prediction, local_jacobian = zero_d_prediction_and_jacobian(
        simulator, np.zeros(PIXEL_COUNT), position
    )

    np.testing.assert_allclose(prediction, np.full(8, 2.0))
    assert local_jacobian.shape == (8, GRID_HEIGHT)
    np.testing.assert_allclose(local_jacobian[:, position[0]], 3.0)
    assert np.count_nonzero(np.delete(local_jacobian, position[0], axis=1)) == 0


class DifferentiableWindowModel:
    rh_mult = torch.ones(1)

    def __init__(self):
        self.window_width = None

    def forward_from_facies(self, facies, index_vector):
        self.window_width = index_vector.shape[1]
        resistivity = (
            facies[:, 0] * 4.0 + facies[:, 1] * 171.0 + facies[:, 2] * 55.0
        )
        per_column = resistivity.sum(dim=1)[:, :self.window_width]
        return per_column[:, :, None, None].expand(-1, -1, 1, 8)


class PositionSensitiveWindowModel(DifferentiableWindowModel):
    def __init__(self):
        super().__init__()
        self.index_vector = None

    def forward_from_facies(self, facies, index_vector):
        self.index_vector = index_vector.detach().cpu().numpy()
        resistivity = (
            facies[:, 0] * 4.0 + facies[:, 1] * 171.0 + facies[:, 2] * 55.0
        )
        per_column = resistivity.sum(dim=1)[:, :index_vector.shape[1]]
        offsets = 1000.0 * torch.arange(
            index_vector.shape[1], dtype=per_column.dtype, device=per_column.device
        )
        values = per_column + offsets
        return values[:, :, None, None].expand(-1, -1, 1, 8)


def test_one_d_uses_full_window_and_returns_active_column_gradient():
    model = DifferentiableWindowModel()
    simulator = SimpleNamespace(
        NNmodel=model,
        all_data_types=[TOOL],
        tool_configs=[TOOL],
    )
    position = (11, 6)

    prediction, local_jacobian = one_d_prediction_and_jacobian(
        simulator, np.zeros(PIXEL_COUNT), position
    )

    assert model.window_width == position[1] + 1
    tail = 1.0 / (1.0 + np.exp(0.4307273 / 0.2))
    middle = 1.0 - 2.0 * tail
    relaxed_rh = 4.0 * tail + 171.0 * middle + 55.0 * tail
    np.testing.assert_allclose(
        prediction, np.full(8, GRID_HEIGHT * relaxed_rh), rtol=1e-5
    )
    assert local_jacobian.shape == (8, GRID_HEIGHT)
    assert np.all(np.abs(local_jacobian) > 0.0)


def test_one_d_extracts_current_well_column_and_row_path():
    model = PositionSensitiveWindowModel()
    simulator = SimpleNamespace(
        NNmodel=model,
        all_data_types=[TOOL],
        tool_configs=[TOOL],
    )
    position = (13, 7)
    state = np.zeros(PIXEL_COUNT)

    prediction, local_jacobian = one_d_prediction_and_jacobian(
        simulator, state, position
    )
    current_index_vector = model.index_vector.copy()
    previous_column = one_d_prediction(
        simulator, state, (position[0], position[1] - 1), smoothed=True
    )

    np.testing.assert_array_equal(
        current_index_vector, np.full((1, position[1] + 1), position[0])
    )
    np.testing.assert_allclose(prediction - previous_column, 1000.0)
    assert local_jacobian.shape == (8, GRID_HEIGHT)
    assert np.all(np.abs(local_jacobian) > 0.0)


def test_smoothed_jacobian_is_consistent_with_smoothed_prediction():
    model = DifferentiableWindowModel()
    simulator = SimpleNamespace(
        NNmodel=model,
        all_data_types=[TOOL],
        tool_configs=[TOOL],
    )
    position = (11, 6)
    rng = np.random.default_rng(2)
    state = rng.standard_normal(PIXEL_COUNT) * 0.3

    _, local_jacobian = one_d_prediction_and_jacobian(simulator, state, position)

    h = 1e-3
    for profile_row in (0, 17, 40, 63):
        plus = state.copy()
        minus = state.copy()
        index = profile_row * GRID_WIDTH + position[1]
        plus[index] += h
        minus[index] -= h
        forward = one_d_prediction(simulator, plus, position, smoothed=True)[0]
        backward = one_d_prediction(simulator, minus, position, smoothed=True)[0]
        np.testing.assert_allclose(
            local_jacobian[0, profile_row], (forward - backward) / (2 * h),
            rtol=5e-3, atol=1e-4,
        )


def dense_prior_covariance(prior):
    return prior.variance * np.kron(prior.vertical, prior.horizontal)


def test_map_update_confined_to_spherical_covariance_range():
    prior = AnalyticPixelPrior(prior_config("Spherical", horizontal=4.5, vertical=10.0))
    column = 30
    state_mean = np.full(PIXEL_COUNT, prior.prior_mean)
    bit_row = 5
    jacobian = np.zeros((1, GRID_HEIGHT))
    jacobian[0, bit_row] = 1.0
    prediction = np.array([1.0])
    observed = np.array([6.0])

    map_state = map_estimate(
        state_mean, prior, column, jacobian, prediction, observed, np.array([0.5])
    )

    update = (map_state - state_mean).reshape(GRID_HEIGHT, GRID_WIDTH)
    outside = np.abs(np.arange(GRID_WIDTH) - column) >= 4.5
    assert np.count_nonzero(update[:, outside]) == 0
    assert np.any(update[:, ~outside] != 0.0)
    support_width = np.count_nonzero(np.abs(update).sum(axis=0))
    assert support_width == 9


def test_map_estimate_matches_dense_kalman_gain():
    prior = AnalyticPixelPrior(prior_config("Cubic", horizontal=8.0, vertical=5.0))
    column = 30
    indices = np.arange(GRID_HEIGHT) * GRID_WIDTH + column
    covariance = dense_prior_covariance(prior)
    cross = covariance[:, indices]
    active_block = covariance[np.ix_(indices, indices)]

    rng = np.random.default_rng(7)
    jacobian = rng.standard_normal((6, GRID_HEIGHT))
    variances = np.array([0.5, 1.0, 2.0, 0.3, 1.7, 0.9])
    prediction = rng.standard_normal(6)
    observed = prediction + 2.0
    state_mean = rng.standard_normal(PIXEL_COUNT)

    innovation = jacobian @ active_block @ jacobian.T + np.diag(variances)
    expected = state_mean + (cross @ jacobian.T) @ np.linalg.solve(
        innovation, observed - prediction
    )

    map_state = map_estimate(
        state_mean, prior, column, jacobian, prediction, observed, variances
    )

    np.testing.assert_allclose(map_state, expected, rtol=1e-8, atol=1e-10)


def test_reduction_reproduces_sequential_posterior_block():
    prior = AnalyticPixelPrior(prior_config("Spherical", horizontal=8.0, vertical=5.0))
    column = 30
    indices = np.arange(GRID_HEIGHT) * GRID_WIDTH + column
    active_block = dense_prior_covariance(prior)[np.ix_(indices, indices)]

    rng = np.random.default_rng(4)
    jacobian = rng.standard_normal((6, GRID_HEIGHT))
    variances = np.full(6, 0.8)

    innovation = jacobian @ active_block @ jacobian.T + np.diag(variances)
    sequential = active_block - active_block @ jacobian.T @ np.linalg.solve(
        innovation, jacobian @ active_block
    )
    cross = dense_prior_covariance(prior)[:, indices]
    reduction = update_reduction(prior, column, jacobian, variances)

    assert reduction.shape == (PIXEL_COUNT, 6)
    np.testing.assert_allclose(
        prior.active_block(column, reduction), sequential, rtol=1e-7, atol=1e-10
    )
    sequential_cross = cross - (cross @ jacobian.T) @ np.linalg.solve(
        innovation, jacobian @ active_block
    )
    np.testing.assert_allclose(
        prior.column_covariance(column, reduction),
        sequential_cross,
        rtol=1e-7,
        atol=1e-10,
    )


def test_gaussian_prior_has_decaying_but_nonzero_tails():
    prior = AnalyticPixelPrior(prior_config("Gaussian", horizontal=6.0, vertical=3.0))

    covariance = prior.column_covariance(30)
    magnitudes = np.abs(covariance[np.arange(GRID_WIDTH), 0])

    assert np.all(magnitudes > 0.0)
    assert magnitudes[31] > magnitudes[40] > magnitudes[55]
    right_tail = magnitudes[30:]
    assert np.all(np.diff(right_tail) <= 1e-12)


def test_prior_draws_match_analytic_covariance():
    prior = AnalyticPixelPrior(prior_config("Exponential", horizontal=9.0, vertical=4.0))
    draws = prior.draw(400, np.random.default_rng(6))

    assert draws.shape == (PIXEL_COUNT, 400)
    covariance = dense_prior_covariance(prior)
    sampled = np.cov(draws)
    checked = [(10, 10), (10, 80), (500, 3900), (2000, 2005)]
    for first, second in checked:
        assert abs(sampled[first, second] - covariance[first, second]) < 0.15 * (
            abs(covariance[first, second]) + 0.1
        )
        assert abs(sampled[first, first] - covariance[first, first]) < 0.15 * abs(
            covariance[first, first]
        )


def linear_setup():
    position = (5, 9)
    simulator = LinearZeroDSimulator()
    simulator.state_index = position[0] * GRID_WIDTH + position[1]
    prior = AnalyticPixelPrior(prior_config("Spherical", 6.0, 3.0))
    state_mean = np.zeros(PIXEL_COUNT)
    observed = np.full(8, 14.0)
    variances = np.full(8, 0.01)
    return simulator, position, prior, state_mean, observed, variances


def test_map_solution_iterates_to_exact_map_on_linear_response():
    simulator, position, prior, state_mean, observed, variances = linear_setup()

    map_state, info = map_solution(
        simulator, "0D", position, prior, state_mean, observed, variances,
        max_iterations=5,
    )

    slope, mismatch, n_data = 3.0, 12.0, 8
    exact_bit = (
        prior.variance * slope * n_data * mismatch
    ) / (variances[0] + n_data * slope**2 * prior.variance)
    assert abs(map_state[simulator.state_index] - exact_bit) < 1e-9
    assert info["converged"] is True
    assert info["reason"] in {"zero step", "step tolerance"}
    assert info["iterations"] == 2
    assert info["alphas"] == [1.0, 1.0]
    assert info["chi2"] <= n_data


def test_adaptive_laplace_domain_keeps_columns_behind_bit_unchanged():
    simulator, position, prior, _, observed, variances = linear_setup()
    rng = np.random.default_rng(23)
    state = prior.draw(12, rng)
    state_mean = state.mean(axis=1)

    map_state, _ = map_solution(
        simulator,
        "0D",
        position,
        prior,
        state_mean,
        observed,
        variances,
        max_iterations=3,
    )
    _, jacobian = zero_d_prediction_and_jacobian(
        simulator, map_state, position
    )
    posterior = sample_inverse_hessian(
        state,
        map_state,
        prior,
        position[1],
        jacobian,
        variances,
        rng,
        minimum_column=position[1],
    )
    reduction = update_reduction(
        prior,
        position[1],
        jacobian,
        variances,
        minimum_column=position[1],
    )
    frozen = behind_bit_indices(position[1])

    np.testing.assert_array_equal(map_state[frozen], state_mean[frozen])
    np.testing.assert_allclose(posterior[frozen], state[frozen], atol=1e-12)
    assert np.count_nonzero(reduction[frozen]) == 0


def test_map_solution_single_iteration_matches_capped_map_estimate():
    simulator, position, prior, state_mean, observed, variances = linear_setup()

    prediction, jacobian = zero_d_prediction_and_jacobian(
        simulator, state_mean, position
    )
    single = map_estimate(
        state_mean,
        prior,
        position[1],
        jacobian,
        prediction,
        observed,
        variances,
        minimum_column=position[1],
    )
    raw_step = single - state_mean
    limit = 2.0 * float(np.sqrt(prior.variance))
    scale = min(1.0, limit / float(np.max(np.abs(raw_step))))
    expected = state_mean + scale * raw_step

    iterated, info = map_solution(
        simulator, "0D", position, prior, state_mean, observed, variances,
        max_iterations=1,
    )

    np.testing.assert_allclose(iterated, expected, rtol=1e-12, atol=1e-12)
    assert info["iterations"] == 1
    assert info["alphas"] == [1.0]


def test_line_search_limits_step_on_convex_response():
    position = (5, 9)
    simulator = ExponentialZeroDSimulator()
    simulator.state_index = position[0] * GRID_WIDTH + position[1]
    prior = AnalyticPixelPrior(prior_config("Spherical", 6.0, 3.0, std=2.0))
    state_mean = np.zeros(PIXEL_COUNT)
    observed = np.full(8, 5.0)
    variances = np.full(8, 0.04)
    initial_chi2 = 8.0 * (5.0 - 1.0) ** 2 / 0.04

    map_state, info = map_solution(
        simulator, "0D", position, prior, state_mean, observed, variances,
        max_iterations=3,
    )

    assert info["alphas"][0] < 1.0
    assert info["chi2"] < initial_chi2
    response = np.exp(map_state[simulator.state_index])
    assert abs(response - 5.0) < abs(1.0 - 5.0)


def test_regularizer_quadratic_matches_dense_inverse():
    prior = AnalyticPixelPrior(prior_config("Spherical", 8.0, 5.0))
    rng = np.random.default_rng(3)
    jacobian = rng.standard_normal((6, GRID_HEIGHT))
    reduction = update_reduction(prior, 30, jacobian, np.full(6, 0.5))
    regularizer = PriorRegularizer(prior, reduction)

    difference = rng.standard_normal(PIXEL_COUNT)
    dense = dense_prior_covariance(prior) - reduction @ reduction.T
    expected = difference @ np.linalg.solve(dense, difference)

    np.testing.assert_allclose(regularizer.quadratic(difference), expected, rtol=1e-8)


def test_map_solution_with_reduction_does_not_stall():
    simulator, position, prior, state_mean, observed, variances = linear_setup()
    prediction, jacobian = zero_d_prediction_and_jacobian(
        simulator, state_mean, position
    )
    rng = np.random.default_rng(10)
    previous_jacobian = rng.standard_normal((5, GRID_HEIGHT))
    reduction = update_reduction(
        prior, position[1] - 4, previous_jacobian, np.full(5, 0.4)
    )

    map_state, info = map_solution(
        simulator, "0D", position, prior, state_mean, observed, variances,
        reduction=reduction,
        max_iterations=5,
    )

    assert info["iterations"] > 0
    assert info["reason"] not in {"line search failed", "no predicted decrease"}
    assert info["chi2"] < float(np.sum((prediction - observed) ** 2 / variances))
    assert not np.allclose(map_state, state_mean)
    reduced_variance = prior.active_block(position[1], reduction)[position[0], position[0]]
    expected_bit = (
        reduced_variance * 3.0 * observed.size * 12.0
    ) / (variances[0] + observed.size * 3.0**2 * reduced_variance)
    assert abs(map_state[simulator.state_index] - expected_bit) < 1e-8


def test_gaussian_prior_quadratic_uses_stable_cholesky_whitening():
    prior = AnalyticPixelPrior(
        prior_config("Gaussian", horizontal=16.0, vertical=4.0, std=1.0, mean=0.0)
    )
    rng = np.random.default_rng(22)
    standard = rng.standard_normal((GRID_HEIGHT, GRID_WIDTH))
    difference = (
        prior.chol_vertical @ standard @ prior.chol_horizontal.T
    ).reshape(PIXEL_COUNT)

    quadratic = PriorRegularizer(prior).quadratic(difference)

    assert quadratic >= 0.0
    np.testing.assert_allclose(quadratic, np.sum(standard**2), rtol=1e-6)


def test_reduced_step_reports_actual_capped_model_decrease():
    prior = AnalyticPixelPrior(prior_config("Spherical", 8.0, 5.0))
    rng = np.random.default_rng(11)
    previous_jacobian = rng.standard_normal((5, GRID_HEIGHT))
    reduction = update_reduction(
        prior, 12, previous_jacobian, np.full(5, 0.4)
    )
    regularizer = PriorRegularizer(prior, reduction)
    column = 19
    jacobian = rng.standard_normal((6, GRID_HEIGHT))
    variances = np.linspace(0.3, 0.8, 6)
    delta = rng.standard_normal(PIXEL_COUNT) * 0.05
    residual = rng.standard_normal(6) * 3.0
    step_limit = 0.02

    step, predicted = _inverse_model_step(
        regularizer, column, jacobian, variances,
        delta, residual, step_limit,
    )

    active = active_column_indices(column)
    linearized_residual = residual + jacobian @ step[active]
    current = 0.5 * float(residual @ (residual / variances))
    current += 0.5 * regularizer.quadratic(delta)
    candidate = 0.5 * float(
        linearized_residual @ (linearized_residual / variances)
    )
    candidate += 0.5 * regularizer.quadratic(delta + step)

    assert np.max(np.abs(step)) <= step_limit * (1.0 + 1e-12)
    gain_step = -delta + _gain(
        prior, column, jacobian, variances, reduction
    ) @ (jacobian @ delta[active] - residual)
    gain_scale = min(1.0, step_limit / np.max(np.abs(gain_step)))
    gain_step *= gain_scale
    np.testing.assert_allclose(step, gain_step, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(predicted, current - candidate, rtol=1e-12)
    assert predicted > 0.0


def test_map_solution_does_not_stop_at_nonstationary_low_chi2():
    simulator, position, prior, state_mean, _, variances = linear_setup()
    observed = np.full(8, 2.05)
    initial_chi2 = float(np.sum((np.full(8, 2.0) - observed) ** 2 / variances))
    assert initial_chi2 < observed.size

    map_state, info = map_solution(
        simulator, "0D", position, prior, state_mean, observed, variances,
        max_iterations=5,
    )

    exact_bit = (
        prior.variance * 3.0 * observed.size * 0.05
    ) / (variances[0] + observed.size * 3.0**2 * prior.variance)
    assert abs(map_state[simulator.state_index] - exact_bit) < 1e-9
    assert info["converged"] is True
    assert info["reason"] in {"zero step", "step tolerance"}


def test_failed_map_keeps_ensemble_and_reduction(monkeypatch, tmp_path):
    state = np.arange(PIXEL_COUNT * 3, dtype=float).reshape(PIXEL_COUNT, 3)
    reduction = np.ones((PIXEL_COUNT, 1))
    simulator = SimpleNamespace(
        all_data_types=[TOOL],
        update_bit_pos=lambda positions: None,
    )
    monkeypatch.setattr(
        laplace, "load_observation",
        lambda *args, **kwargs: (np.zeros(8), np.ones(8)),
    )
    monkeypatch.setattr(
        laplace, "_map_response",
        lambda *args, **kwargs: np.zeros(8),
    )
    monkeypatch.setattr(
        laplace, "map_solution",
        lambda *args, **kwargs: (
            state.mean(axis=1),
            {"reason": "no predicted decrease", "iterations": 5},
        ),
    )
    monkeypatch.setattr(
        laplace, "run_predictions",
        lambda simulator, ensemble: np.zeros((8, ensemble.shape[1])),
    )

    posterior, returned_reduction = laplace.assimilate_laplace(
        state,
        simulator,
        "1D",
        (31, 1),
        object(),
        reduction=reduction,
        output_dir=tmp_path,
    )

    np.testing.assert_array_equal(posterior, state)
    assert returned_reduction is reduction
    np.testing.assert_array_equal(
        np.load(tmp_path / "posterior_state_estimate.npz")["x"], state
    )


def test_inverse_hessian_sampling_updates_current_column_cell():
    prior = AnalyticPixelPrior(
        prior_config("Spherical", horizontal=0.5, vertical=0.5, std=1.0, mean=0.0)
    )
    np.testing.assert_allclose(prior.vertical, np.eye(GRID_HEIGHT))
    np.testing.assert_allclose(prior.horizontal, np.eye(GRID_WIDTH))
    rng = np.random.default_rng(15)
    state = prior.draw(400, rng)
    column = 23
    profile_row = 17
    jacobian = np.zeros((1, GRID_HEIGHT))
    jacobian[0, profile_row] = 1.0
    variance = np.array([0.25])
    map_state = np.linspace(-0.5, 0.5, PIXEL_COUNT)

    posterior = sample_inverse_hessian(
        state, map_state, prior, column, jacobian, variance, rng
    )

    target = profile_row * GRID_WIDTH + column
    other_row = (profile_row + 7) * GRID_WIDTH + column
    other_column = profile_row * GRID_WIDTH + column + 3
    np.testing.assert_allclose(posterior.mean(axis=1), map_state, atol=1e-12)
    assert np.var(posterior[target], ddof=1) == pytest.approx(0.2, rel=0.35)
    assert np.var(posterior[other_row], ddof=1) == pytest.approx(1.0, rel=0.35)
    assert np.var(posterior[other_column], ddof=1) == pytest.approx(1.0, rel=0.35)


def test_two_column_reduction_matches_sequential_dense_blocks():
    prior = AnalyticPixelPrior(prior_config("Cubic", 8.0, 5.0))
    rng = np.random.default_rng(16)
    columns = (11, 17, 24)
    selected = np.concatenate([active_column_indices(column) for column in columns])
    rows = selected // GRID_WIDTH
    horizontal = selected % GRID_WIDTH
    base = prior.variance * (
        prior.vertical[rows[:, None], rows[None, :]]
        * prior.horizontal[horizontal[:, None], horizontal[None, :]]
    )
    sequential = base.copy()
    reduction = None
    lookup = {index: offset for offset, index in enumerate(selected)}

    for column, data_count in zip(columns[:2], (5, 7)):
        jacobian = rng.standard_normal((data_count, GRID_HEIGHT))
        variances = np.linspace(0.3, 0.9, data_count)
        local = np.array([lookup[index] for index in active_column_indices(column)])
        cross = sequential[:, local]
        active_block = sequential[np.ix_(local, local)]
        innovation = jacobian @ active_block @ jacobian.T + np.diag(variances)
        sequential -= (cross @ jacobian.T) @ np.linalg.solve(
            innovation, jacobian @ cross.T
        )
        reduction = update_reduction(
            prior, column, jacobian, variances, reduction
        )

    reduced = base - reduction[selected, :] @ reduction[selected, :].T
    np.testing.assert_allclose(reduced, sequential, rtol=1e-7, atol=1e-9)

    regularizer = PriorRegularizer(prior, reduction)
    probe = rng.standard_normal(PIXEL_COUNT)
    reconstructed = regularizer.inverse_apply(regularizer.covariance_apply(probe))
    np.testing.assert_allclose(reconstructed, probe, rtol=1e-7, atol=1e-8)
