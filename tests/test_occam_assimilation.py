from types import SimpleNamespace

import numpy as np
import torch

import wf_demo.occam_assimilation as occam
from wf_demo.laplace_assimilation import zero_d_prediction_and_jacobian
from wf_demo.occam_assimilation import (
    LinearizedOccamSolutions,
    roughness,
    sample_occam_posterior,
    scatter_column_jacobian,
    solve_occam_step,
)
from wf_demo.pixel_model import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PIXEL_COUNT,
    behind_bit_indices,
)


TOOL = ("6kHz", "83ft")


def dense_roughness_gram(anchor):
    lap = occam._axis_laplacian(GRID_HEIGHT)
    return (
        np.kron(np.eye(GRID_WIDTH), lap)
        + np.kron(lap, np.eye(GRID_WIDTH))
        + anchor * np.eye(PIXEL_COUNT)
    )


class LinearZeroDSimulator:
    all_data_types = [TOOL]

    def update_bit_pos(self, bit_pos):
        pass

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


class WindowModel:
    rh_mult = torch.ones(1)

    def forward_from_facies(self, facies, index_vector):
        resistivity = (
            facies[:, 0] * 4.0 + facies[:, 1] * 171.0 + facies[:, 2] * 55.0
        )
        per_column = resistivity.sum(dim=1)[:, :index_vector.shape[1]]
        return per_column[:, :, None, None].expand(-1, -1, 1, 8)


def test_axis_laplacian_matches_first_difference_gram():
    difference = np.zeros((GRID_HEIGHT - 1, GRID_HEIGHT))
    rows = np.arange(GRID_HEIGHT - 1)
    difference[rows, rows] = -1.0
    difference[rows, rows + 1] = 1.0

    np.testing.assert_allclose(
        occam._axis_laplacian(GRID_HEIGHT), difference.T @ difference
    )


def test_q_inverse_apply_matches_dense_kron_solve():
    rng = np.random.default_rng(1)
    stack = rng.standard_normal((GRID_HEIGHT, GRID_WIDTH, 3))
    anchor = 0.03
    expected = np.linalg.solve(
        dense_roughness_gram(anchor), stack.reshape(PIXEL_COUNT, 3)
    ).reshape(stack.shape)

    np.testing.assert_allclose(
        occam._q_inverse_apply(stack, anchor), expected, rtol=1e-8, atol=1e-10
    )


def test_roughness_spans_both_grid_directions():
    rng = np.random.default_rng(2)
    field = rng.standard_normal((GRID_HEIGHT, GRID_WIDTH))
    delta = field.ravel()

    assert roughness(np.zeros(PIXEL_COUNT)) == 0.0
    np.testing.assert_allclose(
        roughness(delta), delta @ dense_roughness_gram(0.0) @ delta, rtol=1e-10
    )
    assert roughness(delta) > 0.0


def test_scatter_column_jacobian_touches_only_active_column():
    rng = np.random.default_rng(3)
    column = 21
    local = rng.standard_normal((6, GRID_HEIGHT))

    full = scatter_column_jacobian(local, column)

    np.testing.assert_array_equal(full[:, column::GRID_WIDTH], local)
    off = np.delete(full, np.arange(column, PIXEL_COUNT, GRID_WIDTH), axis=1)
    assert np.count_nonzero(off) == 0


def test_linearized_solutions_match_dense_regularized_problem():
    rng = np.random.default_rng(4)
    column = 20
    jacobian = scatter_column_jacobian(rng.standard_normal((6, GRID_HEIGHT)), column)
    variances = np.linspace(0.3, 0.9, 6)
    reference = rng.standard_normal(PIXEL_COUNT) * 0.4
    linearized_data = rng.standard_normal(6) * 2.0
    anchor = 0.05

    solutions = LinearizedOccamSolutions(
        jacobian, variances, linearized_data, reference, anchor
    )
    weighted = jacobian / np.sqrt(variances)[:, None]
    q = dense_roughness_gram(anchor)

    for mu in (1e-4, 1.0, 100.0, 1e6):
        model = solutions.solve(mu)
        expected = np.linalg.solve(
            mu * q + weighted.T @ weighted,
            weighted.T @ (linearized_data / np.sqrt(variances)) + mu * q @ reference,
        )
        np.testing.assert_allclose(model, expected, rtol=1e-7, atol=1e-9)

    assert np.max(np.abs(solutions.solve(1e10) - reference)) < 1e-3


def test_search_mu_prefers_smoothest_target_fitting_candidate():
    settings = occam.OccamSettings(target_value=1.0)
    target = 50.0
    upper = occam._target_upper_bound(target, settings)

    def solve_for_mu(mu):
        return np.full(PIXEL_COUNT, mu)

    def evaluate_model(model):
        mu = float(model[0])
        chi2 = 10.0 * mu if mu > 0.0 else 1e6
        return chi2, (1.0 / mu if mu > 0.0 else 1e9)

    result = occam._search_mu(
        np.zeros(PIXEL_COUNT),
        1e6,
        target,
        settings,
        solve_for_mu,
        evaluate_model,
        lambda candidate: candidate,
    )

    assert result.target_reached is True
    assert result.selected is not None
    assert result.selected.chi2 <= upper
    assert result.selected.mu >= 1.0
    fitting = [c for c in result.candidates if c.chi2 <= upper]
    assert result.selected.roughness == min(c.roughness for c in fitting)


def test_search_mu_falls_back_to_best_misfit_when_target_unreachable():
    settings = occam.OccamSettings(target_value=1.0)

    result = occam._search_mu(
        np.zeros(PIXEL_COUNT),
        1e3,
        50.0,
        settings,
        lambda mu: np.full(PIXEL_COUNT, mu),
        lambda model: (2e3, 1.0),
        lambda candidate: candidate,
    )

    assert result.target_reached is False
    assert result.stalled is True
    assert result.selected is not None
    assert result.selected.chi2 == 2e3


class WindowSimulator:
    def __init__(self):
        self.NNmodel = WindowModel()
        self.all_data_types = [TOOL]
        self.tool_configs = [TOOL]
        self.bit_pos = [(0, 0)]

    def update_bit_pos(self, bit_pos):
        self.bit_pos = list(bit_pos)

    def run_fwd_sim(self, state, member_i):
        from wf_demo.laplace_assimilation import one_d_prediction

        predictions = []
        for member in np.asarray(state["x"]):
            response = one_d_prediction(self, member, self.bit_pos[0], smoothed=False)
            predictions.append([{TOOL: response}])
        return predictions


def linear_zero_d_setup(position, observed_value):
    simulator = LinearZeroDSimulator()
    simulator.state_index = position[0] * GRID_WIDTH + position[1]
    observed = np.full(8, observed_value)
    variances = np.full(8, 0.01)
    return simulator, observed, variances


def linear_zero_d_prediction_and_jacobian(simulator, model, position):
    prediction, local = zero_d_prediction_and_jacobian(simulator, model, position)
    return prediction, scatter_column_jacobian(local, position[1])


def test_solve_occam_step_fits_target_and_spreads_in_2d():
    position = (31, 10)
    simulator, observed, variances = linear_zero_d_setup(position, 3.0 * 1.2 + 2.0)
    reference = np.zeros(PIXEL_COUNT)
    bit = simulator.state_index

    def evaluate_model(model):
        response = np.repeat(3.0 * model[bit] + 2.0, 8)
        return occam._chi_square(response, observed, variances), roughness(
            model - reference
        )

    result = solve_occam_step(
        reference,
        reference,
        observed,
        variances,
        lambda model: linear_zero_d_prediction_and_jacobian(simulator, model, position),
        evaluate_model,
        occam.OccamSettings(target_value=1.0, max_iterations=6),
        sigma_prior=1.0,
        minimum_column=position[1],
    )

    assert result.target_reached is True
    assert abs(result.model[bit] - 1.2) < 0.05
    update = (result.model - reference).reshape(GRID_HEIGHT, GRID_WIDTH)
    assert np.max(np.abs(update[:, position[1] + 1])) > 1e-3
    assert np.max(np.abs(update[position[0] + 1, :])) > 1e-3
    far = update[:, position[1] + 30:]
    assert np.max(np.abs(far)) < 0.15 * abs(result.model[bit])
    assert np.count_nonzero(result.model[behind_bit_indices(position[1])]) == 0


def test_solve_occam_step_keeps_already_fitting_reference():
    position = (15, 5)
    simulator, observed, variances = linear_zero_d_setup(position, 2.0)
    reference = np.zeros(PIXEL_COUNT)
    bit = simulator.state_index

    def evaluate_model(model):
        response = np.repeat(3.0 * model[bit] + 2.0, 8)
        return occam._chi_square(response, observed, variances), roughness(
            model - reference
        )

    result = solve_occam_step(
        reference,
        reference,
        observed,
        variances,
        lambda model: linear_zero_d_prediction_and_jacobian(simulator, model, position),
        evaluate_model,
        occam.OccamSettings(target_value=1.0, max_iterations=4),
        sigma_prior=1.0,
    )

    assert result.iterations == 0
    assert result.target_reached is True
    np.testing.assert_array_equal(result.model, reference)


def test_posterior_sampler_matches_dense_inverse_covariance():
    rng = np.random.default_rng(5)
    jacobian = scatter_column_jacobian(rng.standard_normal((4, GRID_HEIGHT)), 15)
    variances = np.full(4, 0.5)
    mu, anchor, sigma_prior = 50.0, 0.05, 1.3
    assert mu > 1.0 / (anchor * sigma_prior**2)
    map_model = rng.standard_normal(PIXEL_COUNT) * 0.2

    posterior = sample_occam_posterior(
        map_model, jacobian, variances, mu, sigma_prior, 300, rng, anchor
    )

    assert posterior.shape == (PIXEL_COUNT, 300)
    np.testing.assert_allclose(posterior.mean(axis=1), map_model, atol=1e-12)
    hessian = mu * dense_roughness_gram(anchor) + (jacobian.T / variances) @ jacobian
    inverse = np.linalg.inv(hessian)
    draws = posterior - map_model[:, None]
    covariance = np.cov(draws)
    for cell in (0, 500, 2000, 4095):
        assert abs(covariance[cell, cell] - inverse[cell, cell]) < 0.3 * abs(
            inverse[cell, cell]
        )
    quadratic = float(np.mean(np.einsum("ij,jk,ik->i", draws.T, hessian, draws.T)))
    assert abs(quadratic - PIXEL_COUNT) / PIXEL_COUNT < 0.1


def test_sampler_mu_floor_keeps_far_field_within_prior_sigma():
    rng = np.random.default_rng(6)
    jacobian = scatter_column_jacobian(rng.standard_normal((4, GRID_HEIGHT)), 15)

    posterior = sample_occam_posterior(
        np.zeros(PIXEL_COUNT), jacobian, np.full(4, 0.5), 0.0, 1.0, 200, rng, 0.01
    )

    far_std = posterior[::64, :].std(axis=1)
    assert np.all(far_std <= 1.05)


def test_assimilate_occam_returns_ensemble_and_writes_outputs(monkeypatch, tmp_path):
    position = (31, 9)
    simulator = LinearZeroDSimulator()
    simulator.state_index = position[0] * GRID_WIDTH + position[1]
    monkeypatch.setattr(
        occam,
        "load_observation",
        lambda *args, **kwargs: (np.full(8, 5.6), np.full(8, 0.01)),
    )
    rng = np.random.default_rng(7)
    state = rng.standard_normal((PIXEL_COUNT, 6))

    posterior = occam.assimilate_occam(
        state,
        simulator,
        "0D",
        position,
        {"prior_standard_deviation": 1.0},
        output_dir=tmp_path,
        seed=11,
    )

    assert posterior.shape == (PIXEL_COUNT, 6)
    stored = np.load(tmp_path / "posterior_state_estimate.npz")["x"]
    np.testing.assert_array_equal(stored, posterior)
    forecast = np.load(tmp_path / "posterior_forecast.npz", allow_pickle=True)["pred_data"]
    assert np.asarray(forecast.item()[TOOL]).shape[0] == 8
    bit = simulator.state_index
    assert abs(posterior.mean(axis=1)[bit] - 1.2) < 0.1
    np.testing.assert_array_equal(
        posterior[behind_bit_indices(position[1])],
        state[behind_bit_indices(position[1])],
    )


def test_failed_inversion_keeps_previous_ensemble(monkeypatch, tmp_path):
    state = np.arange(PIXEL_COUNT * 3, dtype=float).reshape(PIXEL_COUNT, 3)
    simulator = SimpleNamespace(
        all_data_types=[TOOL],
        update_bit_pos=lambda positions: None,
    )
    monkeypatch.setattr(
        occam, "load_observation", lambda *a, **k: (np.zeros(8), np.ones(8))
    )
    failed = occam.OccamStepResult(
        model=np.zeros(PIXEL_COUNT),
        chi2=1e9,
        target_chi2=8.0,
        roughness=0.0,
        mu=0.0,
        iterations=0,
        converged=False,
        target_reached=False,
        status="stalled",
        history=(),
    )
    monkeypatch.setattr(occam, "solve_occam_step", lambda *a, **k: failed)
    monkeypatch.setattr(
        occam, "run_predictions", lambda sim, ensemble: np.zeros((8, ensemble.shape[1]))
    )

    posterior = occam.assimilate_occam(
        state,
        simulator,
        "1D",
        (31, 1),
        {"prior_standard_deviation": 1.0},
        output_dir=tmp_path,
    )

    np.testing.assert_array_equal(posterior, state)
    np.testing.assert_array_equal(
        np.load(tmp_path / "posterior_state_estimate.npz")["x"], state
    )


def test_zero_d_then_1d_sequence_runs_on_same_measurement(monkeypatch, tmp_path):
    position = (17, 5)
    rng = np.random.default_rng(8)
    zero_d = LinearZeroDSimulator()
    zero_d.state_index = position[0] * GRID_WIDTH + position[1]
    one_d = WindowSimulator()

    def observation(*args, **kwargs):
        return np.full(8, 8.0), np.full(8, 0.04)

    monkeypatch.setattr(occam, "load_observation", observation)
    state = rng.standard_normal((PIXEL_COUNT, 4)) * 0.1

    after_zero_d = occam.assimilate_occam(
        state, zero_d, "0D", position,
        {"prior_standard_deviation": 1.0}, output_dir=tmp_path, seed=1,
    )
    after_one_d = occam.assimilate_occam(
        after_zero_d, one_d, "1D", position,
        {"prior_standard_deviation": 1.0}, output_dir=tmp_path, seed=2,
    )

    assert after_zero_d.shape == state.shape
    assert after_one_d.shape == state.shape
    assert not np.allclose(after_one_d.mean(axis=1), state.mean(axis=1))
