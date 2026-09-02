from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wf_demo.measurements import measurement_width
from wf_demo.pixel_model import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PIXEL_COUNT,
    active_pixel_indices,
    behind_bit_indices,
    pixel_prior_axis_covariances,
    pixel_state_to_facies,
    preserve_behind_bit,
)
from wf_demo.zero_d import (
    point_prediction,
    point_prediction_and_jacobian,
)


VARIANCE_FLOOR = 1e-12
RELAXATION_TEMPERATURE = 0.2
DEFAULT_MAX_GN_ITERATIONS = 10
ARMIJO_CONSTANT = 1e-4
BACKTRACKING_RATIO = 0.5
MAX_BACKTRACKS = 15
OBJECTIVE_TOLERANCE = 1e-4
STEP_TOLERANCE = 1e-4
TRUST_REGION_SIGMAS = 2.0


def _solve(matrix, rhs):
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        scale = max(1.0, float(np.max(np.abs(np.diag(matrix)))))
        return np.linalg.solve(matrix + np.eye(matrix.shape[0]) * 1e-10 * scale, rhs)


def _parse_variance(cell):
    spec = cell if isinstance(cell[0], str) else cell[0]
    if str(spec[0]).lower() != "abs":
        raise ValueError("Laplace assimilation currently requires absolute data variances")
    return np.maximum(np.asarray(spec[1], dtype=float).ravel(), VARIANCE_FLOOR)


def load_observation(data_path, variance_path, data_types):
    data = pd.read_pickle(data_path)
    variance = pd.read_pickle(variance_path)
    observed_parts = []
    variance_parts = []
    for data_type in data_types:
        if data_type not in data.columns or data_type not in variance.columns:
            raise ValueError(f"Missing observation or variance for {data_type!r}")
        observed_parts.append(np.asarray(data.iloc[0][data_type], dtype=float).ravel())
        variance_parts.append(_parse_variance(variance.iloc[0][data_type]))

    observed = np.concatenate(observed_parts)
    variances = np.concatenate(variance_parts)
    if observed.shape != variances.shape:
        raise ValueError(
            f"Observation and variance sizes differ: {observed.size} and {variances.size}"
        )
    return observed, variances


def flatten_predictions(predictions, data_types):
    columns = []
    for member in predictions:
        columns.append(
            np.concatenate([
                np.asarray(member[0][data_type], dtype=float).ravel()
                for data_type in data_types
            ])
        )
    return np.column_stack(columns)


def run_predictions(simulator, state):
    state = np.asarray(state, dtype=float)
    if state.ndim == 1:
        state = state[:, None]
    predictions = simulator.run_fwd_sim({"x": state.T.copy()}, member_i=None)
    return flatten_predictions(predictions, simulator.all_data_types)


def active_column_indices(column):
    return np.arange(GRID_HEIGHT) * GRID_WIDTH + column


class AnalyticPixelPrior:
    """Separable analytic prior C = sigma^2 (V kron H) from the earth-model page.

    The prior covariance never comes from the finite ensemble, so an update at
    one column is exactly zero wherever the covariance is zero: spherical and
    cubic models confine the update to their range, Gaussian and exponential
    models decay with distance but have infinite tails.
    """

    def __init__(self, config):
        self.prior_mean = float(config["prior_mean"])
        prior_standard_deviation = float(config["prior_standard_deviation"])
        if prior_standard_deviation <= 0:
            raise ValueError("Prior standard deviation must be positive")
        self.variance = prior_standard_deviation ** 2
        self.vertical, self.horizontal = pixel_prior_axis_covariances(
            config["covariance_model"],
            float(config["horizontal_correlation"]),
            float(config["vertical_correlation"]),
        )
        self.chol_vertical = np.linalg.cholesky(self.vertical)
        self.chol_horizontal = np.linalg.cholesky(self.horizontal)

    def column_covariance(self, column, reduction=None, minimum_column=0):
        """Cross-covariance between every pixel and the active column: C[:, idx]."""
        if not 0 <= column < GRID_WIDTH:
            raise ValueError(f"Column {column} is outside the pixel grid")
        horizontal_profile = self.horizontal[:, column]
        block = (
            self.vertical[:, None, :] * horizontal_profile[None, :, None]
        ).reshape(PIXEL_COUNT, GRID_HEIGHT)
        block *= self.variance
        if reduction is not None:
            block -= reduction @ reduction[active_column_indices(column), :].T
        block[behind_bit_indices(minimum_column), :] = 0.0
        return block

    def active_block(self, column, reduction=None):
        """Within-column covariance block C[idx, idx] of the active column."""
        block = self.variance * self.horizontal[column, column] * self.vertical
        if reduction is not None:
            rows = reduction[active_column_indices(column), :]
            block = block - rows @ rows.T
        return block

    def draw(self, count, rng):
        standard = rng.standard_normal((GRID_HEIGHT, GRID_WIDTH, count))
        fields = np.einsum(
            "ij,jkn,lk->iln",
            self.chol_vertical,
            standard,
            self.chol_horizontal,
        )
        return self.prior_mean + np.sqrt(self.variance) * fields.reshape(
            PIXEL_COUNT, count
        )


def _gain(prior, column, local_jacobian, variances, reduction, minimum_column=0):
    cross = prior.column_covariance(column, reduction, minimum_column)
    active_block = prior.active_block(column, reduction)
    projected = local_jacobian @ active_block
    innovation = projected @ local_jacobian.T + np.diag(variances)
    return _solve(innovation, (cross @ local_jacobian.T).T).T


def _inverse_model_step(
    regularizer, column, local_jacobian, variances, delta, residual, step_limit,
):
    """Exact Gauss-Newton step for the reduced-prior quadratic model.

    Write y = delta + s and rho = residual - J delta. The model minimum
        min_y 0.5 (rho + J y_a)^T R^{-1} (rho + J y_a)
              + 0.5 y^T C^{-1} y
    is computed in incremental conditional form,
        s = -delta - C J^T (R + J C J^T)^{-1} rho,
    where C is applied as C0 v - R (R^T v) and never formed or inverted.
    The predicted decrease is evaluated from the actual trust-region-capped
    step, not the uncapped model minimum.
    """
    active = active_column_indices(column)
    rho = residual - local_jacobian @ delta[active]

    scattered = np.zeros((PIXEL_COUNT, local_jacobian.shape[0]))
    scattered[active, :] = local_jacobian.T
    c_jacobian_transpose = regularizer.covariance_apply(scattered)
    innovation_matrix = (
        local_jacobian @ c_jacobian_transpose[active, :] + np.diag(variances)
    )
    solved = _solve(innovation_matrix, rho)
    step = -delta - c_jacobian_transpose @ solved

    largest_move = float(np.max(np.abs(step)))
    if largest_move > step_limit:
        step = step * (step_limit / largest_move)
    linearized_residual = residual + local_jacobian @ step[active]
    current_model = 0.5 * float(residual @ (residual / variances))
    current_model += 0.5 * regularizer.quadratic(delta)
    candidate_model = 0.5 * float(
        linearized_residual @ (linearized_residual / variances)
    )
    candidate_model += 0.5 * regularizer.quadratic(delta + step)
    predicted_decrease = current_model - candidate_model
    return step, predicted_decrease


def _state_deviations(state):
    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape[1] < 2:
        raise ValueError("Laplace assimilation requires a state ensemble with at least two members")
    deviations = state - state.mean(axis=1, keepdims=True)
    return deviations - deviations.mean(axis=1, keepdims=True)


def map_estimate(
    state_mean, prior, column, local_jacobian, prediction, observed, variances,
    reduction=None, minimum_column=0,
):
    gain = _gain(
        prior, column, local_jacobian, variances, reduction, minimum_column
    )
    return state_mean + gain @ (observed - prediction)


class PriorRegularizer:
    """Exact C_prior^{-1} quadratic form without dense 4096^2 matrices.

    C_prior = C0 - L L^T with separable C0 = sigma^2 (V kron H), so
    C_prior^{-1} is applied through the axis inverses and a Woodbury
    correction on the low-rank reduction.
    """

    def __init__(self, prior, reduction=None, minimum_column=0):
        self.prior = prior
        self.inverse_variance = 1.0 / prior.variance
        self.minimum_column = int(minimum_column)
        self.active_indices = active_pixel_indices(self.minimum_column)
        self.inactive_indices = behind_bit_indices(self.minimum_column)
        self.width = GRID_WIDTH - self.minimum_column
        self.horizontal = prior.horizontal[
            self.minimum_column:, self.minimum_column:
        ]
        self.chol_horizontal = np.linalg.cholesky(self.horizontal)
        self.whitened_factor = None
        self.gram = None
        self.gram_eigenvalues = None
        self.gram_eigenvectors = None
        self.reduction_matrix = None
        if reduction is not None and reduction.size:
            self.reduction_matrix = np.zeros_like(reduction)
            self.reduction_matrix[self.active_indices, :] = reduction[
                self.active_indices, :
            ]
            self.whitened_factor = self._base_inverse_apply(
                self.reduction_matrix
            )
            gram = (
                np.eye(reduction.shape[1])
                - self.reduction_matrix.T @ self.whitened_factor
            )
            self.gram = 0.5 * (gram + gram.T)
            self.gram_eigenvalues, self.gram_eigenvectors = np.linalg.eigh(self.gram)
            if self.gram_eigenvalues[0] <= 0.0:
                raise ValueError(
                    "Laplace covariance reduction is not positive definite: "
                    f"minimum Woodbury Gram eigenvalue {self.gram_eigenvalues[0]:.3g}"
                )

    def _base_inverse_apply(self, matrix):
        """C0^{-1} @ matrix using stable triangular solves."""
        matrix = np.asarray(matrix)
        vector = matrix.ndim == 1
        columns = 1 if vector else matrix.shape[1]
        field = matrix[self.active_indices].reshape(
            GRID_HEIGHT, self.width, columns
        )
        left = np.linalg.solve(self.prior.chol_vertical, field.reshape(GRID_HEIGHT, -1))
        left = np.linalg.solve(self.prior.chol_vertical.T, left).reshape(field.shape)
        right = np.linalg.solve(
            self.chol_horizontal,
            left.transpose(1, 0, 2).reshape(self.width, -1),
        )
        right = np.linalg.solve(
            self.chol_horizontal.T, right
        ).reshape(self.width, GRID_HEIGHT, -1).transpose(1, 0, 2)
        active = self.inverse_variance * right.reshape(
            self.active_indices.size, columns
        )
        out = np.zeros((PIXEL_COUNT, columns), dtype=active.dtype)
        out[self.active_indices, :] = active
        return out[:, 0] if vector else out

    def gram_solve(self, rhs):
        if self.gram_eigenvalues is None:
            return rhs
        return self.gram_eigenvectors @ (
            (self.gram_eigenvectors.T @ rhs) / self.gram_eigenvalues[:, None]
            if rhs.ndim > 1
            else (self.gram_eigenvectors.T @ rhs) / self.gram_eigenvalues
        )

    def _prior_apply(self, matrix):
        """Active-domain C0 @ matrix for a vector or block."""
        matrix = np.asarray(matrix)
        vector = matrix.ndim == 1
        columns = 1 if vector else matrix.shape[1]
        field = matrix[self.active_indices].reshape(
            GRID_HEIGHT, self.width, columns
        )
        active = self.prior.variance * np.einsum(
            "ik,ab,kbn->ian",
            self.prior.vertical,
            self.horizontal,
            field,
        )
        out = np.zeros((PIXEL_COUNT, columns), dtype=active.dtype)
        out[self.active_indices, :] = active.reshape(
            self.active_indices.size, columns
        )
        return out[:, 0] if vector else out

    def inverse_apply(self, matrix):
        """C_prior^{-1} @ matrix through the axis inverses and the Woodbury term."""
        out = self._base_inverse_apply(matrix)
        if self.whitened_factor is None:
            return out
        return out + self.whitened_factor @ self.gram_solve(
            self.whitened_factor.T @ matrix
        )

    def covariance_apply(self, matrix):
        """C_prior @ matrix: C_prior is exactly C0 - R R^T, applied separably."""
        base = self._prior_apply(matrix)
        if self.reduction_matrix is None:
            return base
        return base - self.reduction_matrix @ (self.reduction_matrix.T @ matrix)

    def quadratic(self, difference):
        base, correction = self.quadratic_split(difference)
        return base + correction

    def quadratic_split(self, difference):
        """(C0 part, Woodbury part) of the C_prior^{-1} quadratic form."""
        field = np.asarray(difference)[self.active_indices].reshape(
            GRID_HEIGHT, self.width
        )
        whitened = np.linalg.solve(self.prior.chol_vertical, field)
        whitened = np.linalg.solve(
            self.chol_horizontal, whitened.T
        ).T
        base = self.inverse_variance * float(np.sum(whitened**2))
        if self.whitened_factor is None:
            return base, 0.0
        projected = self.whitened_factor.T @ difference
        return base, float(projected @ self.gram_solve(projected))


def _map_response(simulator, simulator_name, state, position, smoothed):
    if simulator_name == "0D":
        return point_prediction(simulator, state, position, smoothed)
    if simulator_name == "1D":
        return one_d_prediction(simulator, state, position, smoothed)
    raise ValueError(f"Unknown Laplace simulator: {simulator_name}")


def map_solution(
    simulator,
    simulator_name,
    position,
    prior,
    state_mean,
    observed,
    variances,
    reduction=None,
    max_iterations=DEFAULT_MAX_GN_ITERATIONS,
):
    """Iterated Gauss-Newton MAP solve with an Armijo backtracking line search.

    The solve is self-consistent: the 1D objective and its Jacobian both use
    the relaxed (sigmoid) facies forward, so the linearization predicts real
    decreases of the minimized function; the hard forward is only used for
    reporting. Steps are capped by a trust region of a few prior standard
    deviations. The expected chi-square is reported as a data-fit diagnostic,
    not used as an optimizer stopping criterion. Convergence requires a
    vanishing step or objective/step tolerance, otherwise the iteration budget
    is spent.
    """
    column = position[1]
    regularizer = PriorRegularizer(prior, reduction, minimum_column=column)
    step_limit = TRUST_REGION_SIGMAS * float(np.sqrt(prior.variance))

    def evaluate(state):
        response = _map_response(simulator, simulator_name, state, position, True)
        chi2 = float(np.sum((response - observed) ** 2 / variances))
        return 0.5 * chi2 + 0.5 * regularizer.quadratic(state - state_mean), chi2

    current = np.asarray(state_mean, dtype=float).copy()
    objective, chi2 = evaluate(current)
    alphas = []
    converged = False
    reason = "max iterations"
    print(
        f"{simulator_name} Laplace MAP iter 0: chi2={chi2:.4g}/{observed.size}, "
        f"objective={objective:.4g}"
    )

    for iteration in range(1, max_iterations + 1):
        prediction, local_jacobian = _prediction_and_jacobian(
            simulator, simulator_name, current, position
        )
        residual = prediction - observed
        delta = current - state_mean
        if reduction is None:
            gain = _gain(
                prior,
                column,
                local_jacobian,
                variances,
                reduction,
                minimum_column=column,
            )
            step = -delta + gain @ (
                local_jacobian @ delta[active_column_indices(column)] - residual
            )
            if not np.any(step):
                converged = True
                reason = "zero step"
                break
            largest_move = float(np.max(np.abs(step)))
            if largest_move > step_limit:
                step = step * (step_limit / largest_move)
            linearized = (
                prediction + local_jacobian @ step[active_column_indices(column)]
            )
            linearized_data = 0.5 * float(
                np.sum((linearized - observed) ** 2 / variances)
            )
            linearized_total = linearized_data + 0.5 * regularizer.quadratic(
                current + step - state_mean
            )
            predicted_decrease = objective - linearized_total
        else:
            step, predicted_decrease = _inverse_model_step(
                regularizer, column, local_jacobian, variances,
                delta, residual, step_limit,
            )
            if not np.any(step):
                converged = True
                reason = "zero step"
                break
        if np.linalg.norm(step) / max(1.0, np.linalg.norm(current)) < STEP_TOLERANCE:
            converged = True
            reason = "step tolerance"
            break
        if predicted_decrease <= 1e-12 * max(1.0, abs(objective)):
            base, correction = regularizer.quadratic_split(step)
            gram_min = (
                float(np.linalg.eigvalsh(regularizer.gram).min())
                if regularizer.gram is not None
                else float("nan")
            )
            print(
                f"{simulator_name} Laplace MAP stall: |J|={np.linalg.norm(local_jacobian):.3g}, "
                f"|J^T resid|={np.linalg.norm(local_jacobian.T @ (residual / variances)):.3g}, "
                f"|step|={np.linalg.norm(step):.3g}, "
                f"reg_C0={base:.3g}, reg_woodbury={correction:.3g}, "
                f"gram_min_eig={gram_min:.3g}, predicted_decrease={predicted_decrease:.3g}"
            )
            reason = "no predicted decrease"
            break

        alpha = 1.0
        accepted = None
        best = None
        for _ in range(MAX_BACKTRACKS):
            candidate = current + alpha * step
            candidate_objective, candidate_chi2 = evaluate(candidate)
            if np.isfinite(candidate_objective):
                if candidate_objective <= (
                    objective - ARMIJO_CONSTANT * alpha * predicted_decrease
                ):
                    accepted = (candidate, candidate_objective, candidate_chi2, alpha)
                    break
                if best is None or candidate_objective < best[1]:
                    best = (candidate, candidate_objective, candidate_chi2, alpha)
            alpha *= BACKTRACKING_RATIO
        if accepted is None and best is not None and best[1] < objective:
            accepted = best
        if accepted is None:
            best_objective = best[1] if best is not None else float("nan")
            print(
                f"{simulator_name} Laplace MAP line search failed: "
                f"objective={objective:.6g}, predicted_decrease={predicted_decrease:.6g}, "
                f"best_objective={best_objective:.6g}, |step|={np.linalg.norm(step):.6g}"
            )
            reason = "line search failed"
            break

        candidate, candidate_objective, candidate_chi2, alpha = accepted
        relative_step = (
            alpha * np.linalg.norm(step) / max(1.0, np.linalg.norm(current))
        )
        relative_drop = (objective - candidate_objective) / max(1.0, abs(objective))
        current, objective, chi2 = candidate, candidate_objective, candidate_chi2
        alphas.append(float(alpha))
        print(
            f"{simulator_name} Laplace MAP iter {iteration}: alpha={alpha:.3g}, "
            f"chi2={chi2:.4g}/{observed.size}, objective={objective:.4g}"
        )
        if relative_drop < OBJECTIVE_TOLERANCE or relative_step < STEP_TOLERANCE:
            converged = True
            reason = "tolerance"
            break

    info = {
        "iterations": len(alphas),
        "alphas": alphas,
        "chi2": chi2,
        "objective": objective,
        "converged": converged,
        "reason": reason,
    }
    return current, info


def sample_inverse_hessian(
    state, map_state, prior, column, local_jacobian, variances, rng,
    reduction=None, minimum_column=0,
):
    gain = _gain(
        prior, column, local_jacobian, variances, reduction, minimum_column
    )
    deviations = _state_deviations(state)
    noise = np.sqrt(variances)[:, None] * rng.standard_normal(
        (variances.size, deviations.shape[1])
    )
    posterior_draws = deviations - gain @ (
        local_jacobian @ deviations[active_column_indices(column), :] + noise
    )
    posterior_draws -= posterior_draws.mean(axis=1, keepdims=True)
    return map_state[:, None] + posterior_draws


def update_reduction(
    prior, column, local_jacobian, variances, reduction=None, minimum_column=0
):
    """Low-rank factor of C_post = C_prior - F F^T for the next drill step."""
    cross = prior.column_covariance(column, reduction, minimum_column)
    active_block = prior.active_block(column, reduction)
    innovation = (
        local_jacobian @ active_block @ local_jacobian.T + np.diag(variances)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(innovation)
    floor = 1e-8 * float(eigenvalues.max())
    root_inverse = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T
    factor = cross @ local_jacobian.T @ root_inverse
    combined = factor if reduction is None else np.hstack([reduction, factor])
    combined[behind_bit_indices(minimum_column), :] = 0.0
    return combined


def _relaxed_facies(state):
    """Continuous sigmoid approximation of the truncated-Gaussian facies map."""
    fields = state.reshape(-1, GRID_HEIGHT, GRID_WIDTH)
    lower = -0.4307273
    upper = 0.4307273
    low = torch.sigmoid((lower - fields) / RELAXATION_TEMPERATURE)
    high = torch.sigmoid((fields - upper) / RELAXATION_TEMPERATURE)
    middle = 1.0 - low - high
    facies = torch.stack((low, middle, high), dim=1)
    auxiliary = torch.zeros(
        (state.shape[0], 3, GRID_HEIGHT, GRID_WIDTH),
        dtype=state.dtype,
        device=state.device,
    )
    return torch.cat((facies, auxiliary), dim=1)


def _one_d_columns(simulator, position, map_state, device):
    base = torch.as_tensor(map_state, dtype=torch.float32, device=device).reshape(
        GRID_HEIGHT, GRID_WIDTH
    )
    row, column = position
    index_vector = torch.full(
        (1, column + 1), row, dtype=torch.long, device=device
    )
    return base, column, index_vector


def _extract_one_d_prediction(simulator, logs, column):
    tool_indices = [simulator.tool_configs.index(key) for key in simulator.all_data_types]
    return logs[tool_indices].flatten()


def one_d_prediction(simulator, map_state, position, smoothed):
    model = simulator.NNmodel
    row, column = position
    if not (0 <= row < GRID_HEIGHT and 0 <= column < GRID_WIDTH):
        raise ValueError(f"Bit position {position} is outside the pixel grid")

    with torch.no_grad():
        base, _, index_vector = _one_d_columns(simulator, position, map_state, model.rh_mult.device)
        if smoothed:
            facies = _relaxed_facies(base.reshape(1, -1))
        else:
            facies = pixel_state_to_facies(base.reshape(1, -1))
        logs = model.forward_from_facies(facies, index_vector)[0, column]
        prediction_tensor = _extract_one_d_prediction(simulator, logs, column)
    return prediction_tensor.detach().cpu().numpy().astype(float)


def one_d_prediction_and_jacobian(simulator, map_state, position):
    """Smoothed 1D prediction with a consistent autodiff column Jacobian."""
    row, column = position
    if not (0 <= row < GRID_HEIGHT and 0 <= column < GRID_WIDTH):
        raise ValueError(f"Bit position {position} is outside the pixel grid")

    model = simulator.NNmodel
    base, _, index_vector = _one_d_columns(simulator, position, map_state, model.rh_mult.device)
    profile = base[:, column].detach().clone().requires_grad_(True)
    full_state = base.clone()
    full_state[:, column] = profile
    facies = _relaxed_facies(full_state.reshape(1, -1))
    logs = model.forward_from_facies(facies, index_vector)[0, column]
    prediction_tensor = _extract_one_d_prediction(simulator, logs, column)

    rows = []
    for output_index in range(prediction_tensor.numel()):
        rows.append(
            torch.autograd.grad(
                prediction_tensor[output_index],
                profile,
                retain_graph=output_index < prediction_tensor.numel() - 1,
            )[0]
        )
    local_jacobian = torch.stack(rows).detach().cpu().numpy().astype(float)
    prediction = prediction_tensor.detach().cpu().numpy().astype(float)
    return prediction, local_jacobian


def zero_d_prediction_and_jacobian(simulator, map_state, position):
    return point_prediction_and_jacobian(simulator, map_state, position)


def _forecast_dictionary(predictions, data_types):
    forecast = {}
    offset = 0
    for data_type in data_types:
        width = measurement_width(data_type)
        forecast[data_type] = predictions[offset:offset + width]
        offset += width
    if offset != predictions.shape[0]:
        raise ValueError("Posterior prediction size does not match configured data types")
    return forecast


def save_posterior_outputs(state, predictions, data_types, output_dir="SaveOutputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "posterior_state_estimate.npz", x=state)
    forecast = np.asarray([_forecast_dictionary(predictions, data_types)], dtype=object)
    np.savez(output_dir / "posterior_forecast.npz", pred_data=forecast)


def _prediction_and_jacobian(simulator, simulator_name, state, position):
    if simulator_name == "0D":
        return zero_d_prediction_and_jacobian(simulator, state, position)
    if simulator_name == "1D":
        return one_d_prediction_and_jacobian(simulator, state, position)
    raise ValueError(f"Unknown Laplace simulator: {simulator_name}")


def assimilate_laplace(
    state,
    simulator,
    simulator_name,
    position,
    prior,
    reduction=None,
    data_path="../data/data.pkl",
    variance_path="../data/var.pkl",
    output_dir="SaveOutputs",
    seed=0,
    max_iterations=DEFAULT_MAX_GN_ITERATIONS,
):
    state = np.asarray(state, dtype=float)
    if state.shape[0] != PIXEL_COUNT:
        raise ValueError(
            f"Laplace assimilation only supports the {PIXEL_COUNT}-parameter pixel earth model"
        )

    simulator.update_bit_pos([position])
    observed, variances = load_observation(
        data_path, variance_path, simulator.all_data_types
    )
    _, column = position
    prior_mean = state.mean(axis=1)

    prior_prediction = _map_response(
        simulator, simulator_name, prior_mean, position, smoothed=False
    )
    if prior_prediction.shape != observed.shape:
        raise ValueError(
            f"Predicted and observed data sizes differ: {prior_prediction.size} and {observed.size}"
        )

    map_state, map_info = map_solution(
        simulator,
        simulator_name,
        position,
        prior,
        prior_mean,
        observed,
        variances,
        reduction=reduction,
        max_iterations=max_iterations,
    )
    if map_info["reason"] in {"line search failed", "no predicted decrease"}:
        fallback_prediction = run_predictions(simulator, state)
        save_posterior_outputs(
            state, fallback_prediction, simulator.all_data_types, output_dir
        )
        print(
            f"{simulator_name} Laplace MAP failed at {position}: "
            f"{map_info['reason']} after {map_info['iterations']} accepted step(s); "
            "keeping prior ensemble and covariance reduction"
        )
        return state.copy(), reduction
    map_prediction = _map_response(
        simulator, simulator_name, map_state, position, smoothed=False
    )
    _, map_jacobian = _prediction_and_jacobian(
        simulator, simulator_name, map_state, position
    )
    posterior = sample_inverse_hessian(
        state,
        map_state,
        prior,
        column,
        map_jacobian,
        variances,
        np.random.default_rng(seed),
        reduction,
        minimum_column=column,
    )
    posterior = preserve_behind_bit(state, posterior, column)
    new_reduction = update_reduction(
        prior,
        column,
        map_jacobian,
        variances,
        reduction,
        minimum_column=column,
    )
    posterior_prediction = run_predictions(simulator, posterior)
    save_posterior_outputs(
        posterior, posterior_prediction, simulator.all_data_types, output_dir
    )
    prior_chi2 = float(np.sum((prior_prediction - observed) ** 2 / variances))
    map_chi2 = float(np.sum((map_prediction - observed) ** 2 / variances))
    ensemble_prediction = posterior_prediction.mean(axis=1)
    posterior_chi2 = float(
        np.sum((ensemble_prediction - observed) ** 2 / variances)
    )
    print(
        f"{simulator_name} Laplace DA: {map_info['iterations']} GN iteration(s) "
        f"[{map_info['reason']}], "
        f"chi2 prior/MAP/ensemble {prior_chi2:.3g}/{map_chi2:.3g}/"
        f"{posterior_chi2:.3g}, MAP norm {np.linalg.norm(map_state):.3g}"
    )
    return posterior, new_reduction
