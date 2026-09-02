"""Classical Occam inversion of the full 2D pixel image while drilling.

Adapted from the sequential 1D Occam implementation in
Jacobian/RML-NN/occam_1d_nn.py. Each assimilation step linearizes the
forward model around the current image, solves the roughness-regularized
least-squares problem for a log grid of trade-off parameters mu, and keeps
the smoothest candidate whose TRUE misfit still fits the target
chi-square (Constable, Parker & Constable, 1987). The roughness operator is
the first-order difference along BOTH grid directions, so an update at the
bit spreads smoothly through the whole 2D image; a weak zeroth-order anchor
keeps the far field at the current reference image between drill steps.

The 0D stage uses the direct point-resistivity observation while 1D uses
the directional proxy response, so the two stages consume distinct data.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from wf_demo.laplace_assimilation import (
    VARIANCE_FLOOR,
    active_column_indices,
    load_observation,
    one_d_prediction,
    one_d_prediction_and_jacobian,
    run_predictions,
    save_posterior_outputs,
    zero_d_prediction_and_jacobian,
)
from wf_demo.pixel_model import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PIXEL_COUNT,
    active_pixel_indices,
    preserve_behind_bit,
)
from wf_demo.zero_d import point_prediction


DEFAULT_TARGET_VALUE = 4.0
DEFAULT_MAX_ITERATIONS = 8
ANCHOR_WEIGHT = 1e-2


@dataclass(frozen=True)
class OccamSettings:
    target_value: float = DEFAULT_TARGET_VALUE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    target_relative_tolerance: float = 0.02
    target_absolute_tolerance: float = 0.0
    model_tolerance: float = 1e-3
    chi2_improvement_tolerance: float = 1e-6
    mu_min: float = 1e-8
    mu_max: float = 1e12
    mu_grid_size: int = 25
    mu_refine_iterations: int = 8
    bound_sigmas: float = 3.0
    trust_sigmas: float = 2.0
    max_bound_saturation_fraction: float = 0.25
    backtrack_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class OccamStepResult:
    model: np.ndarray
    chi2: float
    target_chi2: float
    roughness: float
    mu: float
    iterations: int
    converged: bool
    target_reached: bool
    status: str
    history: tuple[dict, ...]


@dataclass(frozen=True)
class _MuCandidate:
    mu: float
    model: np.ndarray
    chi2: float
    roughness: float


@dataclass(frozen=True)
class _MuSearchResult:
    selected: _MuCandidate | None
    target_reached: bool
    candidates: tuple[_MuCandidate, ...]
    stalled: bool


@dataclass(frozen=True)
class _AcceptedStep:
    model: np.ndarray
    chi2: float
    roughness: float
    alpha: float
    step_norm: float


def validate_occam_settings(settings: OccamSettings) -> OccamSettings:
    if not np.isfinite(settings.target_value) or settings.target_value <= 0:
        raise ValueError("Occam target_value must be finite and positive.")
    if settings.max_iterations <= 0 or settings.model_tolerance <= 0:
        raise ValueError("Occam max_iterations and model_tolerance must be positive.")
    if settings.target_relative_tolerance < 0 or settings.target_absolute_tolerance < 0:
        raise ValueError("Occam target tolerances must be non-negative.")
    if settings.mu_min <= 0 or settings.mu_max <= settings.mu_min:
        raise ValueError("Occam mu_min and mu_max must satisfy 0 < mu_min < mu_max.")
    if settings.mu_grid_size < 3:
        raise ValueError("Occam mu_grid_size must be at least 3.")
    if settings.bound_sigmas <= 0 or settings.trust_sigmas <= 0:
        raise ValueError("Occam bound_sigmas and trust_sigmas must be positive.")
    if not 0.0 <= settings.max_bound_saturation_fraction <= 1.0:
        raise ValueError("Occam max_bound_saturation_fraction must be in [0, 1].")
    if not settings.backtrack_alphas or any(alpha <= 0 for alpha in settings.backtrack_alphas):
        raise ValueError("Occam backtrack_alphas must be positive values.")
    return settings


def _target_upper_bound(target: float, settings: OccamSettings) -> float:
    return target * (1.0 + settings.target_relative_tolerance) + settings.target_absolute_tolerance


def _chi_square(prediction: np.ndarray, observed: np.ndarray, variances: np.ndarray) -> float:
    residual = np.asarray(prediction, dtype=float).ravel() - np.asarray(observed, dtype=float).ravel()
    return float(np.sum(residual * residual / np.asarray(variances, dtype=float).ravel()))


def _axis_laplacian(size: int) -> np.ndarray:
    """Gram of the first-difference operator along one grid axis."""
    laplacian = np.zeros((size, size), dtype=float)
    rows = np.arange(size - 1)
    laplacian[rows, rows] = 1.0
    laplacian[rows + 1, rows + 1] += 1.0
    laplacian[rows, rows + 1] = -1.0
    laplacian[rows + 1, rows] = -1.0
    return laplacian


_EIGEN_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _laplacian_eigen(size=GRID_HEIGHT):
    if size not in _EIGEN_CACHE:
        _EIGEN_CACHE[size] = np.linalg.eigh(_axis_laplacian(size))
    return _EIGEN_CACHE[size]


def _to_eigen(stack: np.ndarray) -> np.ndarray:
    _, row_vectors = _laplacian_eigen(stack.shape[0])
    _, column_vectors = _laplacian_eigen(stack.shape[1])
    working = np.einsum("ij,jwk->iwk", row_vectors.T, stack)
    return np.einsum("iwk,wl->ilk", working, column_vectors)


def _from_eigen(stack: np.ndarray) -> np.ndarray:
    _, row_vectors = _laplacian_eigen(stack.shape[0])
    _, column_vectors = _laplacian_eigen(stack.shape[1])
    working = np.einsum("ij,jwk->iwk", row_vectors, stack)
    return np.einsum("iwk,wl->ilk", working, column_vectors.T)


def _spectral_weights_denominator(shape, anchor: float) -> np.ndarray:
    row_values, _ = _laplacian_eigen(shape[0])
    column_values, _ = _laplacian_eigen(shape[1])
    return row_values[:, None] + column_values[None, :] + anchor


def _q_inverse_apply(stack: np.ndarray, anchor: float) -> np.ndarray:
    """Q^{-1} on (H, W, k) fields; Q = D^T D + anchor * I is the separable
    2D first-roughness Gram plus the zeroth-order anchor."""
    weights = 1.0 / _spectral_weights_denominator(stack.shape[:2], anchor)
    return _from_eigen(_to_eigen(stack) * weights[:, :, None])


def _a0_inverse_sqrt_apply(stack: np.ndarray, mu: float, anchor: float) -> np.ndarray:
    """(mu Q)^{-1/2} on (H, W, k) fields through the Kronecker eigenbasis."""
    weights = 1.0 / np.sqrt(
        mu * _spectral_weights_denominator(stack.shape[:2], anchor)
    )
    return _from_eigen(_to_eigen(stack) * weights[:, :, None])


def roughness(delta: np.ndarray, minimum_column=0) -> float:
    """Squared first-order roughness of a full-image deviation, both directions."""
    field = np.asarray(delta, dtype=float).reshape(GRID_HEIGHT, GRID_WIDTH)[
        :, int(minimum_column):
    ]
    return float(np.sum(np.diff(field, axis=0) ** 2) + np.sum(np.diff(field, axis=1) ** 2))


def scatter_column_jacobian(local_jacobian: np.ndarray, column: int) -> np.ndarray:
    """Embed an active-column Jacobian (n_data, GRID_HEIGHT) into the full grid."""
    local = np.asarray(local_jacobian, dtype=float)
    full = np.zeros((local.shape[0], PIXEL_COUNT), dtype=float)
    full[:, active_column_indices(column)] = local
    return full


class LinearizedOccamSolutions:
    """Exact mu-parameterized solutions of the linearized Occam problem.

    For Q = D^T D + anchor * I and U = J^T W^{1/2}, solves
        min_m ||w - U^T m||^2 + mu (m - m_ref)^T Q (m - m_ref)
    in closed form as m(mu) = m_ref + G (mu I + S)^{-1} (w - r) with
    G = Q^{-1} U, S = U^T G, w = W^{1/2} d_lin and r = U^T m_ref. The
    expression is stable for every mu >= 0 (mu -> 0 gives the minimum
    Q-norm data fit around m_ref, mu -> infinity returns m_ref) and Q is
    applied through the Kronecker eigenbasis, so no dense
    PIXEL_COUNT x PIXEL_COUNT matrix is ever formed.
    """

    def __init__(
        self, jacobian, variances, linearized_data, m_ref, anchor, minimum_column=0
    ):
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.ndim != 2 or jacobian.shape[1] != PIXEL_COUNT:
            raise ValueError(
                f"Expected a (n_data, {PIXEL_COUNT}) full-grid Jacobian; got {jacobian.shape}"
            )
        roots = 1.0 / np.sqrt(np.maximum(np.asarray(variances, dtype=float), VARIANCE_FLOOR))
        self.m_ref = np.asarray(m_ref, dtype=float).ravel().copy()
        self.active = active_pixel_indices(minimum_column)
        width = GRID_WIDTH - int(minimum_column)
        self.u = jacobian[:, self.active].T * roots[None, :]
        self.g = _q_inverse_apply(
            self.u.reshape(GRID_HEIGHT, width, -1), anchor
        ).reshape(self.u.shape)
        gram = self.u.T @ self.g
        self.s = 0.5 * (gram + gram.T)
        self.w = np.asarray(linearized_data, dtype=float).ravel() * roots
        self.r = self.u.T @ self.m_ref[self.active]

    def solve(self, mu: float) -> np.ndarray | None:
        matrix = mu * np.eye(self.s.shape[0]) + self.s
        try:
            correction = np.linalg.solve(matrix, self.w - self.r)
        except np.linalg.LinAlgError:
            return None
        model = self.m_ref.copy()
        model[self.active] += self.g @ correction
        return model if np.all(np.isfinite(model)) else None


def _bound_saturation_fraction(model, lower, upper, tolerance=1e-9) -> float:
    at_bound = (model <= lower + tolerance) | (model >= upper - tolerance)
    return float(np.mean(at_bound))


def _search_mu(
    current_model: np.ndarray,
    current_chi2: float,
    target: float,
    settings: OccamSettings,
    solve_for_mu: Callable[[float], np.ndarray | None],
    evaluate_model: Callable[[np.ndarray], tuple[float, float]],
    apply_bounds: Callable[[np.ndarray], np.ndarray],
) -> _MuSearchResult:
    """Select the true-misfit candidate, preferring the smoothest target-fitting model."""
    candidates: list[_MuCandidate] = []

    def evaluate(mu: float) -> None:
        solved = solve_for_mu(mu)
        if solved is None:
            return
        model = apply_bounds(solved)
        chi2, model_roughness = evaluate_model(model)
        if np.isfinite(chi2) and np.isfinite(model_roughness):
            candidates.append(_MuCandidate(float(mu), model, chi2, model_roughness))

    evaluate(0.0)
    for mu in np.geomspace(settings.mu_min, settings.mu_max, num=settings.mu_grid_size):
        evaluate(float(mu))

    upper = _target_upper_bound(target, settings)
    target_candidates = [item for item in candidates if item.chi2 <= upper]
    if target_candidates:
        anchor_mu = max(target_candidates, key=lambda item: item.mu).mu
    elif candidates:
        anchor_mu = min(candidates, key=lambda item: item.chi2).mu
    else:
        anchor_mu = None
    if anchor_mu is not None and anchor_mu > 0.0 and settings.mu_refine_iterations:
        lo = max(settings.mu_min, anchor_mu / 10.0)
        hi = min(settings.mu_max, anchor_mu * 10.0)
        for mu in np.geomspace(lo, hi, num=max(3, settings.mu_refine_iterations * 4)):
            evaluate(float(mu))

    if np.isfinite(current_chi2) and current_chi2 <= upper:
        current_roughness = evaluate_model(current_model)[1]
        if np.isfinite(current_roughness):
            candidates.append(
                _MuCandidate(0.0, current_model.copy(), current_chi2, current_roughness)
            )

    target_candidates = [item for item in candidates if item.chi2 <= upper]
    if target_candidates:
        selected = min(target_candidates, key=lambda item: (item.roughness, -item.mu))
        if np.linalg.norm(selected.model - current_model) <= 0.0:
            selected = None
        return _MuSearchResult(selected, True, tuple(candidates), selected is None)

    if not candidates:
        return _MuSearchResult(None, False, (), True)
    selected = min(candidates, key=lambda item: item.chi2)
    stalled = selected.chi2 >= current_chi2 * (1.0 - settings.chi2_improvement_tolerance)
    return _MuSearchResult(selected, False, tuple(candidates), stalled)


def _accept_step(
    current_model: np.ndarray,
    current_chi2: float,
    target_upper: float,
    step: np.ndarray,
    evaluate_model: Callable[[np.ndarray], tuple[float, float]],
    settings: OccamSettings,
    lower: np.ndarray,
    upper: np.ndarray,
    step_limit: float,
) -> _AcceptedStep | None:
    """Trust-region cap + bound-saturation guard + backtracking line search.

    A trial is accepted only if its TRUE chi2 fits the target or strictly
    improves on the current model, and its bound saturation stays below the
    configured fraction; otherwise the step is halved.
    """
    step = np.asarray(step, dtype=float)
    largest_move = float(np.max(np.abs(step)))
    if largest_move > step_limit:
        step = step * (step_limit / largest_move)
    if not np.any(step):
        return None

    for alpha in settings.backtrack_alphas:
        trial = np.clip(current_model + alpha * step, lower, upper)
        chi2, model_roughness = evaluate_model(trial)
        if not (np.isfinite(chi2) and np.isfinite(model_roughness)):
            continue
        if _bound_saturation_fraction(trial, lower, upper) > settings.max_bound_saturation_fraction:
            continue
        improving = chi2 < current_chi2 * (1.0 - settings.chi2_improvement_tolerance)
        if chi2 <= target_upper or improving:
            return _AcceptedStep(
                model=trial,
                chi2=chi2,
                roughness=model_roughness,
                alpha=float(alpha),
                step_norm=float(np.max(np.abs(alpha * step))),
            )
    return None


def solve_occam_step(
    initial_model: np.ndarray,
    reference_model: np.ndarray,
    observed: np.ndarray,
    variances: np.ndarray,
    prediction_and_jacobian: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    evaluate_model: Callable[[np.ndarray], tuple[float, float]],
    settings: OccamSettings,
    sigma_prior: float,
    anchor: float = ANCHOR_WEIGHT,
    minimum_column=0,
) -> OccamStepResult:
    """One Occam solve of the full 2D image against a single measurement.

    Iterated linearization with a mu-grid search per iteration, exactly as
    solve_occam_row in occam_1d_nn.py: linearize at the current model, solve
    the roughness-regularized problem for every candidate mu, keep the
    smoothest candidate whose true misfit fits the target chi-square, and
    backtrack on the true misfit. Roughness is measured against
    reference_model (the carried image), in both grid directions.
    """
    settings = validate_occam_settings(settings)
    reference = np.asarray(reference_model, dtype=float).ravel()
    observed = np.asarray(observed, dtype=float).ravel()
    variances = np.asarray(variances, dtype=float).ravel()
    lower = reference - settings.bound_sigmas * sigma_prior
    upper = reference + settings.bound_sigmas * sigma_prior
    model = np.clip(np.asarray(initial_model, dtype=float).copy(), lower, upper)
    step_limit = settings.trust_sigmas * float(sigma_prior)
    target = settings.target_value * observed.size
    target_upper = _target_upper_bound(target, settings)
    current_chi2, current_roughness = evaluate_model(model)
    history: list[dict] = []
    last_mu = 0.0
    converged = False
    status = "max_iterations"

    for iteration in range(1, settings.max_iterations + 1):
        prediction, jacobian = prediction_and_jacobian(model)
        linearized_data = observed - prediction + jacobian @ model
        solutions = LinearizedOccamSolutions(
            jacobian,
            variances,
            linearized_data,
            reference,
            anchor,
            minimum_column,
        )
        result = _search_mu(
            model,
            current_chi2,
            target,
            settings,
            solutions.solve,
            evaluate_model,
            lambda candidate: np.clip(candidate, lower, upper),
        )
        if result.selected is None:
            status = "target_reached" if current_chi2 <= target_upper else "stalled"
            converged = current_chi2 <= target_upper
            break

        chosen = result.selected
        accepted = _accept_step(
            model,
            current_chi2,
            target_upper,
            chosen.model - model,
            evaluate_model,
            settings,
            lower,
            upper,
            step_limit,
        )
        if accepted is None:
            status = "target_reached" if current_chi2 <= target_upper else "stalled"
            converged = current_chi2 <= target_upper
            break

        step_norm = accepted.step_norm
        relative_step = step_norm / max(1.0, float(np.linalg.norm(model)))
        model = accepted.model.copy()
        current_chi2 = accepted.chi2
        current_roughness = accepted.roughness
        last_mu = chosen.mu
        history.append({
            "iteration": iteration,
            "chi2": current_chi2,
            "target_chi2": target,
            "roughness": current_roughness,
            "mu": last_mu,
            "backtrack_alpha": accepted.alpha,
            "step_norm": step_norm,
            "relative_step": relative_step,
            "target_reached": result.target_reached,
            "candidate_count": len(result.candidates),
        })
        print(
            f"Occam iter {iteration}: mu={last_mu:.4g}, alpha={accepted.alpha:.3g}, "
            f"chi2={current_chi2:.4g}/{target:.4g}, roughness={current_roughness:.4g}"
        )

        if result.target_reached and relative_step <= settings.model_tolerance:
            status = "converged"
            converged = True
            break

    target_reached = current_chi2 <= target_upper
    if target_reached and status == "max_iterations":
        status = "target_reached_max_iterations"
    return OccamStepResult(
        model=model,
        chi2=current_chi2,
        target_chi2=target,
        roughness=current_roughness,
        mu=last_mu,
        iterations=len(history),
        converged=converged,
        target_reached=target_reached,
        status=status,
        history=tuple(history),
    )


def sample_occam_posterior(
    map_model: np.ndarray,
    jacobian: np.ndarray,
    variances: np.ndarray,
    mu: float,
    sigma_prior: float,
    count: int,
    rng: np.random.Generator,
    anchor: float = ANCHOR_WEIGHT,
    minimum_column=0,
) -> np.ndarray:
    """Exact draws from the linearized Occam posterior N(map, H^{-1}).

    H = J^T W J + mu Q is the Occam Hessian at the solution (the matrix
    behind linearized_posterior_std in occam_1d_nn.py). With A0 = mu Q,
    B = A0^{-1/2} U and f(x) = (1 + x)^{-1/2}, exact draws are
    A0^{-1/2} f(B B^T) z = A0^{-1/2} [z + B h(B^T B) B^T z] for
    h(x) = (f(x) - 1) / x, applied through the n_data-sized eigenbasis of
    B^T B. mu is floored at 1 / (anchor * sigma_prior^2) so the far-field
    standard deviation stays within one prior sigma; a stalled solve
    carrying mu = 0 would otherwise produce variances of order
    1 / (mu * anchor). Draws are centered, so the ensemble mean is exactly
    the Occam model.
    """
    mu = max(float(mu), 1.0 / (anchor * float(sigma_prior) ** 2))
    roots = 1.0 / np.sqrt(np.maximum(np.asarray(variances, dtype=float), VARIANCE_FLOOR))
    active = active_pixel_indices(minimum_column)
    width = GRID_WIDTH - int(minimum_column)
    u = np.asarray(jacobian, dtype=float)[:, active].T * roots[None, :]
    b = _a0_inverse_sqrt_apply(
        u.reshape(GRID_HEIGHT, width, -1), mu, anchor
    ).reshape(u.shape)
    gram = b.T @ b
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    positive = np.maximum(eigenvalues, 0.0)
    weights = np.where(
        positive > 1e-12,
        (1.0 / np.sqrt(1.0 + positive) - 1.0) / np.where(positive > 1e-12, positive, 1.0),
        -0.5,
    )
    z = rng.standard_normal((active.size, int(count)))
    projected = eigenvectors.T @ (b.T @ z)
    white = z + (b @ eigenvectors) @ (weights[:, None] * projected)
    draws = _a0_inverse_sqrt_apply(
        white.reshape(GRID_HEIGHT, width, -1), mu, anchor
    ).reshape(white.shape)
    draws -= draws.mean(axis=1, keepdims=True)
    posterior = np.repeat(
        np.asarray(map_model, dtype=float)[:, None], int(count), axis=1
    )
    posterior[active, :] += draws
    return posterior


def assimilate_occam(
    state,
    simulator,
    simulator_name,
    position,
    prior_config,
    data_path="../data/data.pkl",
    variance_path="../data/var.pkl",
    output_dir="SaveOutputs",
    seed=0,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    target_value=DEFAULT_TARGET_VALUE,
    anchor=ANCHOR_WEIGHT,
):
    """Occam data assimilation of one measurement against the full 2D image.

    The reference model is the incoming ensemble mean, so the update is the
    smoothest full-image change that fits the measurement; the returned
    ensemble consists of centered linearized-posterior draws around the
    Occam model, keeping the mean exactly at the MAP image.
    """
    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape[0] != PIXEL_COUNT:
        raise ValueError(
            f"Occam assimilation only supports the {PIXEL_COUNT}-parameter pixel earth model"
        )

    simulator.update_bit_pos([position])
    observed, variances = load_observation(
        data_path, variance_path, simulator.all_data_types
    )
    _, column = position
    sigma_prior = float(prior_config["prior_standard_deviation"])
    reference_model = state.mean(axis=1)
    settings = validate_occam_settings(
        OccamSettings(target_value=target_value, max_iterations=max_iterations)
    )

    def response(model):
        if simulator_name == "0D":
            return point_prediction(simulator, model, position, smoothed=True)
        if simulator_name == "1D":
            return one_d_prediction(simulator, model, position, smoothed=True)
        raise ValueError(f"Unknown Occam simulator: {simulator_name}")

    def evaluate_model(model):
        chi2 = _chi_square(response(model), observed, variances)
        return chi2, roughness(model - reference_model, column)

    def prediction_and_jacobian(model):
        if simulator_name == "0D":
            prediction, local = zero_d_prediction_and_jacobian(simulator, model, position)
        elif simulator_name == "1D":
            prediction, local = one_d_prediction_and_jacobian(simulator, model, position)
        else:
            raise ValueError(f"Unknown Occam simulator: {simulator_name}")
        return prediction, scatter_column_jacobian(local, column)

    result = solve_occam_step(
        reference_model,
        reference_model,
        observed,
        variances,
        prediction_and_jacobian,
        evaluate_model,
        settings,
        sigma_prior,
        anchor,
        column,
    )
    if result.iterations == 0 and not result.target_reached:
        fallback_prediction = run_predictions(simulator, state)
        save_posterior_outputs(
            state, fallback_prediction, simulator.all_data_types, output_dir
        )
        print(
            f"{simulator_name} Occam inversion failed at {position}: {result.status} "
            "after 0 accepted step(s); keeping prior ensemble"
        )
        return state.copy()

    prior_chi2 = _chi_square(response(reference_model), observed, variances)
    _, final_jacobian = prediction_and_jacobian(result.model)
    posterior = sample_occam_posterior(
        result.model,
        final_jacobian,
        variances,
        result.mu,
        sigma_prior,
        state.shape[1],
        np.random.default_rng(seed),
        anchor,
        column,
    )
    posterior = preserve_behind_bit(state, posterior, column)
    posterior_prediction = run_predictions(simulator, posterior)
    save_posterior_outputs(
        posterior, posterior_prediction, simulator.all_data_types, output_dir
    )
    occam_chi2 = _chi_square(response(result.model), observed, variances)
    ensemble_chi2 = _chi_square(posterior_prediction.mean(axis=1), observed, variances)
    print(
        f"{simulator_name} Occam DA at column {column}: {result.iterations} iteration(s) "
        f"[{result.status}], mu={result.mu:.4g}, target={result.target_chi2:.3g}, "
        f"chi2 prior/Occam/ensemble {prior_chi2:.3g}/{occam_chi2:.3g}/{ensemble_chi2:.3g}"
    )
    return posterior
