import numpy as np
import torch
import torch.nn.functional as F

from geostat.decomp import Cholesky
from geostat.gaussian_sim import fast_gaussian


GRID_HEIGHT = 64
GRID_WIDTH = 64
PIXEL_COUNT = GRID_HEIGHT * GRID_WIDTH
COVARIANCE_MODELS = {
    "Gaussian": "gau",
    "Spherical": "sph",
    "Exponential": "exp",
    "Cubic": "cub",
}


def active_pixel_indices(bit_column):
    """Flattened pixel indices at and ahead of the current drill bit."""
    column = int(bit_column)
    if not 0 <= column < GRID_WIDTH:
        raise ValueError(f"Bit column {column} is outside the pixel grid")
    columns = np.arange(column, GRID_WIDTH)
    return (np.arange(GRID_HEIGHT)[:, None] * GRID_WIDTH + columns).ravel()


def behind_bit_indices(bit_column):
    """Flattened pixel indices in columns already passed by the drill bit."""
    column = int(bit_column)
    if not 0 <= column < GRID_WIDTH:
        raise ValueError(f"Bit column {column} is outside the pixel grid")
    columns = np.arange(column)
    return (np.arange(GRID_HEIGHT)[:, None] * GRID_WIDTH + columns).ravel()


def preserve_behind_bit(previous_state, updated_state, bit_column):
    """Return an update whose already-drilled columns exactly match its input."""
    previous = np.asarray(previous_state)
    updated = np.asarray(updated_state).copy()
    if previous.shape != updated.shape or previous.ndim != 2:
        raise ValueError(
            "Previous and updated pixel states must be matching two-dimensional arrays"
        )
    if previous.shape[0] != PIXEL_COUNT:
        raise ValueError(
            f"Pixel states must contain {PIXEL_COUNT} rows; got {previous.shape[0]}"
        )
    frozen = behind_bit_indices(bit_column)
    updated[frozen, :] = previous[frozen, :]
    return updated


def gan_facies_to_pixel_state(facies):
    """Convert GAN facies realizations to the thresholded pixel parameterization.

    Class representatives -1, 0 and 1 lie safely inside the three intervals
    used by pixel_state_to_facies, so converting the state back to facies
    reproduces the GAN's dominant class exactly.
    """
    if facies.ndim != 4 or facies.shape[1] < 3:
        raise ValueError(
            "GAN facies must have shape (ensemble, at least 3, height, width)"
        )
    if tuple(facies.shape[2:]) != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"GAN facies grid must be {(GRID_HEIGHT, GRID_WIDTH)}; got {tuple(facies.shape[2:])}"
        )
    classes = facies[:, :3].argmax(dim=1)
    representatives = torch.tensor(
        [-1.0, 0.0, 1.0], dtype=facies.dtype, device=facies.device
    )
    fields = representatives[classes]
    return fields.reshape(facies.shape[0], PIXEL_COUNT).T.detach().cpu().numpy()


def sample_pixel_prior(
    ensemble_size,
    covariance_model,
    horizontal_correlation,
    vertical_correlation,
    prior_mean=0.0,
    prior_standard_deviation=1.0,
    seed=0,
):
    """Sample flattened standard-Gaussian fields as columns of a PET state matrix."""
    if covariance_model not in COVARIANCE_MODELS:
        raise ValueError(f"Unsupported covariance model: {covariance_model}")
    if ensemble_size < 1:
        raise ValueError("Ensemble size must be positive")
    if horizontal_correlation <= 0 or vertical_correlation <= 0:
        raise ValueError("Correlation lengths must be positive")
    if prior_standard_deviation <= 0:
        raise ValueError("Prior standard deviation must be positive")

    random_state = np.random.get_state()
    np.random.seed(seed)
    try:
        if covariance_model == "Gaussian":
            standard_prior = fast_gaussian(
                dimension=np.array([GRID_HEIGHT, GRID_WIDTH]),
                sdev=np.array([1.0]),
                corr=np.array([vertical_correlation, horizontal_correlation]),
                num_samples=ensemble_size,
            )
            fields = standard_prior.T.reshape(
                ensemble_size, GRID_WIDTH, GRID_HEIGHT
            ).transpose(0, 2, 1)
            standard_prior = fields.reshape(ensemble_size, PIXEL_COUNT).T
        else:
            standard_prior = _sample_separable_prior(
                ensemble_size,
                COVARIANCE_MODELS[covariance_model],
                horizontal_correlation,
                vertical_correlation,
            )
        return prior_mean + prior_standard_deviation * standard_prior
    finally:
        np.random.set_state(random_state)


def _sample_separable_prior(
    ensemble_size,
    covariance_model,
    horizontal_correlation,
    vertical_correlation,
):
    sampler = Cholesky()
    vertical_covariance = _axis_covariance(
        sampler, GRID_HEIGHT, vertical_correlation, covariance_model
    )
    horizontal_covariance = _axis_covariance(
        sampler, GRID_WIDTH, horizontal_correlation, covariance_model
    )

    vertical_samples = sampler.gen_real(
        np.zeros(GRID_HEIGHT),
        vertical_covariance,
        GRID_WIDTH * ensemble_size,
    )
    _, horizontal_cholesky = sampler.gen_real(
        np.zeros(GRID_WIDTH),
        horizontal_covariance,
        1,
        return_chol=True,
    )
    fields = vertical_samples.reshape(
        GRID_HEIGHT, ensemble_size, GRID_WIDTH
    ).transpose(0, 2, 1)
    fields = np.einsum("hwn,wj->hjn", fields, horizontal_cholesky)
    return fields.reshape(PIXEL_COUNT, ensemble_size)


def _axis_covariance(sampler, size, correlation_length, covariance_model):
    grid = np.arange(size, dtype=float)
    distance = np.abs(np.subtract.outer(grid, grid))
    covariance = sampler.variogram_model(
        distance,
        correlation_length,
        1.0,
        covariance_model,
    )
    return covariance + np.eye(size) * 1e-10


def pixel_prior_axis_covariances(covariance_model, horizontal_correlation, vertical_correlation):
    """Unit-variance analytic prior covariance along each grid axis.

    The Gaussian branch matches the exp(-(d/a)**2) convention of
    geostat.fast_gaussian used by sample_pixel_prior; the other models match
    Cholesky.variogram_model. Spherical and cubic are exactly zero beyond the
    range, Gaussian and exponential have infinite tails.
    """
    if covariance_model not in COVARIANCE_MODELS:
        raise ValueError(f"Unsupported covariance model: {covariance_model}")
    if horizontal_correlation <= 0 or vertical_correlation <= 0:
        raise ValueError("Correlation lengths must be positive")

    if covariance_model == "Gaussian":
        def axis_covariance(correlation_length, size):
            distance = np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
            return np.exp(-(distance / correlation_length) ** 2) + np.eye(size) * 1e-10

        vertical = axis_covariance(vertical_correlation, GRID_HEIGHT)
        horizontal = axis_covariance(horizontal_correlation, GRID_WIDTH)
    else:
        sampler = Cholesky()
        model_code = COVARIANCE_MODELS[covariance_model]
        vertical = _axis_covariance(sampler, GRID_HEIGHT, vertical_correlation, model_code)
        horizontal = _axis_covariance(sampler, GRID_WIDTH, horizontal_correlation, model_code)
    return vertical, horizontal


def pixel_state_to_facies(state):
    """Map flattened Gaussian fields to six-channel truncated-Gaussian facies."""
    if state.ndim != 2 or state.shape[1] != PIXEL_COUNT:
        raise ValueError(
            f"Pixel state must have shape (ensemble, {PIXEL_COUNT}); got {tuple(state.shape)}"
        )
    fields = state.reshape(-1, GRID_HEIGHT, GRID_WIDTH)
    lower_threshold = -0.4307273
    upper_threshold = 0.4307273
    classes = torch.zeros_like(fields, dtype=torch.long)
    classes[fields >= lower_threshold] = 1
    classes[fields >= upper_threshold] = 2
    facies = F.one_hot(classes, num_classes=3).permute(0, 3, 1, 2).to(state.dtype)
    auxiliary_channels = torch.zeros(
        (state.shape[0], 3, GRID_HEIGHT, GRID_WIDTH),
        dtype=state.dtype,
        device=state.device,
    )
    return torch.cat([facies, auxiliary_channels], dim=1)


def run_forward_model(nn_model, state, index_vector, earth_model_type):
    if earth_model_type == "pixel":
        facies = pixel_state_to_facies(state)
        return nn_model.forward_from_facies(facies, index_vector)
    return nn_model.forward(state, index_vector, output_transien_results=False)
