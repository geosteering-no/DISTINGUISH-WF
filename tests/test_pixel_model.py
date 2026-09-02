import numpy as np
import pytest
import torch

from wf_demo.pixel_model import (
    COVARIANCE_MODELS,
    GRID_HEIGHT,
    GRID_WIDTH,
    PIXEL_COUNT,
    active_pixel_indices,
    behind_bit_indices,
    gan_facies_to_pixel_state,
    pixel_prior_axis_covariances,
    pixel_state_to_facies,
    preserve_behind_bit,
    run_forward_model,
    sample_pixel_prior,
)


@pytest.mark.parametrize("covariance_model", COVARIANCE_MODELS)
def test_pixel_prior_matches_requested_ensemble_size(covariance_model):
    prior = sample_pixel_prior(
        ensemble_size=3,
        covariance_model=covariance_model,
        horizontal_correlation=12,
        vertical_correlation=3,
        seed=17,
    )

    assert prior.shape == (PIXEL_COUNT, 3)
    assert np.isfinite(prior).all()


def test_pixel_prior_is_deterministic_without_changing_global_rng():
    np.random.seed(11)
    expected_next_value = np.random.random()
    np.random.seed(11)

    first = sample_pixel_prior(2, "Gaussian", 8, 2, seed=31)
    next_value = np.random.random()
    second = sample_pixel_prior(2, "Gaussian", 8, 2, seed=31)

    np.testing.assert_array_equal(first, second)
    assert next_value == expected_next_value


def test_pixel_prior_applies_requested_mean_and_standard_deviation():
    standard = sample_pixel_prior(2, "Gaussian", 8, 2, seed=23)
    transformed = sample_pixel_prior(
        2,
        "Gaussian",
        8,
        2,
        prior_mean=7.5,
        prior_standard_deviation=2.25,
        seed=23,
    )

    np.testing.assert_allclose(transformed, 7.5 + 2.25 * standard)


def test_gaussian_prior_preserves_horizontal_and_vertical_correlation_directions():
    prior = sample_pixel_prior(
        100,
        "Gaussian",
        horizontal_correlation=16,
        vertical_correlation=2,
        seed=41,
    )
    fields = prior.T.reshape(-1, 64, 64)
    vertical_neighbor_correlation = np.corrcoef(
        fields[:, :-1, :].ravel(), fields[:, 1:, :].ravel()
    )[0, 1]
    horizontal_neighbor_correlation = np.corrcoef(
        fields[:, :, :-1].ravel(), fields[:, :, 1:].ravel()
    )[0, 1]

    assert horizontal_neighbor_correlation > vertical_neighbor_correlation + 0.1


def test_spherical_axis_covariance_is_correlated_and_range_limited():
    vertical, horizontal = pixel_prior_axis_covariances(
        "Spherical", horizontal_correlation=4.5, vertical_correlation=10.0
    )

    np.testing.assert_allclose(horizontal[30, 31], 1.0 - 1.5 / 4.5 + 0.5 / 4.5**3)
    assert horizontal[30, 26] > 0.0
    assert horizontal[30, 25] == 0.0
    assert vertical[5, 6] > 0.0


def test_separable_prior_sampling_produces_correlated_fields():
    prior = sample_pixel_prior(
        200,
        "Spherical",
        horizontal_correlation=20,
        vertical_correlation=3,
        seed=51,
    )
    fields = prior.T.reshape(-1, 64, 64)
    horizontal_neighbor_correlation = np.corrcoef(
        fields[:, :, :-1].ravel(), fields[:, :, 1:].ravel()
    )[0, 1]

    assert horizontal_neighbor_correlation > 0.5


def test_pixel_prior_rejects_nonpositive_standard_deviation():
    with pytest.raises(ValueError, match="standard deviation must be positive"):
        sample_pixel_prior(2, "Gaussian", 8, 2, prior_standard_deviation=0)


def test_pixel_state_maps_to_three_exclusive_facies_and_auxiliary_channels():
    state = torch.zeros((2, PIXEL_COUNT), dtype=torch.float32)
    state[0, 0] = -1
    state[0, 1] = 1

    facies = pixel_state_to_facies(state)

    assert facies.shape == (2, 6, 64, 64)
    torch.testing.assert_close(facies[:, :3].sum(dim=1), torch.ones((2, 64, 64)))
    assert torch.count_nonzero(facies[:, 3:]) == 0
    assert facies[0, 0, 0, 0] == 1
    assert facies[0, 2, 0, 1] == 1


def test_pixel_state_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Pixel state must have shape"):
        pixel_state_to_facies(torch.zeros((2, 60)))


def test_pixel_forward_bypasses_gan():
    class ForwardRecorder:
        def __init__(self):
            self.facies = None

        def forward(self, *args, **kwargs):
            raise AssertionError("GAN forward must not run for pixel model")

        def forward_from_facies(self, facies, index_vector):
            self.facies = facies
            return index_vector.float()

    model = ForwardRecorder()
    state = torch.zeros((2, PIXEL_COUNT))
    index_vector = torch.ones((2, 4), dtype=torch.long)

    result = run_forward_model(model, state, index_vector, "pixel")

    assert model.facies.shape == (2, 6, 64, 64)
    torch.testing.assert_close(result, index_vector.float())


def test_adaptive_pixel_domain_partitions_grid_at_bit_column():
    column = 17
    active = active_pixel_indices(column)
    behind = behind_bit_indices(column)

    assert active.size == GRID_HEIGHT * (GRID_WIDTH - column)
    assert behind.size == GRID_HEIGHT * column
    assert np.intersect1d(active, behind).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate([active, behind])), np.arange(PIXEL_COUNT)
    )


def test_preserve_behind_bit_restores_only_drilled_columns():
    previous = np.arange(PIXEL_COUNT * 3).reshape(PIXEL_COUNT, 3)
    updated = previous + 10000
    column = 12

    restricted = preserve_behind_bit(previous, updated, column)

    np.testing.assert_array_equal(restricted[behind_bit_indices(column)], previous[behind_bit_indices(column)])
    np.testing.assert_array_equal(restricted[active_pixel_indices(column)], updated[active_pixel_indices(column)])


def test_gan_facies_round_trip_through_pixel_state_preserves_classes():
    classes = torch.arange(PIXEL_COUNT * 2).reshape(2, GRID_HEIGHT, GRID_WIDTH) % 3
    facies = torch.nn.functional.one_hot(classes, num_classes=3).permute(0, 3, 1, 2).float()
    facies = torch.cat([facies, torch.zeros_like(facies)], dim=1)

    pixel_state = gan_facies_to_pixel_state(facies)
    reconstructed = pixel_state_to_facies(torch.as_tensor(pixel_state.T))

    assert pixel_state.shape == (PIXEL_COUNT, 2)
    torch.testing.assert_close(reconstructed[:, :3], facies[:, :3])
