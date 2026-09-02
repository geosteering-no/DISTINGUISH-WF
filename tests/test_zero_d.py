from types import SimpleNamespace

import numpy as np
import torch

from wf_demo.measurements import POINT_DATA_TYPE
from wf_demo.pixel_model import GRID_WIDTH, PIXEL_COUNT
from wf_demo.zero_d import (
    PointSimulator,
    direct_point_prediction,
    direct_point_prediction_and_jacobian,
    point_log_resistivity,
    zero_d_status,
)


def input_dict(earth_model_type):
    return {
        "bit_pos": [(3, 4)],
        "datatype": [("6kHz", "83ft")],
        "earth_model_type": earth_model_type,
        "reporttype": "pos",
        "parallel_internal": True,
    }


def test_direct_point_has_no_optional_runtime_dependency():
    assert zero_d_status() == (True, None)


def test_point_log_resistivity_returns_log_rh_and_log_rv():
    facies = torch.zeros((3, 6, 8, 8))
    facies[0, 0, 2, 5] = 1
    facies[1, 1, 2, 5] = 1
    facies[2, 2, 2, 5] = 1

    values = point_log_resistivity(facies, (2, 5)).numpy()

    np.testing.assert_allclose(
        values,
        np.log(np.array([[4.0, 12.0], [171.0, 171.0], [55.0, 85.0]])),
    )


def test_point_pixel_simulator_returns_two_value_pet_prediction():
    simulator = PointSimulator(input_dict("pixel"), earth_simulator=object())
    state = np.zeros((2, PIXEL_COUNT))
    state[1] = -1.0

    predictions = simulator.run_fwd_sim({"x": state}, member_i=None)

    assert simulator.all_data_types == [POINT_DATA_TYPE]
    assert len(predictions) == 2
    np.testing.assert_allclose(
        predictions[0][0][POINT_DATA_TYPE], np.log([171.0, 171.0])
    )
    np.testing.assert_allclose(
        predictions[1][0][POINT_DATA_TYPE], np.log([4.0, 12.0])
    )


def test_point_gan_simulator_uses_same_local_facies_mapping():
    facies = torch.zeros((1, 6, 64, 64))
    facies[:, 2] = 1

    class FakeEvaluator:
        device = torch.device("cpu")

        def eval(self, state, no_grad):
            assert state.shape == (1, 60)
            assert no_grad is True
            return facies

    earth = SimpleNamespace(NNmodel=SimpleNamespace(gan_evaluator=FakeEvaluator()))
    simulator = PointSimulator(input_dict("gan"), earth_simulator=earth)

    predictions = simulator.run_fwd_sim({"x": np.zeros((1, 60))}, member_i=None)

    np.testing.assert_allclose(
        predictions[0][0][POINT_DATA_TYPE], np.log([55.0, 85.0])
    )


def test_point_sensitivity_is_exact_and_only_touches_bit_cell():
    position = (11, 7)
    state = np.zeros(PIXEL_COUNT)
    prediction, jacobian = direct_point_prediction_and_jacobian(state, position)
    step = 1e-5
    plus = state.copy()
    minus = state.copy()
    index = position[0] * GRID_WIDTH + position[1]
    plus[index] += step
    minus[index] -= step
    finite_difference = (
        direct_point_prediction(plus, position, smoothed=True)
        - direct_point_prediction(minus, position, smoothed=True)
    ) / (2.0 * step)

    np.testing.assert_allclose(prediction, direct_point_prediction(state, position))
    np.testing.assert_allclose(jacobian[:, position[0]], finite_difference, rtol=1e-7)
    assert np.count_nonzero(np.delete(jacobian, position[0], axis=1)) == 0
