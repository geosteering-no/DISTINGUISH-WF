import numpy as np
import torch

from GeoSim.sim import GeoSim

from wf_demo.measurements import POINT_DATA_TYPE
from wf_demo.pixel_model import GRID_HEIGHT, GRID_WIDTH, pixel_state_to_facies


RH_VALUES = (4.0, 171.0, 55.0)
RV_VALUES = (12.0, 171.0, 85.0)
LOWER_THRESHOLD = -0.4307273
UPPER_THRESHOLD = 0.4307273
RELAXATION_TEMPERATURE = 0.2


def zero_d_status():
    """Direct point data has no optional compiled dependencies."""
    return True, None


def point_log_resistivity(facies, position):
    """Hard local [ln Rh, ln Rv] observation for each facies realization."""
    if facies.ndim != 4 or facies.shape[1] < 3:
        raise ValueError(
            f"Expected facies shape (ensemble, channels, rows, columns); got {tuple(facies.shape)}"
        )
    row, column = position
    if not (0 <= row < facies.shape[2] and 0 <= column < facies.shape[3]):
        raise ValueError(
            f"Bit position {position} is outside facies grid {tuple(facies.shape[2:])}"
        )
    classes = facies[:, :3].argmax(dim=1)[:, row, column]
    rh = torch.as_tensor(RH_VALUES, dtype=facies.dtype, device=facies.device)[classes]
    rv = torch.as_tensor(RV_VALUES, dtype=facies.dtype, device=facies.device)[classes]
    return torch.stack((torch.log(rh), torch.log(rv)), dim=1)


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def direct_point_prediction(map_state, position, smoothed=True):
    """Direct pixel-state prediction [ln Rh, ln Rv] at the drill bit."""
    row, column = position
    if not (0 <= row < GRID_HEIGHT and 0 <= column < GRID_WIDTH):
        raise ValueError(f"Bit position {position} is outside the pixel grid")
    value = float(np.asarray(map_state)[row * GRID_WIDTH + column])
    if smoothed:
        low = _sigmoid((LOWER_THRESHOLD - value) / RELAXATION_TEMPERATURE)
        high = _sigmoid((value - UPPER_THRESHOLD) / RELAXATION_TEMPERATURE)
        fractions = np.array([low, 1.0 - low - high, high])
        rh = fractions @ np.asarray(RH_VALUES)
        rv = fractions @ np.asarray(RV_VALUES)
    else:
        facies = 0 if value < LOWER_THRESHOLD else (2 if value >= UPPER_THRESHOLD else 1)
        rh = RH_VALUES[facies]
        rv = RV_VALUES[facies]
    return np.log(np.array([rh, rv], dtype=float))


def direct_point_prediction_and_jacobian(map_state, position):
    """Smoothed point response and its exact bit-cell sensitivity.

    The local Jacobian has GRID_HEIGHT columns to match the 1D column
    interface, but only the current bit row is nonzero.
    """
    row, column = position
    if not (0 <= row < GRID_HEIGHT and 0 <= column < GRID_WIDTH):
        raise ValueError(f"Bit position {position} is outside the pixel grid")
    value = float(np.asarray(map_state)[row * GRID_WIDTH + column])
    low = _sigmoid((LOWER_THRESHOLD - value) / RELAXATION_TEMPERATURE)
    high = _sigmoid((value - UPPER_THRESHOLD) / RELAXATION_TEMPERATURE)
    d_low = -low * (1.0 - low) / RELAXATION_TEMPERATURE
    d_high = high * (1.0 - high) / RELAXATION_TEMPERATURE
    fractions = np.array([low, 1.0 - low - high, high])
    derivatives = np.array([d_low, -d_low - d_high, d_high])
    rh_values = np.asarray(RH_VALUES)
    rv_values = np.asarray(RV_VALUES)
    rh = fractions @ rh_values
    rv = fractions @ rv_values
    prediction = np.log(np.array([rh, rv]))
    sensitivity = np.array([
        derivatives @ rh_values / rh,
        derivatives @ rv_values / rv,
    ])
    local_jacobian = np.zeros((2, GRID_HEIGHT), dtype=float)
    local_jacobian[:, row] = sensitivity
    return prediction, local_jacobian


def point_prediction(simulator, map_state, position, smoothed=True):
    """Use a simulator override when supplied, otherwise the direct point map."""
    if hasattr(simulator, "point_prediction"):
        return simulator.point_prediction(map_state, position, smoothed)
    return direct_point_prediction(map_state, position, smoothed)


def point_prediction_and_jacobian(simulator, map_state, position):
    if hasattr(simulator, "point_prediction_and_jacobian"):
        return simulator.point_prediction_and_jacobian(map_state, position)
    return direct_point_prediction_and_jacobian(map_state, position)


class PointSimulator:
    """PET simulator adapter for direct local [ln Rh, ln Rv] observations."""

    def __init__(self, input_dict, earth_simulator=None):
        self.input_dict = input_dict.copy()
        self.input_dict["datatype"] = [POINT_DATA_TYPE]
        self.input_dict["parallel_internal"] = True
        self.redund_sim = None
        self.bit_pos = list(self.input_dict["bit_pos"])
        self.earth_model_type = self.input_dict.get("earth_model_type", "gan")
        self.all_data_types = [POINT_DATA_TYPE]
        self.l_prim = [0]
        self._earth_simulator = earth_simulator or GeoSim(self.input_dict)

    def update_bit_pos(self, bit_pos):
        self.bit_pos = list(bit_pos)

    def setup_fwd_run(self, **kwargs):
        pass

    def point_prediction(self, map_state, position, smoothed=True):
        return direct_point_prediction(map_state, position, smoothed)

    def point_prediction_and_jacobian(self, map_state, position):
        return direct_point_prediction_and_jacobian(map_state, position)

    def _facies(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if self.earth_model_type == "pixel":
            return pixel_state_to_facies(state_tensor)
        evaluator = self._earth_simulator.NNmodel.gan_evaluator
        return evaluator.eval(state_tensor.to(evaluator.device), no_grad=True)

    def run_fwd_sim(self, state, member_i):
        states = np.asarray(state["x"])
        if states.ndim == 1:
            states = states[np.newaxis, :]
        values = point_log_resistivity(self._facies(states), self.bit_pos[0])
        return [
            [{POINT_DATA_TYPE: value.detach().cpu().numpy().copy()}]
            for value in values
        ]


# Keep the old public name for callers outside the Streamlit page.
ZeroDSimulator = PointSimulator
