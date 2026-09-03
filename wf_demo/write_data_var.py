import csv
import pickle
from pathlib import Path

from GeoSim.sim import GeoSim
import numpy as np
import pandas as pd
import torch

from wf_demo.default_load import input_dict
from wf_demo.measurements import POINT_DATA_TYPE
from wf_demo.zero_d import point_log_resistivity


class SyntheticTruth:
    def __init__(self, latent_truth_vector, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.simulator = GeoSim(input_dict)
        self.simulator.l_prim = [0]
        self.simulator.all_data_types = input_dict["datatype"]
        self.one_d_data_types = list(input_dict["datatype"])
        self.all_data_types = [POINT_DATA_TYPE] + [
            data_type
            for data_type in self.one_d_data_types
            if data_type != POINT_DATA_TYPE
        ]
        self.latent_synthetic_truth = latent_truth_vector

    @staticmethod
    def _write_assimilation_indices():
        with open("../data/assim_index.csv", "w", newline="") as handle:
            handle.write("0\n")

    @staticmethod
    def activate_data_types(data_types):
        """Select the columns PET should consume from the combined data files."""
        with open("../data/datatyp.csv", "w", newline="") as handle:
            csv.writer(handle).writerow([str(value) for value in data_types])

    @staticmethod
    def pet_input_files(data_types, output_dir="SaveOutputs"):
        """Write stage-specific pickles because PET otherwise loads all columns.

        PET's pickle reader replaces its configured datatype list with every
        DataFrame column. Filtering the files is therefore required when the
        0D point and 1D directional observations coexist in the source files.
        """
        selected = list(data_types)
        observations = pd.read_pickle("../data/data.pkl")
        variances = pd.read_pickle("../data/var.pkl")
        missing = [
            data_type
            for data_type in selected
            if data_type not in observations.columns
            or data_type not in variances.columns
        ]
        if missing:
            raise KeyError(f"Missing PET observations or variances for {missing}")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stage = "point" if selected == [POINT_DATA_TYPE] else "one_d"
        data_path = output / f"{stage}_data.pkl"
        variance_path = output / f"{stage}_var.pkl"
        observations[selected].to_pickle(data_path)
        variances[selected].to_pickle(variance_path)
        return str(data_path), str(variance_path)

    def acquire_data(self, keys):
        position = keys["bit_pos"][0]
        index_vector = torch.full(
            size=(1, position[1] + 1),
            fill_value=position[0],
            dtype=torch.long,
            device=self.device,
        )
        logs = self.simulator.NNmodel.forward(
            self.latent_synthetic_truth,
            index_vector,
            output_transien_results=False,
        )

        true_facies = self.simulator.NNmodel.eval_gan(
            self.latent_synthetic_truth
        )
        point = (
            point_log_resistivity(true_facies, position)[0]
            .detach()
            .cpu()
            .numpy()
        )
        data = {POINT_DATA_TYPE: [point]}
        variance = {
            POINT_DATA_TYPE: [
                ["ABS", [(0.001 * abs(value)) ** 2 for value in point]]
            ]
        }

        if self.one_d_data_types != [POINT_DATA_TYPE]:
            logs_at_bit = logs.detach().cpu().numpy()[0, position[1], :, :]
            for index, data_type in enumerate(self.one_d_data_types):
                values = logs_at_bit[index, :]
                data[data_type] = [values]
                scale = 0.1 * np.max(np.abs(values))
                variance[data_type] = [["ABS", [scale**2 for _ in values]]]

        data_frame = pd.DataFrame(data, columns=self.all_data_types, index=[0])
        data_frame.index.name = "tvd"
        data_frame.to_pickle("../data/data.pkl")

        variance_frame = pd.DataFrame(
            variance, columns=self.all_data_types, index=[0]
        )
        variance_frame.index.name = "tvd"
        variance_frame.to_csv("../data/var.csv", index=True)
        with open("../data/var.pkl", "wb") as handle:
            pickle.dump(variance_frame, handle)

        self._write_assimilation_indices()
        self.activate_data_types(self.all_data_types)
