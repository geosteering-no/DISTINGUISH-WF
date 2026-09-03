from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from misc import read_input_csv

from wf_demo.measurements import POINT_DATA_TYPE
from wf_demo.write_data_var import SyntheticTruth


TOOL = ("6kHz", "83ft")


class FakeTruthModel:
    def forward(self, latent, index_vector, output_transien_results):
        assert output_transien_results is False
        width = index_vector.shape[1]
        return torch.arange(width * 8, dtype=torch.float32).reshape(
            1, width, 1, 8
        )

    def eval_gan(self, latent):
        facies = torch.zeros((1, 6, 64, 64))
        facies[:, 2] = 1.0
        return facies


def test_synthetic_truth_writes_distinct_point_and_directional_data(
    monkeypatch, tmp_path
):
    work = tmp_path / "work"
    data_dir = tmp_path / "data"
    work.mkdir()
    data_dir.mkdir()
    monkeypatch.chdir(work)
    truth = SyntheticTruth.__new__(SyntheticTruth)
    truth.device = torch.device("cpu")
    truth.simulator = SimpleNamespace(NNmodel=FakeTruthModel())
    truth.one_d_data_types = [TOOL]
    truth.all_data_types = [POINT_DATA_TYPE, TOOL]
    truth.latent_synthetic_truth = torch.zeros((1, 60))

    truth.acquire_data({"bit_pos": [(7, 3)]})

    observations = pd.read_pickle(data_dir / "data.pkl")
    variances = pd.read_pickle(data_dir / "var.pkl")
    assert list(observations.columns) == [POINT_DATA_TYPE, TOOL]
    np.testing.assert_allclose(
        observations.iloc[0][POINT_DATA_TYPE], np.log([55.0, 85.0])
    )
    assert np.asarray(observations.iloc[0][TOOL]).shape == (8,)
    assert len(variances.iloc[0][POINT_DATA_TYPE][1]) == 2
    assert len(variances.iloc[0][TOOL][1]) == 8

    truth.activate_data_types([POINT_DATA_TYPE])
    assert (data_dir / "datatyp.csv").read_text().strip() == POINT_DATA_TYPE


def test_synthetic_truth_constructor_does_not_rewrite_shared_data_files(
    monkeypatch, tmp_path
):
    work = tmp_path / "work"
    data_dir = tmp_path / "data"
    work.mkdir()
    data_dir.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(
        "wf_demo.write_data_var.GeoSim",
        lambda _: SimpleNamespace(l_prim=None, all_data_types=None),
    )

    SyntheticTruth(torch.zeros((1, 60)), device=torch.device("cpu"))

    assert not (data_dir / "datatyp.csv").exists()
    assert not (data_dir / "assim_index.csv").exists()


def test_pet_stage_files_contain_only_the_selected_simulator_data(
    monkeypatch, tmp_path
):
    work = tmp_path / "work"
    data_dir = tmp_path / "data"
    output_dir = work / "SaveOutputs"
    work.mkdir()
    data_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.chdir(work)
    observations = pd.DataFrame({
        POINT_DATA_TYPE: [np.array([1.0, 2.0])],
        TOOL: [np.arange(8.0)],
    })
    variances = pd.DataFrame({
        POINT_DATA_TYPE: [["ABS", [0.1, 0.1]]],
        TOOL: [["ABS", [0.2] * 8]],
    })
    observations.to_pickle(data_dir / "data.pkl")
    variances.to_pickle(data_dir / "var.pkl")

    point_data, point_variance = SyntheticTruth.pet_input_files(
        [POINT_DATA_TYPE], output_dir
    )

    assert list(pd.read_pickle(point_data).columns) == [POINT_DATA_TYPE]
    assert list(pd.read_pickle(point_variance).columns) == [POINT_DATA_TYPE]
    _, pet_data_types, _ = read_input_csv.read_data_df(
        point_data, outtype="list"
    )
    assert pet_data_types == [POINT_DATA_TYPE]
