import numpy as np
import pandas as pd
import pytest
import torch

from wf_demo.measurements import UDAR_COMPONENTS
from wf_demo.panels import (
    combine_selected_data_types,
    data_history_figure,
    discrete_value_scale,
    new_record,
    parse_var_cell,
    record_from_files,
    records_from_files,
    resistivity_section_figure,
    rh_log_profiles,
    split_data_type,
    uncertainty_wash,
)


def one_hot_facies(classes):
    facies = torch.zeros((classes.shape[0], 6, 64, 64))
    for channel in range(3):
        facies[:, channel] = (classes == channel).float().unsqueeze(-1).unsqueeze(-1)
    return facies


def test_rh_log_profiles_matches_facies_resistivities():
    facies = one_hot_facies(torch.tensor([0, 1, 2]))

    rh = rh_log_profiles(facies)

    assert rh.shape == (3, 64, 64)
    expected = [np.log(4.0), np.log(171.0), np.log(55.0)]
    for member, value in enumerate(expected):
        assert rh[member].max().item() == pytest.approx(value)


def test_rh_log_profiles_clamps_nonpositive_totals():
    facies = torch.full((1, 6, 64, 64), -1.0)

    rh = rh_log_profiles(facies)

    assert torch.isfinite(rh).all()


def test_section_figure_shows_envelope_mean_truth_and_star():
    ens = one_hot_facies(torch.tensor([0, 1, 0, 2]))
    truth = one_hot_facies(torch.tensor([1]))

    fig = resistivity_section_figure(ens, truth, position=(10, 5))

    traces = list(fig.data)
    assert len(traces) == 4
    assert traces[1].name == "Ensemble mean"
    assert traces[2].name == "Truth"
    assert traces[3].name == "Tool position"
    assert traces[3].y == (10,)
    assert traces[2].line.color != "black"
    assert tuple(fig.layout.yaxis.range) == (63.5, -0.5)


def test_section_figure_hardens_soft_gan_output():
    soft = torch.zeros((2, 6, 64, 64))
    soft[:, 0] = -0.9  # negative soft output must not leak into Rh
    soft[:, 1] = -0.1  # argmax still picks channel 1 -> sand everywhere
    soft[:, 2] = -0.5

    fig = resistivity_section_figure(soft, soft[:1].clone(), position=(0, 0))

    mean_x = fig.data[1].x
    assert np.allclose(mean_x, 171.0)


@pytest.mark.parametrize(
    "cell",
    [(["ABS", [0.04, 0.09]]), ([["ABS", [0.04, 0.09]]])],
)
def test_parse_var_cell_accepts_flat_and_nested_formats(cell):
    kind, values = parse_var_cell(cell)

    assert kind == "abs"
    assert list(values) == [0.04, 0.09]


def test_new_record_summarizes_one_geosignal():
    preds = np.random.default_rng(0).normal(size=(8, 100)) + 5.0

    record = new_record(
        position=(30, 7),
        obs_cell=np.arange(8, dtype=float),
        var_cell=["ABS", [0.01] * 8],
        preds_matrix=preds,
    )

    assert record["col"] == 7
    assert record["obs"] == 0.0
    assert record["std"] == pytest.approx(0.1)
    assert record["p10"] <= record["p50"] <= record["p90"]


def test_record_from_files_reads_pet_outputs(tmp_path):
    key = ("6kHz", "83ft")
    pd.DataFrame({key: [np.arange(8, dtype=float)]}, index=[0]).to_pickle(
        tmp_path / "data.pkl"
    )
    pd.DataFrame({key: [["ABS", [0.04] * 8]]}, index=[0]).to_pickle(
        tmp_path / "var.pkl"
    )
    preds = {key: np.random.default_rng(1).normal(size=(8, 50))}
    np.savez(tmp_path / "posterior_forecast.npz", pred_data=[preds])

    record = record_from_files(
        (31, 3), tmp_path / "data.pkl", tmp_path / "var.pkl", tmp_path / "posterior_forecast.npz"
    )

    assert record["col"] == 3
    assert record["obs"] == 0.0

    missing = record_from_files(
        (31, 3), tmp_path / "nope.pkl", tmp_path / "var.pkl", tmp_path / "posterior_forecast.npz"
    )
    assert missing is None


def test_records_from_files_preserves_each_tool_and_data_type(tmp_path):
    point = "point"
    tool = ("6kHz", "83ft")
    pd.DataFrame([{
        point: np.array([2.0]),
        tool: np.arange(8, dtype=float) + 10.0,
    }]).to_pickle(tmp_path / "data.pkl")
    pd.DataFrame([{
        point: ["ABS", [0.04]],
        tool: ["ABS", [0.09] * 8],
    }]).to_pickle(tmp_path / "var.pkl")
    forecasts = {
        point: np.full((1, 10), 3.0),
        tool: np.full((8, 10), 12.0),
    }
    np.savez(tmp_path / "posterior_forecast.npz", pred_data=[forecasts])

    records = records_from_files(
        (31, 3),
        tmp_path / "data.pkl",
        tmp_path / "var.pkl",
        tmp_path / "posterior_forecast.npz",
    )

    assert [record["data_type"] for record in records] == [point] + [
        (tool, component) for component in UDAR_COMPONENTS
    ]
    point_record, usdp_record = records[0], records[1]
    assert point_record["obs"] == 2.0
    assert point_record["p50"] == pytest.approx(3.0)
    assert usdp_record["obs"] == 10.0
    assert usdp_record["std"] == pytest.approx(0.3)
    assert usdp_record["p50"] == pytest.approx(12.0)


def test_data_history_figure_draws_records_or_placeholder():
    records = [
        {"col": 1, "obs": 1.0, "std": 0.1, "p10": 0.8, "p50": 1.0, "p90": 1.2},
        {"col": 2, "obs": 1.1, "std": 0.1, "p10": 0.9, "p50": 1.1, "p90": 1.3},
    ]

    filled = data_history_figure(records, y_label="UDAR")
    assert len(filled.data) == 3
    assert filled.data[2].error_y["array"] == (0.2, 0.2)

    empty = data_history_figure([], y_label="UDAR")
    assert len(empty.data) == 0
    assert len(empty.layout.annotations) == 1


def test_data_history_figure_draws_multiple_selected_datatypes_without_black():
    records = [
        {"col": 1, "data_type": "point", "obs": 1.0, "std": 0.1,
         "p10": 0.8, "p50": 1.0, "p90": 1.2},
        {"col": 1, "data_type": (("6kHz", "83ft"), "USDP"), "obs": 2.0,
         "std": 0.2, "p10": 1.7, "p50": 2.0, "p90": 2.3},
    ]

    fig = data_history_figure(
        records,
        y_label="Data",
        selected_types=["point", (("6kHz", "83ft"), "USDP")],
    )

    assert len(fig.data) == 6
    assert {trace.legendgroup for trace in fig.data} == {
        "Point ln Rh",
        "USDP (6kHz / 83ft)",
    }
    assert all(
        getattr(trace.marker, "color", None) != "black"
        for trace in fig.data
    )

    point_only = data_history_figure(
        records, y_label="Data", selected_types=["point"]
    )
    assert len(point_only.data) == 3
    assert point_only.layout.yaxis.title.text == "Point ln Rh"


def test_data_history_figure_splits_phase_and_attenuation_axes():
    tool = ("6kHz", "83ft")
    records = [
        {"col": 1, "data_type": "point", "obs": 1.0, "std": 0.1,
         "p10": 0.8, "p50": 1.0, "p90": 1.2},
        {"col": 1, "data_type": (tool, "USDP"), "obs": 2.0, "std": 0.2,
         "p10": 1.7, "p50": 2.0, "p90": 2.3},
        {"col": 1, "data_type": (tool, "UADA"), "obs": 20.0, "std": 2.0,
         "p10": 17.0, "p50": 20.0, "p90": 23.0},
    ]

    fig = data_history_figure(
        records,
        y_label="Data",
        selected_types=["point", (tool, "USDP"), (tool, "UADA")],
    )

    assert len(fig.data) == 9
    assert [trace.yaxis for trace in fig.data] == ["y"] * 6 + ["y2"] * 3
    assert fig.layout.yaxis.title.text == "Phase [deg]"
    assert fig.layout.yaxis2.title.text == "Attenuation [dB]"
    assert fig.layout.yaxis2.overlaying == "y"
    assert fig.layout.yaxis2.side == "right"


def test_uncertainty_wash_gates_and_scales_on_normalized_std():
    posterior = np.array([[0.3, 1.0, 1.25, 2.0]])  # sigma: [.15, .5, .625, 1]

    wash = uncertainty_wash(posterior, threshold=0.5)
    assert list(wash[0]) == pytest.approx([0.0, 0.0, 0.25, 1.0])

    saturated = uncertainty_wash(posterior, threshold=0.1)
    assert list(saturated[0]) == pytest.approx([0.5, 1.0, 1.0, 1.0])


def test_uncertainty_wash_handles_degenerate_fields():
    assert uncertainty_wash(np.zeros((2, 2)), threshold=1.0).tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    assert uncertainty_wash(np.array([[0.0, 5.0]]), threshold=0.0).tolist() == [
        [0.0, 0.0]
    ]


def test_discrete_value_scale_is_piecewise_constant():
    scale = discrete_value_scale()

    assert len(scale) == 40
    assert scale[0][0] == 0.0
    assert scale[-1][0] == 1.0
    assert all(
        isinstance(color, str) and color.startswith("rgb(") for _, color in scale
    )
    assert scale[0][1] == scale[1][1]
    assert scale[2][1] == scale[3][1]


def test_split_data_type_separates_tool_from_data_type():
    assert split_data_type((("24kHz", "43ft"), "UADA")) == (
        ("24kHz", "43ft"),
        "UADA",
    )
    assert split_data_type("point") == (None, "point")


def test_combine_selected_data_types_filters_tool_and_data_type():
    available = [
        "point",
        (("6kHz", "83ft"), "USDP"),
        (("24kHz", "43ft"), "USDP"),
        (("24kHz", "43ft"), "UADA"),
    ]

    combined = combine_selected_data_types(
        available, [("24kHz", "43ft")], ["USDP"]
    )
    assert combined == [(("24kHz", "43ft"), "USDP")]

    point_without_tool = combine_selected_data_types(
        available, [], ["USDP", "point"]
    )
    assert point_without_tool == ["point"]

    no_matching_tool = combine_selected_data_types(
        available, [("6kHz", "83ft")], ["UADA"]
    )
    assert no_matching_tool == []
