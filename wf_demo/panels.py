import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from wf_demo.measurements import POINT_DATA_TYPE, UDAR_COMPONENTS

RH_MULTIPLIER = (4.0, 171.0, 55.0, 0.0, 0.0, 0.0)
GEOSIGNAL_INDEX = 0
GRID_SIZE = 64
DEFAULT_TOOL = ("6kHz", "83ft")
DEFAULT_DATA_TYPE = "USDP"
DATA_COLORS = (
    "#38BDF8",
    "#FB923C",
    "#A78BFA",
    "#34D399",
    "#F472B6",
    "#FACC15",
    "#2DD4BF",
)


def data_type_label(data_type):
    if data_type == POINT_DATA_TYPE:
        return "Point ln Rh"
    if isinstance(data_type, tuple) and len(data_type) == 2:
        tool, kind = data_type
        if isinstance(tool, tuple):
            return f"{kind} ({tool[0]} / {tool[1]})"
        return f"{tool} / {kind}"
    return str(data_type)


def split_data_type(data_type):
    """Split a history key into (tool configuration, UDAR data type).

    Point carries no tool configuration.
    """
    if (
        isinstance(data_type, tuple)
        and len(data_type) == 2
        and isinstance(data_type[0], tuple)
    ):
        return data_type[0], data_type[1]
    return None, data_type


def combine_selected_data_types(available_types, selected_tools, selected_data_types):
    """Point records follow the data-type panel alone, UDAR records both panels."""
    chosen = []
    for data_type in available_types:
        tool, kind = split_data_type(data_type)
        if kind not in selected_data_types:
            continue
        if tool is None or tool in selected_tools:
            chosen.append(data_type)
    return chosen


def is_attenuation_data_type(data_type):
    """UDAR attenuation signals (trailing 'A') use the secondary plot axis."""
    kind = split_data_type(data_type)[1]
    return isinstance(kind, str) and kind.endswith("A")


def _rgba(hex_color, alpha):
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def discrete_value_scale():
    """Piecewise-constant tab20b colorscale for the value-map imshow."""
    from matplotlib import colormaps

    colors = [
        f"rgb({int(red * 255)},{int(green * 255)},{int(blue * 255)})"
        for red, green, blue in colormaps["tab20b"].colors
    ]
    count = len(colors)
    scale = []
    for index, color in enumerate(colors):
        low = index / count
        high = (index + 1) / count
        scale += [(low, color), (high, color)]
    return scale


def rh_log_profiles(facies_ensemble):
    """Log-resistivity Rh maps from a six-channel facies ensemble [N, 6, H, W]."""
    weights = torch.tensor(
        RH_MULTIPLIER, dtype=facies_ensemble.dtype, device=facies_ensemble.device
    ).view(1, -1, 1, 1)
    totals = (facies_ensemble * weights).sum(dim=1)
    return torch.log(totals.clamp_min(1e-6))


def _hard_facies(facies_ensemble):
    """Reduce soft GAN output to the one-hot facies used by the EM proxy."""
    classes = facies_ensemble[:, 0:3].argmax(dim=1)
    hard = (
        F.one_hot(classes, num_classes=3)
        .to(dtype=facies_ensemble.dtype)
        .permute(0, 3, 1, 2)
    )
    pad = torch.zeros(
        (facies_ensemble.shape[0], 3) + facies_ensemble.shape[2:],
        dtype=facies_ensemble.dtype,
        device=facies_ensemble.device,
    )
    return torch.cat([hard, pad], dim=1)


def resistivity_section_figure(facies_ensemble, true_facies, position):
    """Vertical Rh section at the current bit column with ensemble statistics."""
    ens = (
        rh_log_profiles(_hard_facies(facies_ensemble))[:, :, int(position[1])]
        .cpu()
        .numpy()
    )
    mean = ens.mean(axis=0)
    p10 = np.percentile(ens, 10, axis=0)
    p90 = np.percentile(ens, 90, axis=0)
    truth = rh_log_profiles(_hard_facies(true_facies))[0, :, int(position[1])].cpu().numpy()
    rows = np.arange(ens.shape[1])

    fig = go.Figure()
    fig.add_scatter(
        x=np.concatenate([np.exp(p90), np.exp(p10)[::-1]]),
        y=np.concatenate([rows, rows[::-1]]),
        fill="toself",
        fillcolor="rgba(31,119,180,0.25)",
        line=dict(width=0),
        hoverinfo="skip",
        name="p10-p90",
        showlegend=False,
    )
    fig.add_scatter(
        x=np.exp(mean), y=rows, mode="lines",
        line=dict(color="rgb(31,119,180)"), name="Ensemble mean",
    )
    fig.add_scatter(
        x=np.exp(truth), y=rows, mode="lines",
        line=dict(color="#F97316", dash="dot", width=2), name="Truth",
    )
    bit_row = int(position[0])
    if 0 <= bit_row < len(mean):
        fig.add_scatter(
            x=[float(np.exp(mean[bit_row]))], y=[bit_row], mode="markers",
            marker=dict(symbol="star", size=16, color="#F43F5E",
                        line=dict(color="#7C3AED", width=1)),
            name="Tool position",
        )
    fig.update_xaxes(title_text="Rh [ohm m]", type="log")
    fig.update_yaxes(
        title_text="TVD [row]",
        autorange=False,
        range=[GRID_SIZE - 0.5, -0.5],
    )
    fig.update_layout(
        height=560,
        margin=dict(t=80, b=70, l=50, r=10),
        legend=dict(y=1.12, x=0, orientation="h"),
    )
    return fig


def uncertainty_wash(posterior_std, threshold):
    """Opacity of the white lightness wash over the mean-model map.

    Scaled against the normalized std of the posterior ensemble,
    sigma = std / max(std): cells with sigma below the threshold keep their
    mapped colors; above it the white blend strengthens proportionally to the
    distance past the threshold and saturates at twice the threshold.
    A threshold of zero disables the wash.
    """
    std = np.asarray(posterior_std, dtype=float)
    scale = float(std.max())
    if scale <= 0 or threshold <= 0:
        return np.zeros_like(std)
    normalized = std / scale
    return np.clip((normalized - threshold) / threshold, 0.0, 1.0)


def parse_var_cell(cell):
    """Accept both flat ['ABS', [...]] and nested [['ABS', [...]]] variance cells."""
    spec = cell if isinstance(cell[0], str) else cell[0]
    return spec[0].lower(), spec[1]


def new_record(position, obs_cell, var_cell, preds_matrix, index=GEOSIGNAL_INDEX):
    """One drilled position: observed value, its std, posterior p10/p50/p90."""
    obs = np.asarray(obs_cell, dtype=float).ravel()
    _, var_values = parse_var_cell(var_cell)
    variances = np.asarray(var_values, dtype=float).ravel()
    series = np.asarray(preds_matrix, dtype=float)

    obs_index = index if obs.size > index else 0
    var_index = index if variances.size > index else 0
    sig_index = index if series.shape[0] > index else 0
    values = series[sig_index, :]
    return {
        "col": int(position[1]),
        "obs": float(obs[obs_index]),
        "std": float(np.sqrt(variances[var_index])),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def records_from_files(position, data_pkl, var_pkl, forecast_npz):
    """One history record per (tool configuration, data type) at a position."""
    try:
        data_df = pd.read_pickle(data_pkl)
        var_df = pd.read_pickle(var_pkl)
        stored = np.load(forecast_npz, allow_pickle=True)["pred_data"]
        forecast = stored.item() if stored.ndim == 0 else stored[0]
        records = []
        for key, prediction in forecast.items():
            if key not in data_df.columns or key not in var_df.columns:
                continue
            obs = np.asarray(data_df.iloc[0][key], dtype=float).ravel()
            series = np.asarray(prediction, dtype=float)
            if key == POINT_DATA_TYPE:
                components = (None,)
            else:
                components = UDAR_COMPONENTS[: min(obs.size, series.shape[0])]
            for index, component in enumerate(components):
                record = new_record(
                    position, obs, var_df.iloc[0][key], series, index=index
                )
                record["data_type"] = (key, component) if component else key
                records.append(record)
        return records
    except (FileNotFoundError, KeyError, IndexError, ValueError):
        return []


def record_from_files(position, data_pkl, var_pkl, forecast_npz):
    """Backward-compatible single-record reader."""
    records = records_from_files(position, data_pkl, var_pkl, forecast_npz)
    return records[0] if records else None


def data_history_figure(records, y_label, selected_types=None):
    """Data and posterior predictions for one or more selected datatypes."""
    fig = go.Figure()
    if records:
        available = list(dict.fromkeys(
            record.get("data_type", y_label) for record in records
        ))
        shown = available if selected_types is None else list(selected_types)
        if not shown:
            fig.add_annotation(
                text="Select one or more data types to compare",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
            )
        drawn = []
        for index, data_type in enumerate(shown):
            group = sorted(
                (
                    record
                    for record in records
                    if record.get("data_type", y_label) == data_type
                ),
                key=lambda record: record["col"],
            )
            if not group:
                continue
            drawn.append(data_type)
            color = DATA_COLORS[index % len(DATA_COLORS)]
            label = data_type_label(data_type)
            axis_ref = "y2" if is_attenuation_data_type(data_type) else "y"
            cols = [record["col"] for record in group]
            p10 = [record["p10"] for record in group]
            p90 = [record["p90"] for record in group]
            p50 = [record["p50"] for record in group]
            obs = [record["obs"] for record in group]
            err = [2.0 * record["std"] for record in group]
            fig.add_scatter(
                x=cols + cols[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor=_rgba(color, 0.18),
                line=dict(width=0),
                hoverinfo="skip",
                name=f"{label} p10-p90",
                legendgroup=label,
                showlegend=False,
                yaxis=axis_ref,
            )
            fig.add_scatter(
                x=cols,
                y=p50,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{label} posterior",
                legendgroup=label,
                yaxis=axis_ref,
            )
            fig.add_scatter(
                x=cols,
                y=obs,
                mode="markers",
                marker=dict(color=color, size=9, symbol="x", line=dict(width=2)),
                error_y=dict(array=err, color=color, thickness=1.5),
                name=f"{label} data (±2 std)",
                legendgroup=label,
                yaxis=axis_ref,
            )
        primary = [dt for dt in drawn if not is_attenuation_data_type(dt)]
        attenuation = [dt for dt in drawn if is_attenuation_data_type(dt)]
        if primary:
            point_only = all(split_data_type(dt)[1] == POINT_DATA_TYPE for dt in primary)
            fig.update_yaxes(
                title_text="Point ln Rh" if point_only else "Phase [deg]",
            )
        if attenuation:
            fig.update_layout(
                yaxis2=dict(
                    title_text="Attenuation [dB]",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
            )
    else:
        fig.add_annotation(
            text="Data and posterior predictions appear after the first assimilation step",
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
        )
    fig.update_xaxes(title_text="VS [column]", range=[-0.5, GRID_SIZE - 0.5])
    fig.update_layout(
        height=250,
        margin=dict(t=35, b=55, l=80, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
