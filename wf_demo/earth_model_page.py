from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st

from wf_demo.panels import discrete_value_scale
from wf_demo.pixel_model import (
    COVARIANCE_MODELS,
    GRID_HEIGHT,
    GRID_WIDTH,
    gan_facies_to_pixel_state,
    sample_pixel_prior,
)


st.title("Earth Model")
st.write("Select prior model used by geosteering. Synthetic truth always uses GAN.")

current_model = st.session_state.get("earth_model_type", "gan")
current_config = st.session_state.get("earth_model_config", {})
model_type = st.radio(
    "Prior model",
    options=["GAN", "Pixel-based"],
    index=0 if current_model == "gan" else 1,
    horizontal=True,
)

gan_prior = np.load(Path(__file__).resolve().parent.parent / "orig_prior_2024.npz")["m"]
ensemble_size = gan_prior.shape[1]

if model_type == "GAN":
    st.caption(f"Existing GAN prior: {ensemble_size} realizations, {gan_prior.shape[0]} latent variables.")
    selected_config = {"type": "gan"}
else:
    pixel_prior_source = st.radio(
        "Pixel prior source",
        options=["Geostatistical", "GAN samples"],
        index=(
            1
            if current_config.get("pixel_prior_source") == "GAN samples"
            else 0
        ),
        horizontal=True,
        help="GAN samples are converted to the three-class pixel state. The "
        "covariance controls below still define Laplace regularization.",
    )
    covariance_model = st.selectbox("Covariance model", options=list(COVARIANCE_MODELS))
    col1, col2 = st.columns(2)
    with col1:
        horizontal_correlation = st.number_input(
            "Horizontal correlation length (cells)",
            min_value=1.0,
            max_value=float(GRID_WIDTH),
            value=16.0,
            step=1.0,
        )
    with col2:
        vertical_correlation = st.number_input(
            "Vertical correlation length (cells)",
            min_value=1.0,
            max_value=float(GRID_HEIGHT),
            value=4.0,
            step=1.0,
        )
    col3, col4 = st.columns(2)
    with col3:
        prior_mean = st.number_input(
            "Prior mean",
            value=0.0,
            step=0.1,
        )
    with col4:
        prior_standard_deviation = st.number_input(
            "Prior standard deviation",
            min_value=0.01,
            value=1.0,
            step=0.1,
        )
    selected_config = {
        "type": "pixel",
        "pixel_prior_source": pixel_prior_source,
        "covariance_model": covariance_model,
        "horizontal_correlation": horizontal_correlation,
        "vertical_correlation": vertical_correlation,
        "prior_mean": prior_mean,
        "prior_standard_deviation": prior_standard_deviation,
    }

start_row_text = st.text_input(
    "Custom well start row (0-63)",
    value=str(current_config.get("start_row", 31)),
    help="Row used for first well position; horizontal column always starts at 0.",
)

st.subheader("Data Assimilation")


def persist_localization_settings():
    # Widget state is dropped when the widget is not rendered (other page);
    # mirror it into plain session keys, which persist across pages.
    st.session_state["localization_enabled"] = st.session_state["localization_enabled_widget"]
    st.session_state["localization_strength"] = st.session_state["localization_strength_widget"]


st.toggle(
    "Use auto-adaptive localization",
    value=st.session_state.get("localization_enabled", False),
    key="localization_enabled_widget",
    on_change=persist_localization_settings,
)
st.number_input(
    "Localization strength",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    value=st.session_state.get("localization_strength", 0.9),
    key="localization_strength_widget",
    on_change=persist_localization_settings,
    disabled=not st.session_state.get("localization_enabled", False),
)
st.caption(
    "Localization uses PET AUTOADALOC with sigmoid tapering (SIGM). "
    "Lower strength = stronger localization: 0.99 is nearly no tapering, "
    "0.01 almost fully suppresses the update."
)

if st.button("Use this earth model", type="primary"):
    try:
        start_row = int(start_row_text)
        if not 0 <= start_row < GRID_HEIGHT:
            raise ValueError
    except ValueError:
        st.error(f"Custom well start row must be an integer from 0 to {GRID_HEIGHT - 1}.")
    else:
        if selected_config["type"] == "gan":
            prior = gan_prior
        elif selected_config["pixel_prior_source"] == "Geostatistical":
            with st.spinner("Sampling pixel prior..."):
                prior = sample_pixel_prior(
                    ensemble_size=ensemble_size,
                    covariance_model=selected_config["covariance_model"],
                    horizontal_correlation=selected_config["horizontal_correlation"],
                    vertical_correlation=selected_config["vertical_correlation"],
                    prior_mean=selected_config["prior_mean"],
                    prior_standard_deviation=selected_config["prior_standard_deviation"],
                )
        else:
            with st.spinner("Sampling GAN and converting to pixel states..."):
                import torch

                from GeoSim.sim import GeoSim
                from wf_demo.default_load import input_dict

                simulator = GeoSim(input_dict)
                evaluator = simulator.NNmodel.gan_evaluator
                latent = torch.as_tensor(
                    gan_prior.T,
                    dtype=torch.float32,
                    device=evaluator.device,
                )
                facies = evaluator.eval(latent, no_grad=True)
                prior = gan_facies_to_pixel_state(facies)

        selected_config["start_row"] = start_row
        st.session_state["earth_model_type"] = selected_config["type"]
        st.session_state["earth_model_config"] = selected_config
        st.session_state["earth_model_prior"] = prior
        st.session_state["first_position"] = True
        for key in (
            "ensemble_state", "start_position_state", "path", "auto_opt",
            "auto_pes", "data_history", "data_history_tools",
            "data_history_data_types", "laplace_reduction", "da_failures",
        ):
            st.session_state.pop(key, None)
        st.success(f"{model_type} prior ready with {prior.shape[1]} realizations.")

@st.cache_data(show_spinner="Evaluating GAN prior model...")
def _gan_prior_values(prior):
    import torch
    from GeoSim.sim import GeoSim
    from pathoptim.DP import evaluate_earth_model_ensemble
    from wf_demo.default_load import input_dict

    simulator = GeoSim(input_dict)
    evaluator = simulator.NNmodel.gan_evaluator
    latent = torch.as_tensor(prior.T, dtype=torch.float32, device=evaluator.device)
    facies = evaluator.eval(latent, no_grad=True)
    values = evaluate_earth_model_ensemble(facies, compute_geobody_sizes=True)
    return values.detach().cpu().numpy()


@st.cache_data(show_spinner="Rendering pixel prior model...")
def _pixel_prior_values(prior):
    import torch
    from pathoptim.DP import evaluate_earth_model_ensemble
    from wf_demo.pixel_model import pixel_state_to_facies

    facies = pixel_state_to_facies(torch.as_tensor(prior.T, dtype=torch.float32))
    values = evaluate_earth_model_ensemble(facies, compute_geobody_sizes=True)
    return values.detach().cpu().numpy()


selected_kind = "gan" if model_type == "GAN" else "pixel"
if (
    "earth_model_prior" in st.session_state
    and st.session_state.get("earth_model_type") == selected_kind
):
    prior = st.session_state["earth_model_prior"]
    if selected_kind == "gan":
        values = _gan_prior_values(prior)
    else:
        values = _pixel_prior_values(prior)
    fig = px.imshow(
        values.mean(axis=0),
        aspect="auto",
        color_continuous_scale=discrete_value_scale(),
        zmin=0.0,
        zmax=10.0,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            orientation="h", x=0.5, y=-0.2, xanchor="center",
            yanchor="top", len=0.8, title="Value",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

st.info("Open geosteering from navigation after applying model.")
