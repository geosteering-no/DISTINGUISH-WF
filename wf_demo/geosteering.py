import os
import fcntl
os.environ["MPLBACKEND"] = "Agg"  # must be before importing matplotlib

import matplotlib
matplotlib.use("Agg")


from matplotlib import colormaps  # noqa: F401  (kept for page compatibility)
import numpy as np
import warnings

import torch

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# warnings.filterwarnings("always")
from pathoptim.pathOPTIM import pathfinder
from pathoptim.DP import perform_dynamic_programming, evaluate_earth_model_ensemble
from GeoSim.sim import GeoSim
from pipt.loop.assimilation import Assimilate
from pipt import pipt_init
from input_output import read_config

from write_data_var import SyntheticTruth

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plot_for_app import earth
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
from copy import deepcopy as dp
import time

from wf_demo.default_load import input_dict, load_default_latent_tensor, load_default_starting_ensemble_state, udar_data_type_array
from wf_demo.assimilation_mode import ordered_assimilation_steps, run_assimilation_sequence
from wf_demo.localization import configure_autoadaloc
from wf_demo.measurements import POINT_DATA_TYPE, UDAR_COMPONENTS
from wf_demo.laplace_assimilation import (
    AnalyticPixelPrior,
    assimilate_laplace,
    run_predictions,
    save_posterior_outputs,
)
from wf_demo.occam_assimilation import assimilate_occam
from wf_demo.panels import (
    DEFAULT_DATA_TYPE,
    DEFAULT_TOOL,
    combine_selected_data_types,
    data_history_figure,
    data_type_label,
    discrete_value_scale,
    missing_record_columns,
    record_is_finite,
    records_from_files,
    resistivity_section_figure,
    split_data_type,
    uncertainty_wash,
)
from wf_demo.pixel_model import pixel_state_to_facies, preserve_behind_bit
from wf_demo.zero_d import PointSimulator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global_extent = [0, 640, -16.25, 15.75]
norm = Normalize(vmin=0.0, vmax=1)

value_range = [0., 10.]

# weights_folder = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
# scalers_folder = weights_folder
# full_em_model_file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"
# file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/gan/netG_epoch_15000.pth"

# build a streamlit app to run the workflow. On the first run of the app we will be in the initial state.
# The user has to specify the start position of the well. Number of decissions are always 1, and the user have to specify
# whether to drill ahead, or to stop drilling. If the user decides to stop drilling, the app will stop.
# If the user decides to drill ahead, the main function will be called.

if 'first_position' not in st.session_state:
    st.session_state['first_position'] = True
# Also, plot the current state as a main feature of the app.


input_dict['datatype'] = udar_data_type_array
dt = input_dict['datatype']
data_type_str = 'UDAR'
dt_first = dt[0] if isinstance(dt, list) and len(dt) > 0 else dt
data_label = str(dt_first)
st.title(f'Geosteering ({data_type_str})')
# GMO refers to Generic Modern [UDAR] Observations

if st.session_state.get('localization_enabled', False):
    st.caption(
        f"Localization: ON (AUTOADALOC/SIGM, strength "
        f"{st.session_state.get('localization_strength', 0.9):.2f})"
    )
else:
    st.caption("Localization: OFF")

next_optimal_o = None
next_optimal_p = None
# this creates an instance if a simulator for synthetic truth


true_sim = SyntheticTruth(latent_truth_vector=load_default_latent_tensor().to(device), device=device)
earth_model_type = st.session_state.get('earth_model_type', 'gan')
ensemble_input_dict = input_dict.copy()
ensemble_input_dict['earth_model_type'] = earth_model_type
if earth_model_type == "pixel":
    st.caption(
        "Adaptive inversion domain: columns behind the drill bit are frozen; "
        "only the current column and columns ahead can be updated."
    )
else:
    st.caption(
        "Behind-bit freezing requires the pixel earth model; GAN latent updates "
        "remain global."
    )

assimilation_method = st.radio(
    "Data-assimilation method",
    options=["ES", "Laplace", "Occam"],
    horizontal=True,
    key="assimilation_method",
)
if assimilation_method in {"Laplace", "Occam"} and st.session_state.get('localization_enabled', False):
    st.caption("AUTOADALOC applies to ES only; Laplace and Occam do not use ensemble localization.")
laplace_max_iterations = 10
if assimilation_method == "Laplace":
    st.caption(
        "Laplace propagates the measurement through the analytic prior covariance: "
        "exactly the correlation range for Spherical/Cubic, decaying tails for "
        "Gaussian/Exponential."
    )
    laplace_max_iterations = st.number_input(
        "Laplace: max GN iterations",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        help="1 keeps the single Gauss-Newton step (as in ES); higher values iterate "
        "the MAP solve with a built-in line search until the objective or update "
        "stalls. Chi-square is reported as a data-fit diagnostic.",
    )
occam_max_iterations = 8
occam_target_value = 4.0
if assimilation_method == "Occam":
    st.caption(
        "Occam inverts the full 2D image: the smoothest update — first-order "
        "roughness in both grid directions, weakly anchored to the current "
        "image — that fits the measurement to the target chi-square. Not a "
        "Gaussian update. The direct 0D point data may run before the 1D "
        "directional response."
    )
    occam_max_iterations = st.number_input(
        "Occam: max iterations",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
        help="Each iteration linearizes the forward model, solves the "
        "roughness-regularized problem over a log grid of trade-off parameters "
        "mu, and keeps the smoothest candidate fitting the target chi-square, "
        "with backtracking on the true misfit.",
    )
    occam_target_value = st.number_input(
        "Occam: target chi-square (x N data)",
        min_value=0.1,
        max_value=50.0,
        value=4.0,
        step=0.5,
        help="Occam stops smoothing once chi2 reaches target x N. Values above 1 "
        "tolerate the bias of the smoothed proxy forward; 4 was best in the "
        "RML-NN benchmark study.",
    )


def get_start():
    configured = st.session_state.get("earth_model_config", {}).get("start_row")
    raw = configured if configured is not None else st.query_params.get("start", "31")
    try:
        start_y = int(raw)
        if start_y > 63:
            start_y = 63
        if start_y < 0:
            start_y = 0
        return start_y
    except (TypeError, ValueError):
        return 31


# Show a slider first to select the start position of the well
if st.session_state.first_position:
    # state = np.load('../orig_prior_small.npz')['x']  # the prior latent vector
    state = st.session_state.get('earth_model_prior', load_default_starting_ensemble_state())
    # the commented code loads the truth as the state for checking correctness
    # state_torch = load_default_latent_tensor().cpu()
    # state = state_torch.permute(1,0).numpy()
    print(f'State tensor shape {state.shape}')
    # start_position = (st.slider(label='Enter the horizontal start position of the well', key='start_position',
    #                             min_value=0, max_value=64, value=int(31)), 0)
    start_y = get_start()
    start_position = (start_y, 0)
    st.session_state['path'] = [start_position]
else:
    state = st.session_state.ensemble_state
    start_position = st.session_state.start_position_state
    st.session_state['path'].append(start_position)

# toggle first step
def toggle_first_step_and_rerun():
    st.session_state['first_position'] = False
    st.rerun()

# plot the current state
@st.cache_data
def get_earth(state, input_dict, model_type):
    if model_type == 'pixel':
        state_torch = torch.tensor(state.T, dtype=torch.float32).to(device)
        return pixel_state_to_facies(state_torch)

    # make state into a tensor
    # TODO fix with passing device
    sim_ensemble = GeoSim(input_dict)
    # print(f"Input for display sim: {input_dict}")
    # facies_ensemble = earth(torch.tensor(state, dtype=torch.float32).to(device), simulator=sim_ensemble)
    state_torch = torch.tensor(state.T, dtype=torch.float32).to(device)
    facies_ensemble = sim_ensemble.NNmodel.gan_evaluator.eval(state_torch, no_grad=True)

    # # TODO fix the weights
    # weights = np.array([-0.1, 1, 0.5])
    # value_ensemble = np.mean(facies_ensemble * weights.reshape(1, 3, 1, 1), axis=1)  # Apply weights to the true facies

    return facies_ensemble

def _da_unlocked(state, input_dict, start_position, localization_enabled,
                 localization_strength, assimilation_steps, assimilation_method,
                 laplace_max_iterations, occam_max_iterations, occam_target_value):
    keys_filter = input_dict.copy()
    keys_filter['bit_pos'] = [start_position]

    # One acquisition writes direct point data for 0D and directional data for 1D.
    try:
        true_sim.acquire_data({'bit_pos': [start_position],
                               'vec_size': 60,
                               'reporttype': 'pos',
                               'reportpoint': [0],
                               'datatype': input_dict['datatype']})
    except Exception as exc:
        record_da_failure(start_position, exc)
        return None

    def assimilate(current_state, simulator_name):
        if simulator_name == '0D':
            sim_ensemble = PointSimulator(keys_filter)
        else:
            sim_ensemble = GeoSim(keys_filter)
        true_sim.activate_data_types(sim_ensemble.all_data_types)

        if assimilation_method in {"Laplace", "Occam"}:
            earth_model_config = st.session_state.get('earth_model_config')
            if earth_model_config is None:
                raise KeyError(
                    f"{assimilation_method} assimilation needs an earth-model "
                    "configuration; apply a pixel prior on the Earth Model page first."
                )
            if assimilation_method == "Laplace":
                prior = AnalyticPixelPrior(earth_model_config)
                posterior, reduction = assimilate_laplace(
                    current_state,
                    sim_ensemble,
                    simulator_name,
                    start_position,
                    prior,
                    reduction=st.session_state.get('laplace_reduction'),
                    seed=start_position[0] * 64 + start_position[1]
                    + (0 if simulator_name == "0D" else 4096),
                    max_iterations=laplace_max_iterations,
                )
                st.session_state['laplace_reduction'] = reduction
                record_data_history(start_position, sim_ensemble.all_data_types)
                return posterior
            st.session_state.pop('laplace_reduction', None)
            posterior = assimilate_occam(
                current_state,
                sim_ensemble,
                simulator_name,
                start_position,
                earth_model_config,
                seed=start_position[0] * 64 + start_position[1]
                + (0 if simulator_name == "0D" else 4096),
                max_iterations=occam_max_iterations,
                target_value=occam_target_value,
            )
            record_data_history(start_position, sim_ensemble.all_data_types)
            return posterior

        st.session_state.pop('laplace_reduction', None)
        np.savez('prior.npz', x=current_state)
        keys_data, _ = read_config.read_txt('DA.pipt')
        pet_data, pet_variance = true_sim.pet_input_files(
            sim_ensemble.all_data_types
        )
        keys_data['truedata'] = pet_data
        keys_data['datavar'] = pet_variance
        keys_data['datatype'] = list(sim_ensemble.all_data_types)
        keys_data = configure_autoadaloc(
            keys_data,
            enabled=localization_enabled,
            strength=localization_strength,
            state_size=current_state.shape[0],
        )
        print(
            f"{simulator_name} DA localization: "
            f"{'ON' if localization_enabled else 'OFF'}"
            f"{f' strength={localization_strength}' if localization_enabled else ''} "
            f"({current_state.shape[0]} state parameters)"
        )
        sim_ensemble.update_bit_pos([start_position])
        analysis = pipt_init.init_da(keys_data, keys_data, sim_ensemble)
        assimilation = Assimilate(analysis)
        assimilation.run()
        posterior = np.load('SaveOutputs/posterior_state_estimate.npz')['x']
        if earth_model_type == "pixel":
            posterior = preserve_behind_bit(
                current_state, posterior, start_position[1]
            )
            posterior_prediction = run_predictions(sim_ensemble, posterior)
            save_posterior_outputs(
                posterior,
                posterior_prediction,
                sim_ensemble.all_data_types,
            )
        record_data_history(start_position, sim_ensemble.all_data_types)
        return posterior

    try:
        return run_assimilation_sequence(state, assimilation_steps, assimilate)
    except Exception as exc:
        record_da_failure(start_position, exc)
        return None


def da(state, input_dict, start_position, localization_enabled, localization_strength,
       assimilation_steps, assimilation_method, laplace_max_iterations,
       occam_max_iterations, occam_target_value):
    # Observations and PET outputs use shared filenames. Serialize the complete
    # acquire/assimilate/record transaction across Streamlit sessions/processes.
    with open('../data/.geosteering-da.lock', 'a') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return _da_unlocked(
            state, input_dict, start_position, localization_enabled,
            localization_strength, assimilation_steps, assimilation_method,
            laplace_max_iterations, occam_max_iterations, occam_target_value,
        )


def record_da_failure(position, error):
    failures = st.session_state.setdefault('da_failures', [])
    message = error if isinstance(error, str) else f"{error.__class__.__name__}: {error}"
    failures.append({'col': int(position[1]), 'error': message})


def record_data_history(position, expected_data_types):
    errors = []
    records = records_from_files(
        position,
        '../data/data.pkl',
        '../data/var.pkl',
        'SaveOutputs/posterior_forecast.npz',
        errors=errors,
    )
    expected_types = set()
    for data_type in expected_data_types:
        if data_type == POINT_DATA_TYPE:
            expected_types.add(data_type)
        else:
            expected_types.update(
                (data_type, component) for component in UDAR_COMPONENTS
            )
    records = [
        record for record in records
        if record['data_type'] in expected_types
    ]
    history = list(st.session_state.get('data_history', []))
    incoming_identities = {
        (record['col'], record['data_type']) for record in records
    }
    history = [
        entry for entry in history
        if (entry['col'], entry.get('data_type')) not in incoming_identities
    ]
    history.extend(records)
    st.session_state['data_history'] = history
    actual_types = {record['data_type'] for record in records}
    missing_count = len(expected_types.difference(actual_types))
    if errors or missing_count:
        record_da_failure(
            position,
            errors[0] if errors else (
                f"forecast contained {len(actual_types)} of "
                f"{len(expected_types)} expected comparison records"
            ),
        )

facies_ensemble_torch = get_earth(state, ensemble_input_dict, earth_model_type)
values_ensemble_torch = evaluate_earth_model_ensemble(facies_ensemble_torch, compute_geobody_sizes=True)
# TODO get the correct visualization

# this is the plotting canvas and the average earth value
value_ensemble = values_ensemble_torch.detach().cpu().numpy()

# vertical Rh section at the current bit column (left panel)
true_facies_torch = true_sim.simulator.NNmodel.eval_gan(true_sim.latent_synthetic_truth)
section_fig = resistivity_section_figure(facies_ensemble_torch, true_facies_torch, start_position)

# build (almost) discrete colorscale for imshow
t_cont = discrete_value_scale()

fig = px.imshow(value_ensemble[:, :, :].mean(axis=0),
                aspect='auto',
                color_continuous_scale=t_cont,
                zmin=value_range[0],
                zmax=value_range[1])
# fig = px.imshow(facies_ensemble[0, :, :], aspect='auto', color_continuous_scale='viridis')

true_values_from_cheat = None
if st.checkbox('Cheat!'):
    # true_gan_output, facies_output = get_gan_truth(true_sim.latent_synthetic_truth)
    true_gan_output = true_sim.simulator.NNmodel.eval_gan(true_sim.latent_synthetic_truth)
    true_values_from_cheat = evaluate_earth_model_ensemble(true_gan_output,
                                                           compute_geobody_sizes=True)
    true_values_np = true_values_from_cheat.detach().cpu().numpy()
    fig = px.imshow(true_values_np[:, :, :].mean(axis=0),
                    aspect='auto',
                    color_continuous_scale=t_cont,
                    zmin=value_range[0],
                    zmax=value_range[1])

show_uncertainty = st.checkbox(
    'Map ensemble uncertainty into lightness',
    key='uncertainty_lightness',
)
uncertainty_threshold = st.number_input(
    'Uncertainty threshold (normalized std)',
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    key='uncertainty_threshold',
    disabled=not show_uncertainty,
)
if show_uncertainty:
    fig.add_trace(go.Heatmap(
        z=uncertainty_wash(
            value_ensemble.std(axis=0),
            float(uncertainty_threshold),
        ),
        zmin=0.0,
        zmax=1.0,
        colorscale=[[0.0, "rgba(255,255,255,0)"], [1.0, "rgba(255,255,255,1)"]],
        showscale=False,
        hoverinfo="skip",
        name="Uncertainty wash",
        showlegend=False,
    ))



# Position the colorbar horizontally below the figure
fig.update_layout(
    coloraxis_colorbar=dict(
        orientation='h',
        x=0.5,
        y=-0.3,
        xanchor='center',
        yanchor='top',
        len=0.8,  # Length of the colorbar
        title="Value"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    height=560,
    margin=dict(t=80,
                l=80,
                r=40,
                b=70
                ), # reserved for legend
    plot_bgcolor="lightgray",
    paper_bgcolor="lightgray",
)

# this draws only the current initial bit position
fig.add_scatter(x=[start_position[1]], y=[start_position[0]],
                mode='markers',
                marker=dict(color='gray', size=10),
                name='Start Position')

# st.write(f'The current position of the well is at: {start_position}')


def apply_user_input(user_choice):
    if not isinstance(user_choice, int):
        user_choice = 0
    next_position = (start_position[0] + user_choice, start_position[1] + 1)
    return next_position

flags_string = ""





def compute_and_apply_robot_suggestion(pessimistic=False, greedy=False):
    # todo maybe we want to remove the if and just pass the argument
    if pessimistic:
        # pessimistic
        next_optimal, paths = pathfinder().no_gan_run(
            weighted_images=values_ensemble_torch,
            start_point=start_position,
            recompute_optimal_paths_from_next=False,
            pessimistic=True
        )
    elif greedy:
        # greedy
        next_optimal, paths = pathfinder().no_gan_run(
            weighted_images=values_ensemble_torch,
            start_point=start_position,
            recompute_optimal_paths_from_next=False,
            greedy=True
        )
    else:
        # optimistic
        next_optimal, paths = pathfinder().no_gan_run(
            weighted_images=values_ensemble_torch,
            start_point=start_position,
            recompute_optimal_paths_from_next=True,
            pessimistic=False
        )
        # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32).to(device),
        #                                    start_position,
        #                                    true_sim.simulator.NNmodel.gan_evaluator)
    return next_optimal, paths

if st.checkbox('Show Greedy suggestion'):
    # let's always show paths with the suggestion
    # let's always show paths with the suggestion
    flags_string += "_greedy"
    # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    next_optimal_g, paths = compute_and_apply_robot_suggestion(
        greedy=True
    )
    if next_optimal_g[0] is None or next_optimal_g[1] is None:
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color='black', size=10, symbol='circle-open'),
            name="Greedy Robot recommends to stop drilling"
        )
    else:
        fig.add_scatter(x=[next_optimal_g[1]], y=[next_optimal_g[0]], mode='markers',
                        marker=dict(color='black', size=10, symbol='circle-open'),
                        name='Greedy Robot suggestion')

if st.checkbox('Show Optimistic DP suggestion and future paths'):
    # let's always show paths with the suggestion
    # let's always show paths with the suggestion
    flags_string += "_optimistic"
    # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    next_optimal_o, paths = compute_and_apply_robot_suggestion(
        pessimistic=False
    )
    if next_optimal_o[0] is None or next_optimal_o[1] is None:
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="black", size=10, symbol="cross"),
            name="Optimistic DP Robot recommends to stop drilling"
        )
    else:
        fig.add_scatter(x=[next_optimal_o[1]], y=[next_optimal_o[0]], mode='markers',
                        marker=dict(color='black', size=10, symbol='cross'),
                        name='Optimistic DP Robot suggestion')
        #
        # # show all the DP paths
        # if st.checkbox('Show Optimistic DP paths'):
        # calculate the robot paths
        # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
        flags_string += "_all"

        # optimal_paths = [perform_dynamic_programming(value_ensemble[j, :, :], next_optimal,
        #                  cost_vector=pathfinder().get_cost_vector())[2] for j in range(state.shape[1])]
        optimal_path = paths
        # plot the optimal paths in the plotly figure
        for j in range(state.shape[1]):
            path_rows, path_cols = zip(*(optimal_path[j]))
            noise_mult = 0.48
            # noise_mult = 0
            path_rows_perturbed = [el + noise_mult * np.random.uniform(-noise_mult, noise_mult) for el in path_rows]
            # path_rows_perturbed = [min(63., max(0., el)) for el in path_rows_perturbed]
            fig.add_trace(
                go.Scatter(x=path_cols, y=path_rows_perturbed, mode='lines',
                           line=dict(color='black', width=0.3),
                           showlegend=False))

if st.checkbox('Show Pessimistic DP suggestion and the future path'):
    flags_string += "_pessimistic"
    # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    next_optimal_p, paths = compute_and_apply_robot_suggestion(pessimistic=True)
    if next_optimal_p[0] is None or next_optimal_p[1] is None:
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color='red', size=10, symbol='x'),
            name="Pessimistic DP Robot recommends to stop drilling"
        )
    else:
        fig.add_scatter(x=[next_optimal_p[1]], y=[next_optimal_p[0]], mode='markers',
                        marker=dict(color='red', size=10, symbol='x'),
                        name='Pessimistic DP Robot suggestion')
        optimal_path = paths
        # plot the optimal paths in the plotly figure
        path_rows, path_cols = zip(*(optimal_path[0]))
        # noise_mult = 0.1
        # # noise_mult = 0
        # path_rows_perturbed = [el + noise_mult * np.random.randn() for el in path_rows]
        fig.add_trace(
            go.Scatter(x=path_cols, y=path_rows, mode='lines',
                       line=dict(color='red', width=2),
                       showlegend=False))

if st.checkbox('Show Human controlls'):
    flags_string += "_human"
    user_selection_dy = st.slider(label='Select drilling direction', key='user_selection',
                                  min_value=-1,
                                  max_value=1,
                                  value=int(0),
                                  step=1)
    # user_step_select = st.radio('What is the next step?', ['Drill up', 'Drill ahead', 'Drill down'])
    # next_position = apply_user_input(user_step_select)
    next_position = apply_user_input(user_selection_dy)
    fig.add_scatter(x=[next_position[1]], y=[next_position[0]], mode='markers',
                    marker=dict(color='blue', size=10), name='Human Selection')

if true_values_from_cheat is not None:
    flags_string += "_cheat"
    # the cheat was activated
    # we draw trajectories over the rest of the interface
    next_optimal_cheat, paths = pathfinder().no_gan_run(
        weighted_images=true_values_from_cheat,
        start_point=start_position
    )
    # fig = px.imshow(1.*np_gan_output[0,1,:,:]+0.5*np_gan_output[0,2,:,:], aspect='auto', color_continuous_scale='viridis')
    fig.add_scatter(x=[next_optimal_cheat[1]], y=[next_optimal_cheat[0]], mode='markers',
                    marker=dict(color='white', size=10, symbol='star'),
                    name='Cheat!')
    optimal_path = paths
    # plot the optimal paths in the plotly figure
    path_rows, path_cols = zip(*(optimal_path[0]))
    # noise_mult = 0.1
    # # noise_mult = 0
    # path_rows_perturbed = [el + noise_mult * np.random.randn() for el in path_rows]
    fig.add_trace(
        go.Scatter(x=path_cols, y=path_rows, mode='lines',
                   line=dict(color='white', width=2),
                   showlegend=False))


path_rows, path_cols = zip(*(st.session_state['path']))
fig.add_trace(go.Scatter(x=path_cols, y=path_rows, mode='lines',
                         line=dict(color='gray', width=4), showlegend=False))

x_values = list(ind*10 for ind in range(1,7))
x_labels = list(f"{x*10} m" for x in x_values)
fig.update_xaxes(
    tickvals=x_values,
    ticktext=x_labels,
    title_text='VS',
    showgrid=False,
    zeroline=False,
    mirror=False,
    minor_ticks=""
)
# TODO check if the axes shape indexes should we swapped - they are the same for now
fig.update_xaxes(
    autorange=False,
    range=[-0.5, value_ensemble.shape[2] - 0.5],
    fixedrange=True,
)


y_values = list(ind*10 for ind in range(1,7))
y_labels = list(f"x{300+int(x/2)} m" for x in y_values)
fig.update_yaxes(
    tickvals=y_values,
    ticktext=y_labels,
    title_text='TVD',
    showgrid=False,
    zeroline=False,
    mirror=False,
    minor_ticks=""
)

# TODO check if the axes shape indexes should we swapped - they are the same for now
fig.update_yaxes(
    autorange=False,
    range=[value_ensemble.shape[1] - 0.5, -0.5],
    fixedrange=True,
)

cur_location = st.session_state['path'][-1]

# Match columns across both rows so data/map x-axes and section/map y-axes align.
history = st.session_state.get('data_history', [])
failures = st.session_state.get('da_failures', [])
failed_cols = sorted({failure['col'] for failure in failures})
available_data_types = list(dict.fromkeys(
    record.get('data_type', data_label) for record in history
))
available_tools = list(dict.fromkeys(
    tool
    for tool in (
        split_data_type(data_type)[0] for data_type in available_data_types
    )
    if tool is not None
))
available_kinds = list(dict.fromkeys(
    split_data_type(data_type)[1] for data_type in available_data_types
))
col_selector, col_data = st.columns([1, 4])
with col_selector:
    if available_data_types:
        if available_tools:
            tool_default = (
                [DEFAULT_TOOL]
                if DEFAULT_TOOL in available_tools
                else available_tools[:1]
            )
            selected_tools = st.multiselect(
                "Tool configuration",
                options=available_tools,
                default=tool_default,
                format_func=data_type_label,
                key="data_history_tools",
            )
        else:
            st.caption("Tool configurations appear once UDAR data is assimilated.")
            selected_tools = []
        kind_default = (
            [DEFAULT_DATA_TYPE]
            if DEFAULT_DATA_TYPE in available_kinds
            else available_kinds[:1]
        )
        selected_kinds = st.multiselect(
            "Data type",
            options=available_kinds,
            default=kind_default,
            format_func=data_type_label,
            key="data_history_data_types",
        )
        selected_data_types = combine_selected_data_types(
            available_data_types, selected_tools, selected_kinds
        )
    else:
        st.caption("Data-match controls appear after the first assimilation step.")
        selected_data_types = []
nonfinite_cols = sorted({
    record['col']
    for record in history
    if record.get('data_type', data_label) in selected_data_types
    and not record_is_finite(record)
})
drilled_cols = {
    int(position[1])
    for position in st.session_state.get('path', [])
    if int(position[1]) > 0
}
missing_cols = missing_record_columns(
    history, drilled_cols, selected_data_types, data_label
)
comparison_failed_cols = sorted(
    set(failed_cols).union(nonfinite_cols, missing_cols)
)
with col_data:
    st.plotly_chart(
        data_history_figure(
            history,
            y_label=data_label,
            selected_types=selected_data_types,
            failed_cols=comparison_failed_cols,
        ),
        use_container_width=True,
    )

col_section, col_map = st.columns([1, 4])
with col_section:
    st.plotly_chart(section_fig, use_container_width=True)
with col_map:
    st.plotly_chart(fig, use_container_width=True)

    fig.write_image(f"figures/output_{int(cur_location[1])}_{int(cur_location[0])}{flags_string}.png",
                    width=700,
                    height=450,
                    scale=4
                    )
    print(f"output_{int(cur_location[1])}_{int(cur_location[0])}{flags_string} saved!")

if comparison_failed_cols:
    last_error = f" Last error: {failures[-1]['error']}" if failures else ""
    st.warning(
        "Data comparison unavailable at column(s) "
        f"{', '.join(str(col) for col in comparison_failed_cols)} because data "
        "or predictions are missing or non-finite (red lines in the data-match plot)."
        f"{last_error}"
    )


selected_assimilation_steps = st.pills(
    "Data-assimilation simulators",
    options=["0D", "1D"],
    default=["1D"],
    selection_mode="multi",
    key="assimilation_simulators",
    help="Select both to assimilate direct point data first and directional data second.",
)
assimilation_steps = ordered_assimilation_steps(selected_assimilation_steps)
assimilation_disabled = not assimilation_steps
if assimilation_method in {"Laplace", "Occam"} and earth_model_type != "pixel":
    assimilation_disabled = True
    st.error(f"{assimilation_method} assimilation currently requires the pixel-based earth model.")
if len(assimilation_steps) > 1:
    st.caption(
        "0D assimilates direct [ln Rh, ln Rv] point data first; 1D then "
        "assimilates the directional response."
    )
if not assimilation_steps:
    st.error("Select at least one data-assimilation simulator.")


def drill_to_position(state, next_position):
    print(f"Shape of state for DA {state.shape}")
    updated_state = da(
        state,
        ensemble_input_dict,
        next_position,
        st.session_state.get('localization_enabled', False),
        st.session_state.get('localization_strength', 0.9),
        assimilation_steps,
        assimilation_method,
        laplace_max_iterations,
        occam_max_iterations,
        occam_target_value,
    )
    if updated_state is None:
        return None
    st.session_state.update({
        'ensemble_state': updated_state,
        'start_position_state': next_position,
    })
    return next_position


def drill_like_human(state):
    return drill_to_position(state, apply_user_input(0))


def drill_like_optimist_robot(state, next_optimal_o):
    return drill_to_position(state, next_optimal_o)


def drill_like_pessimist_robot(state, next_optimal_p):
    return drill_to_position(state, next_optimal_p)


col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Drill like a Human', disabled=assimilation_disabled):
        if drill_like_human(state) is None:
            st.rerun()
        toggle_first_step_and_rerun()
with col2:
    if st.button('Drill like Optimistic Robot', disabled=assimilation_disabled):
        if next_optimal_o is None:
            next_optimal_o, _ = compute_and_apply_robot_suggestion(pessimistic=False)
        if drill_like_optimist_robot(state, next_optimal_o) is None:
            st.rerun()
        toggle_first_step_and_rerun()
with col3:
    if st.button('Drill like Pessimistic Robot', disabled=assimilation_disabled):
        if next_optimal_p is None:
            next_optimal_p, _ = compute_and_apply_robot_suggestion(pessimistic=True)
        if drill_like_pessimist_robot(state, next_optimal_p) is None:
            st.rerun()
        toggle_first_step_and_rerun()


def should_stop_autopilot(start_pos, opt_result):
    print("Checking exit conditions")
    print(f'Starting position {start_pos}')
    print(f'Optimization result {opt_result}')
    print("Going forward")
    if opt_result is None:
        return True
    if opt_result[0] is None or opt_result[1] is None:
        return True
    if opt_result[1] == start_pos[1]:
        return True
    return False


auto_col1, auto_col2 = st.columns(2)
with auto_col1:
    auto_opt = st.checkbox(
        'Activate Optimistic Robot Autopilot',
        key="auto_opt",
        disabled=assimilation_disabled,
    )
    if auto_opt and not assimilation_disabled:
        if next_optimal_o is None:
            next_optimal_o, _ = compute_and_apply_robot_suggestion(pessimistic=False)
        if should_stop_autopilot(start_position, next_optimal_o):
            pass
        else:
            if drill_like_optimist_robot(state, next_optimal_o) is None:
                st.session_state.auto_opt = False
                st.rerun()
            toggle_first_step_and_rerun()
with auto_col2:
    auto_pes = st.checkbox(
        'Activate Pessimistic Robot Autopilot',
        key="auto_pes",
        disabled=assimilation_disabled,
    )
    if auto_pes and not assimilation_disabled:
        if next_optimal_p is None:
            next_optimal_p, _ = compute_and_apply_robot_suggestion(pessimistic=True)
        if should_stop_autopilot(start_position, next_optimal_p):
            pass
            # st.session_state.auto_pes_ = False
        else:
            if drill_like_pessimist_robot(state, next_optimal_p) is None:
                st.session_state.auto_pes = False
                st.rerun()
            toggle_first_step_and_rerun()
