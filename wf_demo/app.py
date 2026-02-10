import os
os.environ["MPLBACKEND"] = "Agg"  # must be before importing matplotlib

import matplotlib
matplotlib.use("Agg")


from matplotlib import colormaps
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

from wf_demo.default_load import input_dict, load_default_latent_tensor, load_default_starting_ensemble_state


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
st.title('Distinguish Open Workflow')
# GMO refers to Generic Modern [UDAR] Observations

next_optimal_o = None
next_optimal_p = None
# this creates an instance if a simulator for synthetic truth


true_sim = SyntheticTruth(latent_truth_vector=load_default_latent_tensor().to(device), device=device)

# Show a slider first to select the start position of the well
if st.session_state.first_position:
    # state = np.load('../orig_prior_small.npz')['x']  # the prior latent vector
    state = load_default_starting_ensemble_state()
    # the commented code loads the truth as the state for checking correctness
    # state_torch = load_default_latent_tensor().cpu()
    # state = state_torch.permute(1,0).numpy()
    print(f'State tensor shape {state.shape}')
    start_position = (st.slider(label='Enter the horizontal start position of the well', key='start_position',
                                min_value=0, max_value=64, value=int(31)), 0)
    st.session_state['path'] = [start_position]
else:
    state = st.session_state.ensemble_state
    start_position = st.session_state.start_position_state
    st.session_state['path'].append(start_position)

# toggle first step
def toggle_first_step():
    st.session_state['first_position'] = False
    st.rerun()

# plot the current state
@st.cache_data
def get_gan_earth(state):
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

@st.cache_data
def da(state, start_position):
    num_decissions = 1  # 64 # number of decissions to make

    # start_position = (32, 0) # initial position of the well
    # state = np.load('orig_prior_2024.npz')['m'] # the prior latent vector
    np.savez('prior.npz', **{'x': state})  # save the prior as a file

    keys_filter = input_dict.copy()

    keys_filter['bit_pos'] = [start_position]

    sim_ensemble = GeoSim(keys_filter)

    keys_data, _ = read_config.read_txt('DA.pipt')  # read the config file.

    for i in range(num_decissions):
        # start by assimilating data at the current position

        # make a set of syntetic data for the current position
        # todo check if we need to pass keys_data
        true_sim.acquire_data({'bit_pos': [start_position],
                              'vec_size': 60,
                              'reporttype': 'pos',
                              'reportpoint': [int(el) for el in range(1)],
                              'datatype': [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),
                                           ('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]
                              })
        # do inversion
        sim_ensemble.update_bit_pos([start_position])
        analysis = pipt_init.init_da(keys_data, keys_data, sim_ensemble)  # Re-initialize the data assimilation to read the new data
        assimilation = Assimilate(analysis)
        assimilation.run()

        state = np.load('posterior_state_estimate.npz')['x']  # import the posterior state estimate
        return state

facies_ensemble_torch = get_gan_earth(state)
values_ensemble_torch = evaluate_earth_model_ensemble(facies_ensemble_torch, compute_geobody_sizes=True)
# TODO get the correct visualization

# this is the plotting canvas and the average earth value
value_ensemble = values_ensemble_torch.detach().cpu().numpy()

# this is a ChatGPT-suggested trick to make continuous palette of discrete
# 20 discrete colors from matplotlib tab20 -> Plotly rgb strings
colors_from_map = colormaps["tab20b"]
rgb = colors_from_map.colors
# cmap = cm.get_cmap("tab20c", 20)   # force 20 discrete entries
# tab20c = [cmap(i) for i in range(20)]
colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r,g,b in rgb]
# colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r,g,b,_ in tab20c]

# build (almost) discrete colorscale for imshow
n = len(colors)
eps = 0
t_cont = []
for i, c in enumerate(colors):
    lo = i / n
    hi = (i + 1) / n
    t_cont += [(lo + eps, c), (hi - eps, c)]  # 2 points per color

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
        orientation="v",  # column
        yanchor="top",
        y=1.5,
        xanchor="center",
        x=0.5,
    ),
    margin=dict(t=120,
                r=30,
                l=30
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

if st.checkbox('Show Human suggestion'):
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



def compute_and_apply_robot_suggestion(pessimistic=False):
    # todo maybe we want to remove the if and just pass the argument
    if pessimistic:
        # pessimistic
        next_optimal, paths = pathfinder().no_gan_run(
            weighted_images=values_ensemble_torch,
            start_point=start_position,
            recompute_optimal_paths_from_next=False,
            pessimistic=True
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

if st.checkbox('Show Optimistic DP suggestion and future paths'):
    # let's always show paths with the suggestion
    # let's always show paths with the suggestion
    flags_string += "_optimistic"
    # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    next_optimal_o, paths = compute_and_apply_robot_suggestion(
        pessimistic=False
    )
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
        noise_mult = 0.1
        # noise_mult = 0
        path_rows_perturbed = [el + noise_mult * np.random.randn() for el in path_rows]
        fig.add_trace(
            go.Scatter(x=path_cols, y=path_rows_perturbed, mode='lines',
                       line=dict(color='black', width=0.5),
                       showlegend=False))

if st.checkbox('Show Pessimistic DP suggestion and the future path'):
    flags_string += "_pessimistic"
    # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    next_optimal_p, paths = compute_and_apply_robot_suggestion(pessimistic=True)
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
    title_text='VS'
)
y_values = list(ind*10 for ind in range(1,7))
y_labels = list(f"x{300+int(x/2)} m" for x in x_values)
fig.update_yaxes(
    tickvals=y_values,
    ticktext=y_labels,
    title_text='TVD'
)

cur_location = st.session_state['path'][-1]
fig.write_image(f"figures/output_{int(cur_location[1])}_{int(cur_location[0])}.png")

st.plotly_chart(fig, use_container_width=True)



col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Drill like a Human'):
        next_position = apply_user_input(0)
        st.session_state.start_position_state = next_position
        print(f"Shape of state for DA {state.shape}")
        state = da(state, next_position)
        st.session_state.ensemble_state = state
        toggle_first_step()
with col2:
    if st.button('Drill like Optimistic Robot'):
        if next_optimal_o is None:
            next_optimal_o, _ = compute_and_apply_robot_suggestion(pessimistic=False)
        st.session_state.start_position_state = next_optimal_o
        print(f"Shape of state for DA {state.shape}")
        state = da(state, next_optimal_o)
        st.session_state.ensemble_state = state
        toggle_first_step()
with col3:
    if st.button('Drill like Pessimistic Robot'):
        if next_optimal_p is None:
            next_optimal_p, _ = compute_and_apply_robot_suggestion(pessimistic=True)
        st.session_state.start_position_state = next_optimal_p
        print(f"Shape of state for DA {state.shape}")
        state = da(state, next_optimal_p)
        st.session_state.ensemble_state = state
        toggle_first_step()
