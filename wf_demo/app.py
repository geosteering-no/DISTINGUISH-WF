import numpy as np
import warnings

import torch

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# warnings.filterwarnings("always")
from pathoptim.pathOPTIM import pathfinder
from pathoptim.DP import perform_dynamic_programming
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

from wf_demo.default_load import input_dict, load_default_latent_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global_extent = [0, 640, -16.25, 15.75]
norm = Normalize(vmin=0.0, vmax=1)

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
st.title('Distinguish Open Workflow (GMO)')
# GMO refers to Generic Modern [UDAR] Observations

next_optimal = None
# this creates an instance if a simulator for synthetic truth


true_sim = SyntheticTruth(latent_truth_vector=load_default_latent_tensor().to(device), device=device)

# Show a slider first to select the start position of the well
if st.session_state.first_position:
    # state = np.load('../orig_prior_small.npz')['x']  # the prior latent vector
    state = np.load('../orig_prior_2024.npz')['m']
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
    facies_ensemble = earth(torch.tensor(state, dtype=torch.float32).to(device), simulator=sim_ensemble)

    weights = np.array([-0.1, 1, 0.5])
    value_ensemble = np.mean(facies_ensemble * weights.reshape(1, 3, 1, 1), axis=1)  # Apply weights to the true facies

    return value_ensemble, facies_ensemble

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

value_ensemble, facies_ensemble = get_gan_earth(state)

# this is the plotting canvas and the average earth value
fig = px.imshow(value_ensemble[:,:,:].mean(axis=0), aspect='auto', color_continuous_scale='viridis')
# fig = px.imshow(facies_ensemble[0, :, :], aspect='auto', color_continuous_scale='viridis')

# Position the colorbar horizontally below the figure
fig.update_layout(coloraxis_colorbar=dict(
    orientation='h',
    x=0.5,
    y=-0.2,
    xanchor='center',
    yanchor='top',
    len=0.8,  # Length of the colorbar
))

# this draws only the current initial bit position
fig.add_scatter(x=[start_position[1]], y=[start_position[0]], mode='markers', marker=dict(color='red', size=10),
                name='Start Position')

st.write(f'The current position of the well is at: {start_position}')


def apply_user_input(user_choice: str = None):
    if user_choice == 'Drill up':
        next_position = (start_position[0] - 1, start_position[1] + 1)
    elif user_choice == 'Drill ahead':
        next_position = (start_position[0], start_position[1] + 1)
    else:
        next_position = (start_position[0] + 1, start_position[1] + 1)
    return next_position

if st.checkbox('Cheat!'):
    # true_gan_output, facies_output = get_gan_truth(true_sim.latent_synthetic_truth)
    true_gan_output = true_sim.simulator.NNmodel.eval_gan(true_sim.latent_synthetic_truth)
    print(f"The output {true_gan_output}")
    print(f"Output shape {true_gan_output.shape}")
    np_gan_output = true_gan_output.cpu().numpy()
    print(f"The output {np_gan_output}")
    print(f"Output shape {np_gan_output.shape}")
    # todo improve from the hard-coded constants, see get_gan_...
    fig = px.imshow(1.*np_gan_output[0,1,:,:]+0.5*np_gan_output[0,2,:,:], aspect='auto', color_continuous_scale='viridis')

    #
    #     x = np.array(range(64))
    #     y = np.array(range(64))
    #     X, Y = np.meshgrid(x, y)
    #     Z = np_gan_output[0,0,:,:]
    #     fig.add_trace(go.Contour(z=Z, x=x, y=y, contours=dict(
    #         start=-1,
    #         end=1,
    #         size=1.9,
    #         coloring='none'
    #     ),
    #                              line=dict(
    #                                  width=2,
    #                                  color='white',
    #                                  dash='dash'
    #                              ),
    #                              showscale=False  # Hide the color scale for the contour
    #                              ))



if st.checkbox('Show Human suggestion'):
    user_step_select = st.radio('What is the next step?', ['Drill up', 'Drill ahead', 'Drill down'])
    next_position = apply_user_input(user_step_select)
    fig.add_scatter(x=[next_position[1]], y=[next_position[0]], mode='markers',
                    marker=dict(color='blue', size=10), name='Next Position')



def compute_and_apply_robot_suggestion():
    next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32).to(device),
                                       start_position,
                                       true_sim.simulator.NNmodel.gan_evaluator)
    return next_optimal


# show all the DP paths
# TODO show paths starting from the suggestion instead
if st.checkbox('Show Robot paths'):
    # calculate the robot paths
    # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    next_optimal = compute_and_apply_robot_suggestion()
    optimal_paths = [perform_dynamic_programming(value_ensemble[j, :, :], start_position,
                                                 cost_vector=pathfinder().get_cost_vector())[2] for j in
                     range(state.shape[1])]
    # plot the optimal paths in the plotly figure
    for j in range(state.shape[1]):
        path_rows, path_cols = zip(*(optimal_paths[j]))
        noise_mult = 0.2
        # noise_mult = 0
        row_list = [el + noise_mult * np.random.randn() for el in path_rows]
        fig.add_trace(
            go.Scatter(x=path_cols, y=path_rows, mode='lines', line=dict(color='black'), showlegend=False))

if st.checkbox('Show Robot suggestion'):
    # next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    next_optimal = compute_and_apply_robot_suggestion()
    fig.add_scatter(x=[next_optimal[1]], y=[next_optimal[0]], mode='markers',
                    marker=dict(color='black', size=10, symbol='cross'),
                    name='Robot suggestion')



path_rows, path_cols = zip(*(st.session_state['path']))
fig.add_trace(go.Scatter(x=path_cols, y=path_rows, mode='lines',
                         line=dict(color='red', width=4), showlegend=False))

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if st.button('Drill like a Human'):
        next_position = apply_user_input()
        st.session_state.start_position_state = next_position
        state = da(state, next_position)
        st.session_state.ensemble_state = state
        toggle_first_step()
with col2:
    if st.button('Drill like a Robot'):
        if next_optimal is None:
            next_optimal = compute_and_apply_robot_suggestion()
        st.session_state.start_position_state = next_optimal
        state = da(state, next_optimal)
        st.session_state.ensemble_state = state
        toggle_first_step()