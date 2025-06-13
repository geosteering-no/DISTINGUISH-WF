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

from write_data_var import new_data

import numpy as np

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plot_for_app import earth
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, rgb_to_hsv, hsv_to_rgb
import matplotlib

from copy import deepcopy as dp
import time

global_extent = [0, 640, -16.25, 15.75]
norm = Normalize(vmin=0.0, vmax=1)

weights_folder = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
scalers_folder = weights_folder
full_em_model_file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"
file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/gan/netG_epoch_15000.pth"

# build a streamlit app to run the workflow. On the first run of the app we will be in the initial state.
# The user has to specify the start position of the well. Number of decissions are always 1, and the user have to specify
# wheter to drill ahead, or to stop drilling. If the user decides to stop drilling, the app will stop. If the user decides to drill ahead, the main function will be called.

if 'first_position' not in st.session_state:
    st.session_state['first_position'] = True
# Also, plot the current state as a main feature of the app.
st.title('DISTIGUISH Workflow')

# Show a slider first to select the start position of the well
if st.session_state.first_position:
    state = np.load('../orig_prior_small.npz')['x']  # the prior latent vector
    start_position = (st.slider(label='Enter the horizontal start position of the well', key='start_position',
                                min_value=0, max_value=64, value=int(32)), 0)
    st.session_state['path'] = [start_position]
else:
    state = st.session_state.state
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
    true_facies = earth(torch.tensor(state,dtype=torch.float32))

    weights = np.array([-0.1, 1, 0.5])
    gan = np.mean(true_facies * weights.reshape(1, 3, 1,1),axis=1)  # Apply weights to the true facies

    return gan, true_facies

@st.cache_data
def da(state, start_position):
    num_decissions = 1  # 64 # number of decissions to make

    # start_position = (32, 0) # initial position of the well
    # state = np.load('orig_prior.npz')['m'] # the prior latent vector
    np.savez('prior.npz', **{'x': state})  # save the prior as a file

    kf = {'bit_pos': [start_position],
          'vec_size': 60,
          'reporttype': 'pos',
          'reportpoint': [int(el) for el in range(1)],
          'datatype': [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),
                       ('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')],
          'parallel': 1,
          'file_name': file_name,
          'full_em_model_file_name': full_em_model_file_name,
         'scalers_folder': scalers_folder}

    sim = GeoSim(kf)

    kd, kf = read_config.read_txt('DA.pipt')  # read the config file.

    for i in range(num_decissions):
        # start by assimilating data at the current position

        # make a set of syntetic data for the current position
        new_data({'bit_pos': [start_position],
                  'vec_size': 60,
                  'reporttype': 'pos',
                  'reportpoint': [int(el) for el in range(1)],
                  'datatype': [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),
                               ('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]
                  })
        # do inversion
        sim.update_bit_pos([start_position])
        analysis = pipt_init.init_da(kd, kd, sim)  # Re-initialize the data assimilation to read the new data
        assimilation = Assimilate(analysis)
        assimilation.run()

        state = np.load('posterior_state_estimate.npz')['x']  # import the posterior state estimate
        return state

gan, true_facies = get_gan_earth(state)

mean_rh = gan.mean(axis=0)
std_rh = gan.std(axis=0)

# Normalize std deviation to 0-1
std_norm = (std_rh - std_rh.min()) / (std_rh.max() - std_rh.min())

# Switch
view_mode = st.radio("Select visualization mode:", ["Standard Heatmap", "Uncertainty Modulated"])

if view_mode == "Standard Heatmap":
    # Plotly visualization
    fig = px.imshow(mean_rh, aspect='auto', color_continuous_scale='viridis')
    fig.update_layout(coloraxis_colorbar=dict(
        orientation='h',
        x=0.5,
        y=-0.2,
        xanchor='center',
        yanchor='top',
        len=0.8,
    ))
    fig.add_trace(go.Scatter(x=[start_position[1]], y=[start_position[0]], mode='markers',
                             marker=dict(color='red', size=10), name='Start Position'))
  #  st.plotly_chart(fig, use_container_width=True)

else:

    # HSV color modulation based on uncertainty
    cmap = plt.cm.jet
    norm = Normalize(vmin=np.min(mean_rh), vmax=np.max(mean_rh))
    rgb_image = cmap(norm(mean_rh))[..., :3]
    hsv_image = rgb_to_hsv(rgb_image)
    hsv_image[..., 1] = 1 - std_norm  # Reduce saturation where uncertainty is high
    rgb_modulated = hsv_to_rgb(hsv_image)

    # Create Plotly figure for the main image
    fig = go.Figure()

    fig.add_trace(go.Image(
    z=(rgb_modulated * 255).astype(np.uint8),
    x0=0,
    dx=2,  # stretch in x: each pixel spans 2 units
    y0=0,
    dy=1   # y spacing remains normal
    ))

    # Add marker
    fig.add_trace(go.Scatter(
    x=[start_position[1]],
    y=[start_position[0]],
    mode='markers',
    marker=dict(color='red', size=10),
    name='Start Position'
    ))

    fig.update_layout( title="Mean Values Modulated by Uncertainty",
    margin=dict(t=50, b=50),
    width=800,   # increase width
    height=400,  # decrease height
    xaxis=dict(
        showticklabels=False,
        constrain="domain",  # avoid axis stretching beyond layout,
        scaleanchor=None
    ),
    yaxis=dict(
        showticklabels=False,
        scaleanchor=None    # ensure no 1:1 lock
    )
    )
#    st.plotly_chart(fig, use_container_width=True)

    # Now build the 2D colorbar as a separate plot
    uncertainty_vals = np.linspace(0, 1, 100)
    resistivity_vals = np.linspace(norm.vmin, norm.vmax, 100)
    U, R = np.meshgrid(uncertainty_vals, resistivity_vals)
    cb_colors = cmap(Normalize()(R))[..., :3]
    hsv_cb_colors = rgb_to_hsv(cb_colors)
    hsv_cb_colors[..., 1] = 1 - U
    cb_colors_sat = hsv_to_rgb(hsv_cb_colors)

    fig_cb = go.Figure()

    fig_cb.add_trace(go.Image(
    z=(cb_colors_sat * 255).astype(np.uint8),
        x0=0,
        dx=3,  # stretch in x: each pixel spans 2 units
        y0=0,
        dy=1  # y spacing remains normal
    ))

    fig_cb.update_layout(
    xaxis=dict(title='Uncertainty (normalized)', tickvals=[0, 50, 100], ticktext=['0', '0.5', '1']),
    yaxis=dict(title='Mean Value'),
    margin=dict(t=50),
    width=300,
    height=200
    )

    #st.plotly_chart(fig_cb, use_container_width=True)

    # Matplotlib HSV saturation-modulated colormap
    #cmap = plt.cm.jet
    #norm = Normalize(vmin=np.min(mean_rh), vmax=np.max(mean_rh))#

    #rgb_image = cmap(norm(mean_rh))[..., :3]
    #hsv_image = rgb_to_hsv(rgb_image)
    #hsv_image[..., 1] = 1 - std_norm  # adjust saturation by uncertainty
    #rgb_modulated = hsv_to_rgb(hsv_image)

#    fig2, ax = plt.subplots()
#    im = ax.imshow(rgb_modulated, origin='lower', aspect='auto')
#    ax.plot(start_position[1], start_position[0], 'ro', label='Start Position')
#    ax.legend()

    # Add custom 2D colorbar
#    cb_ax = fig2.add_axes([0.75, 0.1, 0.15, 0.25])
#    uncertainty_vals = np.linspace(0, 1, 100)
#    resistivity_vals = np.linspace(norm.vmin, norm.vmax, 100)
#    U, R = np.meshgrid(uncertainty_vals, resistivity_vals)
#    cb_colors = cmap(Normalize()(R))[..., :3]
#    hsv_cb_colors = rgb_to_hsv(cb_colors)
#    hsv_cb_colors[..., 1] = 1 - U
#    cb_colors_sat = hsv_to_rgb(hsv_cb_colors)

#    cb_ax.imshow(cb_colors_sat, origin='lower', aspect='auto', extent=[0, 1, norm.vmin, norm.vmax])
#    cb_ax.set_xlabel('Uncertainty (normalized)')
#    cb_ax.set_ylabel('Mean Value')
#    cb_ax.set_title('2D Colorbar (Saturation)')

#    st.pyplot(fig2)

st.write(f'The current position of the well is at: {start_position}')

if st.checkbox('Show Human suggestion'):
    user_step_select = st.radio('What is the next step?', ['Drill up', 'Drill ahead', 'Drill down'])
    if user_step_select == 'Drill up':
        next_position = (start_position[0] - 1, start_position[1] + 1)
    elif user_step_select == 'Drill ahead':
        next_position = (start_position[0], start_position[1] + 1)
    else:
        next_position = (start_position[0] + 1, start_position[1] + 1)

    fig.add_scatter(x=[next_position[1]], y=[next_position[0]], mode='markers',
                    marker=dict(color='blue', size=10), name='Next Position')

# ask the user to show all the DP paths
if st.checkbox('Show Robot paths'):
    # calculate the robot paths
    next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    optimal_paths = [perform_dynamic_programming(gan[j, :, :], start_position,
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
    next_optimal, _ = pathfinder().run(torch.tensor(state,dtype=torch.float32), start_position)
    fig.add_scatter(x=[next_optimal[1]], y=[next_optimal[0]], mode='markers',
                    marker=dict(color='black', size=10, symbol='cross'),
                    name='Robot suggestion')

if st.checkbox('Cheat!'):
    x = np.array(range(64))
    y = np.array(range(64))
    X, Y = np.meshgrid(x, y)
    Z = true_facies[0, :, :]
    fig.add_trace(go.Contour(z=Z, x=x, y=y, contours=dict(
        start=-1,
        end=1,
        size=1.9,
        coloring='none'
    ),
                             line=dict(
                                 width=2,
                                 color='white',
                                 dash='dash'
                             ),
                             showscale=False  # Hide the color scale for the contour
                             ))

path_rows, path_cols = zip(*(st.session_state['path']))
fig.add_trace(go.Scatter(x=path_cols, y=path_rows, mode='lines',
                         line=dict(color='red', width=4), showlegend=False))

st.plotly_chart(fig, use_container_width=True)
if view_mode == "Uncertainty Modulated":
    st.plotly_chart(fig_cb, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if st.button('Drill like a Human'):
        st.session_state.start_position_state = next_position
        state = da(state, next_position)
        st.session_state.state = state
        toggle_first_step()
with col2:
    if st.button('Drill like a Robot'):
        st.session_state.start_position_state = next_optimal
        state = da(state, next_optimal)
        st.session_state.state = state
        toggle_first_step()
