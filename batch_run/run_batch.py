import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
import torch

from wf_demo.default_load import input_dict

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

from wf_demo.default_load import input_dict, load_default_latent_tensor
from wf_demo.write_data_var import new_data



global_extent = [0, 640, -16.25, 15.75]
norm = Normalize(vmin=0.0, vmax=1)



def da(state, start_position, input_dict):
    num_decissions = 1  # 64 # number of decissions to make

    # start_position = (32, 0) # initial position of the well
    # state = np.load('orig_prior.npz')['m'] # the prior latent vector
    np.savez('prior.npz', **{'x': state})  # save the prior as a file

    kf = input_dict.copy()
    kf['start_pos'] = [start_position]

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


def compute_and_apply_robot_suggestion(state, start_position):
    next_optimal, _ = pathfinder().run(torch.tensor(state, dtype=torch.float32), start_position)
    return next_optimal

if __name__ == "__main__":
    # get intitial state
    ensemble_state = np.load('../orig_prior_small.npz')['x']  # the prior latent vector
    start_position = (32, 0)
    path = [start_position]

    # todo change the steps
    for i in range(1, 10):
        next_optimal = compute_and_apply_robot_suggestion(ensemble_state, start_position)
        path.append(next_optimal)
        start_position = next_optimal
        ensemble_state = da(ensemble_state, next_optimal, input_dict=input_dict)

