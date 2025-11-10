import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
import torch

from wf_demo.app import optimal_paths

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

from wf_demo.default_load import input_dict, load_default_latent_tensor, load_default_starting_ensemble_state, \
    gan_file_name, full_em_model_file_name, scalers_folder, proxi_input_shape, proxi_output_shape, gan_input_dim, \
    gan_output_height, gan_num_channals
from wf_demo.write_data_var import SyntheticTruth
from plot_result import plot_results_one_step
from NeuralSim.vector_to_log import FullModel





# global_extent = [0, 640, -16.25, 15.75]
# norm = Normalize(vmin=0.0, vmax=1)



def da(state, start_position, input_dict):
    num_decissions = 1  # 64 # number of decissions to make

    # start_position = (32, 0) # initial position of the well
    # state = np.load('orig_prior_2024.npz')['m'] # the prior latent vector
    np.savez('prior.npz', **{'x': state})  # save the prior as a file

    kf = input_dict.copy()
    kf['start_pos'] = [start_position]



    kd, kf = read_config.read_txt('DA.pipt')  # read the config file.

    # start by assimilating data at the current position

    # make a set of syntetic data for the current position
    true_sim.acquire_data({'bit_pos': [start_position],
                           'vec_size': 60,
                           'reporttype': 'pos',
                           'reportpoint': [int(el) for el in range(1)],
                           'datatype': [('6kHz', '83ft'), ('12kHz', '83ft'), ('24kHz', '83ft'),
                                        ('24kHz', '43ft'), ('48kHz', '43ft'), ('96kHz', '43ft')]
                           })

    sim = GeoSim(kf)
    # do inversion
    sim.update_bit_pos([start_position])
    analysis = pipt_init.init_da(kd, kd, sim)  # Re-initialize the data assimilation to read the new data
    assimilation = Assimilate(analysis)
    assimilation.run()

    state = np.load('posterior_state_estimate.npz')['x']  # import the posterior state estimate
    return state


def compute_and_apply_robot_suggestion(state, start_position, true_gan_sim):
    next_optimal, _ = pathfinder().run(torch.tensor(state, dtype=torch.float32).to(device),
                                       start_position,
                                       true_gan_sim)
    return next_optimal


def batch_run(starting_position=(32,0), true_realization_id="C1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get truth
    synthetic_truth_vector = load_default_latent_tensor(true_realization_id)

    # setting simulators
    true_sim = SyntheticTruth(synthetic_truth_vector.to(device),
                              device=device)
    geosim_ensemble = FullModel(latent_size=gan_input_dim,
                                gan_save_file=gan_file_name,
                                proxi_save_file=full_em_model_file_name,
                                proxi_scalers=scalers_folder,
                                proxi_input_shape=proxi_input_shape,
                                proxi_output_shape=proxi_output_shape,
                                gan_output_height=gan_output_height,
                                num_img_channels=gan_num_channals,
                                gan_correct_orientation=False,
                                device=device
                                )


    true_res_image = true_sim.simulator.NNmodel.gan_evaluator.eval(synthetic_truth_vector)

    # get intitial state
    ensemble_state = load_default_starting_ensemble_state()

    # drilling position
    start_position = starting_position
    path = [start_position]

    # compute optimal path
    optimal_path = None
    all_paths = None


    # pre-job plotting
    plot_results_one_step(ensemble_vectors=ensemble_state,
                          full_nn_model=geosim_ensemble,
                          true_res_image=true_res_image,
                          drilled_path=path,
                          optimal_path=optimal_path,
                          all_paths=all_paths
                          )



    # todo change the steps
    for i in range(1, 10):
        next_optimal = compute_and_apply_robot_suggestion(ensemble_state,
                                                          start_position,
                                                          true_sim.simulator.NNmodel.gan_evaluator)
        path.append(next_optimal)
        start_position = next_optimal
        ensemble_state = da(ensemble_state, next_optimal, input_dict=input_dict)


if __name__ == "__main__":
    # todo add parameters according to the tests
    batch_run()