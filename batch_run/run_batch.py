import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
import torch
import os
import random

from wf_demo.write_data_var import SyntheticTruth

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

from wf_demo.default_load import input_dict, load_default_latent_tensor, load_default_starting_ensemble_state, \
    gan_file_name, full_em_model_file_name, scalers_folder, proxi_input_shape, proxi_output_shape, gan_input_dim, \
    gan_output_height, gan_num_channals
from plot_result import plot_results_one_step
from NeuralSim.vector_to_log import FullModel





# global_extent = [0, 640, -16.25, 15.75]
# norm = Normalize(vmin=0.0, vmax=1)



def fix_seed(seed: int = 0) -> None:
    # 1) Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch (CPU)
    torch.manual_seed(seed)

    # 4) PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU

    # 5) cuDNN / deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional but stronger: error out on non-deterministic ops (PyTorch ≥1.8)
    torch.use_deterministic_algorithms(True)


def da(state_torch, start_position, input_dict, true_sim):
    num_decissions = 1  # 64 # number of decissions to make

    # start_position = (32, 0) # initial position of the well
    # state = np.load('orig_prior_2024.npz')['m'] # the prior latent vector
    state_np = state_torch.to('cpu').numpy().T
    np.savez('prior.npz', **{'x': state_np})  # save the prior as a file

    keys_filter = input_dict.copy()
    keys_filter['bit_pos'] = [start_position]


    keys_data, _ = read_config.read_txt('DA.pipt')  # read the config file.

    # start by assimilating data at the current position

    # make a set of syntetic data for the current position
    true_sim.acquire_data({'bit_pos': [start_position],
                           'vec_size': 60,
                           'reporttype': 'pos',
                           'reportpoint': [int(el) for el in range(1)],
                           'datatype': [('6kHz', '83ft'), ('12kHz', '83ft'), ('24kHz', '83ft'),
                                        ('24kHz', '43ft'), ('48kHz', '43ft'), ('96kHz', '43ft')]
                           })

    sim_ensemble = GeoSim(keys_filter)
    # do inversion
    sim_ensemble.update_bit_pos([start_position])
    analysis = pipt_init.init_da(keys_data, keys_data, sim_ensemble)  # Re-initialize the data assimilation to read the new data
    assimilation = Assimilate(analysis)
    assimilation.run()

    state_np = np.load('posterior_state_estimate.npz')['x']  # import the posterior state estimate

    state_torch = torch.tensor(state_np.T,
                               device=state_torch.device,
                               dtype=state_torch.dtype
                               )
    return state_torch


def compute_and_apply_robot_suggestion(state, start_position, true_gan_sim, device):
    next_optimal, _ = pathfinder().run(torch.tensor(state, dtype=torch.float32).to(device),
                                       start_position,
                                       true_gan_sim)
    return next_optimal


def batch_run(starting_position=(31,0), true_realization_id="C1", seed=0, discount_factor=1.0):

    fix_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get truth
    synthetic_truth_vector = load_default_latent_tensor(true_realization_id).to(device)

    # get intitial state
    ensemble_state = load_default_starting_ensemble_state()
    ensemble_state_torch = torch.tensor(ensemble_state.T, dtype=torch.float32, device=device)

    # setting simulators
    # this simulator is for data assimilation and should perhaps be optimized
    true_sim_for_da = SyntheticTruth(latent_truth_vector=synthetic_truth_vector,
                                     device=device)

    # this is the main mapping system for the batch run (for the truth)
    # (can be different then for the ensembl)
    true_sim_model = FullModel(latent_size=gan_input_dim,
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

    # create an image
    true_facies_image = true_sim_model.gan_evaluator.eval(synthetic_truth_vector,
                                                    no_grad=True)
    # this gives a binary image with shale
    true_facies_model_np = true_facies_image.to("cpu").numpy()[0, 0, :, :]

    # drilling position
    start_position = starting_position
    path = [start_position]


    # calculate objective functions
    # truth
    true_values = evaluate_earth_model_ensemble(true_facies_image)

    pathfinder_obj = pathfinder()

    # Cheat! Compute true optimal path
    true_next_optimal, true_paths = pathfinder_obj.no_gan_run(weighted_images=true_values,
                                                              start_point=start_position,
                                                              discount_for_remainder=1.0,  # keep 1.0 for truth!
                                                              recompute_optimal_paths_from_next=False
                                                              )


    # pre-job mapping and plotting
    # transform to facies
    pri_facies_earth = geosim_ensemble.gan_evaluator.eval(ensemble_state_torch,
                                                          no_grad=True)

    # ensenble
    ensemble_values = evaluate_earth_model_ensemble(pri_facies_earth)

    # compute paths
    next_optimal, paths = pathfinder_obj.no_gan_run(weighted_images=ensemble_values,
                                                    start_point=start_position,
                                                    discount_for_remainder=1.0,
                                                    recompute_optimal_paths_from_next=True
                                                    )

    ensemble_facies_truncated = torch.where(pri_facies_earth >= 0,
                                            torch.tensor(0, dtype=pri_facies_earth.dtype),
                                            torch.tensor(1, dtype=pri_facies_earth.dtype))
    ensemble_facies_model_np = ensemble_facies_truncated.to("cpu").numpy()[:, 0, :, :]

    # pre-job plotting
    plot_results_one_step(ensemble_facies_images=ensemble_facies_model_np,
                          true_facies_image=true_facies_model_np,
                          drilled_path=path,
                          true_optimal_path=true_paths[0],
                          next_optimal_recommendation=next_optimal,
                          all_paths=paths,
                          full_nn_model=geosim_ensemble,
                          )

    # loading data assimilation settings
    da_input_dict = input_dict.copy()

    for i in range(64):
        # actual run step
        # start with DA
        post_state_vectors = da(ensemble_state_torch,
                                start_position,
                                da_input_dict,
                                true_sim_for_da)

        # pre-job mapping and plotting
        # transform to facies
        post_facies_earth = geosim_ensemble.gan_evaluator.eval(post_state_vectors,
                                                               no_grad=True)

        # ensenble
        ensemble_values = evaluate_earth_model_ensemble(post_facies_earth)

        # compute paths
        next_optimal, paths = pathfinder_obj.no_gan_run(weighted_images=ensemble_values,
                                                        start_point=start_position,
                                                        discount_for_remainder=1.0,
                                                        recompute_optimal_paths_from_next=True
                                                        )

        ensemble_facies_truncated = torch.where(pri_facies_earth >= 0,
                                                torch.tensor(0, dtype=pri_facies_earth.dtype),
                                                torch.tensor(1, dtype=pri_facies_earth.dtype))
        ensemble_facies_model_np = ensemble_facies_truncated.to("cpu").numpy()[:, 0, :, :]

        # pre-job plotting
        plot_results_one_step(ensemble_facies_images=ensemble_facies_model_np,
                              true_facies_image=true_facies_model_np,
                              drilled_path=path,
                              true_optimal_path=true_paths[0],
                              next_optimal_recommendation=next_optimal,
                              all_paths=paths,
                              full_nn_model=geosim_ensemble,
                              )



    # # todo change the steps
    # for i in range(1, 10):
    #     next_optimal = compute_and_apply_robot_suggestion(ensemble_state,
    #                                                       start_position,
    #                                                       true_sim.simulator.NNmodel.gan_evaluator)
    #     path.append(next_optimal)
    #     start_position = next_optimal
    #     ensemble_state = da(ensemble_state, next_optimal, input_dict=input_dict)


if __name__ == "__main__":
    # todo add parameters according to the tests
    batch_run(seed=7)
    # example where we need to target lower sequence
    batch_run(starting_position=(54,0), seed=54)