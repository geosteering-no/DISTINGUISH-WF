from GeoSim.sim import GeoSim
import numpy as np
import csv
from scipy.linalg import block_diag
import os, sys

weights_folder = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
scalers_folder = weights_folder
full_em_model_file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"
file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/gan/netG_epoch_15000.pth"

input_dict = {
        'file_name':file_name,
        'full_em_model_file_name':full_em_model_file_name,
        'scalers_folder':scalers_folder
         }

sim = GeoSim(input_dict)

rng_state = np.random.get_state()

reference_model_seed = 777
np.random.seed(reference_model_seed)
reference_model = np.random.normal(size=sim.vec_size)

np.random.set_state(rng_state)


# keys = {'bit_pos': [int(el) for el in range(64)],
#          'vec_size': 60,
#         'reporttype': 'pos',
#         'reportpoint': [int(el) for el in range(64)],
#         'datatype': [f'res{el}' for el in range(1,14)]}

def new_data(keys):

    modelresponse = sim.call_sim(state={'m':reference_model})

    with open('../data/true_data.csv', 'w') as f:
            writer = csv.writer(f)
            for el in range(len(keys['bit_pos'])):
                writer.writerow([str(elem) for elem in modelresponse['res'][el, :]])

    with open('../data/data_index.csv', 'w') as f:
        writer = csv.writer(f)
        for el,_ in enumerate(keys['bit_pos']):
            writer.writerow([str(el)])

    with open('../data/data_types.csv', 'w') as f:
        writer = csv.writer(f)
        for el in keys['datatype']:
            writer.writerow([str(el)])

    with open('../data/bit_pos.csv', 'w') as f:
        writer = csv.writer(f)
        for el,_ in enumerate(keys['bit_pos']):
            writer.writerow([str(el)])

    ### Write correlated variance ####
    rel_var = 0.01
    corrRange = 10
    nD=13 # only correlation along point data
    sub_covD = np.zeros((len(keys['bit_pos']),nD,nD))
    for k in range(len(keys['bit_pos'])):
        sub_data = np.array([modelresponse['res'][k,el] for el in range(13)]).flatten()
        for i in range(nD):
            for j in range(nD):
                sub_covD[k,i,j] = (sub_data[i]*rel_var)*(sub_data[j]*rel_var) * np.exp(-3.*(np.abs(i - j)/corrRange)**1.9)

    Cd = block_diag(*sub_covD[:min(8,len(keys['bit_pos'])),:,:]) # maximum 8 correlated points in time

    np.savez('../data/cd.npz',Cd)

    # m_true = numpy_single.copy()
    # mean_f = m_true * 0.25
    # mean_f[20:44] = 0.
    # np.savez('mean_field.npz', mean_f)
