from GeoSim.sim import GeoSim
import numpy as np
import csv
from scipy.linalg import block_diag
import os, sys
import torch
import pandas as pd
import pickle

weights_folder = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
scalers_folder = weights_folder
full_em_model_file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"
file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/gan/netG_epoch_15000.pth"

input_dict = {
        'file_name':file_name,
        'full_em_model_file_name':full_em_model_file_name,
        'scalers_folder':scalers_folder,
        'bit_pos': [(32, 0)],  # Example bit position, adjust as needed
        'vec_size': 60,  # Example vector size, adjust as needed
        'reporttype': 'pos',
        'reportpoint': [int(el) for el in range(1)],
        'datatype': [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),
                       ('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]
         }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sim = GeoSim(input_dict)

sim.l_prim = [0]
sim.all_data_types = [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]

rng_state = np.random.get_state()

reference_model_seed = 777
np.random.seed(reference_model_seed)
reference_model = np.random.normal(size=sim.vec_size)
latent_reference_model = torch.tensor(reference_model, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and move to device

np.random.set_state(rng_state)


def new_data(keys):

    index_vector = torch.full((1, keys['bit_pos'][0][1]), fill_value=keys['bit_pos'][0][0],
                              dtype=torch.long).to(device)

    logs = sim.NNmodel.forward(latent_reference_model, index_vector, output_transien_results=False)
    logs_np = logs.cpu().detach().numpy()[keys['bit_pos'][0][1]-1,:,-8:]

    k = open('../data/assim_index.csv', 'w', newline='')
    # writer4 = csv.writer(k)
    l = open('../data/datatyp.csv', 'w', newline='')
    writer5 = csv.writer(l)

    # build a pandas dataframe with the data.
    # The tvd is the index and the tuple (freq,dist) is the columns

    data = {}
    var = {}
    for count, di in enumerate(sim.all_data_types):
        freq, dist = di
        data[(freq, dist)] = [logs_np[count, :]]
        # var[(freq, dist)] = [[['REL', 10] if abs(el) > abs(0.1*np.mean(values)) else ['ABS', (0.1*np.mean(values))**2] for el in val] for val in values]
        var[(freq, dist)] = [[['ABS', (0.05 * np.mean(val)) ** 2] for val in logs_np[count, :]]]

    df = pd.DataFrame(data, columns=sim.all_data_types, index=[0])
    df.index.name = 'tvd'
    # df.to_csv('data.csv',index=True)
    df.to_pickle('../data/data.pkl')

    df = pd.DataFrame(var, columns=sim.all_data_types, index=[0])
    df.index.name = 'tvd'
    df.to_csv('../data/var.csv', index=True)
    with open('../data/var.pkl', 'wb') as f:
        pickle.dump(df, f)

    # filt = [i*10 for i in range(50)]
    for c, _ in enumerate([0]):
        # if c in filt:
        k.writelines(str(c) + '\n')
    k.close()

    writer5.writerow([str(el) for el in sim.all_data_types])
    l.close()


