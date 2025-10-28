from GeoSim.sim import GeoSim
import numpy as np
import csv
import torch
import pandas as pd
import pickle


from wf_demo.default_load import input_dict, load_default_latent_tensor

class SyntheticTruth:
    def __init__(self, latent_truth_vector, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # this is simulator of the true data
        self.simulator = GeoSim(input_dict)

        self.simulator.l_prim = [0]
        self.simulator.all_data_types = [('6kHz', '83ft'), ('12kHz', '83ft'), ('24kHz', '83ft'), ('24kHz', '43ft'), ('48kHz', '43ft'), ('96kHz', '43ft')]

        self.latent_synthetic_truth = latent_truth_vector
        # load_default_latent_tensor().to(device))



    def acquire_data(self, keys):

        index_vector = torch.full((1, keys['bit_pos'][0][1]), fill_value=keys['bit_pos'][0][0],
                                  dtype=torch.long).to(self.device)

        logs = self.simulator.NNmodel.forward(self.latent_synthetic_truth, index_vector, output_transien_results=False)
        logs_np = logs.cpu().detach().numpy()[keys['bit_pos'][0][1]-1,:,-8:]

        # bookkeeping
        k = open('../data/assim_index.csv', 'w', newline='')
        # writer4 = csv.writer(k)
        l = open('../data/datatyp.csv', 'w', newline='')
        writer5 = csv.writer(l)

        # build a pandas dataframe with the data.
        # The tvd is the index and the tuple (freq,dist) is the columns

        data = {}
        var = {}
        for count, di in enumerate(self.simulator.all_data_types):
            freq, dist = di
            data[(freq, dist)] = [logs_np[count, :]]
            # var[(freq, dist)] = [[['REL', 10] if abs(el) > abs(0.1*np.mean(values)) else ['ABS', (0.1*np.mean(values))**2] for el in val] for val in values]
            var[(freq, dist)] = [[['ABS', (0.05 * np.mean(val)) ** 2] for val in logs_np[count, :]]]

        df = pd.DataFrame(data, columns=self.simulator.all_data_types, index=[0])
        df.index.name = 'tvd'
        # df.to_csv('data.csv',index=True)
        df.to_pickle('../data/data.pkl')

        df = pd.DataFrame(var, columns=self.simulator.all_data_types, index=[0])
        df.index.name = 'tvd'
        df.to_csv('../data/var.csv', index=True)
        with open('../data/var.pkl', 'wb') as f:
            pickle.dump(df, f)

        # filt = [i*10 for i in range(50)]
        for c, _ in enumerate([0]):
            # if c in filt:
            k.writelines(str(c) + '\n')
        k.close()

        writer5.writerow([str(el) for el in self.simulator.all_data_types])
        l.close()


