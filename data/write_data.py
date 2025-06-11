import numpy as np
from GeoSim.sim import GeoSim

from NeuralSim.vector_to_log import FullModel
from NeuralSim.image_to_log import set_global_seed
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv

# First try. Make data with NN.... Not great.

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

sim.l_prim = [0]
sim.all_data_types = [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]
datatype = 'UDAR'

sim.setup_fwd_run({'test':'foo'})

set_global_seed(123)  # fix seeds for reproducibility

my_latent_vec_np = np.random.normal(size=60)
my_latent_tensor = torch.tensor(my_latent_vec_np, dtype=torch.float32).unsqueeze(0).to(
    device)  # Add batch dimension and move to device
index_vector = torch.full((1, 10), fill_value=32, dtype=torch.long).to(device)
image, resistivity, logs = sim.NNmodel.forward(my_latent_tensor, index_vector, output_transien_results=True)

# Get the last 8 values:
#            ['USDA', 'USDP',
#             'UADA', 'UADP',
#             'UHRA', 'UHRP',
#             'UHAA', 'UHAP'
#         ]

logs_np = logs.cpu().detach().numpy()[0,:,-8:]

# the images are upside down, so we need to flip them
#image = image.flip([2])
#resistivity = resistivity.flip([2])

fig, (ax_img, ax_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True, height_ratios=[1, 2]
)

# plotting the image
num_cols = image.shape[3]
# note, that the image is rotated weirdly in the current training
image_np = image.cpu().detach().numpy()
img = np.permute_dims(image_np[0, 0:3, :, :],(1, 2, 0))  # shape: [64, 64, 3]
ax_img.imshow(img, #extent=(-0.5, num_cols - 0.5, pad_top, pad_top + image.shape[2]),
              aspect='auto', interpolation='none')
ax_img.set_title("Facies image")
#ax_img.set_ylim(pad_top + image.shape[2], pad_top)

# plotting resistivity
resistivity_np = resistivity.cpu().detach().numpy()
img_res = resistivity_np[:, 0, :].T  # shape: [H, W]
ax_res.imshow(img_res, extent=(-0.5, num_cols - 0.5, 0, resistivity.shape[2]),
              aspect='auto', interpolation='none', cmap='summer')
ax_res.set_title("Resistivity input")

fig.savefig("Synth_Truth.png")

k = open('assim_index.csv','w',newline='')
#writer4 = csv.writer(k)
l = open('datatyp.csv','w',newline='')
writer5 = csv.writer(l)

# build a pandas dataframe with the data.
# The tvd is the index and the tuple (freq,dist) is the columns


data = {}
var = {}
for count,di in enumerate(sim.all_data_types):
    freq, dist = di
    data[(freq, dist)] = [logs_np[count,:]]
    #var[(freq, dist)] = [[['REL', 10] if abs(el) > abs(0.1*np.mean(values)) else ['ABS', (0.1*np.mean(values))**2] for el in val] for val in values]
    var[(freq, dist)] = [[['ABS', (0.05*np.mean(val))**2] for val in logs_np[count,:]]]


df = pd.DataFrame(data,columns=sim.all_data_types,index=[0])
df.index.name = 'tvd'
df.to_csv('data.csv',index=True)
#df.to_pickle('data.pkl')

df = pd.DataFrame(var,columns=sim.all_data_types,index=[0])
df.index.name = 'tvd'
df.to_csv('var.csv',index=True)
with open('var.pkl','wb') as f:
    pickle.dump(df,f)

#filt = [i*10 for i in range(50)]
for c,_ in enumerate([0]):
    #if c in filt:
    k.writelines(str(c) + '\n')
k.close()

writer5.writerow([str(el) for el in sim.all_data_types])
l.close()