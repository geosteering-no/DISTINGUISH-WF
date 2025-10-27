import numpy as np
from GeoSim.sim import GeoSim

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import pandas as pd
import pickle
import csv
from wf_demo.default_load import input_dict, load_default_latent_tensor


sim = GeoSim(input_dict)

sim.l_prim = [0]
sim.all_data_types = [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]
datatype = 'UDAR'

sim.setup_fwd_run(**{'test':'foo'})

# these seeds are intended for a newer gan model.
# set_global_seed(42)  # fix seeds for reproducibility
# set_global_seed(43)  # fix seeds for reproducibility
# set_global_seed(55)  # fix seeds for reproducibility
# set_global_seed(55)  # fix seeds for reproducibility

my_latent_tensor = load_default_latent_tensor().to(device)

index_vector = torch.full((1, 10), fill_value=32, dtype=torch.long).to(device)
image, resistivity, logs = sim.NNmodel.forward(my_latent_tensor, index_vector, output_transien_results=True)

# Get the last 8 values:
#            ['USDA', 'USDP',
#             'UADA', 'UADP',
#             'UHRA', 'UHRP',
#             'UHAA', 'UHAP'
#         ]

logs_np = logs.cpu().detach().numpy()[:,:,-8:]

# the images are upside down, so we need to flip them
# image = image.flip([2])
# resistivity = resistivity.flip([2])
# TODO flip is at least required for consistency of the Synth truth images

# fig, (ax_img, ax_res) = plt.subplots(
#     2, 1, figsize=(10, 8), sharex=True, height_ratios=[1, 2]
# )
#
# # plotting the image
# num_cols = image.shape[3]
# # note, that the image is rotated weirdly in the current training
# image_np = image.cpu().detach().numpy()
# img = np.transpose(image_np[0, 0:3, :, :],(1, 2, 0))  # shape: [64, 64, 3]
# ax_img.imshow(img, #extent=(-0.5, num_cols - 0.5, pad_top, pad_top + image.shape[2]),
#               aspect='auto', interpolation='none')
# ax_img.set_title("Facies image")
# #ax_img.set_ylim(pad_top + image.shape[2], pad_top)
#
# # plotting resistivity
# resistivity_np = resistivity.cpu().detach().numpy()
# img_res = resistivity_np[:, 0, :].T  # shape: [H, W]
# ax_res.imshow(img_res, extent=(-0.5, num_cols - 0.5, 0, resistivity.shape[2]),
#               aspect='auto', interpolation='none', cmap='summer')
# ax_res.set_title("Resistivity input")
#
# fig.savefig("Synth_Truth.png")

names = [
    # 'real(xx)', 'img(xx)',
    # 'real(yy)', 'img(yy)',
    # 'real(zz)', 'img(zz)',
    # 'real(xz)', 'img(xz)',
    # 'real(zx)', 'img(zx)',
    'USDA', 'USDP',
    'UADA', 'UADP',
    'UHRA', 'UHRP',
    'UHAA', 'UHAP'
]
tool_configs = [f'{f} khz - {s} ft' for f, s in zip([6., 12., 24., 24., 48., 96.], [83., 83., 83., 43., 43., 43.])]

log_tool_index_in_use = 0



fig = plt.figure(figsize=(10, 8))

# Create a grid with 3 rows and 2 columns
# The second column is narrow for the colorbar
gs = GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[1, 2, 1.2], figure=fig)

# Left column: main plots
ax_img = fig.add_subplot(gs[0, 0])
ax_res = fig.add_subplot(gs[1, 0], sharex=ax_img)
ax_logs = fig.add_subplot(gs[2, 0], sharex=ax_img)

# Right column: color-bars
ax_img_cbar = fig.add_subplot(gs[0, 1])
ax_res_cbar = fig.add_subplot(gs[1, 1])
ax_logs_cbar = fig.add_subplot(gs[2, 1])

pad_top = sim.NNmodel.pad_top
# plotting the image
num_cols = image.shape[3]
# note, that the image is rotated weirdly in the current training
img = image[0, 0:3, :, :].permute(1, 2, 0).cpu().numpy()  # shape: [64, 64, 3]
ax_img.imshow(img,
              extent=(-0.5, num_cols - 0.5, pad_top + image.shape[2], pad_top),
              aspect='auto', interpolation='none')
ax_img.set_title("Facies image")
ax_img.set_ylim(pad_top + image.shape[2], pad_top)

num_cols_res = resistivity.shape[0]
# plotting resistivity
img_res = resistivity[:, 0, :].T.cpu().numpy()  # shape: [H, W]
ax_res.imshow(img_res, extent=(-0.5, num_cols_res - 0.5, resistivity.shape[2], 0),
              aspect='auto', interpolation='none', cmap='summer')
ax_res.set_title("Resistivity input")

# plotting logging locations
mask = resistivity[:, 2, :].T.cpu().numpy()  # shape: [H, W]
rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
rgba[..., 0:3] = 0.0  # black color
rgba[..., 3] = (mask != 0).astype(np.float32)  # alpha: 1 for non-zero, 0 for zero

im_res = ax_res.imshow(rgba, extent=(-0.5, num_cols - 0.5, resistivity.shape[2], 0),
              aspect='auto', interpolation='none')
ax_res.set_ylim(resistivity.shape[2], 0)
# ax_res.tick_params(labelbottom=True)

# add colorbar
cbar = fig.colorbar(im_res, cax=ax_res_cbar)
cbar.set_label("Resistivity (normalized)")

# Hide unused colorbar axes if desired
ax_img_cbar.axis('off')
ax_logs_cbar.axis('off')

#
# plotting the logs
ax_logs.set_title(names[log_tool_index_in_use])
# logs_to_plot = logs_np[:, i]  # take the first batch and first channel
for j, config in enumerate(tool_configs):
    ax_logs.plot(logs_np[:, j, log_tool_index_in_use], label=config)

ax_logs.legend()

# remove visible labels on axes
ax_img.tick_params(labelbottom=False)
ax_res.tick_params(labelbottom=False)

# saving
fig.savefig(f'SynthTruth_updated.png', bbox_inches='tight', dpi=300)

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
#df.to_csv('data.csv',index=True)
df.to_pickle('data.pkl')

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
