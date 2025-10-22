import numpy as np
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec

from data.write_data import log_tool_index_in_use

#TODO consider moving these to a more sensible place with other simulation metadata
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



def plot_workflow_status(image, resistivity, logs, pad_top):
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
    cbar.set_label("Resistivity (Ω·m)")

    # Hide unused colorbar axes if desired
    ax_img_cbar.axis('off')
    ax_logs_cbar.axis('off')

    # TODO: check that this is the log tool index to be plotted, not the DA property
    log_tool_index_in_use = 0
    #
    # plotting the logs
    ax_logs.set_title(names[log_tool_index_in_use])

    logs_np = logs.cpu().detach().numpy()[:, :, -8:]
    # logs_to_plot = logs_np[:, i]  # take the first batch and first channel
    for j, config in enumerate(tool_configs):
        ax_logs.plot(logs_np[:, j, log_tool_index_in_use], label=config)

    ax_logs.legend()

    # remove visible labels on axes
    ax_img.tick_params(labelbottom=False)
    ax_res.tick_params(labelbottom=False)

    # saving
    fig.savefig(f'SynthTruth_updated.png', bbox_inches='tight', dpi=300)
