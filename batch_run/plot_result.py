import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline


from pathoptim.pathOPTIM import pathfinder
from pathoptim.DP import create_weighted_image, evaluate_earth_model


import os,sys

from scipy.ndimage import label

# global font sizes for the whole script
mpl.rcParams.update({
    "font.size": 18,         # base font
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22,
})


save_folder = "figures"
# home = os.path.expanduser("~") # os independent home
#
#
# def convert_facies_to_resistivity(single_facies_model):
#     my_shape = single_facies_model.shape
#     result = np.zeros((my_shape[1], my_shape[2]))
#     for i in range(my_shape[1]):
#         for j in range(my_shape[2]):
#             result[i, j] = get_resistivity_default(single_facies_model[:, i, j])
#     return result
#
# # import warnings
# # # Ignore FutureWarning and UserWarning
# # warnings.filterwarnings(action='ignore', category=FutureWarning)
# # warnings.filterwarnings(action='ignore', category=UserWarning)
#
# # gan_file_name = os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth')
# gan_file_name = '../gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth'
# gan_vec_size = 60
# gan_evaluator = GanEvaluator(gan_file_name, gan_vec_size)
#


# ne = 250

# todo think of how to smartly link this to global extent
def convert_path_to_np_arrays(path,
                              offset_x=0,
                              offset_y=15.5,
                              mult_x=10.0,
                              mult_y=-0.5,
                              add_noise=0.0):
    np_shape = len(path)
    x = np.zeros(np_shape)
    y = np.zeros(np_shape)
    for i, point in enumerate(path):
        x[i] = point[1]*mult_x+offset_x
        y[i] = point[0]*mult_y + offset_y + np.random.uniform(-add_noise, add_noise)
    return x, y


def fmt_depth(tick, pos):
    return f"x{100-int(tick)}"   # or int(tick * 100) if your data is 0–1


def plot_results_one_step(true_facies_image=None,
                          true_value_image=None,
                          ensemble_facies_images=None,
                          value_images=None,
                          full_nn_model=None,
                          drilled_path=None,
                          true_optimal_path=None,
                          next_optimal_recommendation=None,
                          all_paths=None,
                          save_file_flags: str = "",
                          stop_to_show_plots=False):
    # todo fix plotting given the simulated features
    global_extent = [-5, 640-5, -16.25, 15.75]

    # we need to switch to TkAgg to show GUI, something switches it to something else
    # matplotlib.use('TkAgg')
    plot_path = 'figures/'
    #check if the path exists and create it if not
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    saved_legend = False

    # plot the resistivity:
    # Create the plot
    fig_res, ax_res = plt.subplots(figsize=(12, 5))
    # Use the 'viridis' colormap

    # # res_im = ax_res.imshow(true_res_image, aspect='auto', cmap='summer', vmin=1, vmax=200)
    # res_im = ax_res.imshow(-(true_facies_image-1.)/2.,
    #                        aspect='auto',
    #                        cmap='tab20b',
    #                        extent=global_extent)
    # cbar = plt.colorbar(res_im, ax=ax_res)

    ensemble_facies_model_mean = ensemble_facies_images.mean(axis=0)

    prob_im = ax_res.imshow(ensemble_facies_model_mean,
                               aspect='auto',
                               cmap='tab20b',
                               vmin=0,   # lower bound of color scale
                               vmax=1,    # upper bound of color scale
                               extent=global_extent)
    cbar = plt.colorbar(prob_im, ax=ax_res)


    # moving to line elements of the plot
    xmin, xmax, ymin, ymax = global_extent
    ny, nx = true_facies_image.shape  # imshow: (ny, nx)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymax, ymin, ny) # origin=lower

    # Drilled path
    # Measurement locations
    # Proposed direction
    # Further trajectory options
    # Outline of sand+crevasse in the true model

    cs = ax_res.contour(x, y,
                        true_facies_image,
                        levels=[0.5],
                        colors="white",
                        linestyles="dashed",
                        linewidths=1)

    ax_res.plot([0],[0],color="white",
                        linestyle="dashed",
                        linewidth=1,
                label="Outline of sand+crevasse in the true model")
    # cs.collections[0].set_label("Outline of sand+crevasse in the true model")

    # this plots already drilled path
    drilled_path_x, drilled_path_y = convert_path_to_np_arrays(drilled_path)
    ax_res.plot(drilled_path_x, drilled_path_y,
                color='black',
                linewidth=2,
                label="Drilled path")






    # this plots next optimal
    next_opt_x, next_opt_y = convert_path_to_np_arrays([drilled_path[-1],
                                                        next_optimal_recommendation])
    ax_res.plot(next_opt_x, next_opt_y,
                linestyle='--',
                marker='*',
                color='black',
                markersize=3,
                label="Proposed direction")



    legend_out = False
    for path in all_paths:
        path_x, path_y = convert_path_to_np_arrays(path, add_noise=0.1)
        if legend_out:
            ax_res.plot(path_x, path_y,
                        color='black',
                        linewidth=0.2)
        else:
            ax_res.plot(path_x, path_y,
                        color='black',
                        linewidth=0.2,
                        label="Further trajectory options")
            legend_out = True

    # this plots true best path
    true_path_x, true_path_y = convert_path_to_np_arrays(true_optimal_path)
    ax_res.plot(true_path_x, true_path_y,
                linestyle='None',
                marker='o',
                color='gray',
                markersize=3,
                label="Optimal solution (if true model is known)")

    # ticks at -10, -5, 0, 5, 10
    ax_res.set_yticks(np.arange(-10, 11, 5))
    ax_res.yaxis.set_major_formatter(FuncFormatter(fmt_depth))



    if stop_to_show_plots:
        ax_res.legend(
            loc="center left",
            bbox_to_anchor=(1.2, 0.5),
            facecolor="lightgray")
        plt.show()

    plt.savefig(f"{save_folder}/{save_file_flags}_{drilled_path[-1][1]}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_folder}/{save_file_flags}_{drilled_path[-1][1]}.png", dpi=300, bbox_inches="tight")


    # exit()

    #
    # # Load the decision points
    #
    #
    # gan_evaluator = full_nn_model.gan_evaluator
    #
    #
    #
    # post_earth = np.array(
    #     [create_weighted_image(evaluate_earth_model(gan_evaluator, state_vectors[:, el])) for el in
    #      range(ne)])  # range(state.shape[1])])
    #
    # # todo should we switch back to probability of sand ??? (use custom weights)
    # post_earth = np.array(
    #     [create_weighted_image(evaluate_earth_model(gan_evaluator, state_vectors[:, el]), weights=[0., 1., 1.]) for el in
    #      range(ne)])  # range(state.shape[1])])
    #
    # next_optimal, _garbage_path = pathfinder().run(state_vectors, start_position_at_step)
    #
    # # creating the figure
    # fig, ax = plt.subplots(figsize=(10, 5))
    #
    # # visualizing the posterior
    # norm = Normalize(vmin=0.0, vmax=1)
    # im = ax.imshow(post_earth.mean(axis=0), cmap='tab20b', aspect='auto', norm=norm)
    #
    #
    # # visualizing the outline of the truth
    # x = np.array(range(64))
    # y = np.array(range(64))
    # X, Y = np.meshgrid(x, y)
    # Z = true_earth_model_facies[0, :, :]
    # contour_style = 'dashed'
    # contour_color = 'white'
    # contour = plt.contour(X, Y, Z, levels=0, colors=contour_color,
    #                         linestyles=contour_style)
    # # contour.collections[0].set_label('Outline of sand+crevasse in the true model')
    #
    # # visualizing the drilled path
    # path_rows, path_cols = zip(*(drilled_path))
    # # todo consider creating smooth path. Does not work easily when few points.
    # # # Creating a smooth curve
    # # x = np.array(path_cols)
    # # y = np.array(path_rows)
    # # x_smooth = np.linspace(x.min(), x.max(), 300)
    # # spl = make_interp_spline(x, y, k=3)  # k=3 for cubic spline
    # # y_smooth = spl(x_smooth)
    # ax.plot(path_cols, path_rows,
    #         'k-', linewidth=3., label='Drilled path')
    # ax.plot(path_cols, path_rows,
    #         'k*', label='Measurement locations')
    #
    # if i == em_model_for_overlay_plotting_step:
    #     ax_res.plot(path_cols, path_rows,
    #             'k-', linewidth=3., label='Drilled path')
    #     ax_res.plot(path_cols, path_rows,
    #             'k*', label='Measurement locations')
    #     # saving the true resistivity image with overlay
    #     # Define the step-sizes
    #     step_x = 10
    #     step_y = 0.5
    #
    #     # Calculate the tick positions and labels
    #     x_ticks_positions = np.arange(0, 64, 10)
    #     x_ticks_labels = (x_ticks_positions - origin_x) * step_x
    #
    #     # Define the specific y-axis labels
    #     y_ticks_labels = [-10, -5, 0, 5, 10]
    #     y_ticks_labels_str = ['x090', 'x095', 'x100', 'x105', 'x110']
    #     # Calculate the corresponding y tick positions
    #     y_ticks_positions = [label / step_y + origin_y for label in y_ticks_labels]
    #
    #     # Set x-ticks and labels
    #     ax_res.set_xticks(x_ticks_positions)
    #     ax_res.set_xticklabels(x_ticks_labels)
    #
    #     # Set y-ticks and labels
    #     ax_res.set_yticks(y_ticks_positions)
    #     ax_res.set_yticklabels(y_ticks_labels_str)
    #
    #     fig_res.savefig(f'{plot_path}true_resistivity_{i}.png', bbox_inches='tight', dpi=600)
    #     fig_res.savefig(f'{plot_path}true_resistivity_{i}.pdf', bbox_inches='tight')
    #
    # if next_optimal[0] is not None:
    #
    #     # visualizing the next decision
    #     path_rows = [start_position_at_step[0], next_optimal[0]]
    #     path_cols = [start_position_at_step[1], next_optimal[1]]
    #     ax.plot(path_cols, path_rows,
    #             'k:', label='Proposed direction')
    #
    #     optimal_paths = [perform_dynamic_programming(post_earth[j, :, :], next_optimal,
    #                                                  cost_vector=pathfinder().get_cost_vector())[2] for j in range(ne)]
    #
    #     # visualizing the optimal paths' remainders
    #     earth_height = post_earth.shape[2]
    #     max_height = earth_height - 0.5 - 0.01
    #     min_height = 0.0 - 0.5 + 0.01
    #     for j in range(ne):
    #         path_rows, path_cols = zip(*(optimal_paths[j]))
    #         noise_mult = 0.2
    #         # noise_mult = 0
    #         row_list = [el + noise_mult*np.random.randn() for el in path_rows]
    #         row_list_truncated = [el if el < max_height else max_height for el in row_list]
    #         row_list_truncated = [el if el > min_height else min_height for el in row_list_truncated]
    #         if j == labelled_index:
    #             ax.plot(path_cols, tuple(row_list_truncated),
    #                     'k--', linewidth=0.25, label='Further trajectory options')
    #         else:
    #             # continue
    #             ax.plot(path_cols, tuple(row_list_truncated),
    #                     'k--', linewidth=0.25)
    #     # ax.set_title('Result with Optimal Path', fontsize=18)
    #     plt.tight_layout()
    #
    #
    #     # Define the step-sizes
    #     step_x = 10
    #     step_y = 0.5
    #
    #     # Calculate the tick positions and labels
    #     x_ticks_positions = np.arange(0, 64, 10)
    #     x_ticks_labels = (x_ticks_positions - origin_x) * step_x
    #
    #     # Define the specific y-axis labels
    #     y_ticks_labels =     [-10, -5, 0, 5, 10]
    #     y_ticks_labels_str = ['x090', 'x095', 'x100', 'x105', 'x110']
    #     # Calculate the corresponding y tick positions
    #     y_ticks_positions = [label / step_y + origin_y for label in y_ticks_labels]
    #
    #     # Set x-ticks and labels
    #     ax.set_xticks(x_ticks_positions)
    #     ax.set_xticklabels(x_ticks_labels)
    #     # Set y-ticks and labels
    #     ax.set_yticks(y_ticks_positions)
    #     ax.set_yticklabels(y_ticks_labels_str)
    #
    #     fig.savefig(f'{plot_path}mean_earth_{i}.png', bbox_inches='tight')
    #     fig.savefig(f'{plot_path}mean_earth_{i}.pdf', bbox_inches='tight')
    #
    #     print(f'Saved step {i}')
    #
    #     if not saved_legend:
    #         # Adding the legend outside the plot
    #         # manually adding a line to the legend
    #         cbar = plt.colorbar(im, ax=ax, location='bottom')
    #         ax.plot([0], [0], linestyle=contour_style, color=contour_color,
    #                  label='Outline of sand+crevasse in the true model')
    #
    #         legend = ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left')
    #         legend.get_frame().set_facecolor('lightgray')  # Set the background color to light gray
    #         fig.savefig(f'{plot_path}legend.png', bbox_inches='tight')
    #         fig.savefig(f'{plot_path}legend.pdf', bbox_inches='tight')
    #         saved_legend = True
    #
    #     plt.close('all')
    #     # drilled_path.append(checkpoint_at_step['pos'])# note that we compute another one during viosualization
    #
    #     # plt.show()
    #
    #     # plot the mean GAN output for the current decision points
