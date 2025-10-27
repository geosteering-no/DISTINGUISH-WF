import numpy as np
import torch

weights_folder = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
scalers_folder = weights_folder
full_em_model_file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"
gan_file_name = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/gan/netG_epoch_4662.safetensors"

def load_default_latent_tensor():
    # my_latent_vec_np = np.random.normal(size=60)
    numpy_input = np.load("../chosen_realization_C1.npz")
    my_latent_vec_np = numpy_input['arr_0']
    # my_latent_vec_np = np.random.uniform(low=0.1, high=0.2, size=60)
    my_latent_tensor = torch.tensor(my_latent_vec_np, dtype=torch.float32).unsqueeze(0)
    return my_latent_tensor