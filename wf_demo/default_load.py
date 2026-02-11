import numpy as np
import torch
from pathlib import Path
from urllib.request import urlretrieve

weights_folder_url = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
full_em_model_file_name_url = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"
gan_file_name_url = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/gan/netG_epoch_4662.safetensors"


# --- cache dir ---
CACHE_DIR = Path("./weights_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download(url: str, dst: Path):
    if not dst.exists():
        urlretrieve(url, dst)

# 1) Full EM checkpoint
local_full_em_model_file_name = CACHE_DIR / "checkpoint_770.pth"
download(full_em_model_file_name_url, local_full_em_model_file_name)

# 2) GAN weights
local_gan_file_name = CACHE_DIR / "netG_epoch_4662.safetensors"
download(gan_file_name_url, local_gan_file_name)

# 3) EM scalers (fill with actual {} keys you use)
scaler_keys = ["min_x", "max_x", "min_y", "max_y"]

local_em_dir = CACHE_DIR / "em"
local_em_dir.mkdir(exist_ok=True)

for name in scaler_keys:
    url = weights_folder_url.format(name)
    download(url, local_em_dir / f"{name}.pth")

# --- switch to local paths ---
weights_folder = str(local_em_dir)
full_em_model_file_name = str(local_full_em_model_file_name)
gan_file_name = str(local_gan_file_name)

proxi_input_shape = (3,128)
proxi_output_shape = (6, 18)
gan_input_dim = 60
gan_output_height = 64
gan_num_channals = 6

scalers_folder = weights_folder

input_dict = {
    'file_name': gan_file_name,
    'swap_gan_output_dims': False,
    'full_em_model_file_name':full_em_model_file_name,
    'reporttype': 'pos',
    'reportpoint': [int(el) for el in range(1)],
    'scalers_folder':scalers_folder,
    'bit_pos':[(32,0)],
    'datatype': [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),
                 ('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]
    }

def load_default_latent_tensor(realization_id: str ="C1"):
    # my_latent_vec_np = np.random.normal(size=60)
    numpy_input = np.load(f"../chosen_realization_{realization_id}.npz")
    my_latent_vec_np = numpy_input['arr_0']
    # my_latent_vec_np = np.random.uniform(low=0.1, high=0.2, size=60)
    my_latent_tensor = torch.tensor(my_latent_vec_np, dtype=torch.float32).unsqueeze(0)
    return my_latent_tensor

def save_realization_latent_tensor(my_latent_tensor, id_str):
    numpy_vector = my_latent_tensor[0].cpu().numpy()
    np.savez(f"../chosen_realization_{id_str}.npz", numpy_vector)

def load_default_starting_ensemble_state():
    state = np.load('../orig_prior_2024.npz')['m'][:,:]
    return state