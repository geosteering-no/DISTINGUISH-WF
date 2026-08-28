import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path
from urllib.request import urlretrieve

weights_folder_url = "https://gitlab.norceresearch.no/krfo/utaproxyweighs/-/raw/post_ecmor_jac-4th-k128_smooth/em/{}.pth?ref_type=heads"
full_em_model_file_name_url = "https://gitlab.norceresearch.no/krfo/utaproxyweighs/-/raw/post_ecmor_jac-4th-k128_smooth/em/checkpoint_done.pth?ref_type=heads"
gan_file_name_url = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/gan/netG_epoch_4662.safetensors"

# --- adapt the upstream model chain to the smooth EM proxy weights ---
# contract of the post_ecmor_jac-4th-k128_smooth weights:
#   input  [b, w, 4, 128]: [ln(rh), ln(rv), one-hot bit position, drilling angle in degrees]
#   output [b, w, 6, 10]: 6 tool configs x 10 B-field components
#                         [Re/Im x Hxx, Hyy, Hzz, Hxz, Hzx]
#   UDAR geosignals (8 per tool) are derived through the analytic mapping
#   convert_bfield_to_udar_torch
from udar_proxi.utils import convert_bfield_to_udar_torch
from NeuralSim import image_to_log as _image_to_log
from NeuralSim import vector_to_log as _vector_to_log
from GeoSim import sim as _geosim_sim

smooth_proxi_input_shape = (4, 128)
smooth_proxi_output_shape = (6, 10)

_original_em_proxy_init = _image_to_log.EMProxy.__init__

def _smooth_em_proxy_init(self, *args, **kwargs):
    kwargs.setdefault('architecture', 'smooth')
    _original_em_proxy_init(self, *args, **kwargs)

# FullModel/GeoSim do not expose EMProxy's architecture argument, so default it to smooth here
_image_to_log.EMProxy.__init__ = _smooth_em_proxy_init

_original_full_model_init = _vector_to_log.FullModel.__init__

def _smooth_full_model_init(self, *args, **kwargs):
    kwargs['proxi_input_shape'] = smooth_proxi_input_shape
    kwargs['proxi_output_shape'] = smooth_proxi_output_shape
    _original_full_model_init(self, *args, **kwargs)

# GeoSim hardcodes the legacy (3,128)/(6,18) proxy shapes; force the smooth ones
_vector_to_log.FullModel.__init__ = _smooth_full_model_init

def _smooth_convert_to_resistivity_format(self, images, index_vector):
    one_hot = F.one_hot(index_vector, num_classes=images.shape[2]).float()
    one_hot = one_hot.permute(0, 2, 1).unsqueeze(1)  # [b, 1, h, w]

    images = images[:, :, :, 0:index_vector.shape[1]]  # [b, c, h, w]

    rh = torch.log((images * self.rh_mult).sum(dim=1, keepdim=True))
    rv = torch.log((images * self.rv_mult).sum(dim=1, keepdim=True))

    resistivity = torch.cat([rh, rv, one_hot], dim=1)  # [b, 3, h, w]
    resistivity = resistivity.permute(0, 3, 1, 2)  # [b, w, c, h]
    resistivity_padded = F.pad(resistivity, (self.pad_top, self.pad_bottom, 0, 0), mode='replicate')  # [b, w, 3, h_padded]

    # drilling angle per column: 90 deg when drilling ahead, 45 deg for a one-cell step
    d_row = (index_vector[:, 1:] - index_vector[:, :-1]).abs().float()
    angle = 90.0 - 45.0 * torch.cat([torch.zeros_like(d_row[:, :1]), d_row], dim=1)  # [b, w]
    angle_channel = angle[:, :, None, None].expand(-1, -1, 1, resistivity_padded.shape[-1])  # [b, w, 1, h_padded]

    return torch.cat([resistivity_padded, angle_channel], dim=2)  # [b, w, 4, h_padded]

_vector_to_log.FullModel.convert_to_resistivity_format = _smooth_convert_to_resistivity_format

_original_full_model_forward = _vector_to_log.FullModel.forward

def _smooth_full_model_forward(self, x, index_vector, output_transien_results=False):
    # the proxy predicts B-field components; map them analytically to UDAR geosignals
    gan_output, resistivity_padded, response = _original_full_model_forward(
        self, x, index_vector, output_transien_results=True
    )
    if response.shape[-1] == 10:
        response = convert_bfield_to_udar_torch(response)  # [b, w, 6 tools, 8 geosignals]
    if output_transien_results:
        return gan_output, resistivity_padded, response
    return response

_vector_to_log.FullModel.forward = _smooth_full_model_forward

def _smooth_call_sim(self, **kwargs):
    my_latent_vec_np = kwargs['x']

    if my_latent_vec_np.ndim == 1:
        my_latent_tensor = torch.tensor(my_latent_vec_np, dtype=torch.float32).unsqueeze(0).to(_geosim_sim.device)
    else:
        my_latent_tensor = torch.tensor(my_latent_vec_np, dtype=torch.float32).to(_geosim_sim.device)

    self.index_vector = torch.full((my_latent_tensor.shape[0], self.bit_pos[0][1]+1),
                                   fill_value=self.bit_pos[0][0],
                                   dtype=torch.long).to(_geosim_sim.device)

    logs = self.NNmodel.forward(my_latent_tensor, self.index_vector, output_transien_results=False)

    logs_np = logs.cpu().detach().numpy()
    batch_size = logs_np.shape[0]

    self.pred_data = []
    for _ in range(batch_size):
        sample_data = [deepcopy({}) for _ in range(max(self.l_prim)+1)]
        for ind in range(max(self.l_prim)+1):
            for key in self.all_data_types:
                sample_data[ind][key] = None
        self.pred_data.append(sample_data)

    for sample_idx in range(batch_size):
        for prim_ind in range(max(self.l_prim)+1):
            for key in self.all_data_types:
                extract_index = self.tool_configs.index(key)
                if key == 'point':
                    self.pred_data[sample_idx][prim_ind][key] = logs_np[sample_idx, self.bit_pos[0][1], :].flatten()
                else:
                    # last dim holds the 8 UDAR geosignals per tool config
                    self.pred_data[sample_idx][prim_ind][key] = logs_np[sample_idx, self.bit_pos[0][1], extract_index, :].flatten()

    return self.pred_data

_geosim_sim.GeoSim.call_sim = _smooth_call_sim

# --- cache dir ---
CACHE_DIR = Path("./weights_cache") / "post_ecmor_jac-4th-k128_smooth"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download(url: str, dst: Path):
    if not dst.exists():
        urlretrieve(url, dst)

# 1) Full EM checkpoint
local_full_em_model_file_name = CACHE_DIR / "checkpoint_done.pth"
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

udar_data_type_array = [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),
                 ('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]

input_dict = {
    'file_name': gan_file_name,
    'swap_gan_output_dims': False,
    'full_em_model_file_name':full_em_model_file_name,
    'reporttype': 'pos',
    'reportpoint': [int(el) for el in range(1)],
    'scalers_folder':scalers_folder,
    'bit_pos':[(32,0)],
    # 'datatype': ['point'],
    'datatype': udar_data_type_array,
    'parallel_internal': True,
    'parallel':250
    }


# This method selects the realization from one of the presets in the root folder
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