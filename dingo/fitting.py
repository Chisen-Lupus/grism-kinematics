import torch
import logging
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


from . import kinematics, utils, grism

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
# Only add handler once 
if not LOG.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'  # Or '%Y-%m-%d %H:%M:%S' for full timestamp
    )
    console_handler.setFormatter(formatter)
    LOG.addHandler(console_handler)


#%% --------------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------------
@dataclass
class FitParamConfig:
    name: str
    value: float
    lr: float
    vmin: float
    vmax: float
    fit: bool

    def __post_init__(self):
        self.vmin = float(self.vmin)
        self.vmax = float(self.vmax)

    def to_tensor(self, device=None):
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Cannot convert non-numeric value to tensor: {self.value}")
        return torch.tensor(
            self.value, dtype=torch.float32,
            requires_grad=self.fit, device=device
        )

    def __repr__(self):
        return (f'FitParamConfig(name={self.name}, value={self.value}, '
                f'lr={self.lr}, fit={self.fit}, '
                f'vmin={self.vmin}, vmax={self.vmax})')


def collect_named_trainable_params(**kwargs):
    seen_ids = set()
    named_params = {}

    for name, val in kwargs.items():
        if isinstance(val, dict):
            for subkey, p in val.items():
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    if id(p) not in seen_ids:
                        full_key = f'{name}.{subkey}'
                        named_params[full_key] = p
                        seen_ids.add(id(p))
        elif isinstance(val, torch.Tensor) and val.requires_grad:
            if id(val) not in seen_ids:
                named_params[name] = val
                seen_ids.add(id(val))

    return named_params

def build_param_config_dict(param_dict, default_cfg, overrides, prefix, device=None):
    tensor_map = {}
    config_map = {}

    allowed_keys = {
        'velocity': {'V_rot', 'R_v', 'x0_v', 'y0_v', 'theta_v', 'inc_v'},
        'image.R': {'dx', 'dy'},
        'image.C': {'dx', 'dy'}
    }

    for name, value in param_dict.items():
        full_key = f'{prefix}.{name}'
        if name not in allowed_keys.get(prefix, set()):
            tensor_map[name] = value  # Keep non-numerical entries as-is
            continue

        override_cfg = overrides.get(full_key, {})
        param_cfg = FitParamConfig(
            name=full_key,
            value=value,
            lr=override_cfg.get('lr', default_cfg['lr']),
            vmin=override_cfg.get('vmin', default_cfg['vmin']),
            vmax=override_cfg.get('vmax', default_cfg['vmax']),
            fit=override_cfg.get('fit', default_cfg['fit'])
        )

        tensor = param_cfg.to_tensor(device=device)  # ✅ Only call to_tensor once
        config_map[full_key] = param_cfg
        tensor_map[name] = tensor
        param_cfg.tensor = tensor  # ✅ Bind it back to config for param_groups

    return tensor_map, config_map

def load_config_and_initialize(config_path, coadd_R, coadd_C, device=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    summary = config['summary']
    image_config = config['image']
    velocity_config_raw = config['velocity']
    fitting_config = config['fitting'][0]  # TODO: now only use the first strategy
    default_cfg = fitting_config['default']
    overrides = fitting_config['override']

    # Set up r_fit and wavelength
    r_fit = summary['r_fit']
    lambda_rest = torch.tensor([summary['lambda_rest']]) * (1 + summary['z'])

    # Convert images to tensors
    true_grism_R = torch.tensor(coadd_R, dtype=torch.float32, device=device)
    true_grism_C = torch.tensor(coadd_C, dtype=torch.float32, device=device)

    # Build forward models
    fwd_models = {}
    for pupil in ['R', 'C']:
        _, spatial_model, disp_model = grism.load_nircam_wfss_model(
            pupil, image_config[pupil]['module'], summary['filter']
        )
        fwd_models[pupil] = utils.get_grism_model_torch(
            spatial_model, disp_model, pupil, 1024, 1024, direction='forward'
        )

    # Velocity parameters
    velocity_params, velocity_cfg = build_param_config_dict(
        velocity_config_raw, default_cfg, overrides, prefix='velocity', device=device
    )

    # Dispersion parameters
    dispersion_params = {}
    dispersion_cfgs = {}
    for pupil in ['R', 'C']:
        param_dict = {k: image_config[pupil][k] for k in ['dx', 'dy'] if k in image_config[pupil]}
        disp_params, disp_cfg = build_param_config_dict(
            param_dict, default_cfg, overrides, prefix=f'image.{pupil}', device=device
        )
        disp_params['forward_model'] = fwd_models[pupil]
        dispersion_params[pupil] = disp_params
        dispersion_cfgs[pupil] = disp_cfg

    # Pixel grid
    nx_G, ny_G = true_grism_R.shape
    y_G, x_G = torch.meshgrid(torch.arange(nx_G), torch.arange(ny_G), indexing='ij')

    # Cutout centers
    cx_R, cy_R = fwd_models['R'](torch.tensor(r_fit), torch.tensor(r_fit), lambda_rest)
    cx_C, cy_C = fwd_models['C'](torch.tensor(r_fit), torch.tensor(r_fit), lambda_rest)
    cutout_R = (int(cx_R)-r_fit, int(cy_R)-r_fit, 2*r_fit+1, 2*r_fit+1)
    cutout_C = (int(cx_C)-r_fit, int(cy_C)-r_fit, 2*r_fit+1, 2*r_fit+1)

    # Build param groups only for those with fit=True
    param_groups = []
    clamp_list = []  # ✅ collect tensors + clamp info here
    for cfg in list(velocity_cfg.values()) + list(dispersion_cfgs['R'].values()) + list(dispersion_cfgs['C'].values()):
        if cfg.fit:
            tensor = cfg.tensor
            param_groups.append({'params': [tensor], 'lr': cfg.lr})
            clamp_list.append((tensor, cfg.vmin, cfg.vmax))  # ✅ add clamp tuple

    return {
        'true_grism_R': true_grism_R,
        'true_grism_C': true_grism_C,
        'velocity_params': velocity_params,
        'dispersion_params_R': dispersion_params['R'],
        'dispersion_params_C': dispersion_params['C'],
        'cutout_R': cutout_R,
        'cutout_C': cutout_C,
        'lambda_rest': lambda_rest,
        'x_G': x_G,
        'y_G': y_G,
        'param_groups': param_groups,
        'clamp_list': clamp_list,  # ✅ add to output
        'fwd_models': fwd_models,
        'r_fit': r_fit,
        'summary': summary,
        'fitting_config': fitting_config,
    }


#%% --------------------------------------------------------------------------
# basic abstract class as interface
# ----------------------------------------------------------------------------

class BaseFitter(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_gradient(self, *args, **kwargs):
        pass


    
#%% --------------------------------------------------------------------------
# Kinematics fitting class
# ----------------------------------------------------------------------------

class KinematicsFitter(BaseFitter):

    def __init__(self, coadd_R, coadd_C, config: Any, device=None):
        config_dict = load_config_and_initialize(config, coadd_R, coadd_C, device)

        # ─────────────────────────────
        # Summary / Metadata
        # ─────────────────────────────
        summary = config_dict['summary']
        self.name = summary['name']
        self.ra = summary['ra']
        self.dec = summary['dec']
        self.z = summary['z']
        self.filter = summary['filter']
        self.mode = summary['mode']
        self.r_fit = config_dict['r_fit']

        # ─────────────────────────────
        # Core data
        # ─────────────────────────────
        self.true_grism_R = config_dict['true_grism_R']
        self.true_grism_C = config_dict['true_grism_C']
        self.lambda_rest = config_dict['lambda_rest']
        self.cutout_R = config_dict['cutout_R']
        self.cutout_C = config_dict['cutout_C']
        self.x_G = config_dict['x_G']
        self.y_G = config_dict['y_G']

        # ─────────────────────────────
        # Model parameters
        # ─────────────────────────────
        self.velocity_params = config_dict['velocity_params']
        self.dispersion_params_R = config_dict['dispersion_params_R']
        self.dispersion_params_C = config_dict['dispersion_params_C']

        # ─────────────────────────────
        # Temporary variables (state)
        # ─────────────────────────────
        self.x_R, self.y_R = kinematics.forward_dispersion_model(
            self.x_G, self.y_G, self.lambda_rest, **self.dispersion_params_R
        )
        self.x_C, self.y_C = kinematics.forward_dispersion_model(
            self.x_G, self.y_G, self.lambda_rest, **self.dispersion_params_C
        )
        self.iter_R = None
        self.iter_C = None

        # variables to be returned
        self.image_R = None
        self.image_C = None
        self.vz_R = None
        self.vz_C = None

        # ─────────────────────────────
        # Fitting strategy and optimizer
        # ─────────────────────────────
        fitting = config_dict['fitting_config']
        self.maxiter = fitting['maxiter']
        self.losses = []
        self.lrs = []

        self.optimizer = torch.optim.Adam(config_dict['param_groups'])
        self.clamp_list = config_dict['clamp_list']

        scheduler_type = fitting['scheduler']
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.707,
                patience=500,
                min_lr=1e-6
            )
        elif scheduler_type == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=2000,
                gamma=0.1
            )
        else:
            raise ValueError(f'Unsupported scheduler type: {scheduler_type}')


    # loss functions  --------------------------------------------------------

    def loss(self):
        
        # R grism

        self.x_R, self.y_R, self.vz_R, self.iter_R = kinematics.iteratively_find_xy(
            self.x_R, self.y_R, self.cutout_R, self.lambda_rest, self.x_G, self.y_G, 
            **self.velocity_params, **self.dispersion_params_R
        )

        self.image_R = kinematics.bilinear_interpolte_intensity_torch(
            self.x_R, self.y_R, self.true_grism_R, self.cutout_R
        )

        # C grism

        self.x_C, self.y_C, self.vz_C, self.iter_C = kinematics.iteratively_find_xy(
            self.x_C, self.y_C, self.cutout_C, self.lambda_rest, self.x_G, self.y_G, 
            **self.velocity_params, **self.dispersion_params_C
        )

        self.image_C = kinematics.bilinear_interpolte_intensity_torch(
            self.x_C, self.y_C, self.true_grism_C, self.cutout_C
        )

        # loss
        loss = torch.sum((self.image_R - self.image_C)**2)

        return loss

    # fitting loops  ---------------------------------------------------------

    def fit_gradient(self):

        self.losses = []
        self.lrs = []
        try:
            for i in range(self.maxiter):

                self.optimizer.zero_grad()

                # NOTE: reusing x/y_R/C will accelerate finding xy
                loss = self.loss()

                lr = self.optimizer.param_groups[0]['lr']
                if i == 0:
                    LOG.info(f'Starting, loss={loss.item():.3f}, lr={lr:.5f}, maxiters finding xy={(self.iter_R, self.iter_C)}')
                if (i + 1) % 500 == 0:
                    LOG.info(f'Step {i+1}, loss={loss.item():.3f}, lr={lr:.5f}, maxiters finding xy={(self.iter_R, self.iter_C)}')

                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                self.losses.append(loss.item())
                self.lrs.append(lr)

                # ✅ Apply clamping based on config
                for p, vmin, vmax in self.clamp_list:
                    p.data.clamp_(min=vmin, max=vmax)

        except Exception as e:
            print(repr(e))
            pass

        return self.losses, self.lrs

    # getters and setters ----------------------------------------------------

    def get_fitting_results(self): 
        return self.image_R, self.image_C, self.vz_R, self.vz_C

    def get_losses_and_lrs(self): 
        return self.losses, self.lrs
    
    def get_params(self): 
        return self.velocity_params, self.dispersion_params_R, self.dispersion_params_C
    
    def get_true_images(self): 
        return self.true_grism_R, self.true_grism_C
    
    # helper functions -------------------------------------------------------