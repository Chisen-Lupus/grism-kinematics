import torch
import logging
import yaml
from dataclasses import dataclass
from astropy.io import fits
import numpy as np
from typing import Dict, Any, Optional, AnyStr
from abc import ABC, abstractmethod
import os

from . import kinematics, utils, grism

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    LOG.addHandler(console_handler)

#%% --------------------------------------------------------------------------
# helper classes and functions
# ----------------------------------------------------------------------------
@dataclass
class FitParamConfig:
    name: str
    value: float
    lr: float
    vmin: float
    vmax: float
    fit: bool
    alias: Optional[str] = None
    tensor: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.vmin = float(self.vmin)
        self.vmax = float(self.vmax)

    def __repr__(self):
        return (f'FitParamConfig(name={self.name}, value={self.value}, '
                f'lr={self.lr}, fit={self.fit}, alias={self.alias}, '
                f'vmin={self.vmin}, vmax={self.vmax})')

    def __str__(self):
        return self.__repr__()

def build_param_config_dict_with_alias(raw_dict, default_cfg, overrides, prefix):
    cfgs = {}
    alias_map = {}
    allowed_keys = {
        'velocity': {'V_rot', 'R_v', 'x0_v', 'y0_v', 'theta_v', 'inc_v'},
        'image.R': {'dx', 'dy'},
        'image.C': {'dx', 'dy'}
    }
    for name, val in raw_dict.items():
        full_key = f'{prefix}.{name}'
        if name not in allowed_keys.get(prefix, set()):
            continue
        if isinstance(val, str):
            alias_map[full_key] = val
        else:
            override_cfg = overrides.get(full_key, {})
            cfgs[full_key] = FitParamConfig(
                name=full_key,
                value=val,
                lr=override_cfg.get('lr', default_cfg['lr']),
                vmin=override_cfg.get('vmin', default_cfg['vmin']),
                vmax=override_cfg.get('vmax', default_cfg['vmax']),
                fit=override_cfg.get('fit', default_cfg['fit'])
            )
    for key, alias in alias_map.items():
        if alias in cfgs:
            cfgs[key] = cfgs[alias]
        else:
            raise ValueError(f"Alias {alias} not found for {key}")
    return cfgs

def _extract_tensors(cfgs):
    seen = set()
    tensors = []
    clamp_list = []
    for cfg in cfgs.values():
        if not cfg.fit:
            continue
        if cfg.tensor is None:
            cfg.tensor = torch.tensor(cfg.value, dtype=torch.float32, requires_grad=True)
        if id(cfg.tensor) in seen:
            continue
        seen.add(id(cfg.tensor))
        tensors.append({'params': [cfg.tensor], 'lr': cfg.lr})
        clamp_list.append((cfg.tensor, cfg.vmin, cfg.vmax))
    return tensors, clamp_list

def load_fits_data(path_hdu):
    path, hdu_index = path_hdu
    with fits.open(path) as hdu:
        return np.array(hdu[hdu_index].data, dtype=np.float32)

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

    def __init__(self, config_path, device=None):
        # ─────────────────────────────
        # Load configuration
        # ─────────────────────────────
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.device = device
        self.config_path = config_path
        self.config = config

        # ─────────────────────────────
        # Summary / Metadata
        # ─────────────────────────────
        summary = config['summary']
        self.name = summary['name']
        self.ra = summary['ra']
        self.dec = summary['dec']
        self.z = summary['z']
        self.filter = summary['filter']
        self.mode = summary['mode']
        self.r_fit = summary['r_fit']

        # ─────────────────────────────
        # Image loading and wavelength
        # ─────────────────────────────
        image_config = config['image']
        self.lambda_rest = torch.tensor([summary['lambda_rest']]) * (1 + self.z)

        self.true_grism_R = torch.tensor(load_fits_data(image_config['R']['path']), device=device)
        self.true_grism_C = torch.tensor(load_fits_data(image_config['C']['path']), device=device)

        # ─────────────────────────────
        # Forward model loading
        # ─────────────────────────────
        self.fwd_models = {}
        for pupil in ['R', 'C']:
            _, spatial_model, disp_model = grism.load_nircam_wfss_model(
                pupil, image_config[pupil]['module'], self.filter)
            self.fwd_models[pupil] = utils.get_grism_model_torch(
                spatial_model, disp_model, pupil, 1024, 1024, direction='forward')

        # ─────────────────────────────
        # Parameter configuration dictionaries
        # ─────────────────────────────
        self.velocity_cfg = build_param_config_dict_with_alias(
            config['velocity'], config['fitting'][0]['default'], config['fitting'][0]['override'], prefix='velocity')

        self.dispersion_cfg_R = build_param_config_dict_with_alias(
            {k: v for k, v in image_config['R'].items() if k in ['dx', 'dy']},
            config['fitting'][0]['default'], config['fitting'][0]['override'], prefix='image.R')
        self.dispersion_cfg_R['image.R.forward_model'] = FitParamConfig(
            name='image.R.forward_model', value=None, lr=0, vmin=0, vmax=0, fit=False)
        self.dispersion_cfg_R['image.R.forward_model'].tensor = self.fwd_models['R']

        self.dispersion_cfg_C = build_param_config_dict_with_alias(
            {k: v for k, v in image_config['C'].items() if k in ['dx', 'dy']},
            config['fitting'][0]['default'], config['fitting'][0]['override'], prefix='image.C')
        self.dispersion_cfg_C['image.C.forward_model'] = FitParamConfig(
            name='image.C.forward_model', value=None, lr=0, vmin=0, vmax=0, fit=False)
        self.dispersion_cfg_C['image.C.forward_model'].tensor = self.fwd_models['C']

        # ─────────────────────────────
        # Pixel grid and cutout regions
        # ─────────────────────────────
        nx_G, ny_G = self.true_grism_R.shape
        self.y_G, self.x_G = torch.meshgrid(torch.arange(nx_G), torch.arange(ny_G), indexing='ij')

        cx_R, cy_R = self.fwd_models['R'](torch.tensor(self.r_fit), torch.tensor(self.r_fit), self.lambda_rest)
        cx_C, cy_C = self.fwd_models['C'](torch.tensor(self.r_fit), torch.tensor(self.r_fit), self.lambda_rest)
        self.cutout_R = (int(cx_R) - self.r_fit, int(cy_R) - self.r_fit, 2*self.r_fit + 1, 2*self.r_fit + 1)
        self.cutout_C = (int(cx_C) - self.r_fit, int(cy_C) - self.r_fit, 2*self.r_fit + 1, 2*self.r_fit + 1)

        # ─────────────────────────────
        # Fitting state (lazy init)
        # ─────────────────────────────
        self.optimizer = None
        self.scheduler = None
        self.clamp_list = None
        self.maxiter = config['fitting'][0]['maxiter']

        # ─────────────────────────────
        # Fitting outputs and temporary state
        # ─────────────────────────────
        self.image_R = None
        self.image_C = None
        self.vz_R = None
        self.vz_C = None
        self.iter_R = None
        self.iter_C = None

    # loss functions  --------------------------------------------------------

    def loss(self):
        # Unpack current tensor values
        velocity = {k.split('.')[-1]: v.tensor for k, v in self.velocity_cfg.items()}
        disp_R = {k.split('.')[-1]: v.tensor for k, v in self.dispersion_cfg_R.items()}
        disp_C = {k.split('.')[-1]: v.tensor for k, v in self.dispersion_cfg_C.items()}

        # Compute forward models and velocities
        self.x_R, self.y_R, self.vz_R, self.iter_R = kinematics.iteratively_find_xy(
            self.x_R, self.y_R, self.cutout_R, self.lambda_rest, self.x_G, self.y_G,
            **velocity, **disp_R)
        self.image_R = kinematics.bilinear_interpolte_intensity_torch(
            self.x_R, self.y_R, self.true_grism_R, self.cutout_R)

        self.x_C, self.y_C, self.vz_C, self.iter_C = kinematics.iteratively_find_xy(
            self.x_C, self.y_C, self.cutout_C, self.lambda_rest, self.x_G, self.y_G,
            **velocity, **disp_C)
        self.image_C = kinematics.bilinear_interpolte_intensity_torch(
            self.x_C, self.y_C, self.true_grism_C, self.cutout_C)

        # Compute L2 loss between R and C images
        return torch.sum((self.image_R - self.image_C)**2)

    # fitting loops  ---------------------------------------------------------

    def fit_gradient(self):
        # Reset XY grids before fitting
        self.x_R, self.y_R = self.x_G.clone(), self.y_G.clone()
        self.x_C, self.y_C = self.x_G.clone(), self.y_G.clone()

        # Extract tensors and build optimizer groups
        all_cfg = {**self.velocity_cfg, **self.dispersion_cfg_R, **self.dispersion_cfg_C}
        param_groups, clamp_list = _extract_tensors(all_cfg)
        self.optimizer = torch.optim.Adam(param_groups)
        self.clamp_list = clamp_list

        # Configure learning rate scheduler
        scheduler_type = self.config['fitting'][0]['scheduler']
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.707, patience=500, min_lr=1e-6)
        elif scheduler_type == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=2000, gamma=0.1)

        # Optimization loop
        self.losses = []
        self.lrs = []

        for i in range(self.maxiter):
            self.optimizer.zero_grad()
            loss = self.loss()

            if i == 0 or (i + 1) % 500 == 0:
                LOG.info(f'Step {i+1}, loss={loss.item():.3f}, lr={self.optimizer.param_groups[0]["lr"]:.5f}, maxiters finding xy={(self.iter_R, self.iter_C)}')

            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step(loss)

            self.losses.append(loss.item())
            self.lrs.append(self.optimizer.param_groups[0]['lr'])


            # Apply value clamping (vmin/vmax)

            for p, vmin, vmax in self.clamp_list:
                p.data.clamp_(min=vmin, max=vmax)

        # Update FitParamConfig.value to latest tensor values
        for cfg in all_cfg.values():
            # exclude entries like forward_model that aren't tensors
            if hasattr(cfg, 'tensor') and isinstance(cfg.tensor, torch.Tensor):
                if cfg.tensor.ndim == 0:
                    cfg.value = cfg.tensor.item()

        return self.losses, self.lrs

    # getters and setters ----------------------------------------------------

    def get_fitting_results(self): 
        return (
            self.image_R.detach().cpu().numpy(),  # fitted image for grism R
            self.image_C.detach().cpu().numpy(),  # fitted image for grism C
            self.vz_R.detach().cpu().numpy(),     # line-of-sight velocity map for R
            self.vz_C.detach().cpu().numpy()      # line-of-sight velocity map for C
        )

    def get_losses_and_lrs(self): 
        return self.losses, self.lrs

    def get_params(self): 
        velocity = {k.split('.')[-1]: v.value for k, v in self.velocity_cfg.items() if v.tensor is not None}
        disp_R = {k.split('.')[-1]: v.value for k, v in self.dispersion_cfg_R.items() if v.tensor is not None}
        disp_C = {k.split('.')[-1]: v.value for k, v in self.dispersion_cfg_C.items() if v.tensor is not None}
        return velocity, disp_R, disp_C

    def get_true_images(self): 
        return (
            self.true_grism_R.detach().cpu().numpy(),  # true (input) image from grism R
            self.true_grism_C.detach().cpu().numpy()   # true (input) image from grism C
        )

    def save_checkpoint(self, path):
        torch.save({
            'velocity_cfg': self.velocity_cfg,
            'dispersion_cfg_R': self.dispersion_cfg_R,
            'dispersion_cfg_C': self.dispersion_cfg_C,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'losses': self.losses,
            'lrs': self.lrs
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.velocity_cfg = checkpoint['velocity_cfg']
        self.dispersion_cfg_R = checkpoint['dispersion_cfg_R']
        self.dispersion_cfg_C = checkpoint['dispersion_cfg_C']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.losses = checkpoint['losses']
        self.lrs = checkpoint['lrs']

    def set_maxiter(self, new_maxiter):
        self.maxiter = new_maxiter
