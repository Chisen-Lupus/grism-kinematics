import torch
import logging
import yaml
from dataclasses import dataclass
from astropy.io import fits
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

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


def build_param_config_dict_with_alias(
    raw_dict: Dict[str, Any],
    default_cfgs: Any,
    overrides_list: Any,
    prefix: str
) -> list:
    """
    Build a list of parameter configuration dictionaries for each fitting strategy.
    raw_dict: parameter values from config.
    default_cfgs: list of default cfg dicts for each stage.
    overrides_list: list of override dicts for each stage.
    prefix: key prefix (e.g., 'velocity', 'image.R').
    """
    if not isinstance(default_cfgs, list):
        default_cfgs = [default_cfgs]
    if not isinstance(overrides_list, list):
        overrides_list = [overrides_list]

    allowed_keys = {
        'velocity': {'V_rot', 'R_v', 'x0_v', 'y0_v', 'theta_v', 'inc_v'},
        'image.R': {'dx', 'dy'},
        'image.C': {'dx', 'dy'}
    }
    cfgs_list = []
    for default_cfg, overrides in zip(default_cfgs, overrides_list):
        cfgs = {}
        alias_map = {}
        for name, val in raw_dict.items():
            full_key = f'{prefix}.{name}'
            if name not in allowed_keys.get(prefix, set()):
                continue
            if isinstance(val, str):
                alias_map[full_key] = val
            else:
                oc = overrides.get(full_key, {})
                cfgs[full_key] = FitParamConfig(
                    name=full_key,
                    value=val,
                    lr=oc.get('lr', default_cfg['lr']),
                    vmin=oc.get('vmin', default_cfg['vmin']),
                    vmax=oc.get('vmax', default_cfg['vmax']),
                    fit=oc.get('fit', default_cfg['fit'])
                )
        for key, alias in alias_map.items():
            if alias in cfgs:
                cfgs[key] = cfgs[alias]
            else:
                raise ValueError(f"Alias {alias} not found for {key}")
        cfgs_list.append(cfgs)
    return cfgs_list


def _extract_tensors(cfgs: Dict[str, FitParamConfig]):
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
    with fits.open(path) as hdulist:
        return np.array(hdulist[hdu_index].data, dtype=np.float32)

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
        self.name       = summary['name']
        self.ra         = summary['ra']
        self.dec        = summary['dec']
        self.z          = summary['z']
        self.filter     = summary['filter']
        self.mode       = summary['mode']
        self.r_fit      = summary['r_fit']

        # ─────────────────────────────
        # Image loading and wavelength
        # ─────────────────────────────
        image_cfg = config['image']
        self.lambda_rest   = torch.tensor([summary['lambda_rest']]) * (1 + self.z)
        self.true_grism_R  = torch.tensor(load_fits_data(image_cfg['R']['path']), device=device)
        self.true_grism_C  = torch.tensor(load_fits_data(image_cfg['C']['path']), device=device)

        # ─────────────────────────────
        # Forward model loading
        # ─────────────────────────────
        self.fwd_models = {}
        for pupil in ['R', 'C']:
            _, sp, dp = grism.load_nircam_wfss_model(
                pupil, image_cfg[pupil]['module'], self.filter
            )
            self.fwd_models[pupil] = utils.get_grism_model_torch(
                sp, dp, pupil, 1024, 1024, direction='forward'
            )

        # ─────────────────────────────
        # Parameter configuration per stage
        # ─────────────────────────────
        defaults  = [fs['default']  for fs in config['fitting']]
        overrides = [fs['override'] for fs in config['fitting']]

        # velocity params list
        self.velocity_cfg_list = build_param_config_dict_with_alias(
            config['velocity'], defaults, overrides, prefix='velocity'
        )
        # dispersion R params list
        self.dispersion_cfg_R_list = build_param_config_dict_with_alias(
            {k: v for k, v in image_cfg['R'].items() if k in ['dx', 'dy']},
            defaults, overrides, prefix='image.R'
        )
        # attach forward_model constant to R
        for cfg in self.dispersion_cfg_R_list:
            cfg['image.R.forward_model'] = FitParamConfig(
                name='image.R.forward_model', value=None,
                lr=0, vmin=0, vmax=0, fit=False
            )
            cfg['image.R.forward_model'].tensor = self.fwd_models['R']
        # dispersion C params list
        self.dispersion_cfg_C_list = build_param_config_dict_with_alias(
            {k: v for k, v in image_cfg['C'].items() if k in ['dx', 'dy']},
            defaults, overrides, prefix='image.C'
        )
        # attach forward_model constant to C
        for cfg in self.dispersion_cfg_C_list:
            cfg['image.C.forward_model'] = FitParamConfig(
                name='image.C.forward_model', value=None,
                lr=0, vmin=0, vmax=0, fit=False
            )
            cfg['image.C.forward_model'].tensor = self.fwd_models['C']

        # ─────────────────────────────
        # Combine all param lists for generic loops
        # suitable for all subclasses
        # ─────────────────────────────
        self.param_cfg_lists = [
            self.velocity_cfg_list,
            self.dispersion_cfg_R_list,
            self.dispersion_cfg_C_list
        ]

        # ─────────────────────────────
        # Pixel grid and cutout regions
        # ─────────────────────────────
        nx, ny = self.true_grism_R.shape
        self.y_G, self.x_G = torch.meshgrid(
            torch.arange(nx), torch.arange(ny), indexing='ij'
        )
        cxR, cyR = self.fwd_models['R'](
            torch.tensor(self.r_fit), torch.tensor(self.r_fit), self.lambda_rest
        )
        cxC, cyC = self.fwd_models['C'](
            torch.tensor(self.r_fit), torch.tensor(self.r_fit), self.lambda_rest
        )
        self.cutout_R = (int(cxR)-self.r_fit, int(cyR)-self.r_fit,
                         2*self.r_fit+1, 2*self.r_fit+1)
        self.cutout_C = (int(cxC)-self.r_fit, int(cyC)-self.r_fit,
                         2*self.r_fit+1, 2*self.r_fit+1)

        # ─────────────────────────────
        # Fitting state (lazy init)
        # ─────────────────────────────
        self.current_stage = 0
        self.optimizer     = None
        self.scheduler     = None
        self.clamp_list    = None

        # ─────────────────────────────
        # Outputs and temporary state
        # ─────────────────────────────
        self.image_R = None
        self.image_C = None
        self.vz_R    = None
        self.vz_C    = None
        self.iter_R  = None
        self.iter_C  = None

    # suitable for all subclasses
    def _collect_all_cfg(self):
        """Gather all FitParamConfig for current stage across param lists."""
        all_cfg = {}
        for cfg_list in self.param_cfg_lists:
            stage_cfg = cfg_list[self.current_stage]
            all_cfg.update(stage_cfg)
        return all_cfg

    # loss functions (subclass-specific)
    def loss(self):
        """Compute loss for KinematicsFitter using current parameters."""
        all_cfg = self._collect_all_cfg()
        # split by prefix
        vel_cfgs = {k: v for k, v in all_cfg.items() if k.startswith('velocity.')}
        dR_cfgs  = {k: v for k, v in all_cfg.items() if k.startswith('image.R.')}
        dC_cfgs  = {k: v for k, v in all_cfg.items() if k.startswith('image.C.')}
        # convert configs to tensors or constants
        velocity = {}
        for full, cfg in vel_cfgs.items():
            name = full.split('.')[-1]
            velocity[name] = cfg.tensor if cfg.tensor is not None else torch.tensor(cfg.value, dtype=torch.float32)
        disp_R = {}
        for full, cfg in dR_cfgs.items():
            name = full.split('.')[-1]
            disp_R[name] = cfg.tensor if cfg.tensor is not None else torch.tensor(cfg.value, dtype=torch.float32)
        disp_C = {}
        for full, cfg in dC_cfgs.items():
            name = full.split('.')[-1]
            disp_C[name] = cfg.tensor if cfg.tensor is not None else torch.tensor(cfg.value, dtype=torch.float32)

        # R-channel modelling
        self.x_R, self.y_R, self.vz_R, self.iter_R = kinematics.iteratively_find_xy(
            self.x_R, self.y_R, self.cutout_R,
            self.lambda_rest, self.x_G, self.y_G,
            **velocity, **disp_R
        )
        self.image_R = kinematics.bilinear_interpolte_intensity_torch(
            self.x_R, self.y_R, self.true_grism_R, self.cutout_R
        )
        # C-channel modelling
        self.x_C, self.y_C, self.vz_C, self.iter_C = kinematics.iteratively_find_xy(
            self.x_C, self.y_C, self.cutout_C,
            self.lambda_rest, self.x_G, self.y_G,
            **velocity, **disp_C
        )
        self.image_C = kinematics.bilinear_interpolte_intensity_torch(
            self.x_C, self.y_C, self.true_grism_C, self.cutout_C
        )
        # L2 loss
        return torch.sum((self.image_R - self.image_C)**2)

    # suitable for all subclasses
    def fit_gradient(self):
        fs = self.config['fitting'][self.current_stage]
        self.maxiter = fs['maxiter']
        # reset grids
        self.x_R, self.y_R = self.x_G.clone(), self.y_G.clone()
        self.x_C, self.y_C = self.x_G.clone(), self.y_G.clone()
        # collect configs
        all_cfg = self._collect_all_cfg()
        param_groups, clamp_list = _extract_tensors(all_cfg)
        self.optimizer  = torch.optim.Adam(param_groups)
        self.clamp_list = clamp_list
        # scheduler
        sched = fs['scheduler']
        if sched == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.707, patience=500, min_lr=1e-6
            )
        elif sched == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=2000, gamma=0.1
            )
        self.losses = []
        self.lrs    = []
        for i in range(self.maxiter):
            self.optimizer.zero_grad()
            loss = self.loss()
            if i == 0 or (i+1) % 500 == 0:
                LOG.info(
                    f"Stage {self.current_stage+1}, Step {i+1}, "
                    f"loss={loss.item():.3f}, lr={self.optimizer.param_groups[0]['lr']:.5f}, "
                    f"xy_iters={(self.iter_R, self.iter_C)}"
                )
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss.item())
                else:
                    self.scheduler.step()
            self.losses.append(loss.item())
            self.lrs.append(self.optimizer.param_groups[0]['lr'])
            # clamp
            for p, vmin, vmax in self.clamp_list:
                p.data.clamp_(min=vmin, max=vmax)
        # update values
        for cfg in all_cfg.values():
            if hasattr(cfg, 'tensor') and isinstance(cfg.tensor, torch.Tensor) and cfg.tensor.ndim == 0:
                cfg.value = cfg.tensor.item()
        return self.losses, self.lrs

    # suitable for all subclasses
    def fit_all(self):
        """
        Sequentially run all fitting stages.
        Start each stage from previous stage's final values.
        """
        all_losses = []
        all_lrs    = []
        total_stages = len(self.config['fitting'])
        for idx in range(total_stages):
            fs = self.config['fitting'][idx]
            LOG.info(f"Starting fitting stage {idx+1}/{total_stages}: method={fs['method']}")
            # propagate previous values
            if idx > 0:
                for cfg_list in self.param_cfg_lists:
                    prev = cfg_list[idx-1]
                    curr = cfg_list[idx]
                    for key, cfg in curr.items():
                        if key.endswith('forward_model'):
                            continue
                        cfg.value = prev[key].value
                        cfg.tensor = None
            self.current_stage = idx
            losses, lrs = self.fit_gradient()
            all_losses.append(losses)
            all_lrs.append(lrs)
        return all_losses, all_lrs

    # getters --------------------------------------------------------------
    def get_fitting_results(self):
        return (
            self.image_R.detach().cpu().numpy(),  # fitted image for grism R
            self.image_C.detach().cpu().numpy(),  # fitted image for grism C
            self.vz_R.detach().cpu().numpy(),     # line-of-sight velocity map for R
            self.vz_C.detach().cpu().numpy()      # line-of-sight velocity map for C
        )

    # suitable for all subclasses
    def get_params(self):
        all_cfg = self._collect_all_cfg()
        vel = {k.split('.')[-1]: cfg.value for k, cfg in all_cfg.items() if k.startswith('velocity.')}
        dR  = {k.split('.')[-1]: cfg.value for k, cfg in all_cfg.items() if k.startswith('image.R.')}
        dC  = {k.split('.')[-1]: cfg.value for k, cfg in all_cfg.items() if k.startswith('image.C.')}
        return vel, dR, dC

    def get_true_images(self):
        return (
            self.true_grism_R.detach().cpu().numpy(),
            self.true_grism_C.detach().cpu().numpy()
        )