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

    # --- ensure inputs are lists ----------------------------------------------
    if not isinstance(default_cfgs, list):
        default_cfgs = [default_cfgs]
    if not isinstance(overrides_list, list):
        overrides_list = [overrides_list]

    # --- EXPAND grouped override keys -----------------------------------------
    # allow keys like "image.R.dx, image.R.dy, image.C.dx" → apply same settings
    expanded_overrides = []
    for ov in overrides_list:
        exp = {}
        for key, settings in ov.items():
            # split on commas, strip whitespace, re-assign
            for sub in [k.strip() for k in key.split(',')]:
                exp[sub] = settings
        expanded_overrides.append(exp)
    overrides_list = expanded_overrides

    # --- allowed keys & build configs -----------------------------------------
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
                oc = overrides.get(full_key, {})  # now catches any split keys
                cfgs[full_key] = FitParamConfig(
                    name=full_key,
                    value=val,
                    lr=oc.get('lr', default_cfg['lr']),
                    vmin=oc.get('vmin', default_cfg['vmin']),
                    vmax=oc.get('vmax', default_cfg['vmax']),
                    fit=oc.get('fit', default_cfg['fit'])
                )
        # apply any string aliases
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
# generic base class for all fitters
# ----------------------------------------------------------------------------
class BaseFitter(ABC):

    def __init__(self, config_path: AnyStr, device=None):
        # ─────────────────────────────
        # Load configuration & metadata
        # ─────────────────────────────
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config_path = config_path
        self.config = config
        self.device = device

        # ─────────────────────────────
        # Summary / Metadata
        # ─────────────────────────────
        s = config['summary']
        self.name   = s['name']
        self.ra     = s['ra']
        self.dec    = s['dec']
        self.z      = s['z']
        self.filter = s['filter']
        self.mode   = s['mode']
        self.r_fit  = s.get('r_fit', None)

        # ─────────────────────────────
        # Fitting stage state
        # ─────────────────────────────
        self.current_stage = 0
        # container for subclass-defined param configs
        self.param_config_lists: Dict[str, list] = {}

        # ─────────────────────────────
        # Subclass-specific setup:
        # load images, models, build param_config_lists, grid, etc.
        # ─────────────────────────────
        self._setup_data()

        # ─────────────────────────────
        # Assign stage-0 configs to attributes
        # ─────────────────────────────
        self._assign_cfgs_for_stage(0)

        # ─────────────────────────────
        # fitting state (lazy init)
        # ─────────────────────────────
        self.optimizer  = None
        self.scheduler  = None
        self.clamp_list = None

    @abstractmethod
    def _setup_data(self):
        """
        Subclasses must:
          - load any image data into attributes
          - load forward models if needed
          - build self.param_config_lists, a dict mapping names
            to lists of cfg-dicts (one per fitting stage)
        """
        pass

    def _assign_cfgs_for_stage(self, stage: int):
        """
        Shortcut attributes for the current stage's config dicts.
        E.g. self.velocity_cfg = self.param_config_lists['velocity'][stage]
        """
        for name, cfg_list in self.param_config_lists.items():
            setattr(self, f'{name}_cfg', cfg_list[stage])

    def _reset_state(self):
        """
        Subclasses may override to reset any per-stage state
        (e.g. starting positions) before each fit_gradient run.
        By default, do nothing.
        """
        pass

    @abstractmethod
    def loss(self):
        """
        Compute and return a scalar loss Tensor.
        Subclasses implement, using self.*_cfg and any loaded data.
        """
        pass

    def _log(self, i: int, loss: torch.Tensor):
        """
        Default logging hook: logs stage, step, loss, and lr.
        Subclasses can override to include extra info (e.g. xy_iters).
        """
        if i == 0 or (i+1) % 500 == 0:
            LOG.info(
                f"Stage {self.current_stage+1}, "
                f"Step {i+1}, loss={loss.item():.3f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.5f}"
            )

    def fit_gradient(self):
        """
        Generic gradient-based fitting loop, using:
          - self.loss() for loss computation
          - self.param_config_lists for parameter groups
          - schedulers as specified in config['fitting'][stage]['scheduler']
        """
        fs = self.config['fitting'][self.current_stage]
        self.maxiter = fs['maxiter']

        # reset any subclass-specific state
        self._reset_state()

        # collect params & build optimizer
        all_cfg = {}
        for cfg_list in self.param_config_lists.values():
            all_cfg.update(cfg_list[self.current_stage])
        param_groups, clamp_list = _extract_tensors(all_cfg)
        self.optimizer  = torch.optim.Adam(param_groups)
        self.clamp_list = clamp_list

        # scheduler setup
        sched_type = fs['scheduler']
        if sched_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.707, patience=500, min_lr=1e-6
            )
        elif sched_type == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=2000, gamma=0.1
            )
        else: 
            LOG.warning(f'{sched_type} is not a currently supported scheduler!')

        self.losses = []
        self.lrs    = []

        for i in range(self.maxiter):
            self.optimizer.zero_grad()
            loss = self.loss()

            # logging hook
            self._log(i, loss)

            loss.backward()
            self.optimizer.step()

            # scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss.item())
                else:
                    self.scheduler.step()

            self.losses.append(loss.item())
            self.lrs.append(self.optimizer.param_groups[0]['lr'])

            # clamp values to their vmin/vmax
            for p, vmin, vmax in clamp_list:
                p.data.clamp_(min=vmin, max=vmax)

        # write back final tensor values
        for cfg in all_cfg.values():
            if hasattr(cfg, 'tensor') and isinstance(cfg.tensor, torch.Tensor) and cfg.tensor.ndim == 0:
                cfg.value = cfg.tensor.item()

        return self.losses, self.lrs

    def fit_all(self):
        """
        Run sequential fitting for all configured stages.
        After stage 0, propagate updated values to next-stage cfg
        so that each subsequent fit starts from the last-fitted values.
        Returns:
          all_losses: list of loss histories per stage
          all_lrs:    list of lr histories per stage
        """
        all_losses = []
        all_lrs    = []

        nstages = len(self.config['fitting'])
        for idx in range(nstages):
            LOG.info(f"Starting fitting stage {idx+1}/{nstages}: method={self.config['fitting'][idx]['method']}")
            self.current_stage = idx

            if idx > 0:
                # propagate values from stage idx-1 → idx
                for cfg_list in self.param_config_lists.values():
                    prev_cfg = cfg_list[idx-1]
                    curr_cfg = cfg_list[idx]
                    for key, cfg in curr_cfg.items():
                        # skip forward_model entries
                        if key.endswith('forward_model'):
                            continue
                        if key in prev_cfg:
                            cfg.value = prev_cfg[key].value
                            cfg.tensor = None

            # assign the up-to-date cfgs for this stage
            self._assign_cfgs_for_stage(idx)

            losses, lrs = self.fit_gradient()
            all_losses.append(losses)
            all_lrs.append(lrs)

        return all_losses, all_lrs

    def get_losses_and_lrs(self):
        return self.losses, self.lrs

#%% --------------------------------------------------------------------------
# Kinematics fitting subclass
# ----------------------------------------------------------------------------
class KinematicsFitter(BaseFitter):

    def __init__(self, config_path: AnyStr, device=None):
        super().__init__(config_path, device)

    def _setup_data(self):
        # ─────────────────────────────
        # Image loading and wavelength
        # ─────────────────────────────
        img_cfg = self.config['image']
        self.lambda_rest = torch.tensor([self.config['summary']['lambda_rest']]) * (1 + self.z)

        self.true_grism_R = torch.tensor(load_fits_data(img_cfg['R']['path']), device=self.device)
        self.true_grism_C = torch.tensor(load_fits_data(img_cfg['C']['path']), device=self.device)

        # ─────────────────────────────
        # Forward model loading
        # ─────────────────────────────
        self.fwd_models = {}
        for pupil in ['R', 'C']:
            _, sp, dp = grism.load_nircam_wfss_model(pupil, img_cfg[pupil]['module'], self.filter)
            self.fwd_models[pupil] = utils.get_grism_model_torch(
                sp, dp, pupil, 1024, 1024, direction='forward'
            )

        # ─────────────────────────────
        # Parameter configs for all stages
        # ─────────────────────────────
        defaults  = [fs['default']  for fs in self.config['fitting']]
        overrides = [fs['override'] for fs in self.config['fitting']]

        # velocity params
        self.velocity_cfg_list = build_param_config_dict_with_alias(
            self.config['velocity'], defaults, overrides, prefix='velocity'
        )

        # dispersion R params
        self.dispersion_R_cfg_list = build_param_config_dict_with_alias(
            {k:v for k,v in img_cfg['R'].items() if k in ['dx','dy']},
            defaults, overrides, prefix='image.R'
        )
        for cfg in self.dispersion_R_cfg_list:
            cfg['image.R.forward_model'] = FitParamConfig(
                name='image.R.forward_model', value=None,
                lr=0, vmin=0, vmax=0, fit=False
            )
            cfg['image.R.forward_model'].tensor = self.fwd_models['R']

        # dispersion C params
        self.dispersion_C_cfg_list = build_param_config_dict_with_alias(
            {k:v for k,v in img_cfg['C'].items() if k in ['dx','dy']},
            defaults, overrides, prefix='image.C'
        )
        for cfg in self.dispersion_C_cfg_list:
            cfg['image.C.forward_model'] = FitParamConfig(
                name='image.C.forward_model', value=None,
                lr=0, vmin=0, vmax=0, fit=False
            )
            cfg['image.C.forward_model'].tensor = self.fwd_models['C']

        # register into the generic param_config_lists
        self.param_config_lists = {
            'velocity':       self.velocity_cfg_list,
            'dispersion_R':   self.dispersion_R_cfg_list,
            'dispersion_C':   self.dispersion_C_cfg_list,
        }

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
        self.cutout_R = (int(cxR)-self.r_fit, int(cyR)-self.r_fit, 2*self.r_fit+1, 2*self.r_fit+1)
        self.cutout_C = (int(cxC)-self.r_fit, int(cyC)-self.r_fit, 2*self.r_fit+1, 2*self.r_fit+1)

    def _reset_state(self):
        # reset XY before each stage
        self.x_R, self.y_R = self.x_G.clone(), self.y_G.clone()
        self.x_C, self.y_C = self.x_G.clone(), self.y_G.clone()

    def loss(self):
        # unpack both fitted and fixed params as tensors
        velocity = {
            k.split('.')[-1]:
            (v.tensor if v.tensor is not None else torch.tensor(v.value, dtype=torch.float32))
            for k,v in self.velocity_cfg.items()
        }
        disp_R = {
            k.split('.')[-1]:
            (v.tensor if v.tensor is not None else torch.tensor(v.value, dtype=torch.float32))
            for k,v in self.dispersion_R_cfg.items()
        }
        disp_C = {
            k.split('.')[-1]:
            (v.tensor if v.tensor is not None else torch.tensor(v.value, dtype=torch.float32))
            for k,v in self.dispersion_C_cfg.items()
        }

        # compute R‐channel
        self.x_R, self.y_R, self.vz_R, self.iter_R = kinematics.iteratively_find_xy(
            self.x_R, self.y_R, self.cutout_R,
            self.lambda_rest, self.x_G, self.y_G,
            **velocity, **disp_R
        )
        self.image_R = kinematics.bilinear_interpolte_intensity_torch(
            self.x_R, self.y_R, self.true_grism_R, self.cutout_R
        )

        # compute C‐channel
        self.x_C, self.y_C, self.vz_C, self.iter_C = kinematics.iteratively_find_xy(
            self.x_C, self.y_C, self.cutout_C,
            self.lambda_rest, self.x_G, self.y_G,
            **velocity, **disp_C
        )
        self.image_C = kinematics.bilinear_interpolte_intensity_torch(
            self.x_C, self.y_C, self.true_grism_C, self.cutout_C
        )

        # L2 loss between the two grism channels
        return torch.sum((self.image_R - self.image_C)**2)

    def _log(self, i: int, loss: torch.Tensor):
        # preserve original xy_iters logging
        if i == 0 or (i+1) % 500 == 0:
            LOG.info(
                f"Stage {self.current_stage+1}, "
                f"Step {i+1}, loss={loss.item():.3f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.5f}, "
                f"xy_iters={(self.iter_R, self.iter_C)}"
            )

    def get_fitting_results(self):
        return (
            self.image_R.detach().cpu().numpy(),
            self.image_C.detach().cpu().numpy(),
            self.vz_R.detach().cpu().numpy(),
            self.vz_C.detach().cpu().numpy()
        )

    def get_params(self):
        velocity = {k.split('.')[-1]: v.value for k,v in self.velocity_cfg.items()}
        disp_R   = {k.split('.')[-1]: v.value for k,v in self.dispersion_R_cfg.items()}
        disp_C   = {k.split('.')[-1]: v.value for k,v in self.dispersion_C_cfg.items()}
        return velocity, disp_R, disp_C

    def get_true_images(self):
        return (
            self.true_grism_R.detach().cpu().numpy(),
            self.true_grism_C.detach().cpu().numpy()
        )

    def save_checkpoint(self, path):
        raise NotImplementedError()

    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        # restore cfgs & optimizer/scheduler state...
        self.velocity_cfg     = ckpt['velocity_cfg']
        self.dispersion_R_cfg = ckpt['dispersion_R_cfg']
        self.dispersion_C_cfg = ckpt['dispersion_C_cfg']
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.losses = ckpt['losses']
        self.lrs    = ckpt['lrs']
