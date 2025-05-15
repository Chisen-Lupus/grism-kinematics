import torch
import logging
import yaml
from dataclasses import dataclass
from astropy.io import fits
import numpy as np
from typing import Dict, Any, Optional, AnyStr
from abc import ABC, abstractmethod
from pprint import pprint
import os

from . import kinematics, utils, grism, galaxy

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

def log_call(func):
    def wrapper(*args, **kwargs):
        # print(f'Calling "{func.__name__}"')
        LOG.info(f'Calling "{func.__name__}"')
        return func(*args, **kwargs)
    return wrapper

#%% --------------------------------------------------------------------------
# helper classes and functions
# ----------------------------------------------------------------------------

@dataclass
class FitParamConfig:
    name: str
    value: float
    lr: float
    min: float
    max: float
    fit: bool
    tensor: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.min = float(self.min)
        self.max = float(self.max)

    def __repr__(self):
        return (
            f'FitParamConfig('
            f'name={self.name!r}, value={self.value}, lr={self.lr}, '
            f'fit={self.fit}, min={self.min:.2e}, max={self.max:.2e})'
        )

    def __str__(self):
        return self.__repr__()
    
def build_param_config_dict_with_alias(
    raw_dict: Dict[str, Any],
    default_cfgs: Any,
    overrides_list: Any,
    prefix: str, 
    all_cfgs: dict = None
) -> list:
    '''
    Build a list of parameter configuration dictionaries for each fitting strategy.
    raw_dict: parameter values from config.
    default_cfgs: list of default cfg dicts for each stage.
    overrides_list: list of override dicts for each stage.
    prefix: key prefix (e.g., 'velocity', 'image.R').
    '''

    if all_cfgs is None:
        all_cfgs = [{}]*len(default_cfgs)  # Create a fresh dict if not provided

    # --- ensure inputs are lists ----------------------------------------------
    if not isinstance(default_cfgs, list):
        default_cfgs = [default_cfgs]
    if not isinstance(overrides_list, list):
        overrides_list = [overrides_list]

    # --- EXPAND grouped override keys -----------------------------------------
    # allow keys like 'image.R.dx, image.R.dy, image.C.dx' → apply same settings
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
        'image': {'dx', 'dy'},
        'sersic': {'I_e', 'R_e', 'n', 'x0', 'y0', 'q', 'theta'}, 
        'psf': {'I_psf', 'x_psf', 'y_psf'}, 
        'direct': {'dx', 'dy'} # TODO: change image to grism and alas to direct??
    }

    cfgs_list = []
    # process each batch of fitting
    for default_cfg, overrides, all_cfg in zip(default_cfgs, overrides_list, all_cfgs):
        cfgs = {}
        alias_map = {}
        for name, val in raw_dict.items():
            full_key = f'{prefix}.{name}'
            # NOTE: full_key may be prefix.name or prefix.cid.name. In the latter case prefix.cid is treated as a prefix
            if name not in allowed_keys.get(prefix.split('.')[0], set()):
                continue
            if isinstance(val, str):
                alias_map[full_key] = val
            else:
                oc = overrides.get(full_key, {})  # now catches any split keys
                cfgs[full_key] = FitParamConfig(
                    name=full_key,
                    value=val,
                    lr=oc.get('lr', default_cfg['lr']),
                    min=oc.get('min', default_cfg['min']),
                    max=oc.get('max', default_cfg['max']),
                    fit=oc.get('fit', default_cfg['fit'])
                )
        # apply any string aliases
        all_cfg.update(cfgs) #joint known keys as alias candidates
        for key, alias in alias_map.items():
            if alias in all_cfg:
                cfgs[key] = all_cfg[alias]
            else:
                raise ValueError(f'Alias {alias} not found for {key}')
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
        clamp_list.append((cfg.tensor, cfg.min, cfg.max))
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
        if not device: 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        '''
        NOTE: Flat list of fitting parameters. It should looks like:
            {'im0': [{'direct.f356w.im0.dx': FitParamConfig(...),
                      'direct.f356w.im0.dy': FitParamConfig(...)},
                     {'direct.f356w.im0.dx': FitParamConfig(...),
                      'direct.f356w.im0.dy': FitParamConfig(...)}],
             'p0': [{'psf.p0.I_psf': FitParamConfig(...),
                     'psf.p0.x_psf': FitParamConfig(...),
                     'psf.p0.y_psf': FitParamConfig(...)},
                    {'psf.p0.I_psf': FitParamConfig(...),
                     'psf.p0.x_psf': FitParamConfig(...),
                     'psf.p0.y_psf': FitParamConfig(...)}],
             's0': [{'sersic.s0.I_e': FitParamConfig(...),
                     'sersic.s0.R_e': FitParamConfig(...),
                     'sersic.s0.n': FitParamConfig(...),
                     'sersic.s0.q': FitParamConfig(...),
                     'sersic.s0.theta': FitParamConfig(...),
                     'sersic.s0.x0': FitParamConfig(...),
                     'sersic.s0.y0': FitParamConfig(...)},
                    {'sersic.s0.I_e': FitParamConfig(...),
                     'sersic.s0.R_e': FitParamConfig(...),
                     'sersic.s0.n': FitParamConfig(...),
                     'sersic.s0.q': FitParamConfig(...),
                     'sersic.s0.theta': FitParamConfig(...),
                     'sersic.s0.x0': FitParamConfig(...),
                     'sersic.s0.y0': FitParamConfig(...)}]}
        '''

        # ─────────────────────────────
        # Subclass-specific setup:
        # load images, models, build param_config_lists, grid, etc.
        # ─────────────────────────────
        self._setup_data()

        # setup completeness check
        if not self.param_config_lists:
            LOG.warning(
                f'[{self.__class__.__name__}] ⚠️ `param_config_lists` is empty. '
                'You should assign parameter config lists in `_setup_data()` or subclass `__init__`.'
            )

        # ─────────────────────────────
        # Assign stage-0 configs to attributes
        # ─────────────────────────────
        self._assign_cfgs_for_stage(0)

    @abstractmethod
    def _setup_data(self):
        '''
        Subclasses must:
          - load any image data into attributes
          - load forward models if needed
          - build self.param_config_lists, a dict mapping names
            to lists of cfg-dicts (one per fitting stage)
        '''
        pass

    def _assign_cfgs_for_stage(self, stage: int):
        '''
        Shortcut attributes for the current stage's config dicts.
        E.g. self.velocity_cfg = self.param_config_lists['velocity'][stage]
        Example generated attributes: self.im0_cfg, self.s0_cfg, self.p0_cfg
        '''
        for name, cfg_list in self.param_config_lists.items():
            setattr(self, f'{name}_cfg', cfg_list[stage])

    def _get_model_params(self, key): 
        params = {
            k.split('.')[-1]:
            (v.tensor if v.tensor is not None else torch.tensor(v.value, dtype=torch.float32))
            for k,v in getattr(self, f'{key}_cfg').items()
        }
        return params

    def _reset_state(self):
        '''
        Subclasses may override to reset any per-stage state
        (e.g. starting positions) before each fit_gradient run.
        By default, do nothing.
        '''
        pass

    @abstractmethod
    def loss(self):
        '''
        Compute and return a scalar loss Tensor.
        Subclasses implement, using self.*_cfg and any loaded data.
        '''
        pass

    def _log(self, i: int, loss: torch.Tensor):
        '''
        Default logging hook: logs stage, step, loss, and lr.
        Subclasses can override to include extra info (e.g. xy_iters).
        '''
        if i == 0 or (i+1) % 500 == 0:
            LOG.info(
                f'Stage {self.current_stage+1}, '
                f'Step {i+1}, loss={loss.item():.3f}, '
                f'lr={self.optimizer.param_groups[0]['lr']:.5f}'
            )

    def fit_gradient(self):
        '''
        Generic gradient-based fitting loop, using:
          - self.loss() for loss computation
          - self.param_config_lists for parameter groups
          - schedulers as specified in config['fitting'][stage]['scheduler']
        '''
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

        try: 

            for i in range(self.maxiter):
                self.optimizer.zero_grad()
                loss = self.loss()

                # logging hook
                self._log(i, loss)

                self.losses.append(loss.item())
                self.lrs.append(self.optimizer.param_groups[0]['lr'])

                loss.backward()
                self.optimizer.step()
                
                # scheduler step
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(loss.item())
                    else:
                        self.scheduler.step()

                # clamp values to their min/max
                for p, min, max in clamp_list:
                    p.data.clamp_(min=min, max=max)

        except KeyboardInterrupt: 
            LOG.info(f'KeyboadrInterrupt, last step: {i}')

        # write back final tensor values
        for cfg in all_cfg.values():
            if hasattr(cfg, 'tensor') and isinstance(cfg.tensor, torch.Tensor) and cfg.tensor.ndim == 0:
                cfg.value = cfg.tensor.item()

        return self.losses, self.lrs

    def fit_all(self):
        '''
        Run sequential fitting for all configured stages.
        After stage 0, propagate updated values to next-stage cfg
        so that each subsequent fit starts from the last-fitted values.
        Returns:
          all_losses: list of loss histories per stage
          all_lrs:    list of lr histories per stage
        '''
        all_losses = []
        all_lrs    = []

        nstages = len(self.config['fitting'])
        for idx in range(nstages):
            LOG.info(f'Starting fitting stage {idx+1}/{nstages}: method={self.config['fitting'][idx]['method']}')
            self.current_stage = idx

            if idx > 0:
                # propagate values from stage idx-1 -> idx
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
        # Parameter configs for ALL stages
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
                lr=0, min=0, max=0, fit=False
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
                lr=0, min=0, max=0, fit=False
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
        velocity = self._get_model_params('velocity')
        disp_R = self._get_model_params('dispersion_R')
        disp_C = self._get_model_params('dispersion_C')

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
                f'Stage {self.current_stage+1}, '
                f'Step {i+1}, loss={loss.item():.3f}, '
                f'lr={self.optimizer.param_groups[0]['lr']:.5f}, '
                f'xy_iters={(self.iter_R, self.iter_C)}'
            )

    # getters ----------------------------------------------------------------

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

#%% --------------------------------------------------------------------------
# Image fitting subclass
# ----------------------------------------------------------------------------

class ImagesFitter(BaseFitter):

    def __init__(self, config_path: str, device=None):
        super().__init__(config_path, device)

    def _setup_data(self):

        # 1) Load psf and direct image data (except dx dy)
        self.all_filters = set(self.config['direct'].keys())

        # load psf data
        self.psfs = {} # {pid: {property: value}}
        for filter, psfs_cfg in self.config['psfs'].items():
            self.psfs[filter] = {}
            for pid, psf_cfg in psfs_cfg.items():
                psf_data = load_fits_data(psf_cfg['path'])
                psf_tensor = torch.tensor(
                    psf_data, dtype=torch.float32, device=self.device
                )
                psf_info = {}
                psf_info['psf'] = psf_tensor
                psf_info['oversample'] = psf_cfg['oversample']
                psf_info['image_map'] = [] # to be added in the next step
                self.psfs[filter][pid] = psf_info

        # load image data
        self.direct_images = {} # {filter: {iid: {property: value}}}
        for filter, imgs_cfg in self.config['direct'].items():
            self.direct_images[filter] = {}
            for iid, img_cfg in imgs_cfg.items():
                # register psf
                pid = img_cfg['psf']
                self.psfs[filter][pid]['image_map'].append(iid)
                # add image data
                img_data = load_fits_data(img_cfg['path'])
                if 'cutout' in img_cfg: 
                    x, y, dx, dy = img_cfg['cutout']
                    img_data = img_data[x:x+dx, y:y+dy]
                img_tensor = torch.tensor(
                    img_data, dtype=torch.float32, device=self.device
                )
                direct_info = {}
                direct_info['pid'] = pid
                direct_info['image'] = img_tensor
                direct_info['oversample'] = img_cfg['oversample']
                if self.psfs[filter][pid]['oversample']<direct_info['oversample']: 
                    raise ValueError('PSF is not well-sampled!')
                self.direct_images[filter][iid] = direct_info

        # 2) Grid # TODO: specify for each filter??
        self.nx, self.ny = self.r_fit*2+1, self.r_fit*2+1
        self.yy, self.xx = torch.meshgrid(
            torch.linspace(0, self.nx-1, self.nx, device=self.device),
            torch.linspace(0, self.ny-1, self.ny, device=self.device),
            indexing='ij'
        )

        # 3) Defaults & overrides
        defaults = [fs['default'] for fs in self.config['fitting']]
        overrides = [fs['override'] for fs in self.config['fitting']]

        # 4) Raw params
        _all_cfgs = [{} for _ in range(len(self.config['fitting']))]  # temp cfg for matching alias
        self.direct_cfgs_lists = {}
        self.sersic_cfgs_lists = {}
        self.psf_cfgs_lists = {}

        # add image params
        for filter, imgs_cfg in self.config['direct'].items():
            for iid, raw_dict in imgs_cfg.items():
                i_cfgs = build_param_config_dict_with_alias(
                    raw_dict=raw_dict,
                    default_cfgs=defaults,
                    overrides_list=overrides,
                    prefix=f'direct.{filter}.{iid}',
                    all_cfgs=_all_cfgs
                )
                self.direct_cfgs_lists[iid] = i_cfgs

        # add sersic params
        for cid, raw_dict in self.config['sersic'].items():
            s_cfgs = build_param_config_dict_with_alias(
                raw_dict=raw_dict,
                default_cfgs=defaults,
                overrides_list=overrides,
                prefix=f'sersic.{cid}',
                all_cfgs=_all_cfgs
            )
            self.sersic_cfgs_lists[cid] = s_cfgs

        # add psf params
        for cid, raw_dict in self.config['psf'].items():
            p_cfgs = build_param_config_dict_with_alias(
                raw_dict=raw_dict,
                default_cfgs=defaults,
                overrides_list=overrides,
                prefix=f'psf.{cid}',
                all_cfgs=_all_cfgs
            )
            self.psf_cfgs_lists[cid] = p_cfgs

        # 5) Register
        self.param_config_lists = self.psf_cfgs_lists | self.sersic_cfgs_lists | self.direct_cfgs_lists

    def loss(self):

        # NOTE: self.*_cfg is already generated by _assign_cfgs_for_stage

        loss = 0

        # construct true image per psf
        for filter in self.all_filters:
            for pid, psf_info in self.psfs[filter].items():
                # print(psf_info)
                psf_tensor = psf_info['psf']
                img_list = psf_info['image_map']

                # compute psf-level sampled model at this filter

                model = 0

                psf_cid_list = self.config['psf'].keys()
                for psf_cid in psf_cid_list: 
                    psf_params = self._get_model_params(psf_cid)
                    psf_model = model + galaxy.full_psf_model_torch(
                        self.nx, self.ny, psf_tensor, **psf_params
                    )
                    model += psf_model
                
                sersic_cid_list = self.config['sersic'].keys()
                for sersic_cid in sersic_cid_list: 
                    sersic_params = self._get_model_params(sersic_cid)
                    sersic_model = galaxy.full_sersic_model_torch(
                        self.xx, self.yy, psf_tensor, **sersic_params
                    )
                    model += sersic_model

                # downsample to each image and compute loss

                for iid in img_list: 
                    direct_info = self.direct_images[filter][iid]
                    this_image = direct_info['image']
                    factor = psf_info['oversample']//direct_info['oversample']
                    direct_params = self._get_model_params(iid)
                    nx, ny = this_image.shape
                    dx, dy = direct_params.values()
                    this_model = utils.downsample_with_shift_and_size(
                        x=model, factor=factor, out_size=(nx, ny), shift=(dx, dy), 
                    )
                    this_loss = torch.sum((this_model - this_image)**2)
                    loss += this_loss

        return loss
    
    # getters ----------------------------------------------------------------

    def get_params(self):
        '''
        Return two dicts of final parameter values (pure Python floats),
        first for all Sérsic components, then for all PSF components.
        Each is a mapping cid -> { param_name: value, … }.
        '''
        sersic = {
            cid: {
                k.split('.')[-1]: v.value
                for k, v in cfg_list[self.current_stage].items()
            }
            for cid, cfg_list in self.sersic_cfgs_lists.items()
        }
        psf = {
            cid: {
                k.split('.')[-1]: v.value
                for k, v in cfg_list[self.current_stage].items()
            }
            for cid, cfg_list in self.psf_cfgs_lists.items()
        }
        return sersic, psf

    def get_true_image(self, filter=None, iid=None): 
        '''
        Return the original (data) image as a NumPy array.
        '''
        if not filter: 
            filter = next(iter(self.direct_images.keys()))
        if not iid: 
            iid = next(iter(self.direct_images[filter].keys()))
        true_image = self.direct_images[filter][iid]['image']
        return true_image.detach().cpu().numpy()

    def get_true_images(self):
        '''
        Return the original (data) image as a dict of NumPy array.
        '''
        true_images = {}
        for filter in self.all_filters:
            true_images[filter] = {}
            for iid in self.direct_images[filter].keys(): 
                true_image = self.get_true_image(filter, iid) 
                true_images[filter][iid] = true_image

        return true_images

    def get_fitted_component(self, filter=None, iid=None):
        '''
        Reconstruct the fitted models (sum of PSF and/or Sérsic components)
        and return it as a NumPy array.

        Parameters
        ----------
        filter: str
            The Filter of the model to be returned. Default to be the first item
        iid: str
            The Image ID of the model to be returned. Default to be the first item
        '''

        if not filter: 
            filter = next(iter(self.direct_images.keys()))
        if not iid: 
            iid = next(iter(self.direct_images[filter].keys()))

        # image-specifig configs
        
        direct_info = self.direct_images[filter][iid]
        pid = direct_info['pid']
        psf_info = self.psfs[filter][pid]
        psf_tensor = psf_info['psf']
        this_image = direct_info['image']
        factor = psf_info['oversample']//direct_info['oversample']
        direct_params = self._get_model_params(iid)
        nx, ny = this_image.shape
        dx, dy = direct_params.values()

        # compute models and downsample

        psf_models = {}
        sersic_models = {}

        psf_cid_list = self.config['psf'].keys()
        for psf_cid in psf_cid_list: 
            psf_params = self._get_model_params(psf_cid)
            psf_model = galaxy.full_psf_model_torch(
                self.nx, self.ny, psf_tensor, **psf_params
            )
            this_component = utils.downsample_with_shift_and_size(
                x=psf_model, factor=factor, out_size=(nx, ny), shift=(dx, dy), 
            )
            psf_models[psf_cid] = this_component.detach().cpu().numpy()
        
        sersic_cid_list = self.config['sersic'].keys()
        for sersic_cid in sersic_cid_list: 
            sersic_params = self._get_model_params(sersic_cid)
            sersic_model = galaxy.full_sersic_model_torch(
                self.xx, self.yy, psf_tensor, **sersic_params
            )
            this_component = utils.downsample_with_shift_and_size(
                x=sersic_model, factor=factor, out_size=(nx, ny), shift=(dx, dy), 
            )
            sersic_models[sersic_cid] = this_component.detach().cpu().numpy()
        
        return sersic_models, psf_models

    def get_fitted_components(self):
        '''
        return all the get_fitted_component
        '''
        models = {}
        all_sersic_models = {}
        all_psf_models = {}
        for filter in self.all_filters:
            models[filter] = {}
            all_psf_models[filter] = {}
            for iid in self.direct_images[filter].keys(): 
                sersic_models, psf_models = self.get_fitted_component(filter, iid)
                all_sersic_models[filter][iid] = sersic_models
                all_psf_models[filter][iid] = psf_models

        return all_sersic_models, all_psf_models

    def get_fitted_model(self, filter=None, iid=None):
        '''
        Reconstruct the final fitted model (sum of PSF + Sérsic components)
        and return it as a NumPy array.
        '''
        sersic_models, psf_models = self.get_fitted_component(filter, iid)
        model = np.sum(list(sersic_models.values())+list(psf_models.values()), axis=0)
        return model

    def get_fitted_models(self):
        '''
        return all the get_fitted_model
        '''
        models = {}
        for filter in self.all_filters:
            models[filter] = {}
            for iid in self.direct_images[filter].keys(): 
                model = self.get_fitted_model(filter, iid)
                models[filter][iid] = model
        return models