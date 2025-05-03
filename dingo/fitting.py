import torch
import logging
import yaml
from typing import Dict, AnyStr
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


def load_config_and_initialize(config_path, coadd_R, coadd_C, device=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    summary = config['summary']
    image_config = config['image']
    velocity_config = config['velocity']
    fitting_config = config['fitting'][0]  # use the first strategy for now

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

    # Build velocity params
    velocity_params = {}
    for key in ['V_rot', 'R_v', 'x0_v', 'y0_v', 'theta_v', 'inc_v']:
        val = velocity_config[key]
        velocity_params[key] = torch.tensor(val, dtype=torch.float32, requires_grad=True, device=device)

    # Pixel grid
    nx_G, ny_G = true_grism_R.shape
    y_G, x_G = torch.meshgrid(torch.arange(nx_G), torch.arange(ny_G), indexing='ij')

    # Cutout centers
    cx_R, cy_R = fwd_models['R'](torch.tensor(r_fit), torch.tensor(r_fit), lambda_rest)
    cx_C, cy_C = fwd_models['C'](torch.tensor(r_fit), torch.tensor(r_fit), lambda_rest)
    cutout_R = (int(cx_R)-r_fit, int(cy_R)-r_fit, 2*r_fit+1, 2*r_fit+1)
    cutout_C = (int(cx_C)-r_fit, int(cy_C)-r_fit, 2*r_fit+1, 2*r_fit+1)

    # Dispersion parameters
    dispersion_params = {}
    for pupil in ['R', 'C']:
        dispersion_params[pupil] = {
            'forward_model': fwd_models[pupil],
            'dx': torch.tensor(image_config[pupil]['dx'], dtype=torch.float32, requires_grad=True, device=device),
            'dy': torch.tensor(image_config[pupil]['dy'], dtype=torch.float32, requires_grad=True, device=device),
        }

    # Learning rate settings
    default_lr = fitting_config['default']['lr']
    overrides = fitting_config['override']
    param_groups = []
    named_params = collect_named_trainable_params(
        velocity_params=velocity_params,
        dispersion_params_R=dispersion_params['R'],
        dispersion_params_C=dispersion_params['C'],
    )

    for name, p in named_params.items():
        if name in overrides and 'lr' in overrides[name]:
            param_groups.append({'params': [p], 'lr': overrides[name]['lr']})
        else:
            param_groups.append({'params': [p], 'lr': default_lr})

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

    def __init__(self, coadd_R, coadd_C, config: AnyStr, device=None):
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
                if i==0:
                    LOG.info(f'Starting, loss={loss.item():.3f}, lr={lr:.5f}, maxiters finding xy={(self.iter_R, self.iter_C)}')
                if (i+1)%500==0:
                    LOG.info(f'Step {i+1}, loss={loss.item():.3f}, lr={lr:.5f}, maxiters finding xy={(self.iter_R, self.iter_C)}')

                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                self.losses.append(loss.item())
                self.lrs.append(lr)
                
                # TODO: apply data clamp
                # theta_v.data.clamp_(min=0., max=torch.pi)
                # inc_v.data.clamp_(min=0., max=torch.pi/2)

                # break 

        # make sure the result is saved when interrupted
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