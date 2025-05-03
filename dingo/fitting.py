# TODO: logging

import torch
import logging
from typing import Dict
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

class KinemtaicsFitter(BaseFitter):

    def __init__(self, coadd_R, coadd_C, config: Dict, device=None): 

        # Data Preparation

        # R data

        self.true_grism_R = torch.tensor(coadd_R, dtype=torch.float32, device=device)

        # BUG: R+B will cause the x and y not converge
        _, spatial_model, disp_model = grism.load_nircam_wfss_model('R', 'A', 'F444W')
        self.forward_model_R = utils.get_grism_model_torch(
            spatial_model, disp_model, 'R', 1024, 1024, direction='forward'
        )

        # C data

        self.true_grism_C = torch.tensor(coadd_C, dtype=torch.float32, device=device)

        _, spatial_model, disp_model = grism.load_nircam_wfss_model('C', 'A', 'F444W')
        self.forward_model_C = utils.get_grism_model_torch(
            spatial_model, disp_model, 'C', 1024, 1024, direction='forward'
        )


        if not device: 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        r_fit = 40

        # params to be fitted

        self.V_rot = torch.tensor(-500.0, requires_grad=True, device=device)
        self.R_v = torch.tensor(10.0, requires_grad=True, device=device)
        self.x0_v = torch.tensor(r_fit+0., requires_grad=False, device=device)
        self.y0_v = torch.tensor(r_fit+0., requires_grad=False, device=device)
        self.theta_v = torch.tensor(-1., requires_grad=True, device=device)
        self.inc_v = torch.tensor(1., requires_grad=True, device=device)

        self.dx_R = torch.tensor(-2., requires_grad=True, device=device)
        self.dy_R = torch.tensor(-4., requires_grad=True, device=device)

        self.dx_C = torch.tensor(-2., requires_grad=True, device=device)
        self.dy_C = torch.tensor(-5., requires_grad=True, device=device)


        self.velocity_params = {
            'V_rot': self.V_rot,
            'R_v': self.R_v,
            'x0_v': self.x0_v,
            'y0_v': self.y0_v,
            'theta_v': self.theta_v,
            'inc_v': self.inc_v,
        }

        self.dispersion_params_R = {
            'forward_model': self.forward_model_R, 
            'dx': self.dx_R,
            'dy': self.dy_R,
        }

        self.dispersion_params_C = {
            'forward_model': self.forward_model_C, 
            'dx': self.dx_C,
            'dy': self.dy_C,
        }

        # spectral information
        z = 1.123
        self.lambda_rest = torch.tensor([1.875])*(1 + z) # nm, Pa alpha
        nx_G, ny_G = self.true_grism_R.shape
        self.y_G, self.x_G = torch.meshgrid(
            torch.arange(nx_G), torch.arange(ny_G), indexing='ij')
        # fitting area XXX
        cx_R, cy_R = self.forward_model_R(
            torch.tensor(r_fit), torch.tensor(r_fit), self.lambda_rest)
        cx_C, cy_C = self.forward_model_C(
            torch.tensor(r_fit), torch.tensor(r_fit), self.lambda_rest)
        self.cutout_R = (int(cx_R)-r_fit, int(cy_R)-r_fit, 2*r_fit+1, 2*r_fit+1)
        self.cutout_C = (int(cx_C)-r_fit, int(cy_C)-r_fit, 2*r_fit+1, 2*r_fit+1)

        
        # temporary parameters to be used in the training loop
        self.x_R, self.y_R = kinematics.forward_dispersion_model(
            self.x_G, self.y_G, self.lambda_rest, **self.dispersion_params_R)
        self.x_C, self.y_C = kinematics.forward_dispersion_model(
            self.x_G, self.y_G, self.lambda_rest, **self.dispersion_params_C)
        self.iter_R = None
        self.iter_C = None

        
        # outputs 
        self.image_R = None
        self.image_C = None
        self.vz_R = None
        self.vz_C = None

        # fitting checkpoints
        self.lrs = None
        self.losses = None

        # prepare optimizer, scheduler, and maxiter

        named_params = collect_named_trainable_params(
            velocity_params=self.velocity_params,
            dispersion_params_R=self.dispersion_params_R,
            dispersion_params_C=self.dispersion_params_C,
        )

        param_groups = []
        for name, p in named_params.items():
            if name in ['velocity_params.V_rot', 'velocity_params.R_v']:
                param_groups.append({'params': [p], 'lr': 0.03})
            elif name in ['dispersion_params_R.dx', 'dispersion_params_R.dy']:
                param_groups.append({'params': [p], 'lr': 0.3})
            elif name in ['dispersion_params_C.dx', 'dispersion_params_C.dy']:
                param_groups.append({'params': [p], 'lr': 0.3})
            else:
                param_groups.append({'params': [p], 'lr': 0.03})

        self.optimizer = torch.optim.Adam(param_groups)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.707,
            patience=500,
            min_lr=1e-6
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=np.sqrt(0.1))


        self.maxiter = 10000


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