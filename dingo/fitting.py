# TODO: logging

import torch
import logging

from . import kinematics

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
# loss functions 
# ----------------------------------------------------------------------------

def loss_kinematics(
    x_R, y_R, x_C, y_C, cutout_R, cutout_C, true_grism_R, true_grism_C, lambda_rest, 
    x_G, y_G, velocity_params, dispersion_params_R, dispersion_params_C
):
    
    # R grism

    x_R, y_R, vz_R, iter_R = kinematics.iteratively_find_xy(
        x_R, y_R, cutout_R, lambda_rest, x_G, y_G, 
        **velocity_params, **dispersion_params_R
    )

    image_R = kinematics.bilinear_interpolte_intensity_torch(
        x_R, y_R, true_grism_R, cutout_R
    )

    # C grism

    x_C, y_C, vz_C, iter_C = kinematics.iteratively_find_xy(
        x_C, y_C, cutout_C, lambda_rest, x_G, y_G, 
        **velocity_params, **dispersion_params_C
    )

    image_C = kinematics.bilinear_interpolte_intensity_torch(
        x_C, y_C, true_grism_C, cutout_C
    )

    # loss
    loss = torch.sum((image_R - image_C)**2)

    return loss, x_R, y_R, x_C, y_C, iter_R, iter_C, image_R, image_C, vz_R, vz_C


#%% --------------------------------------------------------------------------
# fitting functions 
# ----------------------------------------------------------------------------

def fit_kinematics_gradient(
        optimizer, scheduler, maxiter, cutout_R, cutout_C, true_grism_R, true_grism_C, 
        lambda_rest, x_G, y_G, velocity_params, dispersion_params_R, dispersion_params_C
):

    x_R, y_R = kinematics.forward_dispersion_model(x_G, y_G, lambda_rest, **dispersion_params_R)
    x_C, y_C = kinematics.forward_dispersion_model(x_G, y_G, lambda_rest, **dispersion_params_C)

    losses = []
    lrs = []
    try: 
        for i in range(maxiter):

            optimizer.zero_grad()
            
            # NOTE: reusing x/y_R/C will accelerate finding xy
            loss, x_R, y_R, x_C, y_C, iter_R, iter_C, image_R, image_C, vz_R, vz_C = loss_kinematics(
                x_R, y_R, x_C, y_C, cutout_R, cutout_C, true_grism_R, true_grism_C, 
                lambda_rest, x_G, y_G, velocity_params, dispersion_params_R, dispersion_params_C
            )

            if i==0:
                LOG.info(f'Starting, loss={loss.item():.3f}, lr={optimizer.param_groups[0]['lr']:.5f}, maxiters finding xy={(iter_R, iter_C)}')
            
            if (i+1)%500==0:
                LOG.info(f'Step {i+1}, loss={loss.item():.3f}, lr={optimizer.param_groups[0]['lr']:.5f}, maxiters finding xy={(iter_R, iter_C)}')

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            
            # TODO: apply data clamp
            # theta_v.data.clamp_(min=0., max=torch.pi)
            # inc_v.data.clamp_(min=0., max=torch.pi/2)

            # break 
    except Exception as e: 
        print(repr(e))
        pass

    return losses, lrs, image_R, image_C, vz_R, vz_C