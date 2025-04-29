# TODO: logging

import torch

from . import kinematics

def loss_kinematics_gradient(cutout_R, cutout_C, true_grism_R, true_grism_C, lambda_rest, x_G, y_G, velocity_params, dispersion_params_R, dispersion_params_C):
    
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

    return loss



def fit_kinematics_gradient():
    pass