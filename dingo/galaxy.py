import torch
import torch.nn.functional as F
import numpy as np
from astropy.modeling.models import Gaussian2D
from scipy.special import gammaincinv
from scipy.signal import fftconvolve
import torch.nn.functional as F

from . import utils

#%% numpy version of galaxy fitting models

def sersic_model(xs, ys, I_e, R_e, n, x0=0.0, y0=0.0, q=1.0, theta=0.0, **kwargs):
    """
    Compute the Sérsic profile intensity on a grid of positions (xs, ys).
    
    Parameters
    ----------
    xs, ys : ndarray
        2D arrays of the same shape representing x and y coordinates.
    I_e : float
        Intensity at the effective radius R_e.
    R_e : float
        Effective (half-light) radius.
    n : float
        Sérsic index.
    x0, y0 : float, optional
        Center of the profile. Default is (0, 0).
    q : float, optional
        Axis ratio (b/a), must be between 0 and 1. Default is 1 (circular).
    theta : float, optional
        Position angle in radians (CCW from x-axis). Default is 0.

    Returns
    -------
    I : ndarray
        Intensity at each (x, y) location.
    """
    # Sérsic constant b_n satisfies: Γ(2n) = 2γ(2n, b_n)
    # Approximation from Ciotti & Bertin (1999)
    b_n = gammaincinv(2*n, 0.5)

    # Shift coordinates
    dx = xs - x0
    dy = ys - y0

    # Rotate coordinates
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = cos_theta*dx + sin_theta*dy
    y_rot = -sin_theta*dx + cos_theta*dy

    # Elliptical radius
    R = np.sqrt(x_rot**2 + (y_rot/q)**2)

    # Sérsic profile
    exponent = -b_n * ((R/R_e)**(1/n) - 1)
    I = I_e * np.exp(exponent)

    return I


def full_image_model(x, y, psf, **kwargs):
    """
    Generate a full model image by evaluating a Sérsic profile and convolving it with a PSF.

    Parameters
    ----------
    psf : 2D ndarray
        PSF kernel array. Should be centered and normalized.
    kwargs : dict
        Parameters for the Sérsic model, including:
        - I_e : float       — intensity at effective radius
        - R_e : float       — effective radius
        - n   : float       — Sérsic index
        - x0, y0 : float    — center position
        - q   : float       — axis ratio
        - theta : float     — position angle in radians
        - shape : tuple     — shape of output image (ny, nx)

    Returns
    -------
    image_conv : 2D ndarray
        The convolved model image.
    """

    # Generate raw Sérsic image
    model = sersic_model(x, y, **kwargs)

    # Normalize PSF if not already
    psf_norm = psf / psf.sum()

    # Convolve model with PSF
    image_conv = fftconvolve(model, psf_norm, mode='same')

    return image_conv


#%% torch version of galaxy fitting models

def sersic_bn(n):
    """
    Accurate approximation of b_n used in the Sérsic profile:
    gammaincinv(2n, 0.5) ≈ series expansion.

    Parameters
    ----------
    n : torch.Tensor or float
        Sérsic index.

    Returns
    -------
    b_n : torch.Tensor or float
        Approximate solution to gammaincinv(2n, 0.5).
    """
    return 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3)


def sersic_model_torch(x, y, I_e, R_e, n, x0, y0, q, theta, **kwargs):
    """
    Evaluate a Sérsic profile in PyTorch.
    Parameters can be tensors requiring gradients.
    """
    bn = sersic_bn(n)

    dx = x - x0
    dy = y - y0
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    x_rot = cos_t*dx + sin_t*dy
    y_rot = -sin_t*dx + cos_t*dy

    q = torch.clamp(q, min=0.1, max=1.0)
    R = torch.sqrt(x_rot**2 + (y_rot / q)**2 + 1e-6)  # added epsilon
    R_e = torch.clamp(R_e, min=1e-2)
    n = torch.clamp(n, min=0.3)

    exponent = -bn * ((R / R_e).pow(1/n) - 1)
    exponent = torch.clamp(exponent, max=100)  # prevent overflow

    model = I_e * torch.exp(exponent)

    return model


def full_image_model_torch(x, y, psf, add_psf=False, x_psf=None, y_psf=None, I_psf=None, **kwargs):
    """
    Generate a full model image using centered FFT-based convolution.

    Parameters
    ----------
    x, y : 2D torch tensors
        Coordinate grid.
    psf : 2D torch tensor
        PSF kernel, assumed centered and normalized.
    kwargs : dict
        Sérsic model parameters: I_e, R_e, n, x0, y0, q, theta

    Returns
    -------
    image_conv : 2D torch tensor
        The convolved model image.
    """

    model = sersic_model_torch(x, y, **kwargs)

    # Normalize PSF
    psf = psf / psf.sum()

    image_conv = utils.fftconvolve_torch(model, psf, mode='same')

    if add_psf: 
        nx, ny = x.shape
        frame = utils.insert_shifted_psf_into_frame(psf, x0=x_psf, y0=y_psf, nx=nx, ny=ny)
        image_conv = image_conv + frame*I_psf

    return image_conv
