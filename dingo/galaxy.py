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


def full_sersic_model_torch(x, y, psf, **kwargs):
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

    return image_conv

# def full_psf_model_torch(x, y, psf, x_psf, y_psf, I_psf):
#     """
#     Generate a PSF-only model image on the same grid as (x, y).

#     Parameters
#     ----------
#     x, y : 2D torch tensors
#         Coordinate grid; only shape is used.
#     psf : 2D torch tensor
#         PSF kernel.
#     x_psf, y_psf : float / tensor
#         PSF center position in output image coordinates.
#     I_psf : float / tensor
#         Total PSF flux (scaling).

#     Returns
#     -------
#     image_conv : 2D torch tensor
#         PSF image on the given grid.
#     """
#     nx, ny = x.shape  # 或 y.shape

#     frame = utils.insert_shifted_psf_into_frame(
#         psf, x0=x_psf, y0=y_psf, nx=nx, ny=ny
#     )
#     image_conv = frame*I_psf

#     return image_conv


def full_psf_model_torch(x, y, psf, x_psf, y_psf, I_psf):
    """
    使用 FFT phase shift 在整帧上生成一个带 sub-pixel 位置的 PSF 模型。

    Parameters
    ----------
    x, y : (H, W) torch.Tensor
        坐标网格（这里只用 shape，不用具体数值）。
    psf : (P, P) torch.Tensor
        PSF kernel（居中）。
    x_psf, y_psf : float or 0D tensor
        PSF 在输出图像坐标中的中心位置（列、行）。
    I_psf : float or 0D tensor
        PSF 总 flux（缩放因子）。

    Returns
    -------
    image : (H, W) torch.Tensor
        在 (x_psf, y_psf) 处的 PSF 图像。
    """
    H, W = x.shape
    device = x.device
    dtype  = x.dtype

    # ---- 1) 把 psf 放在帧中心 ----
    frame = torch.zeros((H, W), dtype=dtype, device=device)

    P = psf.shape[-1]
    # 帧中心坐标（浮点）
    cx = (W - 1)/2.0
    cy = (H - 1)/2.0

    # 以中心为参考，把 PSF 的整数部分放进去
    x_base = int(cx) - P//2
    y_base = int(cy) - P//2

    x_start = max(x_base, 0)
    y_start = max(y_base, 0)
    x_end   = min(x_base + P, W)
    y_end   = min(y_base + P, H)

    psf_x_start = max(0, -x_base)
    psf_y_start = max(0, -y_base)
    psf_x_end   = psf_x_start + (x_end - x_start)
    psf_y_end   = psf_y_start + (y_end - y_start)

    frame[y_start:y_end, x_start:x_end] = psf[psf_y_start:psf_y_end, psf_x_start:psf_x_end].to(dtype)

    # ---- 2) 在频域做平移：中心 -> (x_psf, y_psf) ----
    # shifts: 输出坐标系里, 列方向 dx, 行方向 dy
    x0 = torch.as_tensor(x_psf, device=device, dtype=torch.float64)
    y0 = torch.as_tensor(y_psf, device=device, dtype=torch.float64)
    dx = x0 - cx
    dy = y0 - cy

    F0 = torch.fft.fft2(frame)                 # complex
    F_shifted = utils.fft_phase_shift(F0, dx, dy)
    image = torch.fft.ifft2(F_shifted).real.to(dtype)

    # ---- 3) 乘上 I_psf ----
    I0 = torch.as_tensor(I_psf, device=device, dtype=dtype)
    image = image*I0

    return image
