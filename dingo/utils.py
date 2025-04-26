import torch
import torch.nn.functional as F
import numpy as np
from astropy.modeling.models import Gaussian2D
from scipy.special import gammaincinv
from scipy.signal import fftconvolve
import torchinterp1d

from . import grism

#%% utility functions in numpy

def create_gaussian_psf(fwhm, size=25, ratio=1.0, theta=0.0):
    """
    Create a 2D Gaussian PSF using Astropy.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum (FWHM) of the PSF (major axis).
    size : int
        Size of the PSF array (size x size).
    ratio : float
        Axis ratio b/a (1 = circular, <1 = elliptical).
    theta : float
        Position angle in radians (CCW from x-axis).

    Returns
    -------
    psf : 2D ndarray
        Normalized PSF array of shape (size, size).
    """
    sigma_major = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_minor = sigma_major * ratio

    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(x, y)

    g2d = Gaussian2D(
        amplitude=1.0,
        x_mean=0, y_mean=0,
        x_stddev=sigma_major,
        y_stddev=sigma_minor,
        theta=theta
    )

    psf = g2d(xx, yy)
    psf /= psf.sum()  # Normalize

    return psf


def coadd_images(images, weights=None):
    """
    Coadd a stack of images with NaNs, dropping weights at NaN pixels.

    Parameters
    ----------
    images : (N, H, W) ndarray
        Stack of N images with shape H x W. May contain NaNs.
    weights : (N,) or (N, H, W) ndarray, optional
        Weights for each image. If None, use uniform weights of 1.

    Returns
    -------
    coadded : (H, W) ndarray
        Weighted coadded image, skipping NaN pixels.
    """

    images = np.asarray(images)
    mask = np.isnan(images)
    valid = ~mask

    weights = np.ones(images.shape, dtype=np.float64)

    # Set weights of NaNs to 0
    weights = weights * valid

    # Zero out NaNs in the images to avoid propagation
    images_filled = np.where(valid, images, 0)

    # Weighted sum and normalize
    weighted_sum = np.sum(images_filled * weights, axis=0)
    weight_sum = np.sum(weights, axis=0)

    # Avoid division by 0
    with np.errstate(invalid='ignore', divide='ignore'):
        coadded = weighted_sum / weight_sum
        coadded[weight_sum == 0] = np.nan

    return coadded




#%% utility functions in torch

def fftconvolve_torch(in1, in2, mode='same'):
    """
    N-dimensional FFT-based convolution using PyTorch.
    
    Parameters
    ----------
    in1 : torch.Tensor
        First input array.
    in2 : torch.Tensor
        Second input array. Must be same ndim as in1.
    mode : str {'full', 'same', 'valid'}
        Output size mode:
        - 'full': full convolution (default)
        - 'same': output same shape as in1
        - 'valid': only regions where in2 fully overlaps in1

    Returns
    -------
    out : torch.Tensor
        Convolution result.
    """
    if in1.ndim != in2.ndim:
        raise ValueError("Inputs must have same number of dimensions")

    # Compute padded shape
    s1 = torch.tensor(in1.shape)
    s2 = torch.tensor(in2.shape)
    shape = s1 + s2 - 1

    # FFTs
    fsize = [int(n) for n in shape]
    fft1 = torch.fft.fftn(in1, s=fsize)
    fft2 = torch.fft.fftn(in2, s=fsize)
    out_fft = fft1 * fft2
    out = torch.fft.ifftn(out_fft).real

    if mode == 'full':
        return out

    elif mode == 'same':
        # Crop to same shape as in1
        start = (shape - s1) // 2
        slices = tuple(slice(int(start[d]), int(start[d] + s1[d])) for d in range(in1.ndim))
        return out[slices]

    elif mode == 'valid':
        valid_shape = s1 - s2 + 1
        if torch.any(valid_shape < 1):
            raise ValueError("in2 must not be larger than in1 in any dimension for 'valid' mode")
        start = (s2 - 1)
        end = start + valid_shape
        slices = tuple(slice(int(start[d]), int(end[d])) for d in range(in1.ndim))
        return out[slices]

    else:
        raise ValueError("mode must be 'full', 'same', or 'valid'")

def get_grism_model_torch(this_spatial_model, this_disp_model, this_pupil, pixelx, pixely, direction='backward'):
    """
    Fully differentiable version of find_pixel_location_torch using torchinterp1d.

    Parameters
    ----------
    WRANGE : tuple
        Wavelength range (min, max).
    this_spatial_model : callable
        Spatial model for the trace.
    this_disp_model : callable
        Dispersion model for the wavelength.
    this_pupil : str
        Pupil label for the grism.
    pixelx, pixely : float
        Central pixel coordinates.
    line_wavelengths : torch.Tensor
        Wavelengths to compute positions for (1D tensor).
    grism : object
        Should have a method `grism_conf_preparation(...)` returning (dxs, dys, wavs)

    Returns
    -------
    xline_on_G_img, yline_on_G_img : torch.Tensor
        Pixel positions of the given wavelengths on the grism image.
    """

    # Get trace and wavelength model output from grism config
    dxs_np, dys_np, wavs_np = grism.grism_conf_preparation(
        x0=pixelx, y0=pixely,
        pupil=this_pupil,
        fit_opt_fit=this_spatial_model,
        w_opt=this_disp_model,
    )
    # print(dxs_np)
    # print(dys_np)

    # Convert to torch tensors
    dxs = torch.tensor(dxs_np, dtype=torch.float32, requires_grad=False)
    dys = torch.tensor(dys_np, dtype=torch.float32, requires_grad=False)
    wavs = torch.tensor(wavs_np, dtype=torch.float32, requires_grad=False)

    # Sort by wavelength
    order = torch.argsort(wavs)
    wavs = wavs[order]
    dxs = dxs[order]
    dys = dys[order]
    # print('Min Δλ in wavs:', (wavs[1:] - wavs[:-1]).min().item())

    # Use torch-native linear interpolation
    def interp1d(x, xp, fp):
        # x: target values
        idx = torch.searchsorted(xp, x, right=True).clamp(1, len(xp)-1)
        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]
        weight = (x - x0) / (x1 - x0 + 1e-8)
        return y0 + weight * (y1 - y0)

    # dxs_dense = interp1d(wavs_dense, wavs, dxs)
    # dys_dense = interp1d(wavs_dense, wavs, dys)

    # Interpolators (differentiable)

    if direction.lower()=='backward': 

        def grism_model(x, y, line_wavelengths):
            """
            Compute grism image coordinates (x_G, y_G) from source position (x, y)
            and wavelength image using interpolated trace model.

            Parameters
            ----------
            x, y : (H, W) torch.Tensor
                Source positions in detector space.
            line_wavelengths : (H, W) torch.Tensor
                Wavelength image at each pixel.
            wavs : (M,) torch.Tensor
                Wavelength grid of the trace model.
            dxs, dys : (M,) torch.Tensor
                Corresponding dx, dy offsets for each wav.

            Returns
            -------
            x_G, y_G : (H, W) torch.Tensor
                Grism image coordinates.
            """
            # Flatten input for vectorized interpolation
            x_flat = x.flatten()
            y_flat = y.flatten()
            wl_flat = line_wavelengths.flatten()

            # Interpolate dx and dy at each wavelength
            # dx_interp = torchinterp1d.interp1d(wavs, dxs, wl_flat)
            # dy_interp = torchinterp1d.interp1d(wavs, dys, wl_flat)
            dx_interp = interp1d(wl_flat, wavs, dxs)
            dy_interp = interp1d(wl_flat, wavs, dys)

            # Add offset to source position
            x_G = x_flat + dx_interp
            y_G = y_flat + dy_interp

            # Reshape back to image
            x_G = x_G.view_as(x)
            y_G = y_G.view_as(y)

            return x_G, y_G

        return grism_model 

    elif direction.lower() == 'forward':

        def grism_inverse_model(x_G, y_G, line_wavelengths):
            """
            Compute source plane coordinates (x, y) from grism image coordinates (x_G, y_G)
            and wavelength using interpolated trace model.

            Parameters
            ----------
            x_G, y_G : (H, W) torch.Tensor
                Grism image coordinates.
            line_wavelengths : (H, W) torch.Tensor
                Wavelength image at each pixel.

            Returns
            -------
            x, y : (H, W) torch.Tensor
                Estimated source coordinates.
            """
            xG_flat = x_G.flatten()
            yG_flat = y_G.flatten()
            wl_flat = line_wavelengths.flatten()

            # Interpolate dx and dy at each wavelength
            dx_interp = interp1d(wl_flat, wavs, dxs)
            dy_interp = interp1d(wl_flat, wavs, dys)

            # Invert the transformation
            x_flat = xG_flat - dx_interp
            y_flat = yG_flat - dy_interp

            # Reshape to image
            x = x_flat.view_as(x_G)
            y = y_flat.view_as(y_G)

            return x, y

        return grism_inverse_model
    
    else: 
        raise ValueError('unknown transform direction')

def bilinear_shift_psf_torch(psf, dx, dy):
    """
    Bilinearly shift a PSF kernel by (dx, dy) within [-0.5, 0.5].
    Uses torch's grid_sample.
    """
    P = psf.shape[-1]
    device = psf.device

    # Normalized grid for grid_sample: [-1, 1]
    coords = torch.linspace(-1, 1, P, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')

    # Shift the grid by normalized dx, dy
    shift_x = 2 * dx / (P - 1)
    shift_y = 2 * dy / (P - 1)

    grid = torch.stack((xx - shift_x, yy - shift_y), dim=-1).to(torch.float64)[None, ...]  # (1, P, P, 2)

    psf = psf[None, None, ...]  # (1, 1, P, P)
    # print(psf.dtype, grid.dtype)
    shifted = F.grid_sample(psf, grid, align_corners=True, padding_mode='zeros')[0, 0]
    return shifted


def insert_shifted_psf_into_frame(psf, x0, y0, nx, ny):
    """
    Create an (ny, nx) image with the PSF centered at subpixel location (x0, y0).

    Parameters
    ----------
    psf : (P, P) torch.Tensor
        PSF kernel (assumed square and centered at (P//2, P//2)).
    x0, y0 : float
        Subpixel center location in the output image where the PSF peak should be placed.
    nx, ny : int
        Output image shape (width, height).

    Returns
    -------
    frame : (ny, nx) torch.Tensor
        Output image with the shifted PSF inserted.
    """
    P = psf.shape[-1]
    device = psf.device

    # Compute integer base location (top-left corner) for PSF insertion
    x_base = int(torch.floor(x0)) - P // 2
    y_base = int(torch.floor(y0)) - P // 2

    # Subpixel offset from center of PSF
    dx = x0 - torch.floor(x0)
    dy = y0 - torch.floor(y0)

    # Normalize shifts to [-1, 1] for grid_sample
    shift_x = 2 * dx / (P - 1)
    shift_y = 2 * dy / (P - 1)

    # Create normalized sampling grid
    coords = torch.linspace(-1, 1, P, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    grid = torch.stack((xx - shift_x, yy - shift_y), dim=-1).to(torch.float32)[None, ...]  # (1, P, P, 2)

    # Apply bilinear shift via grid_sample
    psf = psf[None, None, ...]  # (1, 1, P, P)
    shifted_psf = F.grid_sample(psf, grid, align_corners=True, padding_mode='zeros')[0, 0]  # (P, P)

    # Create blank frame
    frame = torch.zeros((ny, nx), dtype=psf.dtype, device=device)

    # Insert PSF into frame (clipped to boundaries if needed)
    x_start = max(x_base, 0)
    y_start = max(y_base, 0)
    x_end = min(x_base + P, nx)
    y_end = min(y_base + P, ny)

    psf_x_start = max(0, -x_base)
    psf_y_start = max(0, -y_base)
    psf_x_end = psf_x_start + (x_end - x_start)
    psf_y_end = psf_y_start + (y_end - y_start)

    frame[y_start:y_end, x_start:x_end] = shifted_psf[psf_y_start:psf_y_end, psf_x_start:psf_x_end]

    return frame
