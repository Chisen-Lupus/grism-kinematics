import torch
import torch.nn.functional as F
import numpy as np
from astropy.modeling.models import Gaussian2D
from scipy.special import gammaincinv
from scipy.signal import fftconvolve
import logging

from . import grism

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

def downsample_with_shift_and_size(
    x: torch.Tensor,
    factor: int,
    out_size: tuple[int,int],
    shift: tuple[float,float],
    mode: str = 'bicubic'
) -> torch.Tensor:
    """
    Downsample a 2D image by integer `factor` with exact block-averaging,
    into an array of shape out_size=(H_out, W_out), applying a relative
    shift in *output*‐pixel units.

    Args:
        x        (H, W)          : single‐channel input image
        factor   int>1           : downsampling factor f
        out_size (H_out, W_out)  : desired output shape
        shift    (sy_out, sx_out): shift *in output pixels* between
                                  input‐ and output‐centers

    Returns:
        y       (H_out, W_out)   : downsampled & averaged result
    """
    H, W = x.shape
    f = factor
    H_out, W_out = out_size
    sy_out, sx_out = shift
    device, dtype = x.device, x.dtype

    # 1) build a grouped‐conv kernel of ones to integrate each f×f block
    x_in = x.unsqueeze(0).unsqueeze(0)  # → (1,1,H,W)
    kernel = torch.ones(1, 1, f, f, device=device, dtype=dtype)
    conv = F.conv2d(x_in, kernel, stride=1, padding=0)
    # conv has shape (1,1,H-f+1, W-f+1)
    Hc, Wc = conv.shape[2], conv.shape[3]

    # 2) figure out the *starting‐corner* of the block that sits
    #    at the *center* of the output; then march in steps of f.
    #    A block starting at (i,j) covers input[i:i+f, j:j+f] and
    #    its "center" is at i + (f-1)/2.  To have that sit at input
    #    center (at (H-1)/2), you need i = (H - f)/2.
    center_i = (H - f) / 2.0
    center_j = (W - f) / 2.0

    # convert the *output‐pixel* shift into input‐pixel units:
    sy_in = sy_out * f
    sx_in = sx_out * f

    # we want H_out blocks, centered, stepping by f:
    idx_i = torch.arange(H_out, device=device, dtype=dtype) - (H_out - 1)/2.0
    idx_j = torch.arange(W_out, device=device, dtype=dtype) - (W_out - 1)/2.0

    # absolute block‐start positions in *conv*‐coords:
    ys = center_i - sy_in + f * idx_i    # shape (H_out,)
    xs = center_j - sx_in + f * idx_j    # shape (W_out,)

    # 3) normalize to [-1,1] over the conv map
    ys_norm = ys / (Hc - 1) * 2 - 1       # still shape (H_out,)
    xs_norm = xs / (Wc - 1) * 2 - 1       # shape (W_out,)

    # 4) build a (1, H_out, W_out, 2) sampling grid
    grid_y = ys_norm.unsqueeze(1).expand(H_out, W_out)
    grid_x = xs_norm.unsqueeze(0).expand(H_out, W_out)
    grid   = torch.stack((grid_x, grid_y), dim=-1)  # (H_out, W_out, 2)
    grid   = grid.unsqueeze(0)                      # (1, H_out, W_out, 2)

    # 5) bilinearly sample the *integrated* map at those fractional starts
    sampled = F.grid_sample(
        conv, grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=True
    )
    # → shape (1,1,H_out,W_out)

    # 6) divide by f^2 to turn sums into averages
    y = sampled / (f*f)   # still (1,1,H_out,W_out)

    return y[0,0]


def fft_bin(ps, n):
    """
    Perform binning in Fourier space by summing over all nxn frequency blocks,
    simulating alias folding from real-space binning.

    Assumes ps is fftshifted and shape is (n*k, n*k).

    Parameters
    ----------
    ps : torch.Tensor
        2D fftshifted complex FFT of shape (N, N), where N = n * k.
    n : int
        Binning factor.

    Returns
    -------
    c_hat : torch.Tensor
        Folded (binned) Fourier image of shape (N//n, N//n),
        still in fftshifted form.
    """
    N = ps.shape[0]
    assert ps.shape[0] == ps.shape[1], "Input must be square"
    assert N % n == 0, "Size must be divisible by binning factor"

    k = N // n
    c_hat = torch.zeros((k, k), dtype=ps.dtype, device=ps.device)
    for i in range(n):
        for j in range(n):
            c_hat += ps[k*i:k*(i+1), k*j:k*(j+1)]

    return c_hat/n**2

def fft_phase_shift(F, dx, dy):
    """
    Shift a 2D image by (dx, dy) using FFT phase shift.

    Parameters
    ----------
    F : torch.Tensor
        2D tensor (H, W), assumed float32 or float64.
    dx, dy : float
        Subpixel shifts in x (columns) and y (rows).

    Returns
    -------
    torch.Tensor
        Shifted image of same shape.
    """
    H, W = F.shape
    device = F.device
    dtype = F.dtype

    # Frequencies in cycles per pixel (normalized)
    fx = torch.fft.fftfreq(W, d=1.0).to(device=device, dtype=dtype)
    fy = torch.fft.fftfreq(H, d=1.0).to(device=device, dtype=dtype)

    # Meshgrid of frequencies
    v, u = torch.meshgrid(fy, fx, indexing='ij')  # (H, W)

    # Apply phase shift
    phase_shift = torch.exp(-2j * torch.pi * (u * dx + v * dy))
    F_shifted = F * phase_shift

    return F_shifted
