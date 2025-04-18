import torch
from . import galaxy, utils

#%% kinematics models in torch
def arctangent_disk_velocity_model(x, y, V_rot=200., R_v=2., x0_v=0., y0_v=0., theta_v=0., inc_v=0., **kwargs):
    """
    Compute vx, vy, vz on the sky for a rotating disk using arctangent rotation curve.

    Parameters
    ----------
    x, y : torch.Tensor
        Coordinates (float32) of shape (H, W) or (N,) in arcsec or pixels.
    V_rot : float or torch.Tensor
        Maximum rotational velocity.
    R_v : float or torch.Tensor
        Turnover radius (scale radius).
    x0_v, y0_v : float or torch.Tensor
        Dynamical center (same units as x, y).
    theta : float (radians)
        Position angle of major axis (CCW from x-axis).
    inc : float (radians)
        Inclination angle (0=face-on, pi/2=edge-on)

    Returns
    -------
    vz : torch.Tensor
        Velocity components (same shape as x).
    """
    # Shift to center
    dx = x - x0_v
    dy = y - y0_v

    # Rotate coordinates by -theta (align major axis with x')
    cos_t, sin_t = torch.cos(theta_v), torch.sin(theta_v)
    x_p = cos_t * dx + sin_t * dy
    y_p = -sin_t * dx + cos_t * dy

    # Deproject radius (inclination)
    r = torch.sqrt(x_p**2 + (y_p / torch.cos(inc_v))**2 + 1e-8)

    # Rotation curve
    V_circ = (2 / torch.pi) * V_rot * torch.atan(r / R_v)

    # Tangential velocity projected along line-of-sight
    # Only vz (los velocity) is non-zero, vx = vy = 0
    vz = V_circ * torch.sin(inc_v) * (x_p / (r + 1e-8))

    # # Optionally return full velocity field (vx, vy = 0 for circular disk)
    # vx = torch.zeros_like(vz)
    # vy = torch.zeros_like(vz)

    # return vx, vy, vz
    return vz

def dispersion_model(x, y, vz, wavelength_rest, grism_model, dx=0., dy=0., **kwargs):
    """
    Map rest-frame emission line to grism image using Doppler shift and spatial model.

    Parameters
    ----------
    x, y : (H, W) torch.Tensor
        Image coordinates.
    vz : (H, W) torch.Tensor
        Line-of-sight velocity in km/s.
    wavelength_rest : float
        Rest-frame wavelength in microns (e.g., 0.6563 for Hα).
    grism_model : function
        Function grism_model(x, y, wavelength) returning (x_G, y_G).
    wavs, dxs, dys : (M,) torch.Tensor
        Grism trace model used inside grism_model.

    Returns
    -------
    x_G, y_G : (H, W) torch.Tensor
        Projected grism coordinates accounting for Doppler shift.
    """
    x = x + dx
    y = y + dy
    c = 299792.458  # speed of light in km/s
    lambda_obs = wavelength_rest * (1.0 + vz / c)
    x_G, y_G = grism_model(x, y, lambda_obs)

    return x_G, y_G


def scatter_shifted_psf(xg, yg, intensities, psf, cutout=(0, 75, 100, 100)):
    """
    Scatter intensity-weighted, subpixel-shifted PSFs into a full image.

    Parameters
    ----------
    xg, yg : (N,) float tensors - Subpixel locations in grism image.
    intensities : (N,) float tensor - Flux values.
    psf : (P, P) tensor - Centered PSF kernel.
    image_shape : (H, W) - Shape of full image canvas.
    x0, y0 : int - Starting coordinates of cutout region.
    w, h : int or None - Width and height of cutout region. If None, use full image.

    Returns
    -------
    image : (h, w) tensor - Model image of specified cutout region.
    """
    device = xg.device
    x0, y0, w, h = cutout
    H, W = h, w
    image = torch.zeros(h, w, device=device)

    # Transform coordinates to cutout frame
    xg_cut = xg - x0
    yg_cut = yg - y0

    # Scatter intensities using bilinear interpolation
    x0 = torch.floor(xg_cut).long()
    y0 = torch.floor(yg_cut).long()
    dx = xg_cut - x0.float()
    dy = yg_cut - y0.float()

    for i in range(2):
        for j in range(2):
            w = ((1 - dx) if i == 0 else dx) * ((1 - dy) if j == 0 else dy)
            xi = x0 + i
            yj = y0 + j
            valid = (xi >= 0) & (xi < W) & (yj >= 0) & (yj < H)
            image.index_put_((yj[valid], xi[valid]), intensities[valid] * w[valid], accumulate=True)

    convolved = utils.fftconvolve_torch(image, psf, mode='same')

    return convolved


def full_grism_model_torch(x, y, psf, cutout, emline_mask, **kwargs):

    """
    Generate a full model image using centered FFT-based convolution.

    Parameters
    ----------
    x, y : 2D torch tensors
        Coordinate grid.
    psf : 2D torch tensor
        PSF kernel, assumed centered and normalized.
    kwargs : dict
        velocity model parameters: V_rot, R_v, x0_v, y0_v, theta_v, inc_v
        dispersion model parameters: wavelength_rest, grism_model
        Sérsic model parameters: I_e, R_e, n, x0, y0, q, theta
    emline_mask: 2D torch tensor or float
        Filter the galaxy flux to the emline flux
        If tensor then the shape should be the same as x, y

    Returns
    -------
    image_conv : 2D torch tensor
        The convolved model image.
    """


    vz = arctangent_disk_velocity_model(x, y, **kwargs)
    x_G, y_G = dispersion_model(x, y, vz, **kwargs)
    image_model = galaxy.sersic_model_torch(x, y, **kwargs)
    
    # plt.scatter(x_G.detach().numpy(), 
    #             y_G.detach().numpy(), 
    #             c=image_model.detach().numpy())
    # plt.show()

    # convolve with psf
    I_flat = image_model.flatten()
    xG_flat = x_G.flatten()
    yG_flat = y_G.flatten()
    # Build PSF model image
    model_img = scatter_shifted_psf(xG_flat, yG_flat, I_flat, psf, cutout)
    model_img = model_img*emline_mask

    return model_img

def full_grism_model_nonparametric_torch(image_model, psf, cutout, fratio, **kwargs):

    """
    Generate a full model image using centered FFT-based convolution.

    Parameters TODO: change it
    ----------
    x, y : 2D torch tensors
        Coordinate grid.
    psf : 2D torch tensor
        PSF kernel, assumed centered and normalized.
    kwargs : dict
        velocity model parameters: V_rot, R_v, x0_v, y0_v, theta_v, inc_v
        dispersion model parameters: wavelength_rest, grism_model
        Sérsic model parameters: I_e, R_e, n, x0, y0, q, theta

    Returns
    -------
    image_conv : 2D torch tensor
        The convolved model image.
    """
    device = image_model.device
    nx, ny = image_model.shape
    y, x = torch.meshgrid(
        torch.linspace(0, nx - 1, nx, device=device),
        torch.linspace(0, ny - 1, ny, device=device),
        indexing='ij'
    )

    vz = arctangent_disk_velocity_model(x, y, **kwargs)
    x_G, y_G = dispersion_model(x, y, vz, **kwargs)
    
    # plt.scatter(x_G.detach().numpy(), 
    #             y_G.detach().numpy(), 
    #             c=image_model.detach().numpy())
    # plt.show()

    # convolve with psf
    I_flat = image_model.flatten()
    xG_flat = x_G.flatten()
    yG_flat = y_G.flatten()
    # Build PSF model image
    model_img = scatter_shifted_psf(xG_flat, yG_flat, I_flat, psf, cutout)
    model_img = model_img*fratio

    return model_img