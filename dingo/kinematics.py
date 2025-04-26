import torch
from . import galaxy, utils

import matplotlib.pyplot as plt

import logging
LOG = logging.getLogger(__name__)  # This will inherit the root logger's config if set
LOG.addHandler(logging.NullHandler())  # Prevents "No handler found" warning if root isn't configured


#%% kinematics models in torch
def arctangent_disk_velocity_model(x, y, V_rot=200., R_v=2., x0_v=0., y0_v=0., theta_v=0., inc_v=0., **kwargs):
    """
    TODO: change the name to torch
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

def bilinear_interpolte_intensity_torch(xg, yg, intensities, cutout): 
            
        device = intensities.device
        x0, y0, w, h = cutout
        # print(x0, y0, w, h )
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

        return image

def scatter_shifted_psf(xg, yg, intensities, psf, cutout):
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

    image = bilinear_interpolte_intensity_torch(xg, yg, intensities, cutout)

    convolved = utils.fftconvolve_torch(image, psf, mode='same')

    return convolved

    # return image

#%% full grism models

def full_grism_model_torch(x, y, psf, cutout, emline_mask=1., **kwargs):

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

def full_grism_model_nonparametric_torch(image_model, psf, cutout, emline_mask=1., **kwargs):

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
    model_img = model_img*emline_mask

    return model_img

# model ok; idea does not work

def forward_dispersion_model(x_G, y_G, lambda_obs, forward_model, dx=0., dy=0., **kwargs):

    x_G = x_G + dx
    y_G = y_G + dy
    x, y = forward_model(x_G, y_G, lambda_obs)

    return x, y

def full_forward_grism_model_torch(grism_image, lambda_test, cutout, oversample=2, 
                                       **kwargs):
    # TODO: oversample
    nx_G, ny_G = grism_image.shape
    y_G, x_G = torch.meshgrid(torch.arange(nx_G), torch.arange(ny_G))
    x, y = forward_dispersion_model(x_G, y_G, lambda_test, **kwargs)

    # print('cutout', cutout)
    # plt.scatter(x.detach().numpy(), 
    #             y.detach().numpy(), 
    #             c=grism_image.detach().numpy())
    # plt.show()


    image = bilinear_interpolte_intensity_torch(x, y, grism_image, cutout)

    dx = cutout[0] #+ cutout[2]//2
    dy = cutout[1] #+ cutout[3]//2
    vz = arctangent_disk_velocity_model(x-dx, y-dy, **kwargs)


    return image, vz

# finally worked version, iteratively find xy

def iteratively_find_xy(x_init, y_init, cutout, lambda_rest, x_G, y_G, 
                        maxiter=50, tol=1e-3, alpha=0.5, **kwargs):
    c = 299792.458  # speed of light in km/s

    # find x and y
    dx = cutout[0] #+ cutout[2]//2
    dy = cutout[1] #+ cutout[3]//2
    x = x_init.detach()
    y = y_init.detach()
    for k in range(maxiter):
        vz_new = arctangent_disk_velocity_model(x-dx, y-dy, **kwargs)
        lambda_new = lambda_rest * (1.0 + vz_new / c)
        x_new, y_new = forward_dispersion_model(x_G, y_G, lambda_new, **kwargs)
        # print(torch.max(torch.abs(x_new - x)), torch.max(torch.abs(y_new - y)))
        if torch.max(torch.abs(x_new - x)) < tol and \
           torch.max(torch.abs(y_new - y)) < tol:
            x = alpha*x + (1 - alpha)*x_new
            y = alpha*y + (1 - alpha)*y_new
            break
        x = alpha*x + (1 - alpha)*x_new
        y = alpha*y + (1 - alpha)*y_new
    if k+1>=maxiter: LOG.warning('maxiter reached but xy do not converge to 0')

    return x, y, vz_new, k 