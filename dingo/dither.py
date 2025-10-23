arr = []

import os, sys
import torch 
import numpy as np
import torch
import copy
import scipy
from typing import Callable, Any, Tuple, List, Optional
from numpy.typing import NDArray

import matplotlib.pyplot as plt

def combine_image(
    normalized_atlas: List[NDArray[torch.float64]], 
    centroids: List[Tuple[float, float]], 
    wts: Optional[List[float]] = None, 
    oversample: int = 2, 
    device: torch.device = 'cpu',
    # for tesing purpose:
    return_full_array: bool = False, 
    overpadding: int = 0
) -> NDArray[torch.float64]:
    """
    Apply phase shifts to the itorchut data.

    Parameters
    ----------
    normalized_atlas : list-like container of arrays
        Itorchut 2D array TODO
    centroids : list 
        TODO
    wts : int
        TODO
    oversample : int
        TODO

    Returns
    -------
    ndarray
        TODO
    """

    # REGULARIZE INPUT

    if wts is None:
        wts = torch.ones(len(normalized_atlas), dtype=torch.complex64, device=device)

    # ASSERTATION

    assert len(normalized_atlas)==len(centroids)
    assert len(centroids)==len(wts)
    assert len(set([im.shape for im in normalized_atlas]))==1

    # TEMPORARY SOLUTION WHEN 2X DITHER IS PENDING REFRACTOR
    
    if oversample==2: 
        combined_image = combine_image_test(
            normalized_atlas, centroids, wts, oversample, device, 
            return_full_array, overpadding
        )

    elif oversample%2==1:
        combined_image = combine_image_3x(
            normalized_atlas, centroids, wts, oversample, device
        )

    else:
        raise NotImplementedError()


    return combined_image


def combine_image_2x(
    normalized_atlas: List[NDArray[torch.float64]], 
    centroids: List[Tuple[float, float]], 
    wts: Optional[List[float]] = None, 
    oversample: int = 2, 
    device: torch.device = 'cpu',
    # for tesing purpose:
    return_full_array: bool = False, 
    overpadding: int = 0
) -> NDArray[torch.float64]:
    '''
    Accelerated method from Lauer (99), does not handle dither more than 2x
    '''

    # SOME GLOBAL FACTORS

    NSUB = oversample
    NPP = len(normalized_atlas)
    NX, NY = normalized_atlas[0].shape
    NX_LARGE = NX*NSUB
    NY_LARGE = NY*NSUB
    NC_FREQ = int(oversample**np.ceil(np.emath.logn(oversample, NX_LARGE))) # find the next 2^N
    NR_FREQ = int(oversample**np.ceil(np.emath.logn(oversample, NY_LARGE))) # 4x of NX_LARGE can 
    # NC_FREQ = NX_LARGE
    # NR_FREQ = NY_LARGE
    # print(NC_FREQ)
    NC_FREQ *= 2**overpadding
    NR_FREQ *= 2**overpadding


    A_total = torch.zeros((NC_FREQ//2+1, NR_FREQ), dtype=torch.complex64, device=device)

    for npos in range(len(normalized_atlas)): 

        data = normalized_atlas[npos]
        data_large = torch.zeros((NC_FREQ, NR_FREQ), device=device)
        data_large[:NX*NSUB:NSUB, :NY*NSUB:NSUB] = data
        coef = torch.zeros((NSUB, NSUB), dtype=torch.complex64, device=device)

        dx = centroids[:, 1]
        dy = centroids[:, 0]
        phix = NSUB*torch.pi*dx
        phiy = NSUB*torch.pi*dy

        # BEGIN COEFFICIENT COMPUTATION

        # NOTE: Only half of the coefficients calculated here are used for now.
        for iy in range(NSUB): 
            for ix in range(NSUB): 

                # Precompute normalized phase shifts
                px = -2 * phix / NSUB
                py = -2 * phiy / NSUB

                # Compute base indices and initial phases
                nuin = ix - (NSUB - 1) // 2
                nvin = iy
                pxi = nuin * px
                pyi = -nvin * py

                # Generate sub-grid indices
                isatx, isaty = torch.meshgrid(
                    torch.arange(NSUB, device=device), 
                    torch.arange(NSUB, device=device), 
                    indexing='xy'
                )
                isatx = isatx.flatten()
                isaty = isaty.flatten()

                # Calculate total phase using broadcasting
                phit = torch.outer(isatx, px) + pxi + torch.outer(isaty, py) + pyi

                # Compute complex phases and normalize
                phases = (torch.cos(phit) + 1j * torch.sin(phit)) / NSUB**2

                # if ix==iy==npos==0: 
                #     print(ix, iy, npos)
                #     print(phases.numpy())

                # Pivot the fundamental component to the first row
                nfund = NSUB * nvin - nuin
                phases[[0, nfund], :] = phases[[nfund, 0], :]

                # Add weighting factor
                if NPP==NSUB**2:
                    phasem = phases
                else: 
                    phasem = phases @ torch.diag(wts) @ torch.conj(phases).T

                vec = torch.linalg.pinv(phasem)
                # print(np.angle(phasem))

                # For NSUB2 images, we are done
                if NPP==NSUB**2:
                    coef[iy, ix] = vec[npos, 0]
                # Otherwise, we need to do a little more work. Here we just solve for the fundamental image.
                else: 
                    coef[iy, ix] = 0
                    for i in range(NSUB**2):
                        coef[iy, ix] += vec[i, 0]*torch.conj(phases[i, npos])

                    # XXX: Moving it to the else branch means totally ignore wts for NSUB**2 images
                    # Add weighting factor
                    coef[iy, ix] *= wts[npos]

                # print(f'Image {npos}, power {coef[isec]*torch.conj(coef[isec])}, sector {isec}')
                
    
        # global arr
        # arr.append(coef.numpy())
        # print(coef.numpy())

        # print('---')

        # END COEFFICIENT COMPUTATION

        # BEGIN FFT2

        # We only need half of the transformed array since we are doing real transform
        A_hat = torch.conj(torch.fft.rfft2(data_large, dim=(1, 0)))

        # END FFT2

        # BEGIN PHASE SHIFT APPLICATION

        for iy in range(NSUB):
            for ix in range(NSUB):

                # process columns

                # Starting and ending points of this sector
                nu = NC_FREQ//NSUB
                isu = min(nu*ix, NC_FREQ//2+1)
                ieu = min(nu*(ix+1), NC_FREQ//2+1)
                if isu==ieu: 
                    continue

                # Compute the normalized column positions (U)
                cols = torch.arange(isu, ieu, device=device)
                U = cols / NC_FREQ  # Multiply back by 2 to match original scale

                # Compute the column phase shift (as a complex exponential)
                cphase = torch.exp(-2j * phix[npos] * U)

                # process rows

                nv = NR_FREQ//NSUB 
                isv = NR_FREQ//2 - nv*iy
                iev = NR_FREQ//2 - nv*(iy+1) if iy<NSUB-1 else NR_FREQ//2 - NR_FREQ
                # print(isv-NR_FREQ//2, iev-NR_FREQ//2)

                # Extract the complex coefficient
                coef_complex = coef[iy, ix]

                # Compute the normalized row positions (V)
                rows = torch.arange(isv-1, iev-1, -1, device=device)
                V = torch.where(rows >= NR_FREQ // 2, (rows - NR_FREQ) / NR_FREQ, rows / NR_FREQ)
                # Compute the row phase shift (as a complex exponential)
                rphase = torch.exp(-2j * phiy[npos] * V)

                # apply shift

                # print(coef_complex)
                # Compute the overall phase shift (outer product for broadcasting)
                phase_shift = coef_complex * torch.outer(cphase, rphase)
                # print(coef_complex)
                # print(np.angle(rphase))
                # print(np.angle(cphase)) 

                cols_idx = cols % NC_FREQ
                rows_idx = rows % NR_FREQ
                cols_idx = cols_idx.unsqueeze(1)
                rows_idx = rows_idx.unsqueeze(0)
                A_hat[cols_idx, rows_idx] *= phase_shift

        A_total += A_hat

    # END PHASE SHIFT APPLICATION

    # BEGIN IFFT2
    
    data_rec = torch.fft.irfft2(torch.conj(A_total), s=(NC_FREQ, NR_FREQ), dim=(1, 0))
    data_real = data_rec.real

    # END IFFT2

    combined_image = data_real[:NX_LARGE, :NY_LARGE]

    if return_full_array:
        return data_real
    else:
        return combined_image


import torch.nn.functional as F


def pad_to_square(image_array, target_size):
    """
    Pads each 2D image in a 3D array to a target square size.

    Parameters
    ----------
    image_array : np.ndarray
        Input array of shape (n, h, w), where h and w are the current image dimensions.
    target_size : int
        Desired edge length of the output square images (target_size x target_size).

    Returns
    -------
    np.ndarray
        Padded array of shape (n, target_size, target_size).
    """
    if image_array.ndim != 3:
        raise ValueError('Input array must be 3D (n, h, w)')
    
    n, h, w = image_array.shape

    if target_size < max(h, w):
        raise ValueError('Target size must be >= current image dimensions')

    pad_top = (target_size - h)
    pad_left = (target_size - w)



    padded = F.pad(
        image_array,
        pad=(0, pad_left, 0, pad_top),  # pad=(left, right, top, bottom)
        mode='constant',
        value=0  # for constant padding
    )
    return padded

def combine_image_test(
    normalized_atlas: List[NDArray[torch.float64]], 
    centroids: List[Tuple[float, float]], 
    wts: Optional[List[float]] = None, 
    oversample: int = 2, 
    device: torch.device = 'cpu',
    # for tesing purpose:
    return_full_array: bool = False, 
    overpadding: int = 0
) -> NDArray[torch.float64]:
    """
    Apply phase shifts to the itorchut data.

    Parameters
    ----------
    normalized_atlas : list-like container of arrays
        Itorchut 2D array TODO
    centroids : list 
        TODO
    wts : int
        TODO
    oversample : int
        TODO

    Returns
    -------
    ndarray
        TODO
    """

    # REGULARIZE INPUT

    if wts is None:
        wts = torch.ones(len(normalized_atlas), dtype=torch.complex64, device=device)

    # ASSERTATION

    assert len(normalized_atlas)==len(centroids)
    assert len(centroids)==len(wts)
    assert len(set([im.shape for im in normalized_atlas]))==1

    # CALCULATE PARAMETERS
    
    NSUB = oversample

    N = centroids.shape[0]
    J = NSUB * NSUB

    # Create j indices
    j_indices = torch.arange(J)
    jx = j_indices % NSUB
    jy = j_indices // NSUB

    # Compute dot product using broadcasting
    dxs = centroids[:, 0]
    dys = centroids[:, 1]
    phase = torch.outer(dxs, jx) + torch.outer(dys, jy)
    # Apply exponential
    Phi = torch.exp(-1j * torch.pi * phase * 2).T/NSUB**2
    # Phi = Phi @ torch.diag(wts) @ torch.conj(Phi).T
    Phi *= wts

    Phi_inv = torch.linalg.pinv(Phi) # N x NSUB^2

    # COMBINE IMAGE
    
    nx = normalized_atlas[0].shape[0]
    if return_full_array:
        nx = int(2**(np.ceil(np.emath.logn(2, nx))+overpadding))
    # print(nx)
    normalized_atlas_pad = pad_to_square(normalized_atlas, nx)


    coadd_hat = torch.zeros((nx*NSUB, nx*NSUB), dtype=torch.complex128)
    # coadd_hat = 0

    for idx in range(len(normalized_atlas_pad)):

        dy, dx = torch.tensor(centroids[idx])
        im = normalized_atlas_pad[idx]
        coef = Phi_inv[idx]
        wt = wts[idx]
        # print(coef)
        
        im_hat = torch.fft.fft2(im)
        im_hat_large = torch.tile(im_hat, (NSUB, NSUB))
        
        # print(torch.linalg.norm(im_hat_large - torch.tile(im_hat, (NSUB, NSUB))))
        fx = torch.fft.fftfreq(nx*NSUB, d=1.0)
        fy = torch.fft.fftfreq(nx*NSUB, d=1.0)

        # Meshgrid of frequencies
        v, u = torch.meshgrid(fy, fx, indexing='ij')  # (H, W)

        # Apply phase shift
        phase_shift = torch.exp(-2j * torch.pi * (-u * dy - v * dx) * NSUB)
        
        rgrid = torch.linspace(-1, 0, steps=nx+1)[:-1]  # remove last point
        rphase1 = torch.exp(1j * dy * rgrid * 2 * torch.pi)
        cgrid = torch.linspace(-1, 0, steps=nx+1)[:-1]
        cphase1 = torch.exp(1j * dx * cgrid * 2 * torch.pi)
        
        im_hat_large *= phase_shift 
        
        for i in range(NSUB):
            for j in range(NSUB):
                    index = i*NSUB+j
                    sec_hat = im_hat*coef[index]*torch.outer(cphase1, rphase1)*wt
                    coadd_hat[i*nx:(i+1)*nx, j*nx:(j+1)*nx] += sec_hat
                    
    coadd_hat = torch.roll(torch.roll(coadd_hat, shifts=-nx, dims=1), shifts=-nx, dims=0)
    
    return torch.fft.ifft2(coadd_hat).real


def combine_image_3x(
    normalized_atlas: List[torch.Tensor], 
    centroids: List[Tuple[float, float]], 
    wts: Optional[List[float]] = None, 
    oversample: int = 3, 
    device: torch.device = 'cpu',
):
    '''
    Refractored implementation for 3x dither and more
    NOTE: wt is the same scale as SNR, not 1/sigma**2
    '''

    NSUB = oversample
    N = len(centroids)
    dither_images = normalized_atlas
    
    # Create j indices
    center_offset = NSUB // 2
    jy, jx = torch.tensor(np.indices((NSUB, NSUB)), device=device)
    jx = (jx - center_offset).ravel()
    jy = (jy - center_offset).ravel()
    # Compute dot product using broadcasting
    dxs = -centroids[:, 0]
    dys = -centroids[:, 1]
    phase = torch.outer(jx, dxs) + torch.outer(jy, dys)
    # Apply exponential
    Phi = torch.exp(2j * torch.pi * phase * NSUB / NSUB)/NSUB**2
    D = wts[None,:] # works as diag(wts) but faster
    Phi_inv = (D.T) * torch.linalg.pinv(Phi * D)

    nx, ny = dither_images[0].shape
    nx_in = scipy.fft.next_fast_len(nx)
    ny_in = scipy.fft.next_fast_len(ny)
    dither_images_in = torch.zeros((N, nx_in, ny_in), dtype=torch.complex64, device=device)
    dither_images_in[:, :nx, :ny] = dither_images
    coadd_hat = torch.zeros((nx_in*NSUB, ny_in*NSUB), dtype=torch.complex64, device=device)

    # ================

    # proceed each image KEEP THIS FOE DEBUG USE

    # for iim in range(N):

    #     dy, dx = centroids[iim]
    #     im = dither_images_in[iim][:nx_in, :ny_in]
    #     im_hat = torch.fft.fft2(im)
    #     im_hat = torch.fft.fftshift(im_hat)
        
    #     fx = torch.fft.fftfreq(nx_in, d=1.0, device=device)
    #     fy = torch.fft.fftfreq(ny_in, d=1.0, device=device)
    #     fx = torch.fft.fftshift(fx)
    #     fy = torch.fft.fftshift(fy)
    #     v, u = torch.meshgrid(fx, fy, indexing='ij')  # (H, W)

    #     phase_shift = torch.exp(2j * np.pi * (u * dy + v * dx))
        
    #     # proceed each tile
    #     for i in range(NSUB):
    #         for j in range(NSUB):
    #                 itile = i*NSUB+j
    #                 sec_coef = Phi_inv[iim, itile]
                    
    #                 sec_hat = im_hat.clone()
    #                 sec_hat *= sec_coef
    #                 sec_hat *= phase_shift                
                    
    #                 coadd_hat[i*nx_in:(i+1)*nx_in, j*ny_in:(j+1)*ny_in] += sec_hat

    # ================
    
    # EQUIVALENT BUT FASTER:

    # 1) frequency grids
    fx = torch.fft.fftfreq(nx_in, d=1.0, device=device)
    fy = torch.fft.fftfreq(ny_in, d=1.0, device=device)
    fx = torch.fft.fftshift(fx)  # (nx_in,)
    fy = torch.fft.fftshift(fy)  # (ny_in,)
    v, u = torch.meshgrid(fx, fy, indexing='ij')  # (nx_in, ny_in)

    # 2) batch FFT
    im_hat = torch.fft.fftshift(torch.fft.fft2(dither_images_in, dim=(-2,-1)), dim=(-2,-1))  # (N, nx_in, ny_in)

    # 3) batch calculate phase shift
    dy = centroids[:, 0].reshape(N, 1, 1)
    dx = centroids[:, 1].reshape(N, 1, 1)
    phase = torch.exp(2j*torch.pi*(u*dy + v*dx))  # (N, nx_in, ny_in), 复数

    # 4) apply phase shift 
    hat_shifted = im_hat * phase  # (N, nx_in, ny_in)

    # 5) apply phase shift and coefficient
    S = torch.einsum('nj,nhw->jhw', Phi_inv, hat_shifted)  # (J, nx_in, ny_in)

    # 6) merge the array to a large power spectrum
    coadd_hat = (
        S.view(NSUB, NSUB, nx_in, ny_in)    # (NSUB, NSUB, nx_in, ny_in)
        .permute(0, 2, 1, 3)               # (NSUB, nx_in, NSUB, ny_in)
        .reshape(NSUB*nx_in, NSUB*ny_in)   # (NSUB*nx_in, NSUB*ny_in)
    )
    
    # ================

    coadd = torch.fft.ifft2(torch.fft.ifftshift(coadd_hat))
    combined_image = coadd.real[:nx*NSUB, :ny*NSUB]

    return combined_image