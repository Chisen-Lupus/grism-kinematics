arr = []

import os, sys
import torch 
import numpy as np
import torch
import copy
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