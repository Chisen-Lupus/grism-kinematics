
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
        wts = torch.ones(len(normalized_atlas), device=device)

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
    NC_FREQ = int(2**np.ceil(np.log2(NX_LARGE))) # find the next 2^N+2 e.g. 514
    NR_FREQ = int(2**np.ceil(np.log2(NY_LARGE))) # 4x of NX_LARGE can effectlively dissipate the noise
    NC_FREQ *= 2**overpadding
    NR_FREQ *= 2**overpadding


    A_total = torch.zeros((NC_FREQ//2+1, NR_FREQ), dtype=torch.complex64, device=device)

    for torchos in range(len(normalized_atlas)): 

        data = normalized_atlas[torchos]
        data_large = torch.zeros((NC_FREQ, NR_FREQ))
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

                # Pivot the fundamental component to the first row
                nfund = NSUB * nvin - nuin
                phases[[0, nfund], :] = phases[[nfund, 0], :]

                # Add weighting factor
                if NPP==NSUB**2:
                    phasem = phases
                else: 
                    phasem = phases @ torch.diag(wts) @ torch.conj(phases).T

                vec = torch.linalg.inv(phasem)

                # For NSUB2 images, we are done
                if NPP==NSUB**2:
                    coef[iy, ix] = vec[torchos, 0]
                # Otherwise, we need to do a little more work. Here we just solve for the fundamental image.
                else: 
                    coef[iy, ix] = 0
                    for i in range(NSUB**2):
                        coef[iy, ix] += vec[i, 0]*torch.conj(phases[i, torchos])

                    # XXX: Moving it to the else branch means totally ignore wts for NSUB**2 images
                    # Add weighting factor
                    coef[iy, ix] *= wts[torchos]

                # print(f'Image {torchos}, power {coef[isec]*torch.conj(coef[isec])}, sector {isec}')

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
                    break

                # Compute the normalized column positions (U)
                cols = torch.arange(isu, ieu)
                U = cols / NC_FREQ  # Multiply back by 2 to match original scale

                # Compute the column phase shift (as a complex exponential)
                cphase = torch.exp(-2j * phix[torchos] * U)

                # process rows

                nv = NR_FREQ//NSUB 
                isv = NR_FREQ//2 - nv*iy
                iev = NR_FREQ//2 - nv*(iy+1) if iy<NSUB-1 else NR_FREQ//2 - NR_FREQ

                # Extract the complex coefficient
                coef_complex = coef[iy, ix]

                # Compute the normalized row positions (V)
                rows = torch.arange(isv-1, iev-1, -1)
                V = torch.where(rows >= NR_FREQ // 2, (rows - NR_FREQ) / NR_FREQ, rows / NR_FREQ)

                # Compute the row phase shift (as a complex exponential)
                rphase = torch.exp(-2j * phiy[torchos] * V)

                # apply shift

                # Compute the overall phase shift (outer product for broadcasting)
                phase_shift = coef_complex * torch.outer(cphase, rphase)

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
