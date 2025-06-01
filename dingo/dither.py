
import os, sys
import torch 
import numpy as np
import scipy
import copy
from typing import Callable, Any, Tuple, List, Optional
from numpy.typing import NDArray


def combine_image(
    normalized_atlas: List[NDArray[torch.float64]], 
    centroids: List[Tuple[float, float]], 
    wts: Optional[List[float]] = None, 
    oversample: int = 2, 
    return_full_array = False
) -> NDArray[torch.float64]:
    """
    Apply phase shifts to the input data.

    Parameters
    ----------
    normalized_atlas : list-like container of arrays
        Input 2D array TODO
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
        wts = torch.ones(len(normalized_atlas), dtype=torch.complex64)
    # centroids = torch.stack(centroids)
    # wts = torch.stack(wts, dtype=torch.complex64)

    # ASSERTATION

    assert len(normalized_atlas)==len(centroids)
    assert len(centroids)==len(wts)
    assert len(set([im.shape for im in normalized_atlas]))==1

    # SOME GLOBAL FACTORS

    # centroids = centroids.numpy()
    NSUB = oversample
    NPP = len(normalized_atlas)
    NX, NY = normalized_atlas[0].shape
    NX_LARGE = NX*NSUB
    NY_LARGE = NY*NSUB
    NC_FREQ = int(2**np.ceil(np.log2(NX_LARGE))) + 2 # find the next 2^N+2 e.g. 514
    NR_FREQ = int(2**np.ceil(np.log2(NY_LARGE))) # 4x of NX_LARGE can effectlively dissipate the noise
    # NC_FREQ = 130
    # NR_FREQ = 128
    DEVICE = normalized_atlas.device


    Atotal = torch.zeros((NC_FREQ//2, NR_FREQ), dtype=torch.complex64, device=DEVICE)
    F = torch.zeros((NC_FREQ, NR_FREQ), dtype=torch.complex64, device=DEVICE)

    for npos in range(len(normalized_atlas)): 

        data = normalized_atlas[npos]
        data_large = torch.zeros((NC_FREQ, NR_FREQ), device=DEVICE)
        data_large[:NX*NSUB:NSUB, :NY*NSUB:NSUB] = data
        coef = torch.zeros((NSUB, NSUB), dtype=torch.complex64, device=DEVICE)

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
                isatx, isaty = torch.meshgrid(torch.arange(NSUB, device=DEVICE), torch.arange(NSUB, device=DEVICE),indexing='xy')
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
                if NPP>NSUB**2: 
                    phasem = phases @ torch.diag(wts) @ torch.conj(phases).T
                else: 
                    phasem = phases

                # vec = torch.linalg.inv(phasem)
                vec = torch.linalg.pinv(phasem) # pseudo inverse
                # I = torch.eye(phasem.shape[0], dtype=phasem.dtype, device=phasem.device)
                # vec = torch.linalg.solve(phasem, I)

                # For NSUB2 images, we are done
                if NPP==NSUB**2:
                    coef[iy, ix] = vec[npos, 0]
                # Otherwise, we need to do a little more work. Here we just solve for the fundamental image.
                else: 
                    coef[iy, ix] = 0
                    for i in range(NSUB**2):
                        coef[iy, ix] += vec[i, 0]*torch.conj(phases[i, npos])

                # Add weighting factor
                coef[iy, ix] *= wts[npos]

                # print(f'Image {npos}, power {coef[isec]*np.conj(coef[isec])}, sector {isec}')

        # print('---')

        # END COEFFICIENT COMPUTATION

        # BEGIN FFT2

        # We only need half of the transformed array since we are doing real transform
        A_hat = torch.fft.fft2(data_large) # data_large must be (2^N, 2^N) for now
        A_unique = A_hat[:NC_FREQ//2, :]  # shape (NC_FREQ//2, NR_FREQ)
        A_complex = torch.conj(A_unique)

        # END FFT2

        # BEGIN PHASE SHIFT APPLICATION

        for iy in range(NSUB):
            for ix in range(NSUB):

                # process columns

                # Starting and ending points of this sector
                nu = NC_FREQ//NSUB
                isu = min(nu*ix, NC_FREQ//2)
                ieu = min(nu*(ix+1), NC_FREQ//2)
                if isu==ieu: 
                    break

                # Compute the normalized column positions (U)
                cols = torch.arange(isu, ieu, device=DEVICE)
                U = cols / NC_FREQ  # Multiply back by 2 to match original scale

                # Compute the column phase shift (as a complex exponential)
                cphase = torch.exp(-2j * phix[npos] * U)

                # process rows

                nv = NR_FREQ//NSUB
                isv = NR_FREQ//2 - nv*iy
                iev = NR_FREQ//2 - nv*(iy+1) if iy<NSUB-1 else NR_FREQ//2 - NR_FREQ

                # Extract the complex coefficient
                coef_complex = coef[iy, ix]

                # Compute the normalized row positions (V)
                # print('ix', ix, 'iy', iy)
                # print('isu', isu, 'ieu', ieu, 'isv', isv, 'iev', iev)
                rows = torch.arange(isv-1, iev-1, -1, device=DEVICE)
                # rows = torch.where(rows >= 0, rows, NR_FREQ + rows) # numpy array can take negative index
                V = torch.where(rows >= NR_FREQ // 2, (rows - NR_FREQ) / NR_FREQ, rows / NR_FREQ)

                # Compute the row phase shift (as a complex exponential)
                rphase = torch.exp(-2j * phiy[npos] * V)

                # apply shift

                # Compute the overall phase shift (outer product for broadcasting)
                phase_shift = coef_complex * torch.outer(cphase, rphase)

                cols_idx = cols % NC_FREQ
                rows_idx = rows % NR_FREQ
                cols_idx = cols_idx.unsqueeze(1)
                rows_idx = rows_idx.unsqueeze(0)
                A_complex[cols_idx, rows_idx] *= phase_shift

        Atotal += torch.conj(A_complex)

        # print('------')

    # END PHASE SHIFT APPLICATION

    # BEGIN IFFT2


    F[:NC_FREQ//2, :] = Atotal
    F[NC_FREQ//2+1:, 0] = torch.conj(Atotal[1:NC_FREQ//2])[:, 0].flip(dims=[0])
    F[NC_FREQ//2+1:, 1:] = torch.conj(Atotal[1:NC_FREQ//2])[:, 1:].flip(dims=[0, 1])
    data_rec = torch.fft.ifft2(F)
    data_real = data_rec.real

    # END IFFT2

    combined_image = data_real[:NX_LARGE, :NY_LARGE]

    if return_full_array:
        return data_real
    else:
        return combined_image
