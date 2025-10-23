
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import logging
import torch
import time
import functools

def timeit(func):
    @functools.wraps(func)  # 保留原函数信息
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} elapsed {elapsed*1000:.3f} ms")
        return result
    return wrapper

from dingo.fitting import ImagesFitter, build_param_config_dict_with_alias, load_fits_data
from dingo import grism, utils, galaxy, kinematics, fitting, plot, dither

# Set up logger

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

#%% --------------------------------------------------------------------------

def plot_fits_footprints(agn_coord, star_coords, used_files):
    fig, ax = plt.subplots(figsize=(8, 6))

    for filename in used_files:
        with fits.open(filename) as hdul:
            w = WCS(hdul['SCI'].header)
            footprint = w.calc_footprint()

        ra = footprint[:, 0]
        dec = footprint[:, 1]
        ra = np.append(ra, ra[0])
        dec = np.append(dec, dec[0])
        ax.plot(ra, dec, '-', color='gray', lw=1)

    # Plot stars and AGN
    
    for idx in range(len(star_coords)):
        ax.text(star_coords.ra.deg[idx], star_coords.dec.deg[idx], idx)
    ax.scatter(star_coords.ra.deg, star_coords.dec.deg, s=10, color='blue', label='Stars')
    ax.scatter(agn_coord.ra.deg, agn_coord.dec.deg, marker='*', s=100, color='red', label='AGN')

    # Add a 10 arcsec (1/360 deg) radius circle around AGN
    radius_deg = 80 / 3600  # 10 arcsec in degrees
    circle = Circle(
        (agn_coord.ra.deg, agn_coord.dec.deg),
        radius=radius_deg,
        edgecolor='red',
        facecolor='none',
        lw=1,
        linestyle='--',
        zorder=3
    )
    ax.add_patch(circle)

    # Axis formatting
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.set_title('FITS File Footprints and Source Locations')

    cos_dec = np.cos(np.deg2rad(agn_coord.dec.deg))
    ax.set_aspect(1 / cos_dec, adjustable='datalim')

    ax.legend(frameon=True)
    ax.grid(True)
    ax.xaxis.set_inverted(True)
    plt.tight_layout()
    plt.show()
    
def sigma_clipped_std(arr, sigma=3.0, max_iter=5):
    """
    Compute sigma-clipped standard deviation (default 3σ) in PyTorch.
    Input can be multi-dimensional but is flattened internally.
    
    Parameters
    ----------
    arr : torch.Tensor
        Input tensor (any shape).
    sigma : float
        Sigma threshold for clipping.
    max_iter : int
        Number of iterations for clipping.
    
    Returns
    -------
    std : torch.Tensor (scalar)
        The clipped standard deviation.
    """
    data = arr.reshape(-1).clone()
    
    for _ in range(max_iter):
        mean = data.mean()
        std = data.std(unbiased=True)
        mask = (data > mean - sigma*std) & (data < mean + sigma*std)
        new_data = data[mask]
        # if no change, stop early
        if new_data.numel() == data.numel():
            break
        data = new_data
    
    return data.std(unbiased=True)

#%% --------------------------------------------------------------------------

def asinhstretch(im): return np.arcsinh(im/sigma_clipped_stats(im[np.isfinite(im)])[2])

def simple_grid_plot(cutouts, stretch=True):
    ny = int(np.sqrt(len(cutouts)-1))+1
    nx = (len(cutouts)-1)//ny+1
    fig, axs = plt.subplots(nx, ny, figsize=(12, 12))
    axs = axs.flatten()
    for ax in axs: ax.axis('off')
    for i, cutout in enumerate(cutouts):
        if stretch:
            axs[i].imshow(asinhstretch(cutout))
        else:
            axs[i].imshow(cutout, origin='lower', vmin=0, vmax=np.nanmax(cutout)/5)
        axs[i].axis('off')
        axs[i].set_title(i, size=15)

#%% --------------------------------------------------------------------------

def bin(im, n=2):
    """
    Bin a 2D image tensor by averaging over nxn non-overlapping blocks.

    Parameters
    ----------
    im : torch.Tensor
        2D tensor of shape (H, W)
    n : int
        Binning factor (block size)

    Returns
    -------
    torch.Tensor
        Binned image of shape (H//n, W//n)
    """
    H, W = im.shape
    if H % n != 0 or W % n != 0:
        raise ValueError(f'Image dimensions ({H}, {W}) must be divisible by n={n}')
    
    im = im.reshape(H//n, n, W//n, n)
    return im.mean(dim=(1, 3))

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

#%% --------------------------------------------------------------------------

class PSFFitter(ImagesFitter):
    
    def __init__(self, config_path, device=None, oversample=None):
        self.oversample = oversample if oversample else 2
        super().__init__(config_path, device)
    
    def _setup_data(self):

        # 1) Load psf and direct image data (except dx dy)
        self.all_filters = set(self.config['direct'].keys())

        # load image data
        self.direct_images = {} # {filter: {iid: {property: value}}}
        for filter, imgs_cfg in self.config['direct'].items():
            self.direct_images[filter] = {}
            for iid, img_cfg in imgs_cfg.items():
                # add image data
                img_data = load_fits_data(img_cfg['path'])
                if '_cutout' in img_cfg: 
                    x, y, dx, dy = img_cfg['_cutout']
                    img_data = img_data[x:x+dx, y:y+dy]
                img_tensor = torch.tensor(
                    img_data, dtype=torch.float32, device=self.device
                )
                direct_info = {}
                direct_info['image'] = img_tensor
                self.direct_images[filter][iid] = direct_info

        # 3) Defaults & overrides
        defaults = [fs['default'] for fs in self.config['fitting']]
        overrides = [fs['override'] for fs in self.config['fitting']]

        # 4) Raw params
        _all_cfgs = [{} for _ in range(len(self.config['fitting']))]  # temp cfg for matching alias
        self.direct_cfgs_lists = {}

        # add image params
        # NOTE: overrides will be mutated during iteration
        for filter, imgs_cfg in self.config['direct'].items():
            for iid, raw_dict in imgs_cfg.items():
                i_cfgs = build_param_config_dict_with_alias(
                    raw_dict=raw_dict,
                    default_cfgs=defaults,
                    overrides_list=overrides,
                    prefix=f'direct.{filter}.{iid}',
                    allowed_keys_extra = {'direct': {'dx', 'dy', 'wt', 'zp', 'scale'}},
                    all_cfgs=_all_cfgs,
                    device=self.device
                )
                self.direct_cfgs_lists[iid] = i_cfgs

        unused_keys = set([key for stage in overrides for key in stage.keys()])
        if len(unused_keys)>0: 
            LOG.warning(f'The following override keys are not used: {unused_keys}')

        # 5) Register
        self.sersic_cfgs_lists = {}
        self.psf_cfgs_lists = {}
        self.param_config_lists = self.direct_cfgs_lists

    # @timeit
    def loss(self):

        # NOTE: self.*_cfg is already generated by _assign_cfgs_for_stage

        loss = 0
        oversample = self.oversample

        cutouts_in = []
        centroids = []
        wts = []
        zps = []
        scales = []
        # construct true image per psf
        for filter in self.all_filters:
            img_list = [iid for iid in self.direct_images[filter].keys()]
            for iid in img_list: 
                direct_info = self.direct_images[filter][iid]
                this_image = direct_info['image']
                direct_params = self._get_model_params(iid)
                dx, dy, wt, zp, scale = direct_params.values()
                # print(dx.requires_grad)
                # centroids -= 0.25
                centroids.append(torch.stack([dy, dx]))
                wts.append(wt)
                zps.append(zp)
                scales.append(scale)
                # cutouts_in.append((this_image-zp)/wt)
                cutouts_in.append(this_image)
            cutouts_in = torch.stack(cutouts_in)#.to(dtype=torch.complex128)
            centroids = torch.stack(centroids)#.to(dtype=torch.complex128)
            wts = torch.stack(wts)
            zps = torch.stack(zps)
            scales = torch.stack(scales)
            # normalize centroids
            centroids -= torch.mean(centroids, dim=0)
            zps -= torch.mean(zps)

            # compute psf-level sampled model at this filter
            wts_c = wts.to(dtype=torch.complex64)
            combined_image = dither.combine_image(
                torch.stack([(c-z)/s for c, z, s in zip(cutouts_in, zps, scales)]),
                # torch.stack([(c-z)/torch.sum(c-z) for c, z in zip(cutouts_in, zps)]),
                # cutouts_in, 
                centroids, 
                wts=wts_c, 
                oversample=oversample, 
                device=self.device, 
                return_full_array=False,
                overpadding=1,
            )
                
            combined_image_hat = torch.fft.fft2(combined_image)
            N, h, w = cutouts_in.shape
            H, W = combined_image_hat.shape
            n = oversample
            assert H == W and H % n == 0 and W % n == 0, '尺寸需为方阵，且可被 oversample 整除'
            k = H // n
            assert h == k and w == k, '下采样后尺寸需与 cutouts 匹配'

            # 1) 与原先一致的 cutout 归一化（逐样本）
            #    this_cutout = (cutouts_in[i]-zps[i])/scales[i]
            this_cutouts = (cutouts_in - zps[:, None, None]) / scales[:, None, None]   # [N, h, w]

            # 2) 批量化的频域相位平移（严格复刻你的 fft_phase_shift）
            #    你的函数：F * exp(-2πi(u*dx + v*dy))，
            #    调用时传的是 (dy*oversample, dx*oversample) 映射到 (dx, dy) 形参，因此我们保持：
            #      dx_param = dy*oversample, dy_param = dx*oversample
            #    meshgrid 用 v,u = meshgrid(fy, fx, indexing='ij')，也保持一致。
            real_dtype = combined_image_hat.real.dtype
            fx = torch.fft.fftfreq(W, d=1.0).to(device=combined_image_hat.device, dtype=real_dtype)
            fy = torch.fft.fftfreq(H, d=1.0).to(device=combined_image_hat.device, dtype=real_dtype)
            v, u = torch.meshgrid(fy, fx, indexing='ij')                     # [H, W]

            # 提取 (dy, dx)，并按你原来的调用方式乘 oversample 后映射到 (dx_param, dy_param)
            dy = centroids[:, 0].to(real_dtype)   # 原 centroids 存的是 [dy, dx]
            dx = centroids[:, 1].to(real_dtype)
            dx_param = dy * n
            dy_param = dx * n

            # 相位：exp(-2πi(u*dx_param + v*dy_param))。为确保 dtype 完全一致，强制到 combined_image_hat.dtype。
            phase = torch.exp(-2j*torch.pi * (
                u[None, :, :] * dx_param[:, None, None] +
                v[None, :, :] * dy_param[:, None, None]
            )).to(dtype=combined_image_hat.dtype)                                  # [N, H, W] complex

            shifted = combined_image_hat[None, :, :] * phase                       # [N, H, W] complex

            # 3) 频域 bin（严格等价于你的 fft_bin：把 N=n*k 拆成 (n,k)，对 n 维求和，再除以 n**2）
            #    你的 fft_bin 等价于：对索引 (a*k+b, c*k+d) 按 å(b,d) 聚合所有 a,c。
            binned_hat = shifted.reshape(N, n, k, n, k).sum(dim=(1, 3)) / (n**2)   # [N, k, k] complex

            # 4) 批量 ifft2 与取实部（与你原代码一致）
            models = torch.fft.ifft2(binned_hat).real                               # [N, h, w] float

            # 5) 批量 L2 残差（与你原来 loss 累加完全一致的定义）
            
            # loss part ----------
            
            # residual = (models - this_cutouts)*scales[:, None, None]+zps[:, None, None] # [N, h, w]
            residual = models - this_cutouts # [N, h, w]
            # loss = torch.sum(residual * residual) # 标量
            residual_loss = torch.sum(torch.abs(residual))
            loss += residual_loss
            
            # loss = (sigma_clipped_std(combined_image)*N*h*w)**2
            sigma_loss = sigma_clipped_std(combined_image, sigma=2)*N*h*w
            factor = (residual_loss/sigma_loss).detach() # NOTE: detach so that it will not degenerate to 2*residual_loss
            # factor = 1
            # print(factor)
            # raise RuntimeError
            loss += sigma_loss * factor * 8
            

        return loss

#%% --------------------------------------------------------------------------

class SpikesRemover(PSFFitter):
    
    def __init__(self, config_path, device=None, oversample=None):
        super().__init__(config_path, device, oversample)

    def _setup_data(self):
        super()._setup_data()
        self._post_setup()

    def _post_setup(self):
        
        # In this fitter we only fit an error array and fix everything else

        # disable all fitting parameters

        for name, params in self.param_config_lists.items():
            for fs in params: 
                for full_name, cfg in fs.items(): 
                    cfg.fit = False
        
        self._assign_cfgs_for_stage(0)

        # make a combined image and store it in self.combined_images_full

        self.combined_images_full = {}
        self.cutouts = {}
        self.centroids = {}
        self.wts = {}
        self.zps = {}
        self.scales = {}

        cutouts = []
        centroids = []
        wts = []
        zps = []
        scales = []
        # construct true image per psf
        for i, filter in enumerate(self.all_filters):
            img_list = [iid for iid in self.direct_images[filter].keys()]
            for iid in img_list: 
                direct_info = self.direct_images[filter][iid]
                this_image = direct_info['image']
                direct_params = self._get_model_params(iid)
                dx, dy, wt, zp, scale = direct_params.values()
                centroids.append(torch.stack([dy, dx]))
                wts.append(wt)
                zps.append(zp)
                scales.append(scale)
                cutouts.append(this_image)
            cutouts = torch.stack(cutouts)
            centroids = torch.stack(centroids)
            wts = torch.stack(wts)
            zps = torch.stack(zps)
            scales = torch.stack(scales)
            # normalize centroids
            centroids -= torch.mean(centroids, dim=0)
            zps -= torch.mean(zps)

            # compute psf-level sampled model at this filter
            wts_c = wts.to(dtype=torch.complex64)
            N, nx, ny= cutouts.shape
            cutouts_full = torch.zeros((N, nx*2, ny*2), dtype=cutouts.dtype)
            cutouts_full[:N, :nx, :ny] = cutouts
            combined_image_full = dither.combine_image(
                torch.stack([(c-z)/s for c, z, s in zip(cutouts_full, zps, scales)]),
                # (cutouts-zps[:, None, None])/torch.sum(cutouts-zps[:, None, None]),
                # (cutouts-zps[:, None, None])/wts[:, None, None], 
                centroids, 
                wts=wts_c, 
                oversample=self.oversample, 
                device=self.device
            )
            
            self.combined_images_full[filter] = combined_image_full
            self.cutouts[filter] = cutouts
            self.centroids[filter] = centroids
            self.wts[filter] = wts
            self.zps[filter] = zps
            self.scales[filter] = scales

            # create an error array and make this the only fitting parameter
        
            nim = len(cutouts)
            nx_raw, ny_raw = cutouts[0].shape
            vmax = np.abs(cutouts)*5e-2 + sigma_clipped_stats(cutouts)[2]*2e-1
            raw_dict = {'image': np.zeros((nim, nx_raw, ny_raw))}
            defaults = {
                'lr': self.config['fitting'][0]['default']['lr'],
                # 'min': -1e10, 
                # 'max': 1e10, 
                'min': -vmax, 
                'max': vmax, 
                'fit': True
            }
            overrides = {}
            e_cfgs = build_param_config_dict_with_alias(
                raw_dict=raw_dict,
                default_cfgs=defaults,
                overrides_list=overrides,
                prefix=f'err_image.{filter}',
                allowed_keys_extra = {'err_image': {'image'}}, 
                device=self.device
            )
            self.direct_cfgs_lists[f'e{i}'] = e_cfgs

        self._assign_cfgs_for_stage(0)

        # result container
        self.combined_err_full = {}

    def loss(self):

        # NOTE: self.*_cfg is already generated by _assign_cfgs_for_stage

        loss = 0

        for i, filter in enumerate(self.all_filters):
            
            cutouts = self.cutouts[filter]
            err_cutouts = self._get_model_params(f'e{i}')['image']
            combined_image_full = self.combined_images_full[filter]
            centroids = self.centroids[filter]
            wts = self.wts[filter]
            zps = self.zps[filter]
            scales = self.scales[filter]

            N, nx, ny= err_cutouts.shape
            err_cutouts_full = torch.zeros((N, nx*2, ny*2), dtype=err_cutouts.dtype)
            err_cutouts_full[:N, :nx, :ny] = err_cutouts
            combined_err_full = dither.combine_image(
                torch.stack([(c-z)/s for c, z, s in zip(err_cutouts_full, zps, scales)]),
                centroids, 
                wts=wts.to(dtype=torch.complex64), 
                oversample=self.oversample, 
                device=self.device
            )

            residual = combined_image_full - combined_err_full
            nx_raw, ny_raw = cutouts[0].shape
            # plt.imshow(residual[nx_raw*2:, :].detach().numpy())
            # raise RuntimeError
            loss += torch.sum(residual[nx_raw*self.oversample:, :]**2)
            loss += torch.sum(residual[:, ny_raw*self.oversample:]**2)

            self.combined_err_full[filter] = combined_err_full

        return loss

