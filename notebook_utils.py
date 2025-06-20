
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import logging
import torch

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

#%% --------------------------------------------------------------------------

def asinhstretch(im): return np.arcsinh(im/sigma_clipped_stats(im)[2])

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

    def loss(self):

        # NOTE: self.*_cfg is already generated by _assign_cfgs_for_stage

        loss = 0

        cutouts_in = []
        centroids = []
        wts = []
        zps = []
        # construct true image per psf
        for filter in self.all_filters:
            img_list = [iid for iid in self.direct_images[filter].keys()]
            for iid in img_list: 
                direct_info = self.direct_images[filter][iid]
                this_image = direct_info['image']
                direct_params = self._get_model_params(iid)
                dx, dy, wt, zp = direct_params.values()
                # print(dx.requires_grad)
                # centroids -= 0.25
                centroids.append(torch.stack([dy, dx]))
                wts.append(wt)
                zps.append(zp)
                # cutouts_in.append((this_image-zp)/wt)
                cutouts_in.append(this_image)
            cutouts_in = torch.stack(cutouts_in)#.to(dtype=torch.complex128)
            centroids = torch.stack(centroids)#.to(dtype=torch.complex128)
            wts = torch.stack(wts)
            zps = torch.stack(zps)
            # compute psf-level sampled model at this filter

            wts_c = wts.to(dtype=torch.complex64)
            combined_image_full = dither.combine_image(
                torch.stack([c/w for c, w in zip(cutouts_in, wts)]),
                # torch.stack([(c-z)/torch.sum(c-z) for c, z in zip(cutouts_in, zps)]),
                # cutouts_in, 
                centroids, 
                wts=wts_c, 
                oversample=self.oversample, 
                device=self.device, 
                return_full_array=True,
                overpadding=1,
            )

            for i, iid in enumerate(img_list): 

                # this_cutout = (cutouts_in[i]-zps[i, None, None])/torch.sum(cutouts_in[i]-zps[i, None, None])
                this_cutout = cutouts_in[i]/zps[i, None, None]

                dc = 1/2 - 1/2/self.oversample
                nx, ny = this_cutout.shape
                nx_large = nx*self.oversample
                ny_large = ny*self.oversample
                dy, dx = centroids[i]
                combined_image = combined_image_full[:nx_large, :ny_large]
                this_model = utils.downsample_with_shift_and_size(
                    x=combined_image, factor=self.oversample, out_size=(nx, ny), shift=(dx+dc, dy+dc), 
                )

                # this_centroid = centroids - centroids[i] - 0.25
                # wts_c = wts.to(dtype=torch.complex64)
                # combined_image = dither.combine_image(cutouts_in, this_centroid, wts=wts_c)
                # this_model = bin2(combined_image)

                this_loss = torch.sum((this_model - this_cutout)**2)
                # print(this_loss)
                mismatch_loss = torch.sum(combined_image_full[nx_large:, :]**2) \
                              + torch.sum(combined_image_full[:, ny_large:]**2)
                loss += this_loss + mismatch_loss*100
                # loss += this_loss

            # loss += torch.sum(zps)**2

        # print(loss.dtype)

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

        cutouts = []
        centroids = []
        wts = []
        zps = []
        # construct true image per psf
        for i, filter in enumerate(self.all_filters):
            img_list = [iid for iid in self.direct_images[filter].keys()]
            for iid in img_list: 
                direct_info = self.direct_images[filter][iid]
                this_image = direct_info['image']
                direct_params = self._get_model_params(iid)
                dx, dy, wt, zp = direct_params.values()
                centroids.append(torch.stack([dy, dx]))
                wts.append(wt)
                zps.append(zp)
                cutouts.append(this_image)
            cutouts = torch.stack(cutouts)
            centroids = torch.stack(centroids)
            wts = torch.stack(wts)
            zps = torch.stack(zps)

            # compute psf-level sampled model at this filter
            wts_c = wts.to(dtype=torch.complex64)
            combined_image_full = dither.combine_image(
                cutouts/zps[:, None, None], 
                # (cutouts-zps[:, None, None])/torch.sum(cutouts-zps[:, None, None]),
                # (cutouts-zps[:, None, None])/wts[:, None, None], 
                centroids, 
                wts=wts_c, 
                oversample=self.oversample, 
                return_full_array=True, 
                overpadding=1,
                device=self.device
            )
            
            self.combined_images_full[filter] = combined_image_full
            self.cutouts[filter] = cutouts
            self.centroids[filter] = centroids
            self.wts[filter] = wts
            self.zps[filter] = zps

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

            combined_err_full = dither.combine_image(
                # (err_cutouts-zps[:, None, None])/torch.sum(err_cutouts-zps[:, None, None]),
                err_cutouts/zps[:, None, None], 
                # (err_cutouts-zps[:, None, None])/wts[:, None, None], 
                centroids, 
                wts=wts.to(dtype=torch.complex64), 
                oversample=self.oversample, 
                return_full_array=True, 
                overpadding=1,
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

