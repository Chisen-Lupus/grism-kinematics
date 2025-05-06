# plot.py

from astropy.stats import sigma_clipped_stats
import numpy as np
import matplotlib.pyplot as plt


def asinhstretch(im):
    _, _, std = sigma_clipped_stats(im)
    return np.arcsinh(im / std)

def plot_grism_result_2(true_im, model_im, vz, diff_im, title_prefix, axs_row, 
                        velocity_params, dispersion_params):
    dx = dispersion_params['dx']
    dy = dispersion_params['dy']
    x0_v = velocity_params['x0_v']
    y0_v = velocity_params['y0_v']
    residual = model_im - diff_im
    resid_max = np.max(np.abs(residual))
    vz_max = np.max(np.abs(vz))

    im0 = axs_row[0].imshow(true_im)
    axs_row[0].set_title(f'{title_prefix} Grism Image')
    axs_row[0].grid(False)
    axs_row[0].scatter(x0_v-dx, y0_v-dy, marker='o', s=100, c='none', edgecolor='lime')
    plt.colorbar(im0, ax=axs_row[0], fraction=0.046, pad=0.04)

    im1 = axs_row[1].imshow(model_im)
    axs_row[1].set_title(f'{title_prefix} velocity corrected')
    axs_row[1].grid(False)
    axs_row[1].scatter(x0_v, y0_v, marker='o', s=100, c='none', edgecolor='lime')
    plt.colorbar(im1, ax=axs_row[1], fraction=0.046, pad=0.04)

    im2 = axs_row[2].imshow(residual, cmap='seismic', vmin=-resid_max, vmax=resid_max)
    axs_row[2].set_title(f'{title_prefix} Residual')
    axs_row[2].grid(False)
    plt.colorbar(im2, ax=axs_row[2], fraction=0.046, pad=0.04)

    im2 = axs_row[3].imshow(vz, cmap='seismic', vmin=-vz_max, vmax=vz_max)
    axs_row[3].set_title(f'{title_prefix} velocity field')
    axs_row[3].grid(False)
    plt.colorbar(im2, ax=axs_row[3], fraction=0.046, pad=0.04)

def plot_loss_and_lr(losses, lrs, steps=None):
    if steps is None:
        steps = range(len(losses))
    
    fig, ax1 = plt.subplots(figsize=(8,5))

    color_loss = 'tab:blue'
    ax1.plot(steps, losses, color=color_loss, label='Loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # create second y-axis sharing the same x-axis

    color_lr = 'tab:red'
    ax2.plot(steps, lrs, color=color_lr, linestyle='--', label='Learning Rate')
    ax2.set_yscale('log')
    ax2.set_ylabel('Learning Rate', color=color_lr)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.title('Loss and Learning Rate Evolution')
    plt.show()