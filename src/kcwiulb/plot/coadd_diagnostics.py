import matplotlib.pyplot as plt
import numpy as np


def plot_coadd_diagnostics(
    coadd_data,
    coadd_var,
    t_exp_tot=None,
    exposure_time=300,
    save_path=None,
):
    """
    Make a 3-panel diagnostic plot:

    1. Exposure map
    2. Median flux map
    3. Median variance map

    Works for both:
    - blue coadds: t_exp_tot can be 2D
    - red coadds:  t_exp_tot can be 3D (wavelength-dependent effective exposure)
    """

    # -------------------------------
    # Prepare maps
    # -------------------------------
    flux_map = np.nanmedian(coadd_data, axis=0)
    var_map = np.nanmedian(coadd_var, axis=0)

    # Exposure map
    if t_exp_tot is not None:
        if t_exp_tot.ndim == 3:
            exp_map = np.nanmedian(t_exp_tot, axis=0) / exposure_time
        elif t_exp_tot.ndim == 2:
            exp_map = t_exp_tot / exposure_time
        else:
            raise ValueError(f"t_exp_tot must be 2D or 3D, got shape {t_exp_tot.shape}")
        exp_label = "Number of exposures"
    else:
        exp_map = np.zeros_like(var_map)
        good_exp = np.isfinite(var_map) & (var_map > 0)
        if np.any(good_exp):
            exp_map[good_exp] = 1.0 / var_map[good_exp]
            exp_map /= np.nanmax(exp_map)
        exp_label = "Relative exposure"

    # Auto scaling for flux
    good_flux = np.isfinite(flux_map)
    if np.any(good_flux):
        flux_vmin = np.nanpercentile(flux_map[good_flux], 5)
        flux_vmax = np.nanpercentile(flux_map[good_flux], 95)
    else:
        flux_vmin = None
        flux_vmax = None

    # Variance scaling
    good_var = np.isfinite(var_map) & (var_map > 0)
    if np.any(good_var):
        var_med = np.nanmedian(var_map[good_var])
        var_vmin = 0.9 * var_med
        var_vmax = 1.1 * var_med
    else:
        var_vmin = None
        var_vmax = None

    # -------------------------------
    # Plot
    # -------------------------------
    fig, axes = plt.subplots(3, 1)

    # 1. Exposure
    im0 = axes[0].imshow(
        exp_map,
        origin="lower",
        cmap="viridis",
    )
    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.9, pad=0.02)
    cbar0.set_label(exp_label)
    axes[0].set_title("Exposure Map")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # 2. Flux
    im1 = axes[1].imshow(
        flux_map,
        origin="lower",
        cmap="gray",
        vmin=flux_vmin,
        vmax=flux_vmax,
    )
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.9, pad=0.02)
    cbar1.set_label("Flux")
    axes[1].set_title("Median Flux")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # 3. Variance
    im2 = axes[2].imshow(
        var_map,
        origin="lower",
        cmap="magma",
        vmin=var_vmin,
        vmax=var_vmax,
    )
    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.9, pad=0.02)
    cbar2.set_label("Variance")
    axes[2].set_title("Median Variance")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()