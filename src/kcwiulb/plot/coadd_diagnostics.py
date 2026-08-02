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

    Parameters
    ----------
    coadd_data
        Coadded flux cube with shape (wavelength, y, x).
    coadd_var
        Coadded variance cube with shape (wavelength, y, x).
    t_exp_tot
        Exposure-time map:
        - 2D for blue coadds
        - 3D for wavelength-dependent red coadds
        - None if unavailable
    exposure_time
        Exposure time of one input exposure, used to convert seconds
        into an approximate number of exposures.
    save_path
        Optional output figure path.
    """

    # --------------------------------------------------------
    # Prepare flux and variance maps
    # --------------------------------------------------------
    flux_map = np.nanmedian(coadd_data, axis=0)
    var_map = np.nanmedian(coadd_var, axis=0)

    # --------------------------------------------------------
    # Prepare true exposure map
    # --------------------------------------------------------
    if t_exp_tot is not None:
        t_exp_tot = np.asarray(t_exp_tot, dtype=float)

        if t_exp_tot.ndim == 3:
            exp_seconds = np.nanmedian(t_exp_tot, axis=0)
        elif t_exp_tot.ndim == 2:
            exp_seconds = t_exp_tot
        else:
            raise ValueError(
                f"t_exp_tot must be 2D or 3D, got shape {t_exp_tot.shape}"
            )

        exp_map = exp_seconds / exposure_time
        exp_label = "Number of exposures"
        exposure_available = True

    else:
        exp_map = np.full_like(var_map, np.nan, dtype=float)
        exp_label = "Exposure unavailable"
        exposure_available = False

    # --------------------------------------------------------
    # Auto scaling for flux
    # --------------------------------------------------------
    good_flux = np.isfinite(flux_map)

    if np.any(good_flux):
        flux_vmin = np.nanpercentile(flux_map[good_flux], 5)
        flux_vmax = np.nanpercentile(flux_map[good_flux], 95)
    else:
        flux_vmin = None
        flux_vmax = None

    # --------------------------------------------------------
    # Variance scaling
    # --------------------------------------------------------
    good_var = np.isfinite(var_map) & (var_map > 0)

    if np.any(good_var):
        var_med = np.nanmedian(var_map[good_var])
        var_vmin = 0.9 * var_med
        var_vmax = 1.1 * var_med
    else:
        var_vmin = None
        var_vmax = None

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # 1. Exposure
    if exposure_available:
        im0 = axes[0].imshow(
            exp_map,
            origin="lower",
            cmap="viridis",
        )

        cbar0 = fig.colorbar(
            im0,
            ax=axes[0],
            shrink=0.9,
            pad=0.02,
        )
        cbar0.set_label(exp_label)

    else:
        axes[0].text(
            0.5,
            0.5,
            "Exposure map unavailable",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

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

    cbar1 = fig.colorbar(
        im1,
        ax=axes[1],
        shrink=0.9,
        pad=0.02,
    )
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

    cbar2 = fig.colorbar(
        im2,
        ax=axes[2],
        shrink=0.9,
        pad=0.02,
    )
    cbar2.set_label("Variance")

    axes[2].set_title("Median Variance")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()