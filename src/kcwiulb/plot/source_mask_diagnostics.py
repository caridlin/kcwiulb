from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from kcwiulb.analysis.source_mask import _sky_to_pixel


def save_source_mask_diagnostic(
    result,
    collapse_header,
    keep_circles,
    manual_filter_circles,
    output_path,
    show: bool = False,
    save: bool = True,
    flux_vmin: float | None = None,
    flux_vmax: float | None = None,
    snr_vmin: float = -50,
    snr_vmax: float = 50,
):
    output_path = Path(output_path)

    fig, axes = plt.subplots(2, 2)
    ax0, ax1, ax2, ax3 = axes.flat

    # =========================
    # Panel 1: Collapsed flux
    # =========================
    if flux_vmin is None:
        flux_vmin = np.nanpercentile(result.collapsed_flux, 5)
    if flux_vmax is None:
        flux_vmax = np.nanpercentile(result.collapsed_flux, 99)

    im0 = ax0.imshow(
        result.collapsed_flux,
        origin="lower",
        cmap="RdBu_r",
        vmin=flux_vmin,
        vmax=flux_vmax,
    )
    ax0.set_title("Collapsed Flux")
    
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im0, cax=cax0)

    # =========================
    # Panel 2: SNR
    # =========================
    im1 = ax1.imshow(
        result.collapsed_snr,
        origin="lower",
        cmap="RdBu_r",
        vmin=snr_vmin,
        vmax=snr_vmax,
    )
    ax1.set_title("Collapsed SNR")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im1, cax=cax1)

    # =========================
    # Panel 3: Auto mask
    # =========================
    im2 = ax2.imshow(
        result.auto_mask.astype(float),
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax2.set_title("Automatic Mask")

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.05)
    cax2.axis("off")   # 👈 keeps layout but hides colorbar

    # =========================
    # Panel 4: Final mask
    # =========================
    im3 = ax3.imshow(
        result.final_mask.astype(float),
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax3.set_title("Final Mask")

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="3%", pad=0.05)
    cax3.axis("off")   # 👈 same trick

    # =========================
    # Overlay circles
    # =========================
    for ax in axes.flat:
        if keep_circles is not None:
            for mode, a, b, r in keep_circles:
                if mode == "sky":
                    x_c, y_c = _sky_to_pixel(collapse_header, a, b)
                elif mode == "pixel":
                    x_c, y_c = a, b
                else:
                    raise ValueError(f"Unknown circle mode '{mode}'. Use 'sky' or 'pixel'.")

                circ = plt.Circle(
                    (x_c, y_c),
                    r,
                    color="cyan",
                    fill=False,
                    lw=1.5,
                )
                ax.add_patch(circ)

        if manual_filter_circles is not None:
            for mode, a, b, r in manual_filter_circles:
                if mode == "sky":
                    x_c, y_c = _sky_to_pixel(collapse_header, a, b)
                elif mode == "pixel":
                    x_c, y_c = a, b
                else:
                    raise ValueError(f"Unknown circle mode '{mode}'. Use 'sky' or 'pixel'.")

                circ = plt.Circle(
                    (x_c, y_c),
                    r,
                    color="orange",
                    fill=False,
                    lw=1.5,
                )
                ax.add_patch(circ)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    if save:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path