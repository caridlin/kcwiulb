from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from kcwiulb.wcs import wavelength_to_index


def collapse_cube_for_diagnostic(
    cube: np.ndarray,
    header,
    diag_wav_ranges: Sequence[tuple[float, float]],
) -> np.ndarray:
    """
    Collapse an already-cropped cube over selected wavelength ranges.
    """
    slabs = []
    for wav_min, wav_max in diag_wav_ranges:
        z0 = wavelength_to_index(wav_min, header)
        z1 = wavelength_to_index(wav_max, header)
        slabs.append(cube[z0:z1])

    collapsed = np.sum(np.concatenate(slabs, axis=0), axis=0)
    return collapsed


def plot_crop_diagnostics(
    cube_paths: list[Path],
    cube_ids: list[str],
    diag_wav_ranges: Sequence[tuple[float, float]],
    title: str,
    savepath: Path | None = None,
    show: bool = False,
    ncols: int = 4,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Plot collapsed diagnostic images for already-cropped cubes.

    - 4 panels per row by default
    - aspect ratio fixed to 0.2
    - consistent color scaling across all panels
    """
    collapsed_images = []

    for path in cube_paths:
        with fits.open(path) as hdul:
            cube = hdul[0].data
            header = hdul[0].header

            collapsed = collapse_cube_for_diagnostic(
                cube=cube,
                header=header,
                diag_wav_ranges=diag_wav_ranges,
            )
            collapsed_images.append(collapsed)

    if not collapsed_images:
        return

    if vmin is None or vmax is None:
        all_vals = np.concatenate(
            [img[np.isfinite(img)].ravel() for img in collapsed_images]
        )
        vmin = np.nanpercentile(all_vals, 5)
        vmax = np.nanpercentile(all_vals, 95)

    n = len(collapsed_images)
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, img, cube_id in zip(axes, collapsed_images, cube_ids):
        ax.imshow(
            img,
            origin="lower",
            cmap="RdBu_r",
            aspect=0.2,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(cube_id, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(collapsed_images):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)