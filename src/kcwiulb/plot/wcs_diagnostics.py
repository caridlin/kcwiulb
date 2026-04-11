from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_wcs_diagnostics(
    fit_result,
    row_start,
    col_start,
    n_rows,
    n_cols,
    title="",
    savepath=None,
    show=True,
):
    """
    Single-page WCS diagnostic figure:
    - Cutout selection
    - Fit quality (data/model/residual)
    - Global centroid
    """

    collapsed = fit_result.collapsed_image
    cutout = fit_result.cutout

    ny, nx = cutout.shape

    y, x = np.mgrid[:ny, :nx]

    model = (
        fit_result.amplitude
        * np.exp(
            -(
                ((x - fit_result.x_mean) ** 2) / (2 * fit_result.x_stddev**2)
                + ((y - fit_result.y_mean) ** 2) / (2 * fit_result.y_stddev**2)
            )
        )
        + fit_result.constant
    )

    residual = cutout - model

    x_global = fit_result.x_mean + col_start
    y_global = fit_result.y_mean + row_start

    # ===============================
    # Layout
    # ===============================
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ax1, ax2, ax_unused = axes[0]
    ax3, ax4, ax5 = axes[1]

    # ===============================
    # TOP LEFT — Cutout Selection
    # ===============================
    ax1.imshow(collapsed, origin="lower", cmap="RdBu_r", aspect=0.2)
    ax1.set_title("Cutout Selection")

    rect = plt.Rectangle(
        (col_start, row_start),
        n_cols,
        n_rows,
        edgecolor="red",
        facecolor="none",
        lw=2,
    )
    ax1.add_patch(rect)

    # ===============================
    # TOP MIDDLE — Global Position
    # ===============================
    ax2.imshow(collapsed, origin="lower", cmap="RdBu_r", aspect=0.2)
    ax2.scatter(x_global, y_global, color="red")

    rect2 = plt.Rectangle(
        (col_start, row_start),
        n_cols,
        n_rows,
        edgecolor="red",
        facecolor="none",
        lw=2,
    )
    ax2.add_patch(rect2)

    ax2.set_title("Fitted Centroid")

    # ===============================
    # TOP RIGHT — optional (hide it)
    # ===============================
    ax_unused.axis("off")

    # ===============================
    # BOTTOM ROW
    # ===============================
    vmin = np.nanpercentile(cutout, 5)
    vmax = np.nanpercentile(cutout, 95)

    ax3.imshow(cutout, origin="lower", cmap="RdBu_r", aspect=0.2, vmin=vmin, vmax=vmax)
    ax3.scatter(fit_result.x_mean, fit_result.y_mean, color="red")
    ax3.set_title("Data")

    ax4.imshow(model, origin="lower", cmap="RdBu_r", aspect=0.2, vmin=vmin, vmax=vmax)
    ax4.set_title("Model")

    ax5.imshow(residual, origin="lower", cmap="RdBu_r", aspect=0.2)
    ax5.set_title("Residual")

    # ===============================
    # CLEAN AXES
    # ===============================
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    # ===============================
    # Save
    # ===============================
    if savepath:
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_wcs_diagnostics(
    fit_result,
    row_start: int,
    col_start: int,
    n_rows: int,
    n_cols: int,
    title: str = "",
    savepath: str | Path | None = None,
    show: bool = True,
    full_vmin: float | None = 0.0,
    full_vmax: float | None = 1.8,
    cutout_vmin: float | None = 0.0,
    cutout_vmax: float | None = 1.8,
) -> None:
    """
    WCS diagnostic figure matching the notebook style more closely.

    Panels:
    - Full collapsed image with cutout rectangle
    - Full collapsed image with fitted centroid
    - Cutout data
    - Cutout model
    - Cutout residual
    """
    collapsed = fit_result.collapsed_image
    cutout = fit_result.cutout

    ny, nx = cutout.shape
    y, x = np.mgrid[:ny, :nx]

    model = (
        fit_result.amplitude
        * np.exp(
            -(
                ((x - fit_result.x_mean) ** 2) / (2.0 * fit_result.x_stddev**2)
                + ((y - fit_result.y_mean) ** 2) / (2.0 * fit_result.y_stddev**2)
            )
        )
        + fit_result.constant
    )

    residual = cutout - model

    # notebook convention
    x_global = fit_result.x_mean + col_start
    y_global = fit_result.y_mean + row_start

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax1, ax2, ax_unused = axes[0]
    ax3, ax4, ax5 = axes[1]

    # -------------------------------
    # TOP LEFT — full image with cutout box
    # -------------------------------
    ax1.imshow(
        collapsed,
        origin="lower",
        cmap="RdBu_r",
        aspect=1,
        vmin=full_vmin,
        vmax=full_vmax,
        interpolation="nearest",
    )
    rect1 = plt.Rectangle(
        (col_start, row_start),
        n_cols,
        n_rows,
        edgecolor="red",
        facecolor="none",
        lw=2,
    )
    ax1.add_patch(rect1)
    ax1.set_title("Cutout Selection")

    # -------------------------------
    # TOP MIDDLE — full image with fitted centroid
    # -------------------------------
    ax2.imshow(
        collapsed,
        origin="lower",
        cmap="RdBu_r",
        aspect=1,
        vmin=full_vmin,
        vmax=full_vmax,
        interpolation="nearest",
    )
    rect2 = plt.Rectangle(
        (col_start, row_start),
        n_cols,
        n_rows,
        edgecolor="red",
        facecolor="none",
        lw=2,
    )
    ax2.add_patch(rect2)
    ax2.scatter(x_global, y_global, color="red", s=50)
    ax2.set_title("Fitted Centroid")

    ax_unused.axis("off")

    # -------------------------------
    # BOTTOM LEFT — data
    # -------------------------------
    ax3.imshow(
        cutout,
        origin="lower",
        cmap="RdBu_r",
        aspect=1,
        vmin=cutout_vmin,
        vmax=cutout_vmax,
        interpolation="nearest",
    )
    ax3.scatter(fit_result.x_mean, fit_result.y_mean, color="red", s=50)
    ax3.set_title("Data")

    # -------------------------------
    # BOTTOM MIDDLE — model
    # -------------------------------
    ax4.imshow(
        model,
        origin="lower",
        cmap="RdBu_r",
        aspect=1,
        vmin=cutout_vmin,
        vmax=cutout_vmax,
        interpolation="nearest",
    )
    ax4.set_title("Model")

    # -------------------------------
    # BOTTOM RIGHT — residual
    # -------------------------------
    ax5.imshow(
        residual,
        origin="lower",
        cmap="RdBu_r",
        aspect=1,
        vmin=cutout_vmin,
        vmax=cutout_vmax,
        interpolation="nearest",
    )
    ax5.set_title("Residual")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)