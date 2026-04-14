from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from kcwiulb.coadd.covariance_test import make_covariance_calibration_figure


def _style_axes(ax):
    ax.tick_params(
        axis="both",
        top=True,
        right=True,
        direction="in",
        length=4,
        width=1,
        colors="black",
        grid_color="black",
        grid_alpha=0.5,
    )
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="minor",
        top=True,
        right=True,
        direction="in",
        length=1.5,
        width=1,
        colors="black",
        grid_color="black",
        grid_alpha=0.5,
    )


def make_snr_histogram_figure(
    snr: np.ndarray,
) -> plt.Figure:
    mu, sigma = norm.fit(snr)

    fig, ax = plt.subplots(figsize=(6, 4))

    _, bins, _ = ax.hist(
        snr,
        60,
        density=True,
        facecolor="green",
        alpha=0.75,
    )

    y = norm(mu, sigma).pdf(bins)
    ax.plot(bins, y, "r--", linewidth=2, label="Best-fit Gaussian")

    y1 = norm.pdf(bins)
    ax.plot(bins, y1, "b-", linewidth=2, label="Normal distribution")

    ax.set_xlabel("SNR")
    ax.set_ylabel("Probability")
    ax.set_title(
        r"$\mathrm{Histogram\ of\ sky\ region:}\ \mu=%.3f,\ \sigma=%.3f$"
        % (mu, sigma)
    )
    ax.grid(True)
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()

    return fig


def make_ads_process_figure(
    result,
    snr_min: float,
    snr_max: float,
    r_max: float,
    z_max: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(9, 7))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(
        result.f_snr_array,
        color="red",
        label=r"$(SNR_{min} + SNR_{max}) / 2 SNR_{med}$",
    )
    ax1.plot(
        result.med_snr_array,
        color="black",
        label="Median detected SNR",
    )
    ax1.fill_between(
        np.arange(len(result.med_snr_array)),
        result.min_snr_array,
        result.max_snr_array,
        color="grey",
        alpha=0.5,
    )
    ax1.axhline(snr_min, color="black", linestyle="--", label="SNR min")
    ax1.axhline(snr_max, color="black", linestyle="--", label="SNR max")
    ax1.set_ylim((0.2, 7))
    ax1.set_xlabel("Smoothing Step")
    ax1.set_ylabel("SNR of Detected Pixels")
    ax1.legend()
    _style_axes(ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(result.xy_scale_array, color="blue", label="xy scale")
    ax2.plot(result.z_scale_array, color="red", label="z scale")
    ax2.axhline(r_max, color="blue", linestyle="--", label="xy scale limit")
    ax2.axhline(z_max, color="red", linestyle="--", label="z scale limit")
    ax2.set_xlabel("Smoothing Step")
    ax2.set_ylabel("Smoothing Scales")
    ax2.legend()
    _style_axes(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    denom = max(result.total_voxels - result.n_det_0, 1)
    ax3.plot(
        100 * np.array(result.n_det_array) / denom,
        color="black",
        label="Cumulative percentage of \nvoxels detected",
    )
    ax3.set_xlabel("Smoothing Step")
    ax3.set_ylabel("Percentage")
    ax3.legend()
    _style_axes(ax3)

    fig.tight_layout()
    return fig


def save_adaptive_smoothing_diagnostics_pdf(
    result,
    snr_first_check: np.ndarray,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    calibration_kernel_sizes: list[int],
    sigma_ratio_mean: np.ndarray,
    sigma_ratio_std: np.ndarray,
    fitted_alpha: float,
    fitted_norm: float,
    fitted_thresh: float,
    snr_min: float,
    snr_max: float,
    r_max: float,
    z_max: float,
    output_path: str | Path,
) -> Path:
    from kcwiulb.coadd.covariance_test import covar_curve

    output_path = Path(output_path)
    fitted_curve = covar_curve(
        alpha=fitted_alpha,
        norm=fitted_norm,
        thresh=fitted_thresh,
    )

    with PdfPages(output_path) as pdf:
        fig1 = make_snr_histogram_figure(
            snr=snr_first_check,
        )
        pdf.savefig(fig1, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig1)

        fig2 = make_covariance_calibration_figure(
            fit_x=fit_x,
            fit_y=fit_y,
            kernel_sizes=calibration_kernel_sizes,
            sigma_ratio_mean=sigma_ratio_mean,
            sigma_ratio_std=sigma_ratio_std,
            fitted_curve=fitted_curve,
        )
        pdf.savefig(fig2, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig2)

        fig3 = make_ads_process_figure(
            result=result,
            snr_min=snr_min,
            snr_max=snr_max,
            r_max=r_max,
            z_max=z_max,
        )
        pdf.savefig(fig3, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig3)

    return output_path