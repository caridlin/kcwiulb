from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm

from kcwiulb.coadd.covariance_test import (
    load_coadd_products,
    apply_wavelength_ranges,
    build_blank_sky_mask,
    compute_snr_distribution,
)


@dataclass
class VarianceScalingResult:
    flux_path: Path
    var_input_path: Path
    var_output_path: Path
    cov_data_input_path: Path
    cov_data_output_path: Path
    fitted_mu: float
    fitted_sigma: float
    scale_factor: float
    output_plot_path: Path


def make_variance_scaling_figure(
    snr: np.ndarray,
    mu: float,
    sigma: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    n, bins, _ = ax.hist(
        snr,
        60,
        density=True,
        facecolor="green",
        alpha=0.75,
    )

    ax.plot(bins, norm(mu, sigma).pdf(bins), "r--", linewidth=2, label="Best-fit Gaussian")
    ax.plot(bins, norm.pdf(bins), "b-", linewidth=2, label="Normal distribution")

    ax.set_xlabel("SNR")
    ax.set_ylabel("Probability")
    ax.set_title(rf"$\mu={mu:.3f},\ \sigma={sigma:.3f}$")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig


def run_variance_scaling(
    flux_path: str | Path,
    var_path: str | Path,
    cov_data_path: str | Path,
    cov_coord_path: str | Path,
    wavelength_ranges: list[tuple[float, float]] | None,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    overwrite: bool = True,
) -> VarianceScalingResult:
    flux_path = Path(flux_path)
    var_path = Path(var_path)
    cov_data_path = Path(cov_data_path)
    cov_coord_path = Path(cov_coord_path)

    flux, header, var_diag, cov_data, _ = load_coadd_products(
        flux_path=flux_path,
        var_path=var_path,
        cov_data_path=cov_data_path,
        cov_coord_path=cov_coord_path,
    )

    if output_dir is None:
        output_dir = flux_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = flux_path.stem

    signal = apply_wavelength_ranges(flux, header, wavelength_ranges)
    noise = apply_wavelength_ranges(var_diag, header, wavelength_ranges)

    keep_mask = build_blank_sky_mask(signal, sigma=2.5)
    snr = compute_snr_distribution(signal, noise, keep_mask, snr_clip=5.0)

    mu, sigma = norm.fit(snr)
    scale_factor = sigma**2

    var_scaled = var_diag * scale_factor
    cov_data_scaled = cov_data * scale_factor

    var_output_path = output_dir / f"{var_path.stem}_scaled{var_path.suffix}"
    cov_data_output_path = output_dir / f"{cov_data_path.stem}_scaled{cov_data_path.suffix}"
    plot_path = output_dir / f"{prefix}_variance_normalization.png"

    with fits.open(var_path) as hdul:
        hdul[0].data = var_scaled
        hdul.writeto(var_output_path, overwrite=overwrite)

    np.save(cov_data_output_path, cov_data_scaled)

    fig = make_variance_scaling_figure(snr, mu, sigma)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return VarianceScalingResult(
        flux_path=flux_path,
        var_input_path=var_path,
        var_output_path=var_output_path,
        cov_data_input_path=cov_data_path,
        cov_data_output_path=cov_data_output_path,
        fitted_mu=float(mu),
        fitted_sigma=float(sigma),
        scale_factor=float(scale_factor),
        output_plot_path=plot_path,
    )