from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.modeling import fitting
from astropy.modeling.models import custom_model
from astropy.stats import sigma_clip
from scipy.stats import norm

from kcwiulb.wcs import wavelength_to_index


@dataclass
class CovarianceTestResult:
    flux_path: Path
    var_path: Path
    cov_data_path: Path
    cov_coord_path: Path
    histogram_kernel_sizes: list[int]
    calibration_kernel_sizes: list[int]
    sigma_with_cov: np.ndarray
    sigma_diag_only: np.ndarray
    sigma_ratio_mean: np.ndarray
    sigma_ratio_std: np.ndarray
    fitted_alpha: float
    fitted_norm: float
    fitted_thresh: float
    output_pdf_path: Path


@custom_model
def covar_curve(ksizes, alpha=0.3, norm=2.0, thresh=50.0):
    range1 = ksizes <= thresh
    range2 = ksizes > thresh

    res1 = norm * (1 + alpha * np.log(ksizes))
    res2 = norm * (1 + alpha * np.log(thresh))
    return np.select([range1, range2], [res1, res2])


def load_coadd_products(
    flux_path: str | Path,
    var_path: str | Path,
    cov_data_path: str | Path,
    cov_coord_path: str | Path,
) -> tuple[np.ndarray, fits.Header, np.ndarray, np.ndarray, np.ndarray]:
    flux_path = Path(flux_path)
    var_path = Path(var_path)
    cov_data_path = Path(cov_data_path)
    cov_coord_path = Path(cov_coord_path)

    with fits.open(flux_path) as hdul:
        flux = hdul[0].data.copy()
        header = hdul[0].header.copy()

    with fits.open(var_path) as hdul:
        var_diag = hdul[0].data.copy()

    cov_data = np.load(cov_data_path)
    cov_coord = np.load(cov_coord_path)

    return flux, header, var_diag, cov_data, cov_coord


def apply_wavelength_ranges(
    cube: np.ndarray,
    header: fits.Header,
    wavelength_ranges: list[tuple[float, float]] | None,
) -> np.ndarray:
    """
    Concatenate selected wavelength slabs, matching notebook behavior.
    """
    if wavelength_ranges is None or len(wavelength_ranges) == 0:
        return cube.copy()

    slabs: list[np.ndarray] = []
    for wl_min, wl_max in wavelength_ranges:
        i0 = max(0, wavelength_to_index(wl_min, header))
        i1 = min(cube.shape[0], wavelength_to_index(wl_max, header))
        if i1 > i0:
            slabs.append(cube[i0:i1])

    if len(slabs) == 0:
        raise ValueError("No valid wavelength ranges selected.")

    return np.concatenate(slabs, axis=0)


def reconstruct_variance_cube(
    flux_shape: tuple[int, int, int],
    cov_data: np.ndarray,
    cov_coord: np.ndarray,
) -> np.ndarray:
    n_w, n_y, n_x = flux_shape
    var_full = np.zeros(flux_shape, dtype=float)

    coord_dict = {
        (int(c0), int(c1)): i
        for i, (c0, c1) in enumerate(cov_coord)
    }

    for y in range(n_y):
        for x in range(n_x):
            pix = n_x * y + x
            idx = coord_dict.get((pix, pix))
            if idx is not None:
                var_full[:, y, x] = cov_data[idx]

    return var_full


def rebin_cube_spatial_mean(cube: np.ndarray, bin_xy: int) -> np.ndarray:
    if bin_xy == 1:
        return cube.copy()

    n_w, n_y, n_x = cube.shape
    n_y_new = n_y // bin_xy
    n_x_new = n_x // bin_xy

    out = np.zeros((n_w, n_y_new, n_x_new), dtype=float)

    for yb in range(n_y_new):
        for xb in range(n_x_new):
            y0 = yb * bin_xy
            y1 = y0 + bin_xy
            x0 = xb * bin_xy
            x1 = x0 + bin_xy
            out[:, yb, xb] = np.mean(cube[:, y0:y1, x0:x1], axis=(1, 2))

    return out


def rebin_variance_diag(var_diag: np.ndarray, bin_xy: int) -> np.ndarray:
    if bin_xy == 1:
        return var_diag.copy()

    n_w, n_y, n_x = var_diag.shape
    n_y_new = n_y // bin_xy
    n_x_new = n_x // bin_xy

    out = np.zeros((n_w, n_y_new, n_x_new), dtype=float)

    for yb in range(n_y_new):
        for xb in range(n_x_new):
            y0 = yb * bin_xy
            y1 = y0 + bin_xy
            x0 = xb * bin_xy
            x1 = x0 + bin_xy
            out[:, yb, xb] = np.sum(var_diag[:, y0:y1, x0:x1], axis=(1, 2)) / (bin_xy**4)

    return out


def rebin_variance_full(
    flux_shape: tuple[int, int, int],
    cov_data: np.ndarray,
    cov_coord: np.ndarray,
    bin_xy: int,
) -> np.ndarray:
    """
    Rebin full variance from sparse covariance storage.

    Assumes:
    - diagonal terms are stored once
    - off-diagonal covariance terms are stored once only
    so off-diagonal terms must be multiplied by 2 when reconstructing
    Var(sum x_i) = sum Var_i + 2 sum_{i<j} Cov_ij
    """
    n_w, n_y, n_x = flux_shape

    if bin_xy == 1:
        return reconstruct_variance_cube(flux_shape, cov_data, cov_coord)

    n_y_new = n_y // bin_xy
    n_x_new = n_x // bin_xy
    out = np.zeros((n_w, n_y_new, n_x_new), dtype=float)

    coord_dict = {
        (int(c0), int(c1)): i
        for i, (c0, c1) in enumerate(cov_coord)
    }

    for yb in range(n_y_new):
        for xb in range(n_x_new):
            y0 = yb * bin_xy
            y1 = y0 + bin_xy
            x0 = xb * bin_xy
            x1 = x0 + bin_xy

            pixels = [(yy, xx) for yy in range(y0, y1) for xx in range(x0, x1)]
            cov_sum = np.zeros(n_w, dtype=float)

            for i, (yy1, xx1) in enumerate(pixels):
                p1 = n_x * yy1 + xx1

                idx_diag = coord_dict.get((p1, p1))
                if idx_diag is not None:
                    cov_sum += cov_data[idx_diag]

                for j in range(i + 1, len(pixels)):
                    yy2, xx2 = pixels[j]
                    p2 = n_x * yy2 + xx2
                    key = (p1, p2) if p1 <= p2 else (p2, p1)
                    idx_off = coord_dict.get(key)
                    if idx_off is not None:
                        cov_sum += 2.0 * cov_data[idx_off]

            out[:, yb, xb] = cov_sum / (bin_xy**4)

    return out


def build_blank_sky_mask(signal_cube: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """
    True = keep blank-sky pixel
    False = masked continuum/source pixel
    """
    collapsed = np.sum(signal_cube, axis=0)
    clipped = sigma_clip(collapsed, sigma=sigma, maxiters=None, masked=True)
    return ~clipped.mask


def apply_spatial_mask(cube: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    out = cube.copy()
    for i in range(out.shape[0]):
        out[i] *= keep_mask
    return out


def compute_snr_distribution(
    signal_cube: np.ndarray,
    variance_cube: np.ndarray,
    keep_mask: np.ndarray,
    snr_clip: float = 5.0,
) -> np.ndarray:
    signal_use = apply_spatial_mask(signal_cube, keep_mask)
    var_use = apply_spatial_mask(variance_cube, keep_mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = signal_use / np.sqrt(var_use)

    snr = snr[np.isfinite(snr)]
    snr = snr[np.abs(snr) <= snr_clip]
    return snr


def compute_sigma_ratio_distribution(
    variance_full: np.ndarray,
    variance_diag: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    var_full_use = apply_spatial_mask(variance_full, keep_mask)
    var_diag_use = apply_spatial_mask(variance_diag, keep_mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_ratio = np.sqrt(var_full_use) / np.sqrt(var_diag_use)

    sigma_ratio = sigma_ratio[np.isfinite(sigma_ratio)]
    return sigma_ratio


def make_dual_snr_histogram_figure(
    snr_diag: np.ndarray,
    snr_full: np.ndarray,
    kernel_size: int,
) -> tuple[plt.Figure, float, float, float, float]:
    mu_diag, sigma_diag = norm.fit(snr_diag)
    mu_full, sigma_full = norm.fit(snr_full)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Left: diagonal only
    _, bins_diag, _ = axes[0].hist(
        snr_diag, 60, density=True, facecolor="green", alpha=0.75
    )
    axes[0].plot(bins_diag, norm(mu_diag, sigma_diag).pdf(bins_diag), "r--", lw=2, label="Best-fit Gaussian")
    axes[0].plot(bins_diag, norm.pdf(bins_diag), "b-", lw=2, label="Normal distribution")
    axes[0].set_title(f"Diagonal only ({kernel_size}x{kernel_size})\nμ={mu_diag:.3f}, σ={sigma_diag:.3f}")
    axes[0].set_xlabel("SNR")
    axes[0].set_ylabel("Probability")
    axes[0].grid(True)
    axes[0].legend()

    # Right: with covariance
    _, bins_full, _ = axes[1].hist(
        snr_full, 60, density=True, facecolor="green", alpha=0.75
    )
    axes[1].plot(bins_full, norm(mu_full, sigma_full).pdf(bins_full), "r--", lw=2, label="Best-fit Gaussian")
    axes[1].plot(bins_full, norm.pdf(bins_full), "b-", lw=2, label="Normal distribution")
    axes[1].set_title(f"With covariance ({kernel_size}x{kernel_size})\nμ={mu_full:.3f}, σ={sigma_full:.3f}")
    axes[1].set_xlabel("SNR")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("SNR Comparison", y=1.02)
    fig.tight_layout()

    return fig, mu_diag, sigma_diag, mu_full, sigma_full


def make_covariance_calibration_figure(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    kernel_sizes: list[int],
    sigma_ratio_mean: np.ndarray,
    sigma_ratio_std: np.ndarray,
    fitted_curve,
) -> plt.Figure:
    plot_x = np.arange(1, max(kernel_sizes) + 1) ** 2
    kernel_area = np.array(kernel_sizes) ** 2

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        fit_x,
        fit_y,
        rasterized=True,
        color="black",
        marker="x",
        s=2,
        label="_nolegend_",
    )

    ax.plot(
        kernel_area,
        sigma_ratio_mean,
        label=r"Mean, shaded region $\pm 1\sigma$",
        color="black",
    )

    ax.fill_between(
        kernel_area,
        sigma_ratio_mean - sigma_ratio_std,
        sigma_ratio_mean + sigma_ratio_std,
        color="grey",
        alpha=0.5,
        label="_nolegend_",
    )

    ax.plot(
        plot_x,
        fitted_curve(plot_x),
        color="red",
        label=(
            r"fit, $\sigma_{meas}/\sigma_{nocov} = "
            f"{fitted_curve.norm.value:.2f}"
            r"(1 + "
            f"{fitted_curve.alpha.value:.2f}"
            r"\,\log(N_{kernel}))$, "
            f"thresh = {fitted_curve.thresh.value:.1f}"
        ),
    )

    ax.grid(True)
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
    ax.set_xlabel(r"$N_{kernel}$")
    ax.set_ylabel(r"$\sigma_{meas} / \sigma_{nocov}$")
    ax.set_title("Covariance Calibration")
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig


def _compute_kernel_products(
    flux: np.ndarray,
    header: fits.Header,
    var_diag: np.ndarray,
    cov_data: np.ndarray,
    cov_coord: np.ndarray,
    kernel_size: int,
    wavelength_ranges: list[tuple[float, float]] | None,
    mask_sigma: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_binned = rebin_cube_spatial_mean(flux, kernel_size)
    var_diag_binned = rebin_variance_diag(var_diag, kernel_size)
    var_full_binned = rebin_variance_full(flux.shape, cov_data, cov_coord, kernel_size)

    signal = apply_wavelength_ranges(data_binned, header, wavelength_ranges)
    noise_full = apply_wavelength_ranges(var_full_binned, header, wavelength_ranges)
    noise_diag = apply_wavelength_ranges(var_diag_binned, header, wavelength_ranges)

    keep_mask = build_blank_sky_mask(signal, sigma=mask_sigma)

    return signal, noise_full, noise_diag, keep_mask


def run_covariance_test(
    flux_path: str | Path,
    var_path: str | Path,
    cov_data_path: str | Path,
    cov_coord_path: str | Path,
    kernel_sizes: list[int] | None = None,
    wavelength_ranges: list[tuple[float, float]] | None = None,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
) -> CovarianceTestResult:
    flux_path = Path(flux_path)
    var_path = Path(var_path)
    cov_data_path = Path(cov_data_path)
    cov_coord_path = Path(cov_coord_path)

    flux, header, var_diag, cov_data, cov_coord = load_coadd_products(
        flux_path,
        var_path,
        cov_data_path,
        cov_coord_path,
    )

    # User input controls histogram pages only
    histogram_kernel_sizes = kernel_sizes if kernel_sizes is not None else [1, 2, 3]

    # Calibration curve always uses notebook kernel range
    calibration_kernel_sizes = list(range(1, 12))

    if output_dir is None:
        output_dir = flux_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = flux_path.stem

    output_pdf_path = output_dir / f"{prefix}_covariance_test.pdf"

    sigma_with_cov = []
    sigma_diag_only = []
    sigma_ratio_mean = []
    sigma_ratio_std = []
    fit_x = []
    fit_y = []

    with PdfPages(output_pdf_path) as pdf:
        # Histogram pages: only selected kernels
        for n in histogram_kernel_sizes:
            print(f"[covariance_test] Histogram kernel: {n}x{n}")

            signal, noise_full, noise_diag, keep_mask = _compute_kernel_products(
                flux=flux,
                header=header,
                var_diag=var_diag,
                cov_data=cov_data,
                cov_coord=cov_coord,
                kernel_size=n,
                wavelength_ranges=wavelength_ranges,
                mask_sigma=2.5,
            )

            snr_full = compute_snr_distribution(signal, noise_full, keep_mask)
            snr_diag = compute_snr_distribution(signal, noise_diag, keep_mask)

            fig_hist, _, _, _, _ = make_dual_snr_histogram_figure(
                snr_diag=snr_diag,
                snr_full=snr_full,
                kernel_size=n,
            )
            pdf.savefig(fig_hist, dpi=200, bbox_inches="tight")
            plt.close(fig_hist)

        # Calibration curve: always full kernel range 1..11
        for n in calibration_kernel_sizes:
            print(f"[covariance_test] Calibration kernel: {n}x{n}")

            signal, noise_full, noise_diag, keep_mask = _compute_kernel_products(
                flux=flux,
                header=header,
                var_diag=var_diag,
                cov_data=cov_data,
                cov_coord=cov_coord,
                kernel_size=n,
                wavelength_ranges=wavelength_ranges,
                mask_sigma=2.5,
            )

            snr_full = compute_snr_distribution(signal, noise_full, keep_mask)
            snr_diag = compute_snr_distribution(signal, noise_diag, keep_mask)

            _, sig_full = norm.fit(snr_full)
            _, sig_diag = norm.fit(snr_diag)

            sigma_with_cov.append(sig_full)
            sigma_diag_only.append(sig_diag)

            sigma_ratio = compute_sigma_ratio_distribution(
                noise_full,
                noise_diag,
                keep_mask,
            )

            sigma_ratio_mean.append(np.mean(sigma_ratio))
            sigma_ratio_std.append(np.std(sigma_ratio))

            fit_x.append((n**2) * np.ones(len(sigma_ratio)))
            fit_y.append(sigma_ratio)

        fit_x = np.concatenate(fit_x)
        fit_y = np.concatenate(fit_y)

        sigma_with_cov = np.array(sigma_with_cov)
        sigma_diag_only = np.array(sigma_diag_only)
        sigma_ratio_mean = np.array(sigma_ratio_mean)
        sigma_ratio_std = np.array(sigma_ratio_std)

        fit_model = covar_curve()
        fitter = fitting.LevMarLSQFitter()
        fitted_curve = fitter(fit_model, fit_x, fit_y)

        fig_cal = make_covariance_calibration_figure(
            fit_x=fit_x,
            fit_y=fit_y,
            kernel_sizes=calibration_kernel_sizes,
            sigma_ratio_mean=sigma_ratio_mean,
            sigma_ratio_std=sigma_ratio_std,
            fitted_curve=fitted_curve,
        )
        pdf.savefig(fig_cal, dpi=300, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig_cal)

    return CovarianceTestResult(
        flux_path=flux_path,
        var_path=var_path,
        cov_data_path=cov_data_path,
        cov_coord_path=cov_coord_path,
        histogram_kernel_sizes=histogram_kernel_sizes,
        calibration_kernel_sizes=calibration_kernel_sizes,
        sigma_with_cov=sigma_with_cov,
        sigma_diag_only=sigma_diag_only,
        sigma_ratio_mean=sigma_ratio_mean,
        sigma_ratio_std=sigma_ratio_std,
        fitted_alpha=float(fitted_curve.alpha.value),
        fitted_norm=float(fitted_curve.norm.value),
        fitted_thresh=float(fitted_curve.thresh.value),
        output_pdf_path=output_pdf_path,
    )