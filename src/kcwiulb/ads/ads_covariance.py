from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting
from astropy.stats import sigma_clip
from scipy.stats import norm

from kcwiulb.coadd.covariance_test import (
    _compute_kernel_products,
    apply_wavelength_ranges,
    compute_sigma_ratio_distribution,
    covar_curve,
    load_coadd_products,
)


@dataclass
class ADSCovarianceResult:
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

    fit_x: np.ndarray
    fit_y: np.ndarray

    fitted_alpha: float
    fitted_norm: float
    fitted_thresh: float

    snr_first_check: np.ndarray


def patch_covariance_diagonal_from_variance(
    cov_data: np.ndarray,
    cov_coord: np.ndarray,
    var_diag: np.ndarray,
) -> np.ndarray:
    cov_data_patched = cov_data.copy()

    n_w, n_y, n_x = var_diag.shape
    coord_dict = {
        (int(c0), int(c1)): i
        for i, (c0, c1) in enumerate(cov_coord)
    }

    for y in range(n_y):
        for x in range(n_x):
            pix = n_x * y + x
            idx = coord_dict.get((pix, pix))
            if idx is not None:
                cov_data_patched[idx] = var_diag[:, y, x]

    return cov_data_patched


def compute_first_verification_snr(
    flux: np.ndarray,
    header: fits.Header,
    var_diag: np.ndarray,
    wavelength_ranges: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    signal = apply_wavelength_ranges(flux, header, wavelength_ranges)
    noise = apply_wavelength_ranges(var_diag, header, wavelength_ranges)

    d = np.sum(signal, axis=0)
    d_filtered = sigma_clip(d, sigma=2.5, maxiters=None, masked=True)

    c_mask = d_filtered.mask
    e = signal.copy()
    f = noise.copy()

    for w_ in range(signal.shape[0]):
        e[w_] = np.multiply(signal[w_], np.invert(c_mask))
        f[w_] = np.multiply(noise[w_], np.invert(c_mask))

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.divide(e, np.sqrt(f))

    snr = snr[np.abs(snr) <= 5]

    return np.asarray(snr, dtype=float)


def calibrate_ads_covariance(
    flux: np.ndarray,
    header: fits.Header,
    var_diag: np.ndarray,
    cov_data: np.ndarray,
    cov_coord: np.ndarray,
    wavelength_ranges: list[tuple[float, float]] | None = None,
    histogram_kernel_sizes: list[int] | None = None,
    calibration_kernel_sizes: list[int] | None = None,
    patch_diagonal_from_var: bool = True,
    mask_sigma: float = 2.5,
) -> ADSCovarianceResult:
    if histogram_kernel_sizes is None:
        histogram_kernel_sizes = [1]

    if calibration_kernel_sizes is None:
        calibration_kernel_sizes = list(range(1, 12))

    if patch_diagonal_from_var:
        cov_data_use = patch_covariance_diagonal_from_variance(
            cov_data=cov_data,
            cov_coord=cov_coord,
            var_diag=var_diag,
        )
    else:
        cov_data_use = cov_data.copy()

    snr_first_check = compute_first_verification_snr(
        flux=flux,
        header=header,
        var_diag=var_diag,
        wavelength_ranges=wavelength_ranges,
    )

    sigma_with_cov: list[float] = []
    sigma_diag_only: list[float] = []
    sigma_ratio_mean: list[float] = []
    sigma_ratio_std: list[float] = []
    fit_x_parts: list[np.ndarray] = []
    fit_y_parts: list[np.ndarray] = []

    for n in calibration_kernel_sizes:
        signal, noise_full, noise_diag, keep_mask = _compute_kernel_products(
            flux=flux,
            header=header,
            var_diag=var_diag,
            cov_data=cov_data_use,
            cov_coord=cov_coord,
            kernel_size=n,
            wavelength_ranges=wavelength_ranges,
            mask_sigma=mask_sigma,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            snr_full_n = signal / np.sqrt(noise_full)
            snr_diag_n = signal / np.sqrt(noise_diag)

        snr_full_n = snr_full_n * keep_mask[None, :, :]
        snr_diag_n = snr_diag_n * keep_mask[None, :, :]

        snr_full_n = snr_full_n[np.isfinite(snr_full_n)]
        snr_diag_n = snr_diag_n[np.isfinite(snr_diag_n)]

        snr_full_n = snr_full_n[np.abs(snr_full_n) <= 5.0]
        snr_diag_n = snr_diag_n[np.abs(snr_diag_n) <= 5.0]

        _, sig_full = norm.fit(snr_full_n)
        _, sig_diag = norm.fit(snr_diag_n)

        sigma_with_cov.append(float(sig_full))
        sigma_diag_only.append(float(sig_diag))

        sigma_ratio = compute_sigma_ratio_distribution(
            variance_full=noise_full,
            variance_diag=noise_diag,
            keep_mask=keep_mask,
        )

        sigma_ratio_mean.append(float(np.mean(sigma_ratio)))
        sigma_ratio_std.append(float(np.std(sigma_ratio)))

        fit_x_parts.append((n**2) * np.ones(len(sigma_ratio), dtype=float))
        fit_y_parts.append(np.asarray(sigma_ratio, dtype=float))

    fit_x_arr = np.concatenate(fit_x_parts)
    fit_y_arr = np.concatenate(fit_y_parts)

    fitter = fitting.LevMarLSQFitter()
    fit_model = covar_curve()
    fitted_curve = fitter(fit_model, fit_x_arr, fit_y_arr)

    return ADSCovarianceResult(
        flux_path=Path(""),
        var_path=Path(""),
        cov_data_path=Path(""),
        cov_coord_path=Path(""),
        histogram_kernel_sizes=list(histogram_kernel_sizes),
        calibration_kernel_sizes=list(calibration_kernel_sizes),
        sigma_with_cov=np.array(sigma_with_cov, dtype=float),
        sigma_diag_only=np.array(sigma_diag_only, dtype=float),
        sigma_ratio_mean=np.array(sigma_ratio_mean, dtype=float),
        sigma_ratio_std=np.array(sigma_ratio_std, dtype=float),
        fit_x=fit_x_arr,
        fit_y=fit_y_arr,
        fitted_alpha=float(fitted_curve.alpha.value),
        fitted_norm=float(fitted_curve.norm.value),
        fitted_thresh=float(fitted_curve.thresh.value),
        snr_first_check=np.asarray(snr_first_check, dtype=float),
    )


def calibrate_ads_covariance_from_paths(
    flux_path: str | Path,
    var_path: str | Path,
    cov_data_path: str | Path,
    cov_coord_path: str | Path,
    wavelength_ranges: list[tuple[float, float]] | None = None,
    histogram_kernel_sizes: list[int] | None = None,
    calibration_kernel_sizes: list[int] | None = None,
    patch_diagonal_from_var: bool = True,
    mask_sigma: float = 2.5,
) -> ADSCovarianceResult:
    flux_path = Path(flux_path)
    var_path = Path(var_path)
    cov_data_path = Path(cov_data_path)
    cov_coord_path = Path(cov_coord_path)

    flux, header, var_diag, cov_data, cov_coord = load_coadd_products(
        flux_path=flux_path,
        var_path=var_path,
        cov_data_path=cov_data_path,
        cov_coord_path=cov_coord_path,
    )

    result = calibrate_ads_covariance(
        flux=flux,
        header=header,
        var_diag=var_diag,
        cov_data=cov_data,
        cov_coord=cov_coord,
        wavelength_ranges=wavelength_ranges,
        histogram_kernel_sizes=histogram_kernel_sizes,
        calibration_kernel_sizes=calibration_kernel_sizes,
        patch_diagonal_from_var=patch_diagonal_from_var,
        mask_sigma=mask_sigma,
    )

    result.flux_path = flux_path
    result.var_path = var_path
    result.cov_data_path = cov_data_path
    result.cov_coord_path = cov_coord_path

    return result