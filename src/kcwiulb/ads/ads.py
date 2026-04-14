from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import convolution
from astropy.io import fits
from scipy.ndimage import convolve as sc_ndi_convolve

from kcwiulb.ads.ads_covariance import calibrate_ads_covariance_from_paths
from kcwiulb.coadd.covariance_test import covar_curve
from kcwiulb.plot.ads_diagnostics import (
    make_covariance_calibration_figure,
    make_snr_histogram_figure,
    save_adaptive_smoothing_diagnostics_pdf,
)
from kcwiulb.coadd.covariance_test import covar_curve


@dataclass
class AdaptiveSmoothingResult:
    flux_input_path: Path
    var_input_path: Path

    ads_flux_path: Path
    ads_mask_path: Path
    ads_snr_path: Path
    kernel_r_path: Path
    kernel_w_path: Path
    diagnostic_pdf_path: Path

    fitted_alpha: float
    fitted_norm: float
    fitted_thresh: float

    f_snr_array: list[float]
    med_snr_array: list[float]
    min_snr_array: list[float]
    max_snr_array: list[float]
    xy_scale_array: list[float]
    z_scale_array: list[float]
    n_det_array: list[int]

    n_det_0: int
    total_voxels: int


def fwhm2sigma(fwhm: float) -> float:
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def build_fitted_line(alpha: float, norm: float, thresh: float):
    return covar_curve(alpha=alpha, norm=norm, thresh=thresh)


def smooth_cube_spatial(
    data: np.ndarray,
    var: np.ndarray,
    scale: float,
    fitted_line,
    ktype: str = "box",
) -> tuple[np.ndarray, np.ndarray]:
    if ktype == "box":
        kernel = convolution.Box2DKernel(scale)
        ker_vol = scale**2
    elif ktype == "gaussian":
        sigma = fwhm2sigma(scale)
        kernel = convolution.Gaussian2DKernel(sigma)
        ker_vol = np.pi * scale**2
    else:
        raise ValueError(f"No kernel type '{ktype}' for spatial smoothing")

    kernel = np.array([kernel.array])
    kernel_var = kernel**2

    data_out = sc_ndi_convolve(data, kernel, mode="constant")
    var_out = sc_ndi_convolve(var, kernel_var, mode="constant")
    var_out *= fitted_line(ker_vol) ** 2

    return data_out, var_out


def smooth_cube_wavelength(
    data: np.ndarray,
    var: np.ndarray,
    scale: float,
    ktype: str = "box",
) -> tuple[np.ndarray, np.ndarray]:
    if ktype == "box":
        kernel = convolution.Box1DKernel(scale)
    elif ktype == "gaussian":
        sigma = fwhm2sigma(scale)
        kernel = convolution.Gaussian1DKernel(sigma)
    else:
        raise ValueError(f"No kernel type '{ktype}' for wavelength smoothing")

    kernel = np.array([[kernel.array]]).T
    kernel_var = kernel**2

    data_out = sc_ndi_convolve(data, kernel, mode="constant")
    var_out = sc_ndi_convolve(var, kernel_var, mode="constant")

    return data_out, var_out


def _build_output_path(input_path: Path, snr_min: float, suffix: str) -> Path:
    snr_tag = f"{snr_min:g}"
    name = input_path.name

    if name.endswith(".fits"):
        prefix = name[: -len(".fits")]
        return input_path.with_name(f"{prefix}.ads{snr_tag}{suffix}")

    raise ValueError(f"Expected FITS path, got {input_path}")


def run_adaptive_smoothing(
    flux_path: str | Path,
    var_path: str | Path,
    cov_data_path: str | Path,
    cov_coord_path: str | Path,
    wavelength_ranges: list[tuple[float, float]],
    snr_min: float = 2.5,
    snr_max: float | None = None,
    xy_range: tuple[float, float] = (1, 50),
    xy_step: float = 1,
    xy_step_min: float = 1,
    z_range: tuple[float, float] = (1, 20),
    z_step: float = 1,
    z_step_min: float = 1,
    kernel_type: str = "box",
    covariance_histogram_kernel_sizes: list[int] | None = None,
    covariance_kernel_sizes: list[int] | None = None,
    overwrite: bool = True,
) -> AdaptiveSmoothingResult:
    flux_path = Path(flux_path)
    var_path = Path(var_path)
    cov_data_path = Path(cov_data_path)
    cov_coord_path = Path(cov_coord_path)

    if snr_max is None:
        snr_max = 2.0 * snr_min

    if covariance_histogram_kernel_sizes is None:
        covariance_histogram_kernel_sizes = [1]

    if covariance_kernel_sizes is None:
        covariance_kernel_sizes = list(range(1, 12))

    with fits.open(flux_path) as hdul:
        header = hdul[0].header.copy()
        cube = hdul[0].data.astype(float)

    with fits.open(var_path) as hdul:
        var_cube = hdul[0].data.astype(float)

    cov_result = calibrate_ads_covariance_from_paths(
        flux_path=flux_path,
        var_path=var_path,
        cov_data_path=cov_data_path,
        cov_coord_path=cov_coord_path,
        wavelength_ranges=wavelength_ranges,
        histogram_kernel_sizes=covariance_histogram_kernel_sizes,
        calibration_kernel_sizes=covariance_kernel_sizes,
        patch_diagonal_from_var=True,
        mask_sigma=2.5,
    )

    fig1 = make_snr_histogram_figure(
        snr=cov_result.snr_first_check,
    )
    plt.show()
    plt.close(fig1)


    fitted_curve = covar_curve(
        alpha=cov_result.fitted_alpha,
        norm=cov_result.fitted_norm,
        thresh=cov_result.fitted_thresh,
    )

    fig2 = make_covariance_calibration_figure(
        fit_x=cov_result.fit_x,
        fit_y=cov_result.fit_y,
        kernel_sizes=cov_result.calibration_kernel_sizes,
        sigma_ratio_mean=cov_result.sigma_ratio_mean,
        sigma_ratio_std=cov_result.sigma_ratio_std,
        fitted_curve=fitted_curve,
    )

    plt.show()
    plt.close(fig2)

    fitted_line = build_fitted_line(
        alpha=cov_result.fitted_alpha,
        norm=cov_result.fitted_norm,
        thresh=cov_result.fitted_thresh,
    )

    icube = cube.copy()
    vcube = var_cube.copy()

    icube_det = np.zeros_like(icube)
    vcube_det = np.zeros_like(icube)
    mcube_det = np.zeros_like(icube)
    snr_det = np.zeros_like(icube)
    kr_vals = np.zeros_like(icube)
    kw_vals = np.zeros_like(icube)

    r_min, r_max = xy_range
    z_min, z_max = z_range

    if r_max > (np.min(icube.shape[1:]) - 1) / 2.0:
        r_max = (np.min(icube.shape[1:]) - 1) / 2.0
    if z_max > (icube.shape[0] - 1) / 2.0:
        z_max = (icube.shape[0] - 1) / 2.0

    mask_xy = np.max(icube, axis=0) == 0
    mcube_det = mcube_det.T
    mcube_det[mask_xy.T] = 1
    mcube_det = mcube_det.T

    n_det_0 = int(np.sum(mcube_det))

    n_det_array: list[int] = []
    f_snr_array: list[float] = []
    med_snr_array: list[float] = []
    min_snr_array: list[float] = []
    max_snr_array: list[float] = []
    xy_scale_array: list[float] = []
    z_scale_array: list[float] = []

    z_scale = z_min
    z_step_cur = z_step
    z_scale_old = z_scale
    n_det = 0

    while z_scale < z_max:
        n_det_z = 0

        xy_scale = r_min
        xy_step_cur = xy_step
        xy_scale_old = xy_scale

        while xy_scale < r_max:
            print(
                "z_scale = ",
                z_scale,
                "z_step = ",
                z_step_cur,
                "xy_scale = ",
                xy_scale,
                "xy_step = ",
                xy_step_cur,
            )

            det_flag = False
            break_flag = False
            f_snr = -1.0

            icube_old = icube.copy()
            vcube_old = vcube.copy()

            min_snr = np.nan
            max_snr = np.nan
            med_snr = np.nan

            icube_xy, vcube_xy = smooth_cube_spatial(
                icube,
                vcube,
                xy_scale,
                fitted_line,
                ktype=kernel_type,
            )
            icube_xyz, vcube_xyz = smooth_cube_wavelength(
                icube_xy,
                vcube_xy,
                z_scale,
                ktype=kernel_type,
            )
            snr_xyz = np.divide(
                icube_xyz,
                np.sqrt(vcube_xyz),
                out=np.zeros_like(icube_xyz, dtype=float),
                where=vcube_xyz > 0,
            )

            detections_neg = (snr_xyz <= -1.0 * snr_min) & (mcube_det == 0)
            icube[detections_neg] -= icube_xyz[detections_neg]

            icube_xy, vcube_xy = smooth_cube_spatial(
                icube,
                vcube,
                xy_scale,
                fitted_line,
                ktype=kernel_type,
            )
            icube_xyz, vcube_xyz = smooth_cube_wavelength(
                icube_xy,
                vcube_xy,
                z_scale,
                ktype=kernel_type,
            )
            snr_xyz = np.divide(
                icube_xyz,
                np.sqrt(vcube_xyz),
                out=np.zeros_like(icube_xyz, dtype=float),
                where=vcube_xyz > 0,
            )

            detections = (snr_xyz >= snr_min) & (mcube_det == 0)
            snrs_det = snr_xyz[detections]
            n_vox = len(snrs_det)

            if n_vox >= 5:
                med_snr = float(np.median(snrs_det))
                f_snr = float((snr_min + snr_max) / (2.0 * med_snr))

                if f_snr < 1:
                    if xy_scale > r_min:
                        print("condition 1.1.1")
                        xy_step_cur = (xy_scale - xy_scale_old) / 2.0
                        if xy_step_cur < xy_step_min:
                            xy_step_cur = xy_step_min
                        xy_scale -= xy_step_cur
                        if xy_scale < r_min:
                            xy_scale = r_min

                    elif z_scale > z_min:
                        print("condition 1.1.2")
                        z_step_cur = (z_scale - z_scale_old) / 2.0
                        if z_step_cur < z_step_min:
                            z_step_cur = z_step_min
                        z_scale -= z_step_cur
                        if z_scale < z_min:
                            z_scale = z_min
                        break_flag = True

                    else:
                        print("condition 1.1.3")
                        xy_scale_old = xy_scale
                        det_flag = True
                        xy_step_cur *= 0.5
                        if xy_step_cur < xy_step_min:
                            xy_step_cur = xy_step_min
                        xy_scale += xy_step_cur

                elif f_snr > 1:
                    print("condition 1.2")
                    if xy_scale == r_min:
                        z_scale_old = z_scale
                        z_step_cur = (f_snr - 1.0) * z_scale_old
                        if z_step_cur < z_step_min:
                            z_step_cur = z_step_min

                    xy_scale_old = xy_scale
                    xy_step_cur = (f_snr - 1.0) * xy_scale_old
                    if xy_step_cur < xy_step_min:
                        xy_step_cur = xy_step_min
                    xy_scale += xy_step_cur
                    det_flag = True

                else:
                    print("condition 1.3")
                    xy_scale_old = xy_scale
                    xy_scale += xy_step_cur
                    det_flag = True

            elif n_vox > 0:
                print("condition 2")
                xy_scale_old = xy_scale
                xy_step_cur *= 1.25
                xy_scale += xy_step_cur
                det_flag = True

            else:
                print("condition 3")
                xy_scale_old = xy_scale
                xy_step_cur *= 1.5
                xy_scale += xy_step_cur

            if det_flag:
                icube_det[detections] = icube_xyz[detections]
                vcube_det[detections] = vcube_xyz[detections]
                mcube_det[detections] = 1
                kr_vals[detections] = xy_scale
                kw_vals[detections] = z_scale
                snr_det[detections] = snr_xyz[detections]

                icube[detections] -= icube_xyz[detections]
                vcube[detections] += vcube_xyz[detections]

                xy_scale_array.append(float(xy_scale_old))
                z_scale_array.append(float(z_scale_old))
                f_snr_array.append(float(f_snr))

                if n_vox > 0:
                    min_snr = float(np.min(snrs_det))
                    max_snr = float(np.max(snrs_det))
                    if n_vox >= 5:
                        med_snr = float(np.median(snrs_det))
                    else:
                        med_snr = float(np.mean(snrs_det))

                min_snr_array.append(min_snr)
                max_snr_array.append(max_snr)
                med_snr_array.append(med_snr)

                n_det += n_vox
                n_det_z += n_vox
                n_det_array.append(int(n_det))

                perc = 100.0 * n_det / (icube.size - n_det_0)
                print("n_det = ", n_det, "perc = ", perc)

            else:
                icube = icube_old
                vcube = vcube_old

            if break_flag:
                break

        if n_det_z < 5:
            z_step_cur *= 2.0
        z_scale += z_step_cur

    ads_flux_path = _build_output_path(flux_path, snr_min, ".fits")
    ads_mask_path = _build_output_path(flux_path, snr_min, ".mask.fits")
    ads_snr_path = _build_output_path(flux_path, snr_min, ".snr.fits")
    kernel_r_path = _build_output_path(flux_path, snr_min, ".kernelr.fits")
    kernel_w_path = _build_output_path(flux_path, snr_min, ".kernelw.fits")
    diagnostic_pdf_path = _build_output_path(flux_path, snr_min, ".diagnostic.pdf")

    fits.PrimaryHDU(icube_det, header=header).writeto(ads_flux_path, overwrite=overwrite)
    fits.PrimaryHDU(mcube_det, header=header).writeto(ads_mask_path, overwrite=overwrite)
    fits.PrimaryHDU(snr_det, header=header).writeto(ads_snr_path, overwrite=overwrite)
    fits.PrimaryHDU(kr_vals, header=header).writeto(kernel_r_path, overwrite=overwrite)
    fits.PrimaryHDU(kw_vals, header=header).writeto(kernel_w_path, overwrite=overwrite)

    result = AdaptiveSmoothingResult(
        flux_input_path=flux_path,
        var_input_path=var_path,
        ads_flux_path=ads_flux_path,
        ads_mask_path=ads_mask_path,
        ads_snr_path=ads_snr_path,
        kernel_r_path=kernel_r_path,
        kernel_w_path=kernel_w_path,
        diagnostic_pdf_path=diagnostic_pdf_path,
        fitted_alpha=cov_result.fitted_alpha,
        fitted_norm=cov_result.fitted_norm,
        fitted_thresh=cov_result.fitted_thresh,
        f_snr_array=f_snr_array,
        med_snr_array=med_snr_array,
        min_snr_array=min_snr_array,
        max_snr_array=max_snr_array,
        xy_scale_array=xy_scale_array,
        z_scale_array=z_scale_array,
        n_det_array=n_det_array,
        n_det_0=n_det_0,
        total_voxels=int(icube.size),
    )

    save_adaptive_smoothing_diagnostics_pdf(
        result=result,
        snr_first_check=cov_result.snr_first_check,
        fit_x=cov_result.fit_x,
        fit_y=cov_result.fit_y,
        calibration_kernel_sizes=cov_result.calibration_kernel_sizes,
        sigma_ratio_mean=cov_result.sigma_ratio_mean,
        sigma_ratio_std=cov_result.sigma_ratio_std,
        fitted_alpha=cov_result.fitted_alpha,
        fitted_norm=cov_result.fitted_norm,
        fitted_thresh=cov_result.fitted_thresh,
        snr_min=snr_min,
        snr_max=snr_max,
        r_max=r_max,
        z_max=z_max,
        output_path=diagnostic_pdf_path,
    )

    return result