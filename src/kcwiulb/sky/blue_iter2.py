from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit

from kcwiulb.sky.utils import (
    build_wavelength_axis,
    combine_masks_2d,
    load_cube,
    mask2d_to_mask3d,
    sigma_clip_mask_2d,
    whiteband_image,
    weighted_median_filter_1d,
    write_cube,
)


@dataclass
class BlueIter2Result:
    science_path: Path
    output_path_sky: Path
    output_path_sky2: Path
    wavelength: np.ndarray

    region_bounds: list[tuple[int, int]]
    region_wavelength_bounds: list[tuple[float, float]]
    params_list: list[np.ndarray]
    pcov_list: list[np.ndarray | None]

    master_mask: np.ndarray
    fit_residual: np.ndarray
    fit_residual_regions: list[np.ndarray]
    var_add_list: list[np.ndarray]

    science_whiteband: np.ndarray
    sky1_whiteband: np.ndarray
    sky2_whiteband: np.ndarray
    sky3_whiteband: np.ndarray
    sky4_whiteband: np.ndarray

    science_mask: np.ndarray
    sky1_mask: np.ndarray
    sky2_mask: np.ndarray
    sky3_mask: np.ndarray
    sky4_mask: np.ndarray

    sky1_spec: np.ndarray
    sky2_spec: np.ndarray
    sky3_spec: np.ndarray
    sky4_spec: np.ndarray
    sky5_spec: np.ndarray

    sky1_spec1: np.ndarray
    sky2_spec1: np.ndarray
    sky3_spec1: np.ndarray
    sky4_spec1: np.ndarray
    sky5_spec1: np.ndarray

    sky1_spec2: np.ndarray
    sky2_spec2: np.ndarray
    sky3_spec2: np.ndarray
    sky4_spec2: np.ndarray
    sky5_spec2: np.ndarray

    sky1_spec3: np.ndarray
    sky2_spec3: np.ndarray
    sky3_spec3: np.ndarray
    sky4_spec3: np.ndarray
    sky5_spec3: np.ndarray

    sky1_spec_cfw1: np.ndarray
    sky2_spec_cfw1: np.ndarray
    sky3_spec_cfw1: np.ndarray
    sky4_spec_cfw1: np.ndarray
    sky5_spec_cfw1: np.ndarray

    sky1_spec_cfw2: np.ndarray
    sky2_spec_cfw2: np.ndarray
    sky3_spec_cfw2: np.ndarray
    sky4_spec_cfw2: np.ndarray
    sky5_spec_cfw2: np.ndarray

    sky1_spec_cfw3: np.ndarray
    sky2_spec_cfw3: np.ndarray
    sky3_spec_cfw3: np.ndarray
    sky4_spec_cfw3: np.ndarray
    sky5_spec_cfw3: np.ndarray

    sky1_res_cfw1: np.ndarray
    sky2_res_cfw1: np.ndarray
    sky3_res_cfw1: np.ndarray
    sky4_res_cfw1: np.ndarray
    sky5_res_cfw1: np.ndarray

    sky1_res_cfw2: np.ndarray
    sky2_res_cfw2: np.ndarray
    sky3_res_cfw2: np.ndarray
    sky4_res_cfw2: np.ndarray
    sky5_res_cfw2: np.ndarray

    sky1_res_cfw3: np.ndarray
    sky2_res_cfw3: np.ndarray
    sky3_res_cfw3: np.ndarray
    sky4_res_cfw3: np.ndarray
    sky5_res_cfw3: np.ndarray

    c11: np.ndarray
    c12: np.ndarray
    c11_uncert: np.ndarray
    c12_uncert: np.ndarray

    wavgood0: float
    wavgood1: float


def sky_model_four(
    x_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    a: float = 0.25,
    b: float = 0.25,
    c: float = 0.25,
    d: float = 0.25,
) -> np.ndarray:
    x1, x2, x3, x4 = x_tuple
    return a * x1 + b * x2 + c * x3 + d * x4


def wl_to_index(wavelength: float, header) -> int:
    return int((wavelength - header["CRVAL3"]) / header["CD3_3"] + header["CRPIX3"])


def _clip_index(index: int, nw: int) -> int:
    return max(0, min(index, nw))


def _build_region_bounds(
    header,
    nw: int,
    fit_margin_blue: float,
    split_wavelengths: list[float] | None,
) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
    if split_wavelengths is None:
        split_wavelengths = []

    split_wavelengths = sorted(split_wavelengths)

    start_wl = float(header["WAVGOOD0"]) + fit_margin_blue
    end_wl = float(header["WAVGOOD1"])

    boundaries_wl = [start_wl] + split_wavelengths + [end_wl]
    boundaries_idx = [_clip_index(wl_to_index(w, header), nw) for w in boundaries_wl]

    region_bounds: list[tuple[int, int]] = []
    region_wavelength_bounds: list[tuple[float, float]] = []

    for i in range(len(boundaries_idx) - 1):
        i0, i1 = boundaries_idx[i], boundaries_idx[i + 1]
        w0, w1 = boundaries_wl[i], boundaries_wl[i + 1]

        if i1 <= i0:
            continue

        region_bounds.append((i0, i1))
        region_wavelength_bounds.append((w0, w1))

    if not region_bounds:
        raise ValueError(
            "No valid iter2 wavelength regions were created. "
            f"split_wavelengths={split_wavelengths}, "
            f"WAVGOOD0={header['WAVGOOD0']}, WAVGOOD1={header['WAVGOOD1']}"
        )

    return region_bounds, region_wavelength_bounds


def subtract_blue_iter2(
    science_cropped_path: str | Path,
    first_pass_path: str | Path,
    sky_paths: list[str | Path],
    output_path_sky: str | Path | None = None,
    output_path_sky2: str | Path | None = None,
    collapse_wavelength_ranges: list[tuple[float, float]] | None = None,
    cfwidth: int = 125,
    split_y: int = 12,
    split_wavelengths: list[float] | None = None,
    fit_margin_blue: float = 1.0,
) -> BlueIter2Result:
    """
    Blue iteration 2 sky subtraction.

    Parameters
    ----------
    split_wavelengths
        Internal wavelength boundaries between fit regions.
        Examples:
        - None or []              -> one region
        - [5530]                  -> two regions
        - [4750, 5400, 5530]      -> four regions

        These are used together with:
        - start = WAVGOOD0 + fit_margin_blue
        - end   = WAVGOOD1
    """
    if collapse_wavelength_ranges is None:
        collapse_wavelength_ranges = [(3700, 3980), (4150, 5200)]

    science_cropped_path = Path(science_cropped_path)
    first_pass_path = Path(first_pass_path)
    sky_paths = [Path(p) for p in sky_paths]

    if len(sky_paths) != 4:
        raise ValueError(f"Expected 4 sky paths for iter2, got {len(sky_paths)}")

    c1, h1, c1_uncert = load_cube(science_cropped_path)
    c1_first, _, _ = load_cube(first_pass_path)
    c2, h2, c2_uncert = load_cube(sky_paths[0])
    c3, h3, c3_uncert = load_cube(sky_paths[1])
    c4, h4, c4_uncert = load_cube(sky_paths[2])
    c5, h5, c5_uncert = load_cube(sky_paths[3])

    wl = build_wavelength_axis(h1, c1.shape[0])
    nw = c1.shape[0]

    # --------------------------------------------------------
    # White-band images + masks
    # --------------------------------------------------------
    d1 = whiteband_image(c1_first, h1, collapse_wavelength_ranges)
    d2 = whiteband_image(c2, h2, collapse_wavelength_ranges)
    d3 = whiteband_image(c3, h3, collapse_wavelength_ranges)
    d4 = whiteband_image(c4, h4, collapse_wavelength_ranges)
    d5 = whiteband_image(c5, h5, collapse_wavelength_ranges)

    m1 = sigma_clip_mask_2d(d1, sigma=3.0)
    m2 = sigma_clip_mask_2d(d2, sigma=3.0)
    m3 = sigma_clip_mask_2d(d3, sigma=3.0)
    m4 = sigma_clip_mask_2d(d4, sigma=3.0)
    m5 = sigma_clip_mask_2d(d5, sigma=3.0)

    master_mask = combine_masks_2d(m1, m2, m3, m4, m5)
    master_mask_3d = mask2d_to_mask3d(master_mask, c1.shape)

    c1_masked = ma.masked_array(c1, mask=master_mask_3d)
    c2_masked = ma.masked_array(c2, mask=master_mask_3d)
    c3_masked = ma.masked_array(c3, mask=master_mask_3d)
    c4_masked = ma.masked_array(c4, mask=master_mask_3d)
    c5_masked = ma.masked_array(c5, mask=master_mask_3d)

    # --------------------------------------------------------
    # Median sky spectra
    # --------------------------------------------------------
    sky1_spec = np.ma.median(c1_masked, axis=(1, 2)).filled(np.nan)
    sky2_spec = np.ma.median(c2_masked, axis=(1, 2)).filled(np.nan)
    sky3_spec = np.ma.median(c3_masked, axis=(1, 2)).filled(np.nan)
    sky4_spec = np.ma.median(c4_masked, axis=(1, 2)).filled(np.nan)
    sky5_spec = np.ma.median(c5_masked, axis=(1, 2)).filled(np.nan)

    sky1_spec_std2 = np.nanmedian(c1_uncert, axis=(1, 2))
    sky2_spec_std2 = np.nanmedian(c2_uncert, axis=(1, 2))
    sky3_spec_std2 = np.nanmedian(c3_uncert, axis=(1, 2))
    sky4_spec_std2 = np.nanmedian(c4_uncert, axis=(1, 2))
    sky5_spec_std2 = np.nanmedian(c5_uncert, axis=(1, 2))

    # --------------------------------------------------------
    # Region-specific spectra in y
    # --------------------------------------------------------
    sky1_spec1 = np.ma.median(c1_masked[:, :, 1:split_y], axis=(1, 2)).filled(np.nan)
    sky2_spec1 = np.ma.median(c2_masked[:, :, 1:split_y], axis=(1, 2)).filled(np.nan)
    sky3_spec1 = np.ma.median(c3_masked[:, :, 1:split_y], axis=(1, 2)).filled(np.nan)
    sky4_spec1 = np.ma.median(c4_masked[:, :, 1:split_y], axis=(1, 2)).filled(np.nan)
    sky5_spec1 = np.ma.median(c5_masked[:, :, 1:split_y], axis=(1, 2)).filled(np.nan)

    sky1_spec2 = np.ma.median(c1_masked[:, :, split_y:], axis=(1, 2)).filled(np.nan)
    sky2_spec2 = np.ma.median(c2_masked[:, :, split_y:], axis=(1, 2)).filled(np.nan)
    sky3_spec2 = np.ma.median(c3_masked[:, :, split_y:], axis=(1, 2)).filled(np.nan)
    sky4_spec2 = np.ma.median(c4_masked[:, :, split_y:], axis=(1, 2)).filled(np.nan)
    sky5_spec2 = np.ma.median(c5_masked[:, :, split_y:], axis=(1, 2)).filled(np.nan)

    sky1_spec3 = np.ma.median(c1_masked[:, :, 0], axis=1).filled(np.nan)
    sky2_spec3 = np.ma.median(c2_masked[:, :, 0], axis=1).filled(np.nan)
    sky3_spec3 = np.ma.median(c3_masked[:, :, 0], axis=1).filled(np.nan)
    sky4_spec3 = np.ma.median(c4_masked[:, :, 0], axis=1).filled(np.nan)
    sky5_spec3 = np.ma.median(c5_masked[:, :, 0], axis=1).filled(np.nan)

    # --------------------------------------------------------
    # Continuum filtering
    # --------------------------------------------------------
    sky1_spec_cfw1 = weighted_median_filter_1d(sky1_spec1, sky1_spec_std2, width=cfwidth)
    sky2_spec_cfw1 = weighted_median_filter_1d(sky2_spec1, sky2_spec_std2, width=cfwidth)
    sky3_spec_cfw1 = weighted_median_filter_1d(sky3_spec1, sky3_spec_std2, width=cfwidth)
    sky4_spec_cfw1 = weighted_median_filter_1d(sky4_spec1, sky4_spec_std2, width=cfwidth)
    sky5_spec_cfw1 = weighted_median_filter_1d(sky5_spec1, sky5_spec_std2, width=cfwidth)

    sky1_spec_cfw2 = weighted_median_filter_1d(sky1_spec2, sky1_spec_std2, width=cfwidth)
    sky2_spec_cfw2 = weighted_median_filter_1d(sky2_spec2, sky2_spec_std2, width=cfwidth)
    sky3_spec_cfw2 = weighted_median_filter_1d(sky3_spec2, sky3_spec_std2, width=cfwidth)
    sky4_spec_cfw2 = weighted_median_filter_1d(sky4_spec2, sky4_spec_std2, width=cfwidth)
    sky5_spec_cfw2 = weighted_median_filter_1d(sky5_spec2, sky5_spec_std2, width=cfwidth)

    sky1_spec_cfw3 = weighted_median_filter_1d(sky1_spec3, sky1_spec_std2, width=cfwidth)
    sky2_spec_cfw3 = weighted_median_filter_1d(sky2_spec3, sky2_spec_std2, width=cfwidth)
    sky3_spec_cfw3 = weighted_median_filter_1d(sky3_spec3, sky3_spec_std2, width=cfwidth)
    sky4_spec_cfw3 = weighted_median_filter_1d(sky4_spec3, sky4_spec_std2, width=cfwidth)
    sky5_spec_cfw3 = weighted_median_filter_1d(sky5_spec3, sky5_spec_std2, width=cfwidth)

    # --------------------------------------------------------
    # Residual spectra after continuum removal
    # --------------------------------------------------------
    sky1_res_cfw1 = sky1_spec1 - sky1_spec_cfw1
    sky2_res_cfw1 = sky2_spec1 - sky2_spec_cfw1
    sky3_res_cfw1 = sky3_spec1 - sky3_spec_cfw1
    sky4_res_cfw1 = sky4_spec1 - sky4_spec_cfw1
    sky5_res_cfw1 = sky5_spec1 - sky5_spec_cfw1

    sky1_res_cfw2 = sky1_spec2 - sky1_spec_cfw2
    sky2_res_cfw2 = sky2_spec2 - sky2_spec_cfw2
    sky3_res_cfw2 = sky3_spec2 - sky3_spec_cfw2
    sky4_res_cfw2 = sky4_spec2 - sky4_spec_cfw2
    sky5_res_cfw2 = sky5_spec2 - sky5_spec_cfw2

    sky1_res_cfw3 = sky1_spec3 - sky1_spec_cfw3
    sky2_res_cfw3 = sky2_spec3 - sky2_spec_cfw3
    sky3_res_cfw3 = sky3_spec3 - sky3_spec_cfw3
    sky4_res_cfw3 = sky4_spec3 - sky4_spec_cfw3
    sky5_res_cfw3 = sky5_spec3 - sky5_spec_cfw3

    # --------------------------------------------------------
    # Build full residual cubes using notebook region logic
    # --------------------------------------------------------
    c1_res = np.empty_like(c1)
    c2_res = np.empty_like(c2)
    c3_res = np.empty_like(c3)
    c4_res = np.empty_like(c4)
    c5_res = np.empty_like(c5)

    nx = c1.shape[1]
    ny = c1.shape[2]

    for x_ in range(nx):
        for y_ in range(ny):
            if y_ == 0:
                c1_res[:, x_, y_] = c1[:, x_, y_] - sky1_spec_cfw3
                c2_res[:, x_, y_] = c2[:, x_, y_] - sky2_spec_cfw3
                c3_res[:, x_, y_] = c3[:, x_, y_] - sky3_spec_cfw3
                c4_res[:, x_, y_] = c4[:, x_, y_] - sky4_spec_cfw3
                c5_res[:, x_, y_] = c5[:, x_, y_] - sky5_spec_cfw3
            elif y_ < split_y:
                c1_res[:, x_, y_] = c1[:, x_, y_] - sky1_spec_cfw1
                c2_res[:, x_, y_] = c2[:, x_, y_] - sky2_spec_cfw1
                c3_res[:, x_, y_] = c3[:, x_, y_] - sky3_spec_cfw1
                c4_res[:, x_, y_] = c4[:, x_, y_] - sky4_spec_cfw1
                c5_res[:, x_, y_] = c5[:, x_, y_] - sky5_spec_cfw1
            else:
                c1_res[:, x_, y_] = c1[:, x_, y_] - sky1_spec_cfw2
                c2_res[:, x_, y_] = c2[:, x_, y_] - sky2_spec_cfw2
                c3_res[:, x_, y_] = c3[:, x_, y_] - sky3_spec_cfw2
                c4_res[:, x_, y_] = c4[:, x_, y_] - sky4_spec_cfw2
                c5_res[:, x_, y_] = c5[:, x_, y_] - sky5_spec_cfw2

    c1_res_masked = ma.masked_array(c1_res, mask=master_mask_3d)
    c2_res_masked = ma.masked_array(c2_res, mask=master_mask_3d)
    c3_res_masked = ma.masked_array(c3_res, mask=master_mask_3d)
    c4_res_masked = ma.masked_array(c4_res, mask=master_mask_3d)
    c5_res_masked = ma.masked_array(c5_res, mask=master_mask_3d)

    # --------------------------------------------------------
    # Flexible wavelength-region fits
    # --------------------------------------------------------
    region_bounds, region_wavelength_bounds = _build_region_bounds(
        header=h1,
        nw=nw,
        fit_margin_blue=fit_margin_blue,
        split_wavelengths=split_wavelengths,
    )

    params_list: list[np.ndarray] = []
    pcov_list: list[np.ndarray | None] = []
    fit_residual_regions: list[np.ndarray] = []
    var_add_list: list[np.ndarray] = []

    for i0, i1 in region_bounds:
        x1 = np.ma.median(c2_res_masked[i0:i1], axis=(1, 2)).filled(np.nan)
        x2 = np.ma.median(c3_res_masked[i0:i1], axis=(1, 2)).filled(np.nan)
        x3 = np.ma.median(c4_res_masked[i0:i1], axis=(1, 2)).filled(np.nan)
        x4 = np.ma.median(c5_res_masked[i0:i1], axis=(1, 2)).filled(np.nan)
        y1 = np.ma.median(c1_res_masked[i0:i1], axis=(1, 2)).filled(np.nan)

        valid = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(x3) & np.isfinite(x4) & np.isfinite(y1)
        if np.count_nonzero(valid) < 5:
            raise ValueError(
                f"Too few valid spectral samples in iter2 fit region {i0}:{i1}. "
                f"Region wavelengths: {wl[i0]:.2f}-{wl[i1 - 1]:.2f}"
            )

        params, pcov = curve_fit(
            sky_model_four,
            (x1[valid], x2[valid], x3[valid], x4[valid]),
            y1[valid],
            maxfev=5000,
        )
        params_list.append(params)
        pcov_list.append(pcov)

        # choose the y-region residual family based on wavelength region index
        # keep notebook-like behavior for up to 3 regions; beyond that, reuse the last family
        region_idx = min(len(params_list), 3)

        if region_idx == 1:
            fit_residual_region = sky1_res_cfw1 - sky_model_four(
                (sky2_res_cfw1, sky3_res_cfw1, sky4_res_cfw1, sky5_res_cfw1), *params
            )
        elif region_idx == 2:
            fit_residual_region = sky1_res_cfw2 - sky_model_four(
                (sky2_res_cfw2, sky3_res_cfw2, sky4_res_cfw2, sky5_res_cfw2), *params
            )
        else:
            fit_residual_region = sky1_res_cfw3 - sky_model_four(
                (sky2_res_cfw3, sky3_res_cfw3, sky4_res_cfw3, sky5_res_cfw3), *params
            )

        fit_residual_regions.append(fit_residual_region)

        var_add = (
            np.square(params[0]) * np.square(sky2_spec_std2)
            + np.square(params[1]) * np.square(sky3_spec_std2)
            + np.square(params[2]) * np.square(sky4_spec_std2)
            + np.square(params[3]) * np.square(sky5_spec_std2)
        )
        var_add_list.append(var_add)

    # stitch global fit residual
    fit_residual = np.zeros_like(wl)
    for (i0, i1), region_res in zip(region_bounds, fit_residual_regions):
        fit_residual[i0:i1] = region_res[i0:i1]

    # pad legacy fields for compatibility with plotting code
    zero_spec = np.zeros_like(wl)
    params1 = params_list[0]
    pcov1 = pcov_list[0]
    fit_residual1 = fit_residual_regions[0]
    var_add1 = var_add_list[0]

    params2 = params_list[1] if len(params_list) > 1 else params1.copy()
    pcov2 = pcov_list[1] if len(pcov_list) > 1 else pcov1
    fit_residual2 = fit_residual_regions[1] if len(fit_residual_regions) > 1 else zero_spec.copy()
    var_add2 = var_add_list[1] if len(var_add_list) > 1 else np.zeros_like(var_add1)

    params3 = params_list[2] if len(params_list) > 2 else params2.copy()
    pcov3 = pcov_list[2] if len(pcov_list) > 2 else pcov2
    fit_residual3 = fit_residual_regions[2] if len(fit_residual_regions) > 2 else zero_spec.copy()
    var_add3 = var_add_list[2] if len(var_add_list) > 2 else np.zeros_like(var_add1)

    # --------------------------------------------------------
    # Notebook-style cube construction
    # --------------------------------------------------------
    c11_regions: list[np.ndarray] = []
    c12_regions: list[np.ndarray] = []
    c11_uncert_regions: list[np.ndarray] = []
    c12_uncert_regions: list[np.ndarray] = []

    def _region_residual_vectors(y_index: int):
        if y_index == 0:
            return sky2_res_cfw3, sky3_res_cfw3, sky4_res_cfw3, sky5_res_cfw3
        elif y_index < split_y:
            return sky2_res_cfw1, sky3_res_cfw1, sky4_res_cfw1, sky5_res_cfw1
        else:
            return sky2_res_cfw2, sky3_res_cfw2, sky4_res_cfw2, sky5_res_cfw2

    for region_idx, ((i0, i1), params, var_add) in enumerate(zip(region_bounds, params_list, var_add_list), start=1):
        c11_region = np.empty_like(c1[i0:i1])
        c12_region = np.empty_like(c1[i0:i1])
        c11_uncert_region = np.empty_like(c1_uncert[i0:i1])
        c12_uncert_region = np.empty_like(c1_uncert[i0:i1])

        for x_ in range(nx):
            for y_ in range(ny):
                r2, r3, r4, r5 = _region_residual_vectors(y_)

                c12_region[:, x_, y_] = c1_res[i0:i1, x_, y_] - sky_model_four(
                    (r2[i0:i1], r3[i0:i1], r4[i0:i1], r5[i0:i1]), *params
                )
                c11_region[:, x_, y_] = c1_res[i0:i1, x_, y_] - sky_model_four(
                    (
                        c2_res[i0:i1, x_, y_],
                        c3_res[i0:i1, x_, y_],
                        c4_res[i0:i1, x_, y_],
                        c5_res[i0:i1, x_, y_],
                    ),
                    *params,
                )

                c11_uncert_region[:, x_, y_] = np.sqrt(
                    np.square(c1_uncert[i0:i1, x_, y_])
                    + np.square(params[0]) * np.square(c2_uncert[i0:i1, x_, y_])
                    + np.square(params[1]) * np.square(c3_uncert[i0:i1, x_, y_])
                    + np.square(params[2]) * np.square(c4_uncert[i0:i1, x_, y_])
                    + np.square(params[3]) * np.square(c5_uncert[i0:i1, x_, y_])
                )
                c12_uncert_region[:, x_, y_] = np.sqrt(
                    np.square(c1_uncert[i0:i1, x_, y_]) + var_add[i0:i1]
                )

        c11_regions.append(c11_region)
        c12_regions.append(c12_region)
        c11_uncert_regions.append(c11_uncert_region)
        c12_uncert_regions.append(c12_uncert_region)

    c11 = np.concatenate(c11_regions, axis=0)
    c12 = np.concatenate(c12_regions, axis=0)
    c11_uncert = np.concatenate(c11_uncert_regions, axis=0)
    c12_uncert = np.concatenate(c12_uncert_regions, axis=0)

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------
    if output_path_sky is None:
        output_path_sky = science_cropped_path.with_name(science_cropped_path.stem + ".sky.sky.fits")
    if output_path_sky2 is None:
        output_path_sky2 = science_cropped_path.with_name(science_cropped_path.stem + ".sky.sky2.fits")

    output_path_sky = write_cube(h1, c11, c11_uncert, output_path_sky)
    output_path_sky2 = write_cube(h1, c12, c12_uncert, output_path_sky2)

    return BlueIter2Result(
        science_path=science_cropped_path,
        output_path_sky=output_path_sky,
        output_path_sky2=output_path_sky2,
        wavelength=wl,

        region_bounds=region_bounds,
        region_wavelength_bounds=region_wavelength_bounds,
        params_list=params_list,
        pcov_list=pcov_list,

        master_mask=master_mask,
        fit_residual=fit_residual,
        fit_residual_regions=fit_residual_regions,
        var_add_list=var_add_list,

        science_whiteband=d1,
        sky1_whiteband=d2,
        sky2_whiteband=d3,
        sky3_whiteband=d4,
        sky4_whiteband=d5,

        science_mask=m1,
        sky1_mask=m2,
        sky2_mask=m3,
        sky3_mask=m4,
        sky4_mask=m5,

        sky1_spec=sky1_spec,
        sky2_spec=sky2_spec,
        sky3_spec=sky3_spec,
        sky4_spec=sky4_spec,
        sky5_spec=sky5_spec,

        sky1_spec1=sky1_spec1,
        sky2_spec1=sky2_spec1,
        sky3_spec1=sky3_spec1,
        sky4_spec1=sky4_spec1,
        sky5_spec1=sky5_spec1,

        sky1_spec2=sky1_spec2,
        sky2_spec2=sky2_spec2,
        sky3_spec2=sky3_spec2,
        sky4_spec2=sky4_spec2,
        sky5_spec2=sky5_spec2,

        sky1_spec3=sky1_spec3,
        sky2_spec3=sky2_spec3,
        sky3_spec3=sky3_spec3,
        sky4_spec3=sky4_spec3,
        sky5_spec3=sky5_spec3,

        sky1_spec_cfw1=sky1_spec_cfw1,
        sky2_spec_cfw1=sky2_spec_cfw1,
        sky3_spec_cfw1=sky3_spec_cfw1,
        sky4_spec_cfw1=sky4_spec_cfw1,
        sky5_spec_cfw1=sky5_spec_cfw1,

        sky1_spec_cfw2=sky1_spec_cfw2,
        sky2_spec_cfw2=sky2_spec_cfw2,
        sky3_spec_cfw2=sky3_spec_cfw2,
        sky4_spec_cfw2=sky4_spec_cfw2,
        sky5_spec_cfw2=sky5_spec_cfw2,

        sky1_spec_cfw3=sky1_spec_cfw3,
        sky2_spec_cfw3=sky2_spec_cfw3,
        sky3_spec_cfw3=sky3_spec_cfw3,
        sky4_spec_cfw3=sky4_spec_cfw3,
        sky5_spec_cfw3=sky5_spec_cfw3,

        sky1_res_cfw1=sky1_res_cfw1,
        sky2_res_cfw1=sky2_res_cfw1,
        sky3_res_cfw1=sky3_res_cfw1,
        sky4_res_cfw1=sky4_res_cfw1,
        sky5_res_cfw1=sky5_res_cfw1,

        sky1_res_cfw2=sky1_res_cfw2,
        sky2_res_cfw2=sky2_res_cfw2,
        sky3_res_cfw2=sky3_res_cfw2,
        sky4_res_cfw2=sky4_res_cfw2,
        sky5_res_cfw2=sky5_res_cfw2,

        sky1_res_cfw3=sky1_res_cfw3,
        sky2_res_cfw3=sky2_res_cfw3,
        sky3_res_cfw3=sky3_res_cfw3,
        sky4_res_cfw3=sky4_res_cfw3,
        sky5_res_cfw3=sky5_res_cfw3,

        c11=c11,
        c12=c12,
        c11_uncert=c11_uncert,
        c12_uncert=c12_uncert,

        wavgood0=h1["WAVGOOD0"],
        wavgood1=h1["WAVGOOD1"],
    )