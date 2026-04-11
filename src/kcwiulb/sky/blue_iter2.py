from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
from scipy.ndimage import generic_filter

from kcwiulb.sky.utils import (
    build_wavelength_axis,
    combine_masks_2d,
    load_cube,
    mask2d_to_mask3d,
    sigma_clip_mask_2d,
    whiteband_image,
    write_cube,
)


@dataclass
class BlueIter2Result:
    science_path: Path
    output_path: Path
    wavelength: np.ndarray
    params_left: np.ndarray
    params_right: np.ndarray
    master_mask: np.ndarray
    fit_residual: np.ndarray


def weighted_quantile(values, quantiles=0.5, sample_weight=None):
    values = np.array(values).flatten()
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    sample_weight = np.array(sample_weight).flatten()

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


def weighted_median_filter_1d(spec: np.ndarray, spec_unc: np.ndarray, width: int = 125, quantile: float = 0.3):
    return generic_filter(
        spec,
        weighted_quantile,
        size=width,
        mode="reflect",
        extra_keywords={
            "quantiles": quantile,
            "sample_weight": np.abs(spec_unc),
        },
    )


def sky_model_four(x_tuple, a, b, c, d):
    x1, x2, x3, x4 = x_tuple
    return a * x1 + b * x2 + c * x3 + d * x4


def subtract_blue_iter2(
    science_cropped_path: str | Path,
    first_pass_path: str | Path,
    sky_paths: list[str | Path],
    output_path: str | Path | None,
    collapse_wavelength_ranges: list[tuple[float, float]],
    cfwidth: int = 125,
) -> BlueIter2Result:
    science_cropped_path = Path(science_cropped_path)
    first_pass_path = Path(first_pass_path)
    sky_paths = [Path(p) for p in sky_paths]

    c1, h1, c1u = load_cube(science_cropped_path)
    c1_first, _, _ = load_cube(first_pass_path)
    c2, h2, c2u = load_cube(sky_paths[0])
    c3, h3, c3u = load_cube(sky_paths[1])
    c4, h4, c4u = load_cube(sky_paths[2])
    c5, h5, c5u = load_cube(sky_paths[3])

    wl = build_wavelength_axis(h1, c1.shape[0])

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

    c1m = ma.masked_array(c1, mask=master_mask_3d)
    c2m = ma.masked_array(c2, mask=master_mask_3d)
    c3m = ma.masked_array(c3, mask=master_mask_3d)
    c4m = ma.masked_array(c4, mask=master_mask_3d)
    c5m = ma.masked_array(c5, mask=master_mask_3d)

    # notebook-style left/right spatial split
    s1_left = np.ma.median(c1m[:, :, 1:12], axis=(1, 2)).filled(np.nan)
    s2_left = np.ma.median(c2m[:, :, 1:12], axis=(1, 2)).filled(np.nan)
    s3_left = np.ma.median(c3m[:, :, 1:12], axis=(1, 2)).filled(np.nan)
    s4_left = np.ma.median(c4m[:, :, 1:12], axis=(1, 2)).filled(np.nan)
    s5_left = np.ma.median(c5m[:, :, 1:12], axis=(1, 2)).filled(np.nan)

    s1_right = np.ma.median(c1m[:, :, 12:], axis=(1, 2)).filled(np.nan)
    s2_right = np.ma.median(c2m[:, :, 12:], axis=(1, 2)).filled(np.nan)
    s3_right = np.ma.median(c3m[:, :, 12:], axis=(1, 2)).filled(np.nan)
    s4_right = np.ma.median(c4m[:, :, 12:], axis=(1, 2)).filled(np.nan)
    s5_right = np.ma.median(c5m[:, :, 12:], axis=(1, 2)).filled(np.nan)

    # simple spectral uncertainties from sky cubes
    s2u = np.nanmedian(c2u, axis=(1, 2))
    s3u = np.nanmedian(c3u, axis=(1, 2))
    s4u = np.nanmedian(c4u, axis=(1, 2))
    s5u = np.nanmedian(c5u, axis=(1, 2))

    s1_left_c = weighted_median_filter_1d(s1_left, s2u, width=cfwidth)
    s2_left_c = weighted_median_filter_1d(s2_left, s2u, width=cfwidth)
    s3_left_c = weighted_median_filter_1d(s3_left, s3u, width=cfwidth)
    s4_left_c = weighted_median_filter_1d(s4_left, s4u, width=cfwidth)
    s5_left_c = weighted_median_filter_1d(s5_left, s5u, width=cfwidth)

    s1_right_c = weighted_median_filter_1d(s1_right, s2u, width=cfwidth)
    s2_right_c = weighted_median_filter_1d(s2_right, s2u, width=cfwidth)
    s3_right_c = weighted_median_filter_1d(s3_right, s3u, width=cfwidth)
    s4_right_c = weighted_median_filter_1d(s4_right, s4u, width=cfwidth)
    s5_right_c = weighted_median_filter_1d(s5_right, s5u, width=cfwidth)

    r1_left = s1_left - s1_left_c
    r2_left = s2_left - s2_left_c
    r3_left = s3_left - s3_left_c
    r4_left = s4_left - s4_left_c
    r5_left = s5_left - s5_left_c

    r1_right = s1_right - s1_right_c
    r2_right = s2_right - s2_right_c
    r3_right = s3_right - s3_right_c
    r4_right = s4_right - s4_right_c
    r5_right = s5_right - s5_right_c

    ind1 = max(int((h1["WAVGOOD0"] + 1 - h1["CRVAL3"]) / h1["CD3_3"] + h1["CRPIX3"]), 0)
    ind2 = int((h1["WAVGOOD1"] - h1["CRVAL3"]) / h1["CD3_3"] + h1["CRPIX3"])

    params_left, _ = curve_fit(
        sky_model_four,
        (r2_left[ind1:ind2], r3_left[ind1:ind2], r4_left[ind1:ind2], r5_left[ind1:ind2]),
        r1_left[ind1:ind2],
        maxfev=10000,
    )
    params_right, _ = curve_fit(
        sky_model_four,
        (r2_right[ind1:ind2], r3_right[ind1:ind2], r4_right[ind1:ind2], r5_right[ind1:ind2]),
        r1_right[ind1:ind2],
        maxfev=10000,
    )

    fit_left = sky_model_four((r2_left, r3_left, r4_left, r5_left), *params_left)
    fit_right = sky_model_four((r2_right, r3_right, r4_right, r5_right), *params_right)

    fit_residual = 0.5 * (fit_left + fit_right)

    var_add = (
        np.square(params_left[0]) * np.square(s2u)
        + np.square(params_left[1]) * np.square(s3u)
        + np.square(params_left[2]) * np.square(s4u)
        + np.square(params_left[3]) * np.square(s5u)
    )

    c_out = c1_first - fit_residual[:, None, None]
    c_out_u = np.sqrt(np.square(c1u) + var_add[:, None, None])

    if output_path is None:
        output_path = science_cropped_path.with_name(science_cropped_path.stem + ".sky.sky.fits")

    output_path = write_cube(h1, c_out, c_out_u, output_path)

    return BlueIter2Result(
        science_path=science_cropped_path,
        output_path=output_path,
        wavelength=wl,
        params_left=params_left,
        params_right=params_right,
        master_mask=master_mask,
        fit_residual=fit_residual,
    )