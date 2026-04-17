from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.ma as ma
from astropy.io import fits
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
class RedIter3Result:
    science_path: Path
    science_mask_path: Path
    sky_paths: list[Path]
    sky_mask_paths: list[Path]

    output_path_sky: Path
    output_path_sky2: Path

    wavelength: np.ndarray

    region_bounds: list[tuple[int, int]]
    region_wavelength_bounds: list[tuple[float, float]]
    params_list: list[np.ndarray]
    pcov_list: list[np.ndarray | None]

    master_mask_2d: np.ndarray
    master_mask_3d: np.ndarray
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

    science_spec: np.ndarray
    sky1_spec: np.ndarray
    sky2_spec: np.ndarray
    sky3_spec: np.ndarray
    sky4_spec: np.ndarray

    science_spec_cont: np.ndarray
    sky1_spec_cont: np.ndarray
    sky2_spec_cont: np.ndarray
    sky3_spec_cont: np.ndarray
    sky4_spec_cont: np.ndarray

    science_spec_res: np.ndarray
    sky1_spec_res: np.ndarray
    sky2_spec_res: np.ndarray
    sky3_spec_res: np.ndarray
    sky4_spec_res: np.ndarray

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


def load_cr_masked_cube(
    path: str | Path,
) -> tuple[np.ndarray, fits.Header, np.ndarray, np.ndarray]:
    path = Path(path)
    with fits.open(path) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
        uncert = hdul[1].data.copy()

        if len(hdul) >= 3:
            cr_mask = hdul[2].data.astype(bool)
        else:
            cr_mask = np.zeros_like(data, dtype=bool)

    return data, header, uncert, cr_mask


def wl_to_index(wavelength: float, header) -> int:
    return int((wavelength - header["CRVAL3"]) / header["CD3_3"] + header["CRPIX3"])


def _clip_index(index: int, nw: int) -> int:
    return max(0, min(index, nw))


def _build_region_bounds(
    header,
    nw: int,
    fit_margin_red: float,
    split_wavelengths: list[float] | None,
) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
    if split_wavelengths is None:
        split_wavelengths = []

    split_wavelengths = sorted(split_wavelengths)

    start_wl = float(header["WAVGOOD0"]) + fit_margin_red
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
            "No valid wavelength regions were created. "
            f"split_wavelengths={split_wavelengths}, "
            f"WAVGOOD0={header['WAVGOOD0']}, WAVGOOD1={header['WAVGOOD1']}"
        )

    return region_bounds, region_wavelength_bounds


def masked_median_spectrum_with_uncert(
    cube: np.ndarray,
    cube_uncert: np.ndarray,
    mask3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cube_masked = ma.masked_array(cube, mask=mask3d)
    uncert_masked = ma.masked_array(cube_uncert, mask=mask3d)

    nz = cube.shape[0]
    spec = np.full(nz, np.nan, dtype=float)
    spec_uncert = np.full(nz, np.nan, dtype=float)

    for i in range(nz):
        vals = cube_masked[i].compressed()
        errs = uncert_masked[i].compressed()

        if len(vals) == 0:
            continue

        spec[i] = np.nanmedian(vals)

        if len(errs) >= 2:
            spec_uncert[i] = np.sqrt(
                np.nansum(np.square(errs)) * np.pi / (2.0 * (len(errs) - 1))
            )

    return spec, spec_uncert


def subtract_red_iter3(
    science_path: str | Path,
    science_mask_path: str | Path,
    sky_paths: list[str | Path],
    sky_mask_paths: list[str | Path],
    output_path_sky: str | Path | None = None,
    output_path_sky2: str | Path | None = None,
    collapse_wavelength_ranges: list[tuple[float, float]] | None = None,
    cfwidth: int = 125,
    split_wavelengths: list[float] | None = None,
    fit_margin_red: float = 1.0,
    sigma_clip_sigma: float = 2.0,
) -> RedIter3Result:
    if collapse_wavelength_ranges is None:
        collapse_wavelength_ranges = [(7000, 7500), (7700, 8000)]

    science_path = Path(science_path)
    science_mask_path = Path(science_mask_path)
    sky_paths = [Path(p) for p in sky_paths]
    sky_mask_paths = [Path(p) for p in sky_mask_paths]

    if len(sky_paths) != 4:
        raise ValueError(f"Expected 4 sky paths, got {len(sky_paths)}")
    if len(sky_mask_paths) != 4:
        raise ValueError(f"Expected 4 sky mask paths, got {len(sky_mask_paths)}")

    # Original unsubtracted cubes
    c1, h1, c1_uncert = load_cube(science_path)
    c2, h2, c2_uncert = load_cube(sky_paths[0])
    c3, h3, c3_uncert = load_cube(sky_paths[1])
    c4, h4, c4_uncert = load_cube(sky_paths[2])
    c5, h5, c5_uncert = load_cube(sky_paths[3])

    # Latest CR-masked intermediate products for mask construction
    c1_mask_src, _, _, c1_cr = load_cr_masked_cube(science_mask_path)
    c2_mask_src, _, _, c2_cr = load_cr_masked_cube(sky_mask_paths[0])
    c3_mask_src, _, _, c3_cr = load_cr_masked_cube(sky_mask_paths[1])
    c4_mask_src, _, _, c4_cr = load_cr_masked_cube(sky_mask_paths[2])
    c5_mask_src, _, _, c5_cr = load_cr_masked_cube(sky_mask_paths[3])

    wl = build_wavelength_axis(h1, c1.shape[0])
    nw = c1.shape[0]

    # --------------------------------------------------------
    # Notebook-style CR-cleaned white-band images
    # --------------------------------------------------------
    c1_wb = c1_mask_src.astype(float).copy()
    c2_wb = c2_mask_src.astype(float).copy()
    c3_wb = c3_mask_src.astype(float).copy()
    c4_wb = c4_mask_src.astype(float).copy()
    c5_wb = c5_mask_src.astype(float).copy()

    c1_wb[c1_cr] = np.nan
    c2_wb[c2_cr] = np.nan
    c3_wb[c3_cr] = np.nan
    c4_wb[c4_cr] = np.nan
    c5_wb[c5_cr] = np.nan

    d1 = whiteband_image(c1_wb, h1, collapse_wavelength_ranges)
    d2 = whiteband_image(c2_wb, h2, collapse_wavelength_ranges)
    d3 = whiteband_image(c3_wb, h3, collapse_wavelength_ranges)
    d4 = whiteband_image(c4_wb, h4, collapse_wavelength_ranges)
    d5 = whiteband_image(c5_wb, h5, collapse_wavelength_ranges)

    m1 = sigma_clip_mask_2d(d1, sigma=sigma_clip_sigma)
    m2 = sigma_clip_mask_2d(d2, sigma=sigma_clip_sigma)
    m3 = sigma_clip_mask_2d(d3, sigma=sigma_clip_sigma)
    m4 = sigma_clip_mask_2d(d4, sigma=sigma_clip_sigma)
    m5 = sigma_clip_mask_2d(d5, sigma=sigma_clip_sigma)

    master_mask_2d = combine_masks_2d(m1, m2, m3, m4, m5)
    continuum_mask_3d = mask2d_to_mask3d(master_mask_2d, c1.shape)

    # Notebook-style shared CR mask across all 5 cubes
    cr_mask = c1_cr | c2_cr | c3_cr | c4_cr | c5_cr
    master_mask_3d = continuum_mask_3d | cr_mask

    # --------------------------------------------------------
    # Apply final shared mask to original unsubtracted cubes
    # --------------------------------------------------------
    science_spec, science_std = masked_median_spectrum_with_uncert(c1, c1_uncert, master_mask_3d)
    sky1_spec, sky1_std = masked_median_spectrum_with_uncert(c2, c2_uncert, master_mask_3d)
    sky2_spec, sky2_std = masked_median_spectrum_with_uncert(c3, c3_uncert, master_mask_3d)
    sky3_spec, sky3_std = masked_median_spectrum_with_uncert(c4, c4_uncert, master_mask_3d)
    sky4_spec, sky4_std = masked_median_spectrum_with_uncert(c5, c5_uncert, master_mask_3d)

    science_spec_cont = weighted_median_filter_1d(science_spec, science_std, width=cfwidth)
    sky1_spec_cont = weighted_median_filter_1d(sky1_spec, sky1_std, width=cfwidth)
    sky2_spec_cont = weighted_median_filter_1d(sky2_spec, sky2_std, width=cfwidth)
    sky3_spec_cont = weighted_median_filter_1d(sky3_spec, sky3_std, width=cfwidth)
    sky4_spec_cont = weighted_median_filter_1d(sky4_spec, sky4_std, width=cfwidth)

    science_spec_res = science_spec - science_spec_cont
    sky1_spec_res = sky1_spec - sky1_spec_cont
    sky2_spec_res = sky2_spec - sky2_spec_cont
    sky3_spec_res = sky3_spec - sky3_spec_cont
    sky4_spec_res = sky4_spec - sky4_spec_cont

    c1_res = c1 - science_spec_cont[:, None, None]
    c2_res = c2 - sky1_spec_cont[:, None, None]
    c3_res = c3 - sky2_spec_cont[:, None, None]
    c4_res = c4 - sky3_spec_cont[:, None, None]
    c5_res = c5 - sky4_spec_cont[:, None, None]

    c1_res_masked = ma.masked_array(c1_res, mask=master_mask_3d)
    c2_res_masked = ma.masked_array(c2_res, mask=master_mask_3d)
    c3_res_masked = ma.masked_array(c3_res, mask=master_mask_3d)
    c4_res_masked = ma.masked_array(c4_res, mask=master_mask_3d)
    c5_res_masked = ma.masked_array(c5_res, mask=master_mask_3d)

    region_bounds, region_wavelength_bounds = _build_region_bounds(
        header=h1,
        nw=nw,
        fit_margin_red=fit_margin_red,
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

        valid = (
            np.isfinite(x1)
            & np.isfinite(x2)
            & np.isfinite(x3)
            & np.isfinite(x4)
            & np.isfinite(y1)
        )
        if np.count_nonzero(valid) < 5:
            raise ValueError(
                f"Too few valid samples in fit region {i0}:{i1} "
                f"({wl[i0]:.2f}-{wl[i1 - 1]:.2f})"
            )

        params, pcov = curve_fit(
            sky_model_four,
            (x1[valid], x2[valid], x3[valid], x4[valid]),
            y1[valid],
            maxfev=5000,
        )
        params_list.append(params)
        pcov_list.append(pcov)

        fit_residual_region = science_spec_res - sky_model_four(
            (sky1_spec_res, sky2_spec_res, sky3_spec_res, sky4_spec_res),
            *params,
        )
        fit_residual_regions.append(fit_residual_region)

        var_add = (
            np.square(params[0]) * np.square(sky1_std)
            + np.square(params[1]) * np.square(sky2_std)
            + np.square(params[2]) * np.square(sky3_std)
            + np.square(params[3]) * np.square(sky4_std)
        )
        var_add_list.append(var_add)

    fit_residual = np.zeros_like(wl)
    for (i0, i1), region_res in zip(region_bounds, fit_residual_regions):
        fit_residual[i0:i1] = region_res[i0:i1]

    c11_regions: list[np.ndarray] = []
    c12_regions: list[np.ndarray] = []
    c11_uncert_regions: list[np.ndarray] = []
    c12_uncert_regions: list[np.ndarray] = []

    for (i0, i1), params, var_add in zip(region_bounds, params_list, var_add_list):
        c11_region = np.empty_like(c1[i0:i1])
        c12_region = np.empty_like(c1[i0:i1])
        c11_uncert_region = np.empty_like(c1_uncert[i0:i1])
        c12_uncert_region = np.empty_like(c1_uncert[i0:i1])

        model_region_med = sky_model_four(
            (
                sky1_spec_res[i0:i1],
                sky2_spec_res[i0:i1],
                sky3_spec_res[i0:i1],
                sky4_spec_res[i0:i1],
            ),
            *params,
        )

        for x_ in range(c1.shape[1]):
            for y_ in range(c1.shape[2]):
                model_region_spaxel = sky_model_four(
                    (
                        c2_res[i0:i1, x_, y_],
                        c3_res[i0:i1, x_, y_],
                        c4_res[i0:i1, x_, y_],
                        c5_res[i0:i1, x_, y_],
                    ),
                    *params,
                )

                c11_region[:, x_, y_] = c1_res[i0:i1, x_, y_] - model_region_spaxel
                c12_region[:, x_, y_] = c1_res[i0:i1, x_, y_] - model_region_med

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

    if output_path_sky is None:
        output_path_sky = science_path.with_name(
            science_path.stem + ".sky.cr.sky.cr.sky.fits"
        )
    if output_path_sky2 is None:
        output_path_sky2 = science_path.with_name(
            science_path.stem + ".sky.cr.sky2.cr.sky2.fits"
        )

    output_path_sky = write_cube(h1, c11, c11_uncert, output_path_sky)
    output_path_sky2 = write_cube(h1, c12, c12_uncert, output_path_sky2)

    return RedIter3Result(
        science_path=science_path,
        science_mask_path=science_mask_path,
        sky_paths=sky_paths,
        sky_mask_paths=sky_mask_paths,
        output_path_sky=output_path_sky,
        output_path_sky2=output_path_sky2,
        wavelength=wl,
        region_bounds=region_bounds,
        region_wavelength_bounds=region_wavelength_bounds,
        params_list=params_list,
        pcov_list=pcov_list,
        master_mask_2d=master_mask_2d,
        master_mask_3d=master_mask_3d,
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
        science_spec=science_spec,
        sky1_spec=sky1_spec,
        sky2_spec=sky2_spec,
        sky3_spec=sky3_spec,
        sky4_spec=sky4_spec,
        science_spec_cont=science_spec_cont,
        sky1_spec_cont=sky1_spec_cont,
        sky2_spec_cont=sky2_spec_cont,
        sky3_spec_cont=sky3_spec_cont,
        sky4_spec_cont=sky4_spec_cont,
        science_spec_res=science_spec_res,
        sky1_spec_res=sky1_spec_res,
        sky2_spec_res=sky2_spec_res,
        sky3_spec_res=sky3_spec_res,
        sky4_spec_res=sky4_spec_res,
        c11=c11,
        c12=c12,
        c11_uncert=c11_uncert,
        c12_uncert=c12_uncert,
        wavgood0=h1["WAVGOOD0"],
        wavgood1=h1["WAVGOOD1"],
    )