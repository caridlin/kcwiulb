from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit

from kcwiulb.sky.utils import build_wavelength_axis, write_cube


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


def load_cube_basic(
    path: str | Path,
) -> tuple[np.ndarray, fits.Header, np.ndarray]:
    path = Path(path)
    with fits.open(path) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
        uncert = hdul[1].data.copy()
    return data, header, uncert


def wl_to_index(wavelength: float, header) -> int:
    return int((wavelength - header["CRVAL3"]) / header["CD3_3"] + header["CRPIX3"])


def wavelength_mask(
    cube: np.ndarray | ma.MaskedArray,
    header,
) -> np.ndarray | ma.MaskedArray:
    return np.ma.concatenate(
        (
            cube[wl_to_index(7000, header):wl_to_index(7500, header)],
            cube[wl_to_index(7700, header):wl_to_index(8000, header)],
        )
    )


def notebook_mask2d_to_3d(mask2d: np.ndarray, shape3d: tuple[int, int, int]) -> np.ndarray:
    """
    Mirror the notebook exactly:

    c_mask_3d = np.zeros_like(c1)
    c_mask_3d = c_mask_3d.T
    c_mask_3d[c_mask.T] = 1
    c_mask_3d = c_mask_3d.T
    """
    c_mask_3d = np.zeros(shape3d, dtype=bool)
    c_mask_3d = c_mask_3d.T
    c_mask_3d[mask2d.T] = True
    c_mask_3d = c_mask_3d.T
    return c_mask_3d


def masked_region_median_and_uncert(
    cube: np.ndarray | ma.MaskedArray,
    cube_uncert: np.ndarray,
    mask2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Notebook-style per-cube sky spectrum and uncertainty.
    Uses only the cube's own 2D sigma-clipped mask at this stage.
    """
    e = []
    f = []

    for y_ in range(cube.shape[2]):
        for x_ in range(cube.shape[1]):
            if mask2d[x_, y_] is False:
                e.append(cube[:, x_, y_])
                f.append(cube_uncert[:, x_, y_])

    e = np.asarray(e, dtype=float)
    f = np.asarray(f, dtype=float)

    spec = np.nanmedian(e, axis=0)
    spec_std2 = np.sqrt(np.nansum(np.power(f, 2), axis=0) * np.pi / (2 * (len(f) - 1)))

    return spec, spec_std2


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

    # --------------------------------------------------------
    # Original unsubtracted cubes: c1..c5
    # --------------------------------------------------------
    c1, h1, c1_uncert = load_cube_basic(science_path)
    c2, h2, c2_uncert = load_cube_basic(sky_paths[0])
    c3, h3, c3_uncert = load_cube_basic(sky_paths[1])
    c4, h4, c4_uncert = load_cube_basic(sky_paths[2])
    c5, h5, c5_uncert = load_cube_basic(sky_paths[3])

    # --------------------------------------------------------
    # Corresponding *.sky.cr.sky2.cr.fits cubes: c6..c10
    # --------------------------------------------------------
    c6, _, _, c1_cr_mask = load_cr_masked_cube(science_mask_path)
    c7, _, _, c2_cr_mask = load_cr_masked_cube(sky_mask_paths[0])
    c8, _, _, c3_cr_mask = load_cr_masked_cube(sky_mask_paths[1])
    c9, _, _, c4_cr_mask = load_cr_masked_cube(sky_mask_paths[2])
    c10, _, _, c5_cr_mask = load_cr_masked_cube(sky_mask_paths[3])

    wl = build_wavelength_axis(h1, c1.shape[0])

    # --------------------------------------------------------
    # Apply CR masks exactly like notebook
    # --------------------------------------------------------
    c1 = ma.masked_array(c1, mask=c1_cr_mask)
    c2 = ma.masked_array(c2, mask=c2_cr_mask)
    c3 = ma.masked_array(c3, mask=c3_cr_mask)
    c4 = ma.masked_array(c4, mask=c4_cr_mask)
    c5 = ma.masked_array(c5, mask=c5_cr_mask)

    c6 = ma.masked_array(c6, mask=c1_cr_mask)
    c7 = ma.masked_array(c7, mask=c2_cr_mask)
    c8 = ma.masked_array(c8, mask=c3_cr_mask)
    c9 = ma.masked_array(c9, mask=c4_cr_mask)
    c10 = ma.masked_array(c10, mask=c5_cr_mask)

    # --------------------------------------------------------
    # White-band images: d1..d5 from c6..c10
    # --------------------------------------------------------
    d1 = np.ma.sum(wavelength_mask(c6, h1), axis=0)
    d2 = np.ma.sum(wavelength_mask(c7, h2), axis=0)
    d3 = np.ma.sum(wavelength_mask(c8, h3), axis=0)
    d4 = np.ma.sum(wavelength_mask(c9, h4), axis=0)
    d5 = np.ma.sum(wavelength_mask(c10, h5), axis=0)

    d1_filtered = sigma_clip(d1, sigma=sigma_clip_sigma, maxiters=None, masked=True)
    d2_filtered = sigma_clip(d2, sigma=sigma_clip_sigma, maxiters=None, masked=True)
    d3_filtered = sigma_clip(d3, sigma=sigma_clip_sigma, maxiters=None, masked=True)
    d4_filtered = sigma_clip(d4, sigma=sigma_clip_sigma, maxiters=None, masked=True)
    d5_filtered = sigma_clip(d5, sigma=sigma_clip_sigma, maxiters=None, masked=True)

    c1_mask = np.array(d1_filtered.mask, dtype=bool)
    c2_mask = np.array(d2_filtered.mask, dtype=bool)
    c3_mask = np.array(d3_filtered.mask, dtype=bool)
    c4_mask = np.array(d4_filtered.mask, dtype=bool)
    c5_mask = np.array(d5_filtered.mask, dtype=bool)

    # --------------------------------------------------------
    # Extract notebook-style spectra from c1..c5 using own masks
    # --------------------------------------------------------
    sky1_spec, sky1_spec_std2 = masked_region_median_and_uncert(c1, c1_uncert, c1_mask)
    sky2_spec, sky2_spec_std2 = masked_region_median_and_uncert(c2, c2_uncert, c2_mask)
    sky3_spec, sky3_spec_std2 = masked_region_median_and_uncert(c3, c3_uncert, c3_mask)
    sky4_spec, sky4_spec_std2 = masked_region_median_and_uncert(c4, c4_uncert, c4_mask)
    sky5_spec, sky5_spec_std2 = masked_region_median_and_uncert(c5, c5_uncert, c5_mask)

    # --------------------------------------------------------
    # Continuum filtering
    # --------------------------------------------------------
    from kcwiulb.sky.utils import weighted_median_filter_1d

    sky1_spec_cfw = weighted_median_filter_1d(sky1_spec, sky1_spec_std2, width=cfwidth)
    sky2_spec_cfw = weighted_median_filter_1d(sky2_spec, sky2_spec_std2, width=cfwidth)
    sky3_spec_cfw = weighted_median_filter_1d(sky3_spec, sky3_spec_std2, width=cfwidth)
    sky4_spec_cfw = weighted_median_filter_1d(sky4_spec, sky4_spec_std2, width=cfwidth)
    sky5_spec_cfw = weighted_median_filter_1d(sky5_spec, sky5_spec_std2, width=cfwidth)

    sky1_res_cfw = sky1_spec - sky1_spec_cfw
    sky2_res_cfw = sky2_spec - sky2_spec_cfw
    sky3_res_cfw = sky3_spec - sky3_spec_cfw
    sky4_res_cfw = sky4_spec - sky4_spec_cfw
    sky5_res_cfw = sky5_spec - sky5_spec_cfw

    # residual cubes
    c1_res = c1 - sky1_spec_cfw[:, None, None]
    c2_res = c2 - sky2_spec_cfw[:, None, None]
    c3_res = c3 - sky3_spec_cfw[:, None, None]
    c4_res = c4 - sky4_spec_cfw[:, None, None]
    c5_res = c5 - sky5_spec_cfw[:, None, None]

    # --------------------------------------------------------
    # Master mask exactly like notebook
    # --------------------------------------------------------
    c_mask = c1_mask | c2_mask | c3_mask | c4_mask | c5_mask
    master_mask_2d = c_mask.copy()

    c_mask_3d = notebook_mask2d_to_3d(c_mask, c1.shape)
    cr_mask = np.logical_or(
        c1_cr_mask,
        np.logical_or(
            c2_cr_mask,
            np.logical_or(c3_cr_mask, np.logical_or(c4_cr_mask, c5_cr_mask)),
        ),
    )
    master_mask_3d = np.logical_or(c_mask_3d, cr_mask)

    wl_3d = np.tile(wl, (c1.shape[2], c1.shape[1], 1)).T

    c1_res_masked = ma.masked_array(c1_res, mask=master_mask_3d)
    c2_res_masked = ma.masked_array(c2_res, mask=master_mask_3d)
    c3_res_masked = ma.masked_array(c3_res, mask=master_mask_3d)
    c4_res_masked = ma.masked_array(c4_res, mask=master_mask_3d)
    c5_res_masked = ma.masked_array(c5_res, mask=master_mask_3d)
    wl_3d_masked = ma.masked_array(wl_3d, mask=master_mask_3d)

    # --------------------------------------------------------
    # Exact notebook wavelength regions
    # --------------------------------------------------------
    ind1 = wl_to_index(h1["WAVGOOD0"] + 1, h1)
    ind2 = wl_to_index(7200, h1)
    ind3 = wl_to_index(7700, h1)
    ind4 = wl_to_index(h1["WAVGOOD1"], h1)

    region_bounds = [(ind1, ind2), (ind2, ind3), (ind3, ind4)]
    region_wavelength_bounds = [
        (wl[ind1], wl[ind2 - 1]),
        (wl[ind2], wl[ind3 - 1]),
        (wl[ind3], wl[ind4 - 1]),
    ]

    params_list: list[np.ndarray] = []
    pcov_list: list[np.ndarray | None] = []
    fit_residual_regions: list[np.ndarray] = []
    var_add_list: list[np.ndarray] = []

    for i0, i1 in region_bounds:
        x1 = np.ma.median(c2_res_masked[i0:i1], axis=(1, 2))
        x2 = np.ma.median(c3_res_masked[i0:i1], axis=(1, 2))
        x3 = np.ma.median(c4_res_masked[i0:i1], axis=(1, 2))
        x4 = np.ma.median(c5_res_masked[i0:i1], axis=(1, 2))
        y1 = np.ma.median(c1_res_masked[i0:i1], axis=(1, 2))

        params, pcov = curve_fit(
            sky_model_four,
            (x1, x2, x3, x4),
            y1,
            maxfev=5000,
        )
        params_list.append(params)
        pcov_list.append(pcov)

        fit_residual_region = sky1_res_cfw - sky_model_four(
            (sky2_res_cfw, sky3_res_cfw, sky4_res_cfw, sky5_res_cfw),
            *params,
        )
        fit_residual_regions.append(fit_residual_region)

    params1, params2, params3 = params_list
    pcov1, pcov2, pcov3 = pcov_list

    var_add1 = (
        np.power(params1[0], 2) * np.power(sky2_spec_std2, 2)
        + np.power(params1[1], 2) * np.power(sky3_spec_std2, 2)
        + np.power(params1[2], 2) * np.power(sky4_spec_std2, 2)
        + np.power(params1[3], 2) * np.power(sky5_spec_std2, 2)
    )
    var_add2 = (
        np.power(params2[0], 2) * np.power(sky2_spec_std2, 2)
        + np.power(params2[1], 2) * np.power(sky3_spec_std2, 2)
        + np.power(params2[2], 2) * np.power(sky4_spec_std2, 2)
        + np.power(params2[3], 2) * np.power(sky5_spec_std2, 2)
    )
    var_add3 = (
        np.power(params3[0], 2) * np.power(sky2_spec_std2, 2)
        + np.power(params3[1], 2) * np.power(sky3_spec_std2, 2)
        + np.power(params3[2], 2) * np.power(sky4_spec_std2, 2)
        + np.power(params3[3], 2) * np.power(sky5_spec_std2, 2)
    )
    var_add_list = [var_add1, var_add2, var_add3]

    # --------------------------------------------------------
    # Notebook c111/c121 etc
    # --------------------------------------------------------
    c111 = np.empty_like(c1)
    c121 = np.empty_like(c1)
    c111_uncert = np.empty_like(c1_uncert)
    c121_uncert = np.empty_like(c1_uncert)

    c112 = np.empty_like(c1)
    c122 = np.empty_like(c1)
    c112_uncert = np.empty_like(c1_uncert)
    c122_uncert = np.empty_like(c1_uncert)

    c113 = np.empty_like(c1)
    c123 = np.empty_like(c1)
    c113_uncert = np.empty_like(c1_uncert)
    c123_uncert = np.empty_like(c1_uncert)

    for x_ in range(c1.shape[1]):
        for y_ in range(c1.shape[2]):
            c121[:, x_, y_] = c1_res[:, x_, y_] - sky_model_four(
                (sky2_res_cfw, sky3_res_cfw, sky4_res_cfw, sky5_res_cfw), *params1
            )
            c111[:, x_, y_] = c1_res[:, x_, y_] - sky_model_four(
                (c2_res[:, x_, y_], c3_res[:, x_, y_], c4_res[:, x_, y_], c5_res[:, x_, y_]),
                *params1,
            )
            c111_uncert[:, x_, y_] = np.sqrt(
                np.power(c1_uncert[:, x_, y_], 2)
                + np.power(params1[0], 2) * np.power(c2_uncert[:, x_, y_], 2)
                + np.power(params1[1], 2) * np.power(c3_uncert[:, x_, y_], 2)
                + np.power(params1[2], 2) * np.power(c4_uncert[:, x_, y_], 2)
                + np.power(params1[3], 2) * np.power(c5_uncert[:, x_, y_], 2)
            )
            c121_uncert[:, x_, y_] = np.sqrt(np.power(c1_uncert[:, x_, y_], 2) + var_add1)

    for x_ in range(c1.shape[1]):
        for y_ in range(c1.shape[2]):
            c122[:, x_, y_] = c1_res[:, x_, y_] - sky_model_four(
                (sky2_res_cfw, sky3_res_cfw, sky4_res_cfw, sky5_res_cfw), *params2
            )
            c112[:, x_, y_] = c1_res[:, x_, y_] - sky_model_four(
                (c2_res[:, x_, y_], c3_res[:, x_, y_], c4_res[:, x_, y_], c5_res[:, x_, y_]),
                *params2,
            )
            c112_uncert[:, x_, y_] = np.sqrt(
                np.power(c1_uncert[:, x_, y_], 2)
                + np.power(params2[0], 2) * np.power(c2_uncert[:, x_, y_], 2)
                + np.power(params2[1], 2) * np.power(c3_uncert[:, x_, y_], 2)
                + np.power(params2[2], 2) * np.power(c4_uncert[:, x_, y_], 2)
                + np.power(params2[3], 2) * np.power(c5_uncert[:, x_, y_], 2)
            )
            c122_uncert[:, x_, y_] = np.sqrt(np.power(c1_uncert[:, x_, y_], 2) + var_add2)

    for x_ in range(c1.shape[1]):
        for y_ in range(c1.shape[2]):
            c123[:, x_, y_] = c1_res[:, x_, y_] - sky_model_four(
                (sky2_res_cfw, sky3_res_cfw, sky4_res_cfw, sky5_res_cfw), *params3
            )
            c113[:, x_, y_] = c1_res[:, x_, y_] - sky_model_four(
                (c2_res[:, x_, y_], c3_res[:, x_, y_], c4_res[:, x_, y_], c5_res[:, x_, y_]),
                *params3,
            )
            c113_uncert[:, x_, y_] = np.sqrt(
                np.power(c1_uncert[:, x_, y_], 2)
                + np.power(params3[0], 2) * np.power(c2_uncert[:, x_, y_], 2)
                + np.power(params3[1], 2) * np.power(c3_uncert[:, x_, y_], 2)
                + np.power(params3[2], 2) * np.power(c4_uncert[:, x_, y_], 2)
                + np.power(params3[3], 2) * np.power(c5_uncert[:, x_, y_], 2)
            )
            c123_uncert[:, x_, y_] = np.sqrt(np.power(c1_uncert[:, x_, y_], 2) + var_add3)

    c11 = np.concatenate((c111[ind1:ind2], c112[ind2:ind3], c113[ind3:ind4]), axis=0)
    c12 = np.concatenate((c121[ind1:ind2], c122[ind2:ind3], c123[ind3:ind4]), axis=0)
    c11_uncert = np.concatenate(
        (c111_uncert[ind1:ind2], c112_uncert[ind2:ind3], c113_uncert[ind3:ind4]), axis=0
    )
    c12_uncert = np.concatenate(
        (c121_uncert[ind1:ind2], c122_uncert[ind2:ind3], c123_uncert[ind3:ind4]), axis=0
    )

    fit_residual = np.zeros_like(wl)
    fit_residual[ind1:ind2] = fit_residual_regions[0][ind1:ind2]
    fit_residual[ind2:ind3] = fit_residual_regions[1][ind2:ind3]
    fit_residual[ind3:ind4] = fit_residual_regions[2][ind3:ind4]

    if output_path_sky is None:
        output_path_sky = science_path.with_name(
            science_path.name.replace("icubes.wc.c", "icubes.wc.c.sky.cr.sky.cr.sky")
        )
    if output_path_sky2 is None:
        output_path_sky2 = science_path.with_name(
            science_path.name.replace("icubes.wc.c", "icubes.wc.c.sky.cr.sky2.cr.sky2")
        )

    # notebook writes CRMASK only to c11 product
    hdu1 = fits.PrimaryHDU(c11.data if ma.isMaskedArray(c11) else c11)
    hdu2 = fits.ImageHDU(c11_uncert)
    hdu3 = fits.ImageHDU(cr_mask.astype(int))
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist[0].header = h1
    hdulist.writeto(output_path_sky, overwrite=True)

    hdu1 = fits.PrimaryHDU(c12.data if ma.isMaskedArray(c12) else c12)
    hdu2 = fits.ImageHDU(c12_uncert)
    hdulist = fits.HDUList([hdu1, hdu2])
    hdulist[0].header = h1
    hdulist.writeto(output_path_sky2, overwrite=True)

    return RedIter3Result(
        science_path=science_path,
        science_mask_path=science_mask_path,
        sky_paths=sky_paths,
        sky_mask_paths=sky_mask_paths,
        output_path_sky=Path(output_path_sky),
        output_path_sky2=Path(output_path_sky2),
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
        science_whiteband=np.asarray(d1.filled(np.nan) if ma.isMaskedArray(d1) else d1),
        sky1_whiteband=np.asarray(d2.filled(np.nan) if ma.isMaskedArray(d2) else d2),
        sky2_whiteband=np.asarray(d3.filled(np.nan) if ma.isMaskedArray(d3) else d3),
        sky3_whiteband=np.asarray(d4.filled(np.nan) if ma.isMaskedArray(d4) else d4),
        sky4_whiteband=np.asarray(d5.filled(np.nan) if ma.isMaskedArray(d5) else d5),
        science_mask=c1_mask,
        sky1_mask=c2_mask,
        sky2_mask=c3_mask,
        sky3_mask=c4_mask,
        sky4_mask=c5_mask,
        science_spec=sky1_spec,
        sky1_spec=sky2_spec,
        sky2_spec=sky3_spec,
        sky3_spec=sky4_spec,
        sky4_spec=sky5_spec,
        science_spec_cont=sky1_spec_cfw,
        sky1_spec_cont=sky2_spec_cfw,
        sky2_spec_cont=sky3_spec_cfw,
        sky3_spec_cont=sky4_spec_cfw,
        sky4_spec_cont=sky5_spec_cfw,
        science_spec_res=sky1_res_cfw,
        sky1_spec_res=sky2_res_cfw,
        sky2_spec_res=sky3_res_cfw,
        sky3_spec_res=sky4_res_cfw,
        sky4_spec_res=sky5_res_cfw,
        c11=c11,
        c12=c12,
        c11_uncert=c11_uncert,
        c12_uncert=c12_uncert,
        wavgood0=h1["WAVGOOD0"],
        wavgood1=h1["WAVGOOD1"],
    )