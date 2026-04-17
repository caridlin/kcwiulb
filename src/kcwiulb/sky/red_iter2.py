from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit

from kcwiulb.sky.red_iter1 import sky_model_two
from kcwiulb.sky.utils import build_wavelength_axis


@dataclass
class RedIter2Result:
    science_path: Path
    sky1_path: Path
    sky2_path: Path
    output_path_spaxel: Path
    output_path_median: Path
    wavelength: np.ndarray
    params: np.ndarray
    covariance: np.ndarray
    science_spec: np.ndarray
    sky1_spec: np.ndarray
    sky2_spec: np.ndarray
    model_spec: np.ndarray
    residual_spec: np.ndarray
    science_mask_2d: np.ndarray
    sky1_mask_2d: np.ndarray
    sky2_mask_2d: np.ndarray
    continuum_mask_3d: np.ndarray
    cr_mask_3d: np.ndarray
    master_mask_3d: np.ndarray
    science_whiteband: np.ndarray
    sky1_whiteband: np.ndarray
    sky2_whiteband: np.ndarray
    sky2_spec_std2: np.ndarray
    sky3_spec_std2: np.ndarray
    wavgood0: float
    wavgood1: float


def load_cr_masked_cube(path: str | Path) -> tuple[np.ndarray, fits.Header, np.ndarray, np.ndarray]:
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


def wl_to_index(wavelength: float, header: fits.Header) -> int:
    return int((wavelength - header["CRVAL3"]) / header["CD3_3"] + header["CRPIX3"])


def wavelength_mask(
    cube: np.ndarray | ma.MaskedArray,
    header: fits.Header,
    collapse_wavelength_ranges: list[tuple[float, float]],
) -> np.ndarray | ma.MaskedArray:
    pieces = []
    for w0, w1 in collapse_wavelength_ranges:
        i0 = wl_to_index(w0, header)
        i1 = wl_to_index(w1, header)
        pieces.append(cube[i0:i1])
    return ma.concatenate(pieces, axis=0)


def expand_mask_2d_to_3d(mask2d: np.ndarray, shape3d: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = shape3d
    if mask2d.shape != (ny, nx):
        raise ValueError(f"mask2d has shape {mask2d.shape}, expected {(ny, nx)}")
    return np.broadcast_to(mask2d[None, :, :], shape3d).copy()


def write_cube_with_mask(
    path: str | Path,
    data: np.ndarray,
    header: fits.Header,
    uncert: np.ndarray,
    cr_mask: np.ndarray | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hdus = [
        fits.PrimaryHDU(data=data, header=header),
        fits.ImageHDU(data=uncert, name="UNCERT"),
    ]

    if cr_mask is not None:
        hdus.append(fits.ImageHDU(data=cr_mask.astype(np.uint8), name="CRMASK"))

    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def subtract_red_iter2(
    science_path: str | Path,
    sky1_path: str | Path,
    sky2_path: str | Path,
    output_path_spaxel: str | Path | None,
    output_path_median: str | Path | None,
    collapse_wavelength_ranges: list[tuple[float, float]],
    fit_margin_red: float = 1.0,
    sigma_clip_sigma: float = 2.0,
) -> RedIter2Result:
    """
    Notebook-faithful red iter2 sky subtraction.
    """
    science_path = Path(science_path)
    sky1_path = Path(sky1_path)
    sky2_path = Path(sky2_path)

    # original unsubtracted cubes
    c1, h1, c1_uncert, _ = load_cr_masked_cube(science_path.with_name(science_path.name.replace(".sky.cr.fits", ".fits")))
    c2, h2, c2_uncert, _ = load_cr_masked_cube(sky1_path.with_name(sky1_path.name.replace(".sky.cr.fits", ".fits")))
    c3, h3, c3_uncert, _ = load_cr_masked_cube(sky2_path.with_name(sky2_path.name.replace(".sky.cr.fits", ".fits")))

    # iter1 sky-subtracted CR products
    c4, _, _, c1_cr_mask = load_cr_masked_cube(science_path)
    c5, _, _, c2_cr_mask = load_cr_masked_cube(sky1_path)
    c6, _, _, c3_cr_mask = load_cr_masked_cube(sky2_path)

    wl = build_wavelength_axis(h1, c1.shape[0])

    # notebook: apply corresponding CR mask first
    c1_ma = ma.masked_array(c1, mask=c1_cr_mask)
    c2_ma = ma.masked_array(c2, mask=c2_cr_mask)
    c3_ma = ma.masked_array(c3, mask=c3_cr_mask)

    c4_ma = ma.masked_array(c4, mask=c1_cr_mask)
    c5_ma = ma.masked_array(c5, mask=c2_cr_mask)
    c6_ma = ma.masked_array(c6, mask=c3_cr_mask)

    # white-band images are built from c4/c5/c6
    d1 = ma.sum(wavelength_mask(c4_ma, h1, collapse_wavelength_ranges), axis=0)
    d2 = ma.sum(wavelength_mask(c5_ma, h2, collapse_wavelength_ranges), axis=0)
    d3 = ma.sum(wavelength_mask(c6_ma, h3, collapse_wavelength_ranges), axis=0)

    d1_filtered = sigma_clip(d1, sigma=sigma_clip_sigma, maxiters=None, masked=True)
    d2_filtered = sigma_clip(d2, sigma=sigma_clip_sigma, maxiters=None, masked=True)
    d3_filtered = sigma_clip(d3, sigma=sigma_clip_sigma, maxiters=None, masked=True)

    c1_mask = np.array(d1_filtered.mask, dtype=bool)
    c2_mask = np.array(d2_filtered.mask, dtype=bool)
    c3_mask = np.array(d3_filtered.mask, dtype=bool)

    # notebook-style sky spectra and uncertainty lists
    e1, f1 = [], []
    for y_ in range(c1.shape[2]):
        for x_ in range(c1.shape[1]):
            if not c1_mask[x_, y_]:
                e1.append(c1[:, x_, y_])
                f1.append(c1_uncert[:, x_, y_])

    e2, f2 = [], []
    for y_ in range(c2.shape[2]):
        for x_ in range(c2.shape[1]):
            if not c2_mask[x_, y_]:
                e2.append(c2[:, x_, y_])
                f2.append(c2_uncert[:, x_, y_])

    e3, f3 = [], []
    for y_ in range(c3.shape[2]):
        for x_ in range(c2.shape[1]):
            if not c3_mask[x_, y_]:
                e3.append(c3[:, x_, y_])
                f3.append(c3_uncert[:, x_, y_])

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    e3 = np.asarray(e3, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    f3 = np.asarray(f3, dtype=float)

    sky1_spec = np.nanmedian(e1, axis=0)
    sky2_spec = np.nanmedian(e2, axis=0)
    sky3_spec = np.nanmedian(e3, axis=0)

    sky2_spec_std2 = np.sqrt(np.nansum(np.power(f2, 2), axis=0) * np.pi / (2 * (len(f2) - 1)))
    sky3_spec_std2 = np.sqrt(np.nansum(np.power(f3, 2), axis=0) * np.pi / (2 * (len(f3) - 1)))

    # shared 2d mask -> 3d
    c_mask = c1_mask | c2_mask | c3_mask
    continuum_mask_3d = expand_mask_2d_to_3d(c_mask, c1.shape)

    cr_mask = np.logical_or(c1_cr_mask, np.logical_or(c2_cr_mask, c3_cr_mask))
    master_mask_3d = np.logical_or(continuum_mask_3d, cr_mask)

    c1_masked = ma.masked_array(c1, mask=master_mask_3d)
    c2_masked = ma.masked_array(c2, mask=master_mask_3d)
    c3_masked = ma.masked_array(c3, mask=master_mask_3d)

    wl_3d = np.tile(wl, (c1.shape[2], c1.shape[1], 1)).T
    wl_3d_masked = ma.masked_array(wl_3d, mask=master_mask_3d)

    ind1 = wl_to_index(h1["WAVGOOD0"] + fit_margin_red, h1)
    ind2 = wl_to_index(h1["WAVGOOD1"], h1)

    x1 = c2_masked[ind1:ind2].flatten()
    x2 = c3_masked[ind1:ind2].flatten()
    x3 = wl_3d_masked[ind1:ind2].flatten()
    y1 = c1_masked[ind1:ind2].flatten()

    x1 = x1[~x1.mask]
    x2 = x2[~x2.mask]
    x3 = x3[~x3.mask]
    y1 = y1[~y1.mask]

    params, pcov = curve_fit(
        sky_model_two,
        (x1, x2, x3),
        y1,
        maxfev=10000,
    )

    model = sky_model_two((sky2_spec, sky3_spec, wl), *params)
    residual = sky1_spec - model

    c7 = np.empty_like(c1)
    c8 = np.empty_like(c1)
    c7_uncert = np.empty_like(c1_uncert)
    c8_uncert = np.empty_like(c1_uncert)

    var_add1 = (
        np.power((params[0] + params[1] * wl), 2) * np.power(sky2_spec_std2, 2)
        + np.power((params[2] + params[3] * wl), 2) * np.power(sky3_spec_std2, 2)
    )
    var_add = var_add1

    for x_ in range(c1.shape[1]):
        for y_ in range(c1.shape[2]):
            c8[:, x_, y_] = c1[:, x_, y_] - sky_model_two((sky2_spec, sky3_spec, wl), *params)
            c7[:, x_, y_] = c1[:, x_, y_] - sky_model_two((c2[:, x_, y_], c3[:, x_, y_], wl), *params)

            c7_uncert[:, x_, y_] = np.sqrt(
                np.power(c1_uncert[:, x_, y_], 2)
                + np.power((params[0] + params[1] * wl), 2) * np.power(c2_uncert[:, x_, y_], 2)
                + np.power((params[2] + params[3] * wl), 2) * np.power(c3_uncert[:, x_, y_], 2)
            )
            c8_uncert[:, x_, y_] = np.sqrt(
                np.power(c1_uncert[:, x_, y_], 2) + var_add
            )

    if output_path_spaxel is None:
        output_path_spaxel = science_path.with_name(science_path.stem + ".sky.fits")
    if output_path_median is None:
        output_path_median = science_path.with_name(science_path.stem + ".sky2.fits")

    output_path_spaxel = write_cube_with_mask(
        output_path_spaxel,
        c7,
        h1,
        c7_uncert,
        cr_mask=cr_mask,
    )
    output_path_median = write_cube_with_mask(
        output_path_median,
        c8,
        h1,
        c8_uncert,
        cr_mask=None,
    )

    return RedIter2Result(
        science_path=science_path,
        sky1_path=sky1_path,
        sky2_path=sky2_path,
        output_path_spaxel=output_path_spaxel,
        output_path_median=output_path_median,
        wavelength=wl,
        params=params,
        covariance=pcov,
        science_spec=sky1_spec,
        sky1_spec=sky2_spec,
        sky2_spec=sky3_spec,
        model_spec=model,
        residual_spec=residual,
        science_mask_2d=c1_mask,
        sky1_mask_2d=c2_mask,
        sky2_mask_2d=c3_mask,
        continuum_mask_3d=continuum_mask_3d,
        cr_mask_3d=cr_mask,
        master_mask_3d=master_mask_3d,
        science_whiteband=np.asarray(d1.filled(np.nan) if ma.isMaskedArray(d1) else d1),
        sky1_whiteband=np.asarray(d2.filled(np.nan) if ma.isMaskedArray(d2) else d2),
        sky2_whiteband=np.asarray(d3.filled(np.nan) if ma.isMaskedArray(d3) else d3),
        sky2_spec_std2=sky2_spec_std2,
        sky3_spec_std2=sky3_spec_std2,
        wavgood0=h1["WAVGOOD0"],
        wavgood1=h1["WAVGOOD1"],
    )