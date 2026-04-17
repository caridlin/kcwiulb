from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from scipy.optimize import curve_fit

from kcwiulb.sky.red_iter1 import sky_model_two
from kcwiulb.sky.utils import (
    build_wavelength_axis,
    sigma_clip_mask_2d,
    whiteband_image,
)


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


def masked_median_from_3d(
    cube: np.ndarray,
    mask3d: np.ndarray,
) -> np.ndarray:
    cube_masked = ma.masked_array(cube, mask=mask3d)
    return np.ma.median(cube_masked, axis=(1, 2)).filled(np.nan)


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
    Red-channel refined sky subtraction using CR masks to improve continuum masking.

    Inputs are the Iteration 1 CR-masked products (*.sky.cr.fits).

    Two outputs are written, following the notebook:
    - *.sky.cr.sky.fits   : subtraction using per-spaxel sky cubes
    - *.sky.cr.sky2.fits  : subtraction using median sky spectra
    """
    science_path = Path(science_path)
    sky1_path = Path(sky1_path)
    sky2_path = Path(sky2_path)

    c1, h1, c1u, c1_cr_mask = load_cr_masked_cube(science_path)
    c2, h2, c2u, c2_cr_mask = load_cr_masked_cube(sky1_path)
    c3, h3, c3u, c3_cr_mask = load_cr_masked_cube(sky2_path)

    wl = build_wavelength_axis(h1, c1.shape[0])

    # --------------------------------------------------------
    # White-band images (built from Iteration 1 sky-subtracted cubes)
    # --------------------------------------------------------
    d1 = whiteband_image(c1, h1, collapse_wavelength_ranges)
    d2 = whiteband_image(c2, h2, collapse_wavelength_ranges)
    d3 = whiteband_image(c3, h3, collapse_wavelength_ranges)

    m1_2d = sigma_clip_mask_2d(d1, sigma=sigma_clip_sigma)
    m2_2d = sigma_clip_mask_2d(d2, sigma=sigma_clip_sigma)
    m3_2d = sigma_clip_mask_2d(d3, sigma=sigma_clip_sigma)

    continuum_mask_2d = m1_2d | m2_2d | m3_2d
    continuum_mask_3d = expand_mask_2d_to_3d(continuum_mask_2d, c1.shape)

    cr_mask_3d = c1_cr_mask | c2_cr_mask | c3_cr_mask
    master_mask_3d = continuum_mask_3d | cr_mask_3d

    # --------------------------------------------------------
    # Median spectra from masked cubes
    # --------------------------------------------------------
    s1 = masked_median_from_3d(c1, master_mask_3d)
    s2 = masked_median_from_3d(c2, master_mask_3d)
    s3 = masked_median_from_3d(c3, master_mask_3d)

    # --------------------------------------------------------
    # Wavelength range used for fitting
    # --------------------------------------------------------
    ind1 = max(
        int((h1["WAVGOOD0"] + fit_margin_red - h1["CRVAL3"]) / h1["CD3_3"] + h1["CRPIX3"]),
        0,
    )
    ind2 = int((h1["WAVGOOD1"] - h1["CRVAL3"]) / h1["CD3_3"] + h1["CRPIX3"])

    # --------------------------------------------------------
    # Flatten masked voxel samples for fit
    # --------------------------------------------------------
    wl_3d = np.broadcast_to(wl[:, None, None], c1.shape)

    c1_masked = ma.masked_array(c1, mask=master_mask_3d)
    c2_masked = ma.masked_array(c2, mask=master_mask_3d)
    c3_masked = ma.masked_array(c3, mask=master_mask_3d)
    wl_3d_masked = ma.masked_array(wl_3d, mask=master_mask_3d)

    x1 = c2_masked[ind1:ind2].compressed()
    x2 = c3_masked[ind1:ind2].compressed()
    x3 = wl_3d_masked[ind1:ind2].compressed()
    y1 = c1_masked[ind1:ind2].compressed()

    params, pcov = curve_fit(
        sky_model_two,
        (x1, x2, x3),
        y1,
        maxfev=10000,
    )

    model = sky_model_two((s2, s3, wl), *params)
    residual = s1 - model

    # --------------------------------------------------------
    # Output 1: per-spaxel sky subtraction
    # c7 in notebook
    # --------------------------------------------------------
    c_spaxel = np.empty_like(c1)
    c_spaxel_u = np.empty_like(c1u)

    for y_ in range(c1.shape[1]):
        for x_ in range(c1.shape[2]):
            model_spaxel = sky_model_two((c2[:, y_, x_], c3[:, y_, x_], wl), *params)
            c_spaxel[:, y_, x_] = c1[:, y_, x_] - model_spaxel
            c_spaxel_u[:, y_, x_] = np.sqrt(
                np.square(c1u[:, y_, x_])
                + np.square(params[0] + params[1] * wl) * np.square(c2u[:, y_, x_])
                + np.square(params[2] + params[3] * wl) * np.square(c3u[:, y_, x_])
            )

    # --------------------------------------------------------
    # Output 2: subtraction using median sky spectra
    # c8 in notebook
    # --------------------------------------------------------
    var_add1 = (
        np.square(params[0] + params[1] * wl) * np.nanvar(c2_masked, axis=(1, 2)).filled(np.nan)
        + np.square(params[2] + params[3] * wl) * np.nanvar(c3_masked, axis=(1, 2)).filled(np.nan)
    )

    # keep notebook behavior: use only var_add1-like term for this product
    c_median = c1 - model[:, None, None]
    c_median_u = np.sqrt(np.square(c1u) + var_add1[:, None, None])

    # --------------------------------------------------------
    # Default output paths
    # --------------------------------------------------------
    if output_path_spaxel is None:
        output_path_spaxel = science_path.with_name(science_path.stem + ".sky.fits")
    if output_path_median is None:
        output_path_median = science_path.with_name(science_path.stem + ".sky2.fits")

    output_path_spaxel = write_cube_with_mask(
        output_path_spaxel,
        c_spaxel,
        h1,
        c_spaxel_u,
        cr_mask=cr_mask_3d,
    )
    output_path_median = write_cube_with_mask(
        output_path_median,
        c_median,
        h1,
        c_median_u,
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
        science_spec=s1,
        sky1_spec=s2,
        sky2_spec=s3,
        model_spec=model,
        residual_spec=residual,
        science_mask_2d=m1_2d,
        sky1_mask_2d=m2_2d,
        sky2_mask_2d=m3_2d,
        continuum_mask_3d=continuum_mask_3d,
        cr_mask_3d=cr_mask_3d,
        master_mask_3d=master_mask_3d,
        science_whiteband=d1,
        sky1_whiteband=d2,
        sky2_whiteband=d3,
        wavgood0=h1["WAVGOOD0"],
        wavgood1=h1["WAVGOOD1"],
    )