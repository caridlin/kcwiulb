from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from kcwiulb.sky.utils import (
    build_wavelength_axis,
    load_cube,
    sigma_clip_mask_2d,
    whiteband_image,
    write_cube,
)


@dataclass
class RedIter1Result:
    science_path: Path
    sky1_path: Path
    sky2_path: Path
    output_path: Path
    wavelength: np.ndarray
    params: np.ndarray
    covariance: np.ndarray
    science_spec: np.ndarray
    sky1_spec: np.ndarray
    sky2_spec: np.ndarray
    science_spec_uncert: np.ndarray
    sky1_spec_uncert: np.ndarray
    sky2_spec_uncert: np.ndarray
    model_spec: np.ndarray
    residual_spec: np.ndarray
    science_mask: np.ndarray
    sky1_mask: np.ndarray
    sky2_mask: np.ndarray
    master_mask: np.ndarray
    science_whiteband: np.ndarray
    sky1_whiteband: np.ndarray
    sky2_whiteband: np.ndarray
    wavgood0: float
    wavgood1: float


def sky_model_two(x_tuple, a0, a1, b0, b1, c0, c1):
    x1, x2, wl = x_tuple
    return (a0 + a1 * wl) * x1 + (b0 + b1 * wl) * x2 + c0 + c1 * wl


def masked_median_spectrum_red(
    cube: np.ndarray,
    cube_uncert: np.ndarray,
    mask2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the median sky spectrum and its uncertainty from unmasked spaxels.

    This follows the logic in the red-channel notebook:
    - select all unmasked spaxels
    - take the median spectrum across those spaxels
    - estimate the uncertainty using the propagated per-spaxel uncertainties
      with the median correction factor sqrt(pi / (2 * (N - 1))).

    Parameters
    ----------
    cube
        Flux cube with shape (nz, ny, nx).
    cube_uncert
        Uncertainty cube with shape (nz, ny, nx).
    mask2d
        Boolean 2D mask with shape (ny, nx), where True means masked.

    Returns
    -------
    spec : np.ndarray
        Median spectrum.
    spec_uncert : np.ndarray
        Estimated uncertainty on the median spectrum.
    """
    good = ~mask2d
    if not np.any(good):
        raise ValueError("Mask excludes all spaxels; cannot build sky spectrum.")

    spectra = cube[:, good].T
    spectra_uncert = cube_uncert[:, good].T

    spec = np.nanmedian(spectra, axis=0)

    n_spec = spectra.shape[0]
    if n_spec < 2:
        raise ValueError("Need at least two unmasked spaxels to estimate uncertainty.")

    spec_uncert = np.sqrt(
        np.nansum(np.square(spectra_uncert), axis=0) * np.pi / (2.0 * (n_spec - 1))
    )

    return spec, spec_uncert


def subtract_red_iter1(
    science_path: str | Path,
    sky1_path: str | Path,
    sky2_path: str | Path,
    output_path: str | Path | None,
    collapse_wavelength_ranges: list[tuple[float, float]],
    fit_margin_red: float = 1.0,
) -> RedIter1Result:
    """
    Red-channel iterative sky subtraction using two sky cubes.

    This mirrors the blue iter1 implementation, but uses the red-channel
    spectrum/uncertainty extraction reflected in the notebook.
    """
    science_path = Path(science_path)
    sky1_path = Path(sky1_path)
    sky2_path = Path(sky2_path)

    c1, h1, c1u = load_cube(science_path)
    c2, h2, c2u = load_cube(sky1_path)
    c3, h3, c3u = load_cube(sky2_path)

    wl = build_wavelength_axis(h1, c1.shape[0])

    d1 = whiteband_image(c1, h1, collapse_wavelength_ranges)
    d2 = whiteband_image(c2, h2, collapse_wavelength_ranges)
    d3 = whiteband_image(c3, h3, collapse_wavelength_ranges)

    m1 = sigma_clip_mask_2d(d1, sigma=3.0)
    m2 = sigma_clip_mask_2d(d2, sigma=3.0)
    m3 = sigma_clip_mask_2d(d3, sigma=3.0)

    s1, s1u = masked_median_spectrum_red(c1, c1u, m1)
    s2, s2u = masked_median_spectrum_red(c2, c2u, m2)
    s3, s3u = masked_median_spectrum_red(c3, c3u, m3)

    master_mask = m1 | m2 | m3

    ind1 = max(
        int((h1["WAVGOOD0"] + fit_margin_red - h1["CRVAL3"]) / h1["CD3_3"] + h1["CRPIX3"]),
        0,
    )
    ind2 = int((h1["WAVGOOD1"] - h1["CRVAL3"]) / h1["CD3_3"] + h1["CRPIX3"])

    params, pcov = curve_fit(
        sky_model_two,
        (s2[ind1:ind2], s3[ind1:ind2], wl[ind1:ind2]),
        s1[ind1:ind2],
        maxfev=10000,
    )

    model = sky_model_two((s2, s3, wl), *params)
    residual = s1 - model

    var_add = (
        np.square(params[0] + params[1] * wl) * np.square(s2u)
        + np.square(params[2] + params[3] * wl) * np.square(s3u)
    )

    c_out = c1 - model[:, None, None]
    c_out_u = np.sqrt(np.square(c1u) + var_add[:, None, None])

    if output_path is None:
        output_path = science_path.with_name(science_path.stem + ".sky.fits")

    output_path = write_cube(h1, c_out, c_out_u, output_path)

    return RedIter1Result(
        science_path=science_path,
        sky1_path=sky1_path,
        sky2_path=sky2_path,
        output_path=output_path,
        wavelength=wl,
        params=params,
        covariance=pcov,
        science_spec=s1,
        sky1_spec=s2,
        sky2_spec=s3,
        science_spec_uncert=s1u,
        sky1_spec_uncert=s2u,
        sky2_spec_uncert=s3u,
        model_spec=model,
        residual_spec=residual,
        science_mask=m1,
        sky1_mask=m2,
        sky2_mask=m3,
        master_mask=master_mask,
        science_whiteband=d1,
        sky1_whiteband=d2,
        sky2_whiteband=d3,
        wavgood0=h1["WAVGOOD0"],
        wavgood1=h1["WAVGOOD1"],
    )