from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.wcs import WCS


@dataclass
class GaussianFitResult:
    x_mean: float
    y_mean: float
    amplitude: float
    x_stddev: float
    y_stddev: float
    constant: float
    cutout: np.ndarray
    collapsed_image: np.ndarray


@dataclass
class WCSReferenceResult:
    ra_deg: float
    dec_deg: float
    x_ref: float
    y_ref: float
    fit_result: GaussianFitResult


@dataclass
class WCSPropagationResult:
    input_path: Path
    output_path: Path
    ra_deg: float
    dec_deg: float
    x_ref: float
    y_ref: float


def wavelength_to_index(wavelength: float, header: fits.Header) -> int:
    """Convert wavelength to cube index using FITS WCS keywords."""
    return int((wavelength - header["CRVAL3"]) / header["CD3_3"] + header["CRPIX3"])


def index_to_wavelength(index: int, header: fits.Header) -> float:
    """Convert cube index to wavelength using FITS WCS keywords."""
    return header["CRVAL3"] + header["CD3_3"] * (index - header["CRPIX3"])


def _build_collapsed_image(
    cube: np.ndarray,
    header: fits.Header,
    wavelength_ranges: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Collapse a cube over selected wavelength ranges."""
    slabs: list[np.ndarray] = []
    for wl_min, wl_max in wavelength_ranges:
        i0 = wavelength_to_index(wl_min, header)
        i1 = wavelength_to_index(wl_max, header)
        slabs.append(cube[i0:i1])

    if not slabs:
        raise ValueError("At least one wavelength range must be provided.")

    return np.sum(np.concatenate(slabs, axis=0), axis=0)


def fit_reference_source_2d_gaussian(
    cube: np.ndarray,
    header: fits.Header,
    wavelength_ranges: Sequence[tuple[float, float]],
    row_start: int,
    col_start: int,
    n_rows: int,
    n_cols: int,
    amplitude_init: float,
    x_mean_init: float,
    y_mean_init: float,
    x_stddev_init: float,
    y_stddev_init: float,
) -> GaussianFitResult:
    """Collapse selected wavelength ranges and fit a 2D Gaussian + constant.

    Parameters
    ----------
    cube
        Input data cube with shape (wavelength, y, x).
    header
        FITS header for the cube.
    wavelength_ranges
        Wavelength ranges to sum, typically chosen to avoid bright sky lines and
        strong emission lines while retaining continuum signal.
    row_start, col_start
        Lower-left corner of the fitting cutout in image coordinates.
    n_rows, n_cols
        Size of the fitting cutout.
    amplitude_init, x_mean_init, y_mean_init, x_stddev_init, y_stddev_init
        Initial guesses for the 2D Gaussian model in cutout coordinates.
    """
    collapsed = _build_collapsed_image(cube, header, wavelength_ranges)
    cutout = collapsed[row_start : row_start + n_rows, col_start : col_start + n_cols]

    y_grid, x_grid = np.mgrid[:n_rows, :n_cols]
    fitter = fitting.LevMarLSQFitter()
    model_init = models.Gaussian2D(
        amplitude=amplitude_init,
        x_mean=x_mean_init,
        y_mean=y_mean_init,
        x_stddev=x_stddev_init,
        y_stddev=y_stddev_init,
    ) + models.Const2D()
    fit_model = fitter(model_init, x_grid, y_grid, cutout)

    gaussian = fit_model[0]
    constant = fit_model[1]

    return GaussianFitResult(
        x_mean=float(gaussian.x_mean.value),
        y_mean=float(gaussian.y_mean.value),
        amplitude=float(gaussian.amplitude.value),
        x_stddev=float(gaussian.x_stddev.value),
        y_stddev=float(gaussian.y_stddev.value),
        constant=float(constant.amplitude.value),
        cutout=cutout,
        collapsed_image=collapsed,
    )


def solve_absolute_wcs_from_reference(
    cube_path: str | Path,
    ra_deg: float,
    dec_deg: float,
    wavelength_ranges: Sequence[tuple[float, float]],
    row_start: int,
    col_start: int,
    n_rows: int,
    n_cols: int,
    amplitude_init: float,
    x_mean_init: float,
    y_mean_init: float,
    x_stddev_init: float,
    y_stddev_init: float,
    pixel_origin: int = 1,
) -> WCSReferenceResult:
    """Fit a reference continuum source and solve an absolute WCS anchor.

    This updates the CRVAL/CRPIX pair conceptually by assigning the known sky
    coordinates (ra_deg, dec_deg) to the fitted source centroid.
    """
    cube_path = Path(cube_path)
    with fits.open(cube_path) as hdul:
        cube = hdul[0].data
        header = hdul[0].header.copy()

    fit_result = fit_reference_source_2d_gaussian(
        cube=cube,
        header=header,
        wavelength_ranges=wavelength_ranges,
        row_start=row_start,
        col_start=col_start,
        n_rows=n_rows,
        n_cols=n_cols,
        amplitude_init=amplitude_init,
        x_mean_init=x_mean_init,
        y_mean_init=y_mean_init,
        x_stddev_init=x_stddev_init,
        y_stddev_init=y_stddev_init,
    )

    x_ref = fit_result.x_mean + col_start + pixel_origin
    y_ref = fit_result.y_mean + row_start + pixel_origin

    return WCSReferenceResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        x_ref=x_ref,
        y_ref=y_ref,
        fit_result=fit_result,
    )


def pixel_to_sky_from_reference_solution(
    cube_path: str | Path,
    x_pixel: float,
    y_pixel: float,
    wavelength_index: int = 0,
) -> tuple[float, float]:
    """Convert a pixel coordinate in a reference cube into sky coordinates."""
    cube_path = Path(cube_path)
    with fits.open(cube_path) as hdul:
        header = hdul[0].header.copy()

    wcs = WCS(header)
    sky_coord, _ = wcs.pixel_to_world(x_pixel, y_pixel, wavelength_index)
    return float(sky_coord.ra.deg), float(sky_coord.dec.deg)


def update_header_wcs(
    header: fits.Header,
    ra_deg: float,
    dec_deg: float,
    x_ref: float,
    y_ref: float,
) -> fits.Header:
    """Return a copy of a header with updated CRVAL1/2 and CRPIX1/2."""
    new_header = header.copy()
    new_header["CRVAL1"] = ra_deg
    new_header["CRVAL2"] = dec_deg
    new_header["CRPIX1"] = x_ref
    new_header["CRPIX2"] = y_ref
    return new_header


def write_wcs_corrected_cube(
    input_path: str | Path,
    output_path: str | Path | None,
    ra_deg: float,
    dec_deg: float,
    x_ref: float,
    y_ref: float,
) -> WCSPropagationResult:
    """Write a WCS-corrected FITS cube preserving all HDUs."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix("").as_posix() + ".wc.fits"
    output_path = Path(output_path)

    with fits.open(input_path) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "header") and hdu.header is not None:
                hdu.header = update_header_wcs(hdu.header, ra_deg, dec_deg, x_ref, y_ref)
        hdul.writeto(output_path, overwrite=True)

    return WCSPropagationResult(
        input_path=input_path,
        output_path=output_path,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        x_ref=x_ref,
        y_ref=y_ref,
    )


def propagate_relative_wcs_from_reference_cube(
    reference_cube_path: str | Path,
    target_cube_path: str | Path,
    target_wavelength_ranges: Sequence[tuple[float, float]],
    row_start: int,
    col_start: int,
    n_rows: int,
    n_cols: int,
    amplitude_init: float,
    x_mean_init: float,
    y_mean_init: float,
    x_stddev_init: float,
    y_stddev_init: float,
    output_path: str | Path | None = None,
    pixel_origin: int = 0,
) -> WCSPropagationResult:
    """Use a reference cube with trusted WCS to anchor a second cube.

    This matches your notebook logic for the red cube: fit the same continuum
    source in the reference cube and the target cube, convert the fitted source
    position in the reference cube to sky coordinates, then assign those sky
    coordinates to the fitted source position in the target cube.
    """
    reference_cube_path = Path(reference_cube_path)
    target_cube_path = Path(target_cube_path)

    with fits.open(reference_cube_path) as hdul_ref:
        cube_ref = hdul_ref[0].data
        header_ref = hdul_ref[0].header.copy()

    ref_fit = fit_reference_source_2d_gaussian(
        cube=cube_ref,
        header=header_ref,
        wavelength_ranges=target_wavelength_ranges,
        row_start=row_start,
        col_start=col_start,
        n_rows=n_rows,
        n_cols=n_cols,
        amplitude_init=amplitude_init,
        x_mean_init=x_mean_init,
        y_mean_init=y_mean_init,
        x_stddev_init=x_stddev_init,
        y_stddev_init=y_stddev_init,
    )

    x_ref_cube = ref_fit.x_mean + col_start + pixel_origin
    y_ref_cube = ref_fit.y_mean + row_start + pixel_origin
    ra_deg, dec_deg = pixel_to_sky_from_reference_solution(
        reference_cube_path,
        x_pixel=x_ref_cube,
        y_pixel=y_ref_cube,
        wavelength_index=0,
    )

    with fits.open(target_cube_path) as hdul_target:
        cube_target = hdul_target[0].data
        header_target = hdul_target[0].header.copy()

    target_fit = fit_reference_source_2d_gaussian(
        cube=cube_target,
        header=header_target,
        wavelength_ranges=target_wavelength_ranges,
        row_start=row_start,
        col_start=col_start,
        n_rows=n_rows,
        n_cols=n_cols,
        amplitude_init=amplitude_init,
        x_mean_init=x_mean_init,
        y_mean_init=y_mean_init,
        x_stddev_init=x_stddev_init,
        y_stddev_init=y_stddev_init,
    )

    x_target = target_fit.x_mean + col_start + pixel_origin
    y_target = target_fit.y_mean + row_start + pixel_origin

    return write_wcs_corrected_cube(
        input_path=target_cube_path,
        output_path=output_path,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        x_ref=x_target,
        y_ref=y_target,
    )
