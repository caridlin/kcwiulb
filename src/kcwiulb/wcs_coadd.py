from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits

from kcwiulb.wcs import (
    GaussianFitResult,
    fit_reference_source_2d_gaussian,
    update_header_wcs,
)


@dataclass
class CoaddWCSResult:
    flux_input_path: Path
    flux_output_path: Path
    updated_paths: list[Path]
    ra_deg: float
    dec_deg: float
    x_ref: float
    y_ref: float
    fit_result: GaussianFitResult


def _default_wc_path(path: Path) -> Path:
    return path.with_suffix("").with_name(path.stem + ".wc.fits")


def write_wcs_corrected_fits(
    input_path: str | Path,
    output_path: str | Path | None,
    ra_deg: float,
    dec_deg: float,
    x_ref: float,
    y_ref: float,
) -> Path:
    """
    Write a WCS-corrected FITS file by updating CRVAL1/2 and CRPIX1/2
    in all HDUs that have headers.
    """
    input_path = Path(input_path)
    output_path = _default_wc_path(input_path) if output_path is None else Path(output_path)

    with fits.open(input_path) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "header") and hdu.header is not None:
                hdu.header = update_header_wcs(
                    hdu.header,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    x_ref=x_ref,
                    y_ref=y_ref,
                )
        hdul.writeto(output_path, overwrite=True)

    return output_path


def solve_absolute_wcs_for_coadd(
    flux_cube_path: str | Path,
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
    extra_paths_to_update: Sequence[str | Path] | None = None,
    flux_output_path: str | Path | None = None,
    extra_output_paths: Sequence[str | Path | None] | None = None,
    pixel_origin: int = 1,
) -> CoaddWCSResult:
    """
    Fit a reference source on a coadded flux cube, then propagate the
    corrected WCS to the flux cube itself and any additional matching
    products (e.g. variance cubes).

    Parameters
    ----------
    flux_cube_path
        Coadded flux cube used for source fitting.
    ra_deg, dec_deg
        Known sky coordinates of the reference source.
    wavelength_ranges
        Wavelength ranges used to build the collapsed image.
    row_start, col_start, n_rows, n_cols
        Cutout definition for the Gaussian fit.
    amplitude_init, x_mean_init, y_mean_init, x_stddev_init, y_stddev_init
        Initial guesses in cutout coordinates.
    extra_paths_to_update
        Additional FITS files that should receive the same WCS correction,
        typically the matching variance cube and possibly parallel sky/sky2 products.
    flux_output_path
        Output path for the corrected flux cube. If None, writes *.wc.fits.
    extra_output_paths
        Output paths corresponding to extra_paths_to_update. If None, each file
        is written as *.wc.fits.
    pixel_origin
        Use 1 to match FITS convention and your notebook logic.

    Returns
    -------
    CoaddWCSResult
    """
    flux_cube_path = Path(flux_cube_path)

    with fits.open(flux_cube_path) as hdul:
        flux_cube = hdul[0].data
        flux_header = hdul[0].header.copy()

    fit_result = fit_reference_source_2d_gaussian(
        cube=flux_cube,
        header=flux_header,
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

    updated_paths: list[Path] = []

    flux_output_path = write_wcs_corrected_fits(
        input_path=flux_cube_path,
        output_path=flux_output_path,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        x_ref=x_ref,
        y_ref=y_ref,
    )
    updated_paths.append(flux_output_path)

    if extra_paths_to_update is not None:
        extra_paths = [Path(p) for p in extra_paths_to_update]

        if extra_output_paths is None:
            extra_output_paths = [None] * len(extra_paths)
        else:
            if len(extra_output_paths) != len(extra_paths):
                raise ValueError("extra_output_paths must match extra_paths_to_update in length.")

        for input_path, output_path in zip(extra_paths, extra_output_paths):
            out = write_wcs_corrected_fits(
                input_path=input_path,
                output_path=output_path,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                x_ref=x_ref,
                y_ref=y_ref,
            )
            updated_paths.append(out)

    return CoaddWCSResult(
        flux_input_path=flux_cube_path,
        flux_output_path=flux_output_path,
        updated_paths=updated_paths,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        x_ref=float(x_ref),
        y_ref=float(y_ref),
        fit_result=fit_result,
    )