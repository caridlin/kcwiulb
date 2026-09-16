from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u

from kcwiulb.sky.utils import (
    build_wavelength_axis,
    load_cube,
    sigma_clip_mask_2d,
    whiteband_image,
    write_cube,
)


@dataclass
class BlueSelfSkyResult:
    science_path: Path
    output_path: Path
    wavelength: np.ndarray
    sky_spec: np.ndarray
    sky_uncert: np.ndarray
    residual_spec: np.ndarray
    sky_mask: np.ndarray
    source_mask: np.ndarray
    whiteband: np.ndarray
    qso_x: float
    qso_y: float
    source_mask_radius_arcsec: float
    wavgood0: float
    wavgood1: float


def subtract_blue_selfsky_iter1(
    science_path: str | Path,
    output_path: str | Path | None,
    collapse_wavelength_ranges: list[tuple[float, float]],
    qso_ra_deg: float,
    qso_dec_deg: float,
    source_mask_radius_arcsec: float = 5.0,
    sigma_clip: float = 3.0,
) -> BlueSelfSkyResult:
    """
    Subtract sky using source-free spaxels from the science cube itself.

    Intended for observations without dedicated sky cubes.

    A circular region around the QSO/source is excluded from the sky
    estimate. Additional bright sources outside this region are rejected
    using sigma clipping of the whiteband image.

    The final sky spectrum is the wavelength-by-wavelength median of all
    remaining sky spaxels and is subtracted uniformly from the cube.

    IMPORTANT
    ---------
    For extended-emission science, source_mask_radius_arcsec should be
    chosen conservatively. Real emission outside the mask will enter the
    estimated sky spectrum and can therefore be self-subtracted.
    """
    science_path = Path(science_path)

    # --------------------------------------------------------
    # Load science cube
    # --------------------------------------------------------
    cube, header, uncert = load_cube(science_path)
    wl = build_wavelength_axis(header, cube.shape[0])

    # --------------------------------------------------------
    # Whiteband image
    # --------------------------------------------------------
    whiteband = whiteband_image(
        cube,
        header,
        collapse_wavelength_ranges,
    )

    # --------------------------------------------------------
    # Locate QSO using WCS
    # --------------------------------------------------------
    wcs2d = WCS(header).celestial

    qso_coord = SkyCoord(
        ra=qso_ra_deg * u.deg,
        dec=qso_dec_deg * u.deg,
        frame="icrs",
    )

    qso_x, qso_y = wcs2d.world_to_pixel(qso_coord)

    # --------------------------------------------------------
    # Calculate angular distance of every spaxel from QSO
    #
    # Use the WCS directly rather than assuming a fixed pixel
    # scale, so this remains valid for different configurations.
    # --------------------------------------------------------
    yy, xx = np.indices(whiteband.shape)

    ra, dec = wcs2d.all_pix2world(
        xx,
        yy,
        0,
    )

    spaxel_coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs",
    )

    separation_arcsec = spaxel_coords.separation(qso_coord).arcsec

    source_mask = separation_arcsec <= source_mask_radius_arcsec

    # --------------------------------------------------------
    # Reject other bright sources using whiteband sigma clipping
    # --------------------------------------------------------
    bright_mask = sigma_clip_mask_2d(
        whiteband,
        sigma=sigma_clip,
    )

    # --------------------------------------------------------
    # Final sky mask
    #
    # True = excluded from sky calculation.
    # --------------------------------------------------------
    sky_mask = source_mask | bright_mask

    n_total = sky_mask.size
    n_sky = np.count_nonzero(~sky_mask)

    if n_sky < 2:
        raise ValueError(
            f"Too few sky spaxels remain after masking: "
            f"{n_sky}/{n_total}"
        )

    print("[subtract_blue_selfsky] Self-sky subtraction")
    print(f"  Science: {science_path}")
    print(f"  QSO RA: {qso_ra_deg:.8f}")
    print(f"  QSO Dec: {qso_dec_deg:.8f}")
    print(f"  QSO x: {qso_x:.2f}")
    print(f"  QSO y: {qso_y:.2f}")
    print(f"  Source mask radius: {source_mask_radius_arcsec:.2f} arcsec")
    print(f"  Total spatial pixels: {n_total}")
    print(f"  Sky pixels used: {n_sky}")
    print(f"  Sky fraction: {n_sky / n_total:.3f}")

    # --------------------------------------------------------
    # Extract sky spectrum
    #
    # cube[:, ~sky_mask] has shape:
    #
    #     (N_lambda, N_sky_spaxels)
    #
    # Use the median at each wavelength.
    # --------------------------------------------------------
    sky_values = cube[:, ~sky_mask]
    sky_uncert_values = uncert[:, ~sky_mask]

    sky_spec = np.nanmedian(
        sky_values,
        axis=1,
    )

    # Same median-uncertainty convention used elsewhere in
    # the sky/CR routines.
    n = sky_values.shape[1]

    sky_uncert = np.sqrt(
        np.nansum(np.square(sky_uncert_values), axis=1)
        * np.pi
        / (2.0 * (n - 1))
    )

    bad = ~np.isfinite(sky_uncert) | (sky_uncert <= 0)
    sky_uncert[bad] = np.inf

    # --------------------------------------------------------
    # Subtract spatially uniform self-sky spectrum
    # --------------------------------------------------------
    cube_out = cube - sky_spec[:, None, None]

    cube_out_uncert = np.sqrt(
        np.square(uncert)
        + np.square(sky_uncert[:, None, None])
    )

    # --------------------------------------------------------
    # Diagnostic residual spectrum
    #
    # Median of the sky pixels after subtraction should sit
    # close to zero.
    # --------------------------------------------------------
    residual_spec = np.nanmedian(
        cube_out[:, ~sky_mask],
        axis=1,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    if output_path is None:
        output_path = science_path.with_name(
            science_path.stem + ".sky.fits"
        )

    output_path = write_cube(
        header,
        cube_out,
        cube_out_uncert,
        output_path,
    )

    return BlueSelfSkyResult(
        science_path=science_path,
        output_path=output_path,
        wavelength=wl,
        sky_spec=sky_spec,
        sky_uncert=sky_uncert,
        residual_spec=residual_spec,
        sky_mask=sky_mask,
        source_mask=source_mask,
        whiteband=whiteband,
        qso_x=float(qso_x),
        qso_y=float(qso_y),
        source_mask_radius_arcsec=source_mask_radius_arcsec,
        wavgood0=header["WAVGOOD0"],
        wavgood1=header["WAVGOOD1"],
    )