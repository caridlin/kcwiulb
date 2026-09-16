from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from kcwiulb.sky.utils import (
    build_wavelength_axis,
    load_cube,
    sigma_clip_mask_2d,
    whiteband_image,
    write_cube,
)


@dataclass
class RedSelfSkyIter1Result:
    science_path: Path
    output_path: Path
    wavelength: np.ndarray
    sky_spec: np.ndarray
    sky_uncert: np.ndarray
    residual_spec: np.ndarray
    source_mask: np.ndarray
    bright_mask: np.ndarray
    sky_mask: np.ndarray
    whiteband: np.ndarray
    qso_x: float
    qso_y: float
    source_mask_radius_arcsec: float
    wavgood0: float
    wavgood1: float


def subtract_red_selfsky_iter1(
    science_path: str | Path,
    output_path: str | Path | None,
    collapse_wavelength_ranges: list[tuple[float, float]],
    qso_ra_deg: float,
    qso_dec_deg: float,
    source_mask_radius_arcsec: float = 7.0,
    sigma_clip: float = 3.0,
) -> RedSelfSkyIter1Result:
    """
    Red-channel iter1 self-sky subtraction for observations without
    dedicated sky cubes.

    The sky spectrum is estimated independently for each science cube:

        sky(lambda) = median of source-free outer spaxels

    A circular region around the QSO is excluded, together with additional
    bright spaxels identified from the white-band image.

    The resulting median sky spectrum is subtracted uniformly from every
    spaxel.

    IMPORTANT
    ---------
    Real extended emission outside source_mask_radius_arcsec can enter the
    sky estimate and be partially self-subtracted. The mask radius should
    therefore be tested for convergence when searching for extended emission.
    """
    science_path = Path(science_path)

    cube, header, uncert = load_cube(science_path)
    wl = build_wavelength_axis(header, cube.shape[0])

    # --------------------------------------------------------
    # White-band image
    # --------------------------------------------------------
    whiteband = whiteband_image(
        cube,
        header,
        collapse_wavelength_ranges,
    )

    # --------------------------------------------------------
    # QSO position and radial source mask
    # --------------------------------------------------------
    wcs2d = WCS(header).celestial

    qso_coord = SkyCoord(
        ra=qso_ra_deg * u.deg,
        dec=qso_dec_deg * u.deg,
        frame="icrs",
    )

    qso_x, qso_y = wcs2d.world_to_pixel(qso_coord)

    yy, xx = np.indices(whiteband.shape)
    ra, dec = wcs2d.all_pix2world(xx, yy, 0)

    spaxel_coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs",
    )

    separation_arcsec = spaxel_coords.separation(qso_coord).arcsec
    source_mask = separation_arcsec <= source_mask_radius_arcsec

    # --------------------------------------------------------
    # Reject other bright spatial sources
    # --------------------------------------------------------
    bright_mask = sigma_clip_mask_2d(
        whiteband,
        sigma=sigma_clip,
    )

    # True = excluded from sky estimate
    sky_mask = source_mask | bright_mask

    n_total = sky_mask.size
    n_sky = np.count_nonzero(~sky_mask)

    if n_sky < 2:
        raise ValueError(
            f"Too few sky spaxels remain after masking: "
            f"{n_sky}/{n_total}"
        )

    print("[subtract_red_selfsky_iter1] Self-sky subtraction")
    print(f"  Science: {science_path}")
    print(f"  QSO RA: {qso_ra_deg:.8f}")
    print(f"  QSO Dec: {qso_dec_deg:.8f}")
    print(f"  QSO x: {qso_x:.2f}")
    print(f"  QSO y: {qso_y:.2f}")
    print(f"  Source mask radius: {source_mask_radius_arcsec:.2f} arcsec")
    print(f"  Sky pixels used: {n_sky}/{n_total} ({n_sky / n_total:.1%})")

    # --------------------------------------------------------
    # Median outer-region sky spectrum
    # --------------------------------------------------------
    sky_values = cube[:, ~sky_mask]
    sky_uncert_values = uncert[:, ~sky_mask]

    sky_spec = np.nanmedian(
        sky_values,
        axis=1,
    )

    n = sky_values.shape[1]

    sky_uncert = np.sqrt(
        np.nansum(np.square(sky_uncert_values), axis=1)
        * np.pi
        / (2.0 * (n - 1))
    )

    bad = ~np.isfinite(sky_uncert) | (sky_uncert <= 0)
    sky_uncert[bad] = np.inf

    # --------------------------------------------------------
    # Subtract median self-sky from every spaxel
    # --------------------------------------------------------
    cube_out = cube - sky_spec[:, None, None]

    cube_out_uncert = np.sqrt(
        np.square(uncert)
        + np.square(sky_uncert[:, None, None])
    )

    residual_spec = np.nanmedian(
        cube_out[:, ~sky_mask],
        axis=1,
    )

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

    return RedSelfSkyIter1Result(
        science_path=science_path,
        output_path=output_path,
        wavelength=wl,
        sky_spec=sky_spec,
        sky_uncert=sky_uncert,
        residual_spec=residual_spec,
        source_mask=source_mask,
        bright_mask=bright_mask,
        sky_mask=sky_mask,
        whiteband=whiteband,
        qso_x=float(qso_x),
        qso_y=float(qso_y),
        source_mask_radius_arcsec=source_mask_radius_arcsec,
        wavgood0=header["WAVGOOD0"],
        wavgood1=header["WAVGOOD1"],
    )