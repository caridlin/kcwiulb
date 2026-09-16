from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from kcwiulb.sky.red_iter2 import write_cube_with_mask
from kcwiulb.sky.utils import (
    build_wavelength_axis,
    sigma_clip_mask_2d,
    whiteband_image,
)


@dataclass
class RedSelfSkyIter2Result:
    science_path: Path
    original_path: Path
    output_path: Path
    wavelength: np.ndarray
    sky_spec: np.ndarray
    sky_uncert: np.ndarray
    residual_spec: np.ndarray
    source_mask: np.ndarray
    bright_mask: np.ndarray
    sky_mask_2d: np.ndarray
    cr_mask_3d: np.ndarray
    sky_mask_3d: np.ndarray
    master_mask_3d: np.ndarray
    whiteband: np.ndarray
    qso_x: float
    qso_y: float
    source_mask_radius_arcsec: float
    wavgood0: float
    wavgood1: float


def load_cr_product(
    path: str | Path,
) -> tuple[np.ndarray, fits.Header, np.ndarray, np.ndarray]:
    path = Path(path)

    with fits.open(path) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
        uncert = hdul["UNCERT"].data.copy()

        if "CRMASK" in hdul:
            cr_mask = hdul["CRMASK"].data.astype(bool)
        elif len(hdul) >= 3:
            cr_mask = hdul[2].data.astype(bool)
        else:
            cr_mask = np.zeros_like(data, dtype=bool)

    return data, header, uncert, cr_mask


def subtract_red_selfsky_iter2(
    science_path: str | Path,
    output_path: str | Path | None,
    collapse_wavelength_ranges: list[tuple[float, float]],
    qso_ra_deg: float,
    qso_dec_deg: float,
    source_mask_radius_arcsec: float = 7.0,
    sigma_clip: float = 2.0,
) -> RedSelfSkyIter2Result:
    """
    Red self-sky iteration 2.

    science_path is the iter1 sky-subtracted CR product:

        *_icubes.wc.c.sky.cr.fits

    The routine:

    1. Reads the CR mask from that product.
    2. Goes back to the corresponding ORIGINAL cube:
           *_icubes.wc.c.fits
    3. Builds the spatial self-sky region outside the QSO/source mask.
    4. Re-estimates the wavelength-dependent median sky while excluding
       CR-contaminated voxels.
    5. Subtracts this improved sky estimate directly from the original cube.

    Thus iter2 does not subtract a second full sky spectrum from the
    already-subtracted iter1 cube.
    """
    science_path = Path(science_path)

    # --------------------------------------------------------
    # CR product from iter1
    # --------------------------------------------------------
    iter1_data, _, _, cr_mask = load_cr_product(science_path)

    # --------------------------------------------------------
    # Recover original unsubtracted science cube
    #
    # foo.sky.cr.fits -> foo.fits
    # --------------------------------------------------------
    if not science_path.name.endswith(".sky.cr.fits"):
        raise ValueError(
            "Expected science_path ending in '.sky.cr.fits', got "
            f"{science_path.name}"
        )

    original_path = science_path.with_name(
        science_path.name.replace(".sky.cr.fits", ".fits")
    )

    if not original_path.exists():
        raise FileNotFoundError(
            f"Missing original unsubtracted cube: {original_path}"
        )

    original_data, header, original_uncert, _ = load_cr_product(
        original_path
    )

    wl = build_wavelength_axis(
        header,
        original_data.shape[0],
    )

    # --------------------------------------------------------
    # White-band image
    #
    # Use the iter1-subtracted cube for source identification,
    # analogous to the normal red iter2 workflow.
    # --------------------------------------------------------
    whiteband = whiteband_image(
        iter1_data,
        header,
        collapse_wavelength_ranges,
    )

    # --------------------------------------------------------
    # QSO/source spatial exclusion
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

    source_mask = (
        separation_arcsec
        <= source_mask_radius_arcsec
    )

    # --------------------------------------------------------
    # Additional spatial source mask
    # --------------------------------------------------------
    bright_mask = sigma_clip_mask_2d(
        whiteband,
        sigma=sigma_clip,
    )

    sky_mask_2d = source_mask | bright_mask

    # --------------------------------------------------------
    # Expand spatial mask and combine with CR mask
    # --------------------------------------------------------
    sky_mask_3d = np.broadcast_to(
        sky_mask_2d[None, :, :],
        original_data.shape,
    ).copy()

    master_mask_3d = sky_mask_3d | cr_mask

    # --------------------------------------------------------
    # Build wavelength-dependent median self-sky.
    #
    # Unlike iter1, CR masking varies with wavelength, so we
    # calculate the median using the full 3D master mask.
    # --------------------------------------------------------
    masked_data = np.ma.masked_array(
        original_data,
        mask=master_mask_3d,
    )

    masked_uncert = np.ma.masked_array(
        original_uncert,
        mask=master_mask_3d,
    )

    sky_spec = np.ma.median(
        masked_data,
        axis=(1, 2),
    ).filled(np.nan)

    # --------------------------------------------------------
    # Median uncertainty wavelength-by-wavelength.
    #
    # N can now vary with wavelength because CR voxels are
    # individually masked.
    # --------------------------------------------------------
    valid = ~master_mask_3d
    n_valid = np.sum(valid, axis=(1, 2))

    variance_sum = np.sum(
        np.where(
            valid,
            np.square(original_uncert),
            0.0,
        ),
        axis=(1, 2),
    )

    sky_uncert = np.full_like(
        sky_spec,
        np.inf,
        dtype=float,
    )

    good_n = n_valid >= 2

    sky_uncert[good_n] = np.sqrt(
        variance_sum[good_n]
        * np.pi
        / (2.0 * (n_valid[good_n] - 1))
    )

    # --------------------------------------------------------
    # Improved iter2 subtraction from ORIGINAL cube
    # --------------------------------------------------------
    cube_out = (
        original_data
        - sky_spec[:, None, None]
    )

    cube_out_uncert = np.sqrt(
        np.square(original_uncert)
        + np.square(sky_uncert[:, None, None])
    )

    # --------------------------------------------------------
    # Diagnostic residual:
    # median of valid self-sky voxels after subtraction
    # --------------------------------------------------------
    residual_ma = np.ma.masked_array(
        cube_out,
        mask=master_mask_3d,
    )

    residual_spec = np.ma.median(
        residual_ma,
        axis=(1, 2),
    ).filled(np.nan)

    n_total = sky_mask_2d.size
    n_spatial_sky = np.count_nonzero(~sky_mask_2d)

    print("[subtract_red_selfsky_iter2]")
    print(f"  CR product: {science_path}")
    print(f"  Original: {original_path}")
    print(f"  QSO pixel: ({qso_x:.2f}, {qso_y:.2f})")
    print(f"  Source mask radius: {source_mask_radius_arcsec:.2f} arcsec")
    print(
        f"  Spatial sky pixels: "
        f"{n_spatial_sky}/{n_total} "
        f"({n_spatial_sky / n_total:.1%})"
    )
    print(f"  CR voxels excluded: {int(cr_mask.sum())}")

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    if output_path is None:
        output_path = science_path.with_name(
            science_path.stem + ".sky2.fits"
        )

    output_path = write_cube_with_mask(
        output_path,
        cube_out,
        header,
        cube_out_uncert,
        cr_mask=cr_mask,
    )

    print(f"  Output: {output_path}")

    return RedSelfSkyIter2Result(
        science_path=science_path,
        original_path=original_path,
        output_path=output_path,
        wavelength=wl,
        sky_spec=sky_spec,
        sky_uncert=sky_uncert,
        residual_spec=residual_spec,
        source_mask=source_mask,
        bright_mask=bright_mask,
        sky_mask_2d=sky_mask_2d,
        cr_mask_3d=cr_mask,
        sky_mask_3d=sky_mask_3d,
        master_mask_3d=master_mask_3d,
        whiteband=whiteband,
        qso_x=float(qso_x),
        qso_y=float(qso_y),
        source_mask_radius_arcsec=source_mask_radius_arcsec,
        wavgood0=header["WAVGOOD0"],
        wavgood1=header["WAVGOOD1"],
    )