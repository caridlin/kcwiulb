from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from kcwiulb.wcs import wavelength_to_index


@dataclass
class CropParams:
    xcrop: tuple[int, int]
    ycrop: tuple[int, int]
    wav_crop: tuple[float, float]


@dataclass
class CropResult:
    input_path: Path
    output_path: Path | None
    original_shape: tuple[int, ...]
    cropped_shape: tuple[int, ...]
    xcrop: tuple[int, int]
    ycrop: tuple[int, int]
    zcrop: tuple[int, int]
    wav_crop: tuple[float, float]


def get_wavgood_crop(header: fits.Header) -> tuple[float, float]:
    """
    Return a wavelength crop based on WAVGOOD0/WAVGOOD1, rounded inward
    to the nearest integer Angstrom.
    """
    if "WAVGOOD0" not in header or "WAVGOOD1" not in header:
        raise KeyError("Header missing WAVGOOD0/WAVGOOD1 keywords.")

    wav_min = float(np.ceil(header["WAVGOOD0"]))
    wav_max = float(np.floor(header["WAVGOOD1"]))

    if wav_max <= wav_min:
        raise ValueError(
            f"Invalid WAVGOOD range: WAVGOOD0={header['WAVGOOD0']}, WAVGOOD1={header['WAVGOOD1']}"
        )

    return wav_min, wav_max


def get_wavelength_indices(
    wav_min: float,
    wav_max: float,
    header: fits.Header,
) -> tuple[int, int]:
    """Return wavelength index range corresponding to (wav_min, wav_max)."""
    i0 = wavelength_to_index(wav_min, header)
    i1 = wavelength_to_index(wav_max, header)

    if i1 <= i0:
        raise ValueError(
            f"Invalid wavelength crop: ({wav_min}, {wav_max}) -> ({i0}, {i1})"
        )

    return i0, i1


def update_header_for_crop(
    header: fits.Header,
    xcrop: tuple[int, int],
    ycrop: tuple[int, int],
    zcrop: tuple[int, int],
    new_shape: tuple[int, int, int],
) -> fits.Header:
    """Return a cropped-header copy with updated CRPIX and NAXIS values."""
    new_header = header.copy()

    x0, _ = xcrop
    y0, _ = ycrop
    z0, _ = zcrop

    if "CRPIX1" in new_header:
        new_header["CRPIX1"] -= x0
    if "CRPIX2" in new_header:
        new_header["CRPIX2"] -= y0
    if "CRPIX3" in new_header:
        new_header["CRPIX3"] -= z0

    nz, ny, nx = new_shape
    if "NAXIS1" in new_header:
        new_header["NAXIS1"] = nx
    if "NAXIS2" in new_header:
        new_header["NAXIS2"] = ny
    if "NAXIS3" in new_header:
        new_header["NAXIS3"] = nz

    return new_header


def crop_array_with_indices(
    cube: np.ndarray,
    xcrop: tuple[int, int],
    ycrop: tuple[int, int],
    zcrop: tuple[int, int],
) -> np.ndarray:
    """Crop a 3D array using precomputed x/y/z index ranges."""
    x0, x1 = xcrop
    y0, y1 = ycrop
    z0, z1 = zcrop

    cropped = cube[z0:z1, y0:y1, x0:x1]

    if cropped.size == 0:
        raise ValueError(
            f"Cropping produced an empty cube: x={xcrop}, y={ycrop}, z={zcrop}"
        )

    return cropped


def crop_cube_data(
    cube: np.ndarray,
    header: fits.Header,
    xcrop: tuple[int, int],
    ycrop: tuple[int, int],
    wav_crop: tuple[float, float],
) -> tuple[np.ndarray, fits.Header, tuple[int, int]]:
    """
    Crop a cube in wavelength, y, and x using a single continuous wavelength range.
    """
    zcrop = get_wavelength_indices(wav_crop[0], wav_crop[1], header)

    cropped = crop_array_with_indices(
        cube=cube,
        xcrop=xcrop,
        ycrop=ycrop,
        zcrop=zcrop,
    )

    new_header = update_header_for_crop(
        header=header,
        xcrop=xcrop,
        ycrop=ycrop,
        zcrop=zcrop,
        new_shape=cropped.shape,
    )

    return cropped, new_header, zcrop


def crop_fits_cube(
    input_path: str | Path,
    output_path: str | Path | None,
    xcrop: tuple[int, int],
    ycrop: tuple[int, int],
    wav_crop: tuple[float, float],
) -> CropResult:
    """
    Crop a KCWI FITS cube and write the result.

    Assumes:
    - HDU 0 is the science cube
    - additional 3D HDUs with the same shape are cropped identically
    - non-3D extensions are preserved unchanged
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".c.fits")
    output_path = Path(output_path)

    with fits.open(input_path) as hdul:
        science_data = hdul[0].data
        science_header = hdul[0].header

        original_shape = science_data.shape
        cropped_hdus = []

        # Compute crop from science header only
        zcrop = get_wavelength_indices(wav_crop[0], wav_crop[1], science_header)

        # Crop science cube
        cropped_data = crop_array_with_indices(
            cube=science_data,
            xcrop=xcrop,
            ycrop=ycrop,
            zcrop=zcrop,
        )
        cropped_header = update_header_for_crop(
            header=science_header,
            xcrop=xcrop,
            ycrop=ycrop,
            zcrop=zcrop,
            new_shape=cropped_data.shape,
        )
        cropped_hdus.append(fits.PrimaryHDU(data=cropped_data, header=cropped_header))

        # Crop other matching 3D extensions using same indices
        for hdu in hdul[1:]:
            if getattr(hdu, "data", None) is None:
                cropped_hdus.append(hdu.copy())
                continue

            data = hdu.data
            header = hdu.header

            if (
                isinstance(data, np.ndarray)
                and data.ndim == 3
                and data.shape == original_shape
            ):
                cropped_ext = crop_array_with_indices(
                    cube=data,
                    xcrop=xcrop,
                    ycrop=ycrop,
                    zcrop=zcrop,
                )

                # If extension header has full WCS, update it; otherwise keep minimal header
                if all(k in header for k in ("CRPIX1", "CRPIX2", "CRPIX3")):
                    cropped_ext_header = update_header_for_crop(
                        header=header,
                        xcrop=xcrop,
                        ycrop=ycrop,
                        zcrop=zcrop,
                        new_shape=cropped_ext.shape,
                    )
                    cropped_hdus.append(
                        fits.ImageHDU(
                            data=cropped_ext,
                            header=cropped_ext_header,
                            name=hdu.name,
                        )
                    )
                else:
                    cropped_hdus.append(
                        fits.ImageHDU(
                            data=cropped_ext,
                            name=hdu.name,
                        )
                    )
            else:
                cropped_hdus.append(hdu.copy())

        fits.HDUList(cropped_hdus).writeto(output_path, overwrite=True)

    return CropResult(
        input_path=input_path,
        output_path=output_path,
        original_shape=original_shape,
        cropped_shape=cropped_data.shape,
        xcrop=xcrop,
        ycrop=ycrop,
        zcrop=zcrop,
        wav_crop=wav_crop,
    )