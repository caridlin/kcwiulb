from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from kcwiulb.wcs import wavelength_to_index, index_to_wavelength


@dataclass
class SpectralWindowResult:
    input_paths: list[Path]
    output_paths: list[Path]
    wavelength_min: float
    wavelength_max: float
    index_start: int
    index_end: int
    n_spectral_pixels: int
    wavelength_min_actual: float
    wavelength_max_actual: float


# ===============================
# Filename helpers
# ===============================
def _fits_output_path(path: Path, label: str | None) -> Path:
    """
    Example:
    coadd_blue_a_sky.wc.fits → coadd_blue_a_sky.wc.ha.fits
    """
    if label is None:
        label = "window"

    return path.with_suffix("").with_name(f"{path.stem}.{label}.fits")


def _npy_output_path(path: Path, label: str | None) -> Path:
    """
    Example:
    coadd_blue_a_sky_cov_data.npy → coadd_blue_a_sky_cov_data_ha.npy
    """
    if label is None:
        label = "window"

    return path.with_name(f"{path.stem}_{label}{path.suffix}")


# ===============================
# Header update
# ===============================
def _update_spectral_header(header: fits.Header, index_start: int) -> fits.Header:
    new_header = header.copy()

    if "CRPIX3" in new_header:
        new_header["CRPIX3"] = new_header["CRPIX3"] - index_start

    return new_header


# ===============================
# FITS crop
# ===============================
def crop_spectral_window_fits(
    input_path: str | Path,
    wavelength_min: float,
    wavelength_max: float,
    label: str | None = None,
    overwrite: bool = True,
) -> tuple[Path, int, int, float, float]:

    input_path = Path(input_path)
    output_path = _fits_output_path(input_path, label)

    with fits.open(input_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()

        i0 = max(0, wavelength_to_index(wavelength_min, header))
        i1 = min(data.shape[0], wavelength_to_index(wavelength_max, header))

        if i1 <= i0:
            raise ValueError("Invalid wavelength window")

        cropped = data[i0:i1].copy()
        new_header = _update_spectral_header(header, i0)

        hdul[0].data = cropped
        hdul[0].header = new_header
        hdul.writeto(output_path, overwrite=overwrite)

    wl_min_actual = index_to_wavelength(i0, header)
    wl_max_actual = index_to_wavelength(i1 - 1, header)

    return output_path, i0, i1, wl_min_actual, wl_max_actual


# ===============================
# Covariance crop
# ===============================
def crop_covariance_data(
    cov_data_path: str | Path,
    index_start: int,
    index_end: int,
    label: str | None = None,
    overwrite: bool = True,
) -> Path:
    """
    Crop covariance data along the spectral axis, matching FITS crop.
    
    cov_data shape: (n_pairs, n_wave)
    """
    cov_data_path = Path(cov_data_path)
    output_path = _npy_output_path(cov_data_path, label)

    data = np.load(cov_data_path)

    # --- THIS IS THE CRUCIAL LINE (same as your notebook) ---
    cropped = data[:, index_start:index_end].copy()

    np.save(output_path, cropped)

    return output_path


# ===============================
# Group function
# ===============================
def crop_spectral_window_group(
    flux_path: str | Path,
    var_path: str | Path,
    cov_data_path: str | Path,
    wavelength_min: float,
    wavelength_max: float,
    label: str | None = None,
) -> SpectralWindowResult:

    flux_path = Path(flux_path)
    var_path = Path(var_path)
    cov_data_path = Path(cov_data_path)

    # --- Flux ---
    flux_out, i0, i1, wl0, wl1 = crop_spectral_window_fits(
        flux_path,
        wavelength_min,
        wavelength_max,
        label=label,
    )

    # --- Variance ---
    var_out, _, _, _, _ = crop_spectral_window_fits(
        var_path,
        wavelength_min,
        wavelength_max,
        label=label,
    )

    # --- Covariance ---
    cov_out = crop_covariance_data(
        cov_data_path,
        index_start=i0,
        index_end=i1,
        label=label,
    )

    return SpectralWindowResult(
        input_paths=[flux_path, var_path, cov_data_path],
        output_paths=[flux_out, var_out, cov_out],
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        index_start=i0,
        index_end=i1,
        n_spectral_pixels=i1 - i0,
        wavelength_min_actual=wl0,
        wavelength_max_actual=wl1,
    )