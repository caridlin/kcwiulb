from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from skimage.measure import label, regionprops_table


@dataclass
class PostADSDenoisingResult:
    flux_input_path: Path
    mask_input_path: Path
    kernel_r_input_path: Path

    denoised_flux_path: Path
    denoised_mask_path: Path

    radius_cut_arcsec: float
    min_kernel_pixels: float
    min_region_size: int

    n_detected_voxels_input: int
    n_detected_voxels_after_kernel_cut: int
    n_detected_voxels_final: int

    n_regions_input: int
    n_regions_kept: int


def _get_header2d(header3d: fits.Header) -> fits.Header:
    header2d = header3d.copy()

    keys_to_delete = [key for key in header2d.keys() if "3" in key]
    for key in keys_to_delete:
        del header2d[key]

    header2d["NAXIS"] = 2
    header2d["WCSDIM"] = 2
    return header2d


def _pixel_area_arcsec2(header: fits.Header) -> float:
    if header["NAXIS"] == 3:
        header = _get_header2d(header)

    yscale_deg, xscale_deg = proj_plane_pixel_scales(WCS(header))
    yscale_arcsec = yscale_deg * 3600.0
    xscale_arcsec = xscale_deg * 3600.0
    return float(yscale_arcsec * xscale_arcsec)


def _radius_grid_arcsec(
    header: fits.Header,
    ra_deg: float,
    dec_deg: float,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Return a 2D radius grid in arcsec relative to a sky position.
    """
    header2d = _get_header2d(header) if header["NAXIS"] == 3 else header.copy()
    wcs2d = WCS(header2d)

    y_grid, x_grid = np.mgrid[:ny, :nx]
    ra_grid, dec_grid = wcs2d.all_pix2world(x_grid, y_grid, 0)

    dra_arcsec = (ra_grid - ra_deg) * 3600.0 * np.cos(np.deg2rad(dec_deg))
    ddec_arcsec = (dec_grid - dec_deg) * 3600.0
    rr_arcsec = np.sqrt(dra_arcsec**2 + ddec_arcsec**2)

    return rr_arcsec.astype(float)


def _build_output_path(input_path: Path, suffix: str) -> Path:
    name = input_path.name

    if name.endswith(".fits"):
        prefix = name[:-len(".fits")]
        return input_path.with_name(f"{prefix}{suffix}")

    raise ValueError(f"Expected FITS path, got {input_path}")


def run_post_ads_denoising(
    flux_path: str | Path,
    mask_path: str | Path,
    kernel_r_path: str | Path,
    center_ra_deg: float,
    center_dec_deg: float,
    radius_cut_arcsec: float = 7.0,
    min_kernel_pixels: float = 5.0,
    min_region_size: int = 150,
    connectivity: int = 1,
    overwrite: bool = True,
) -> PostADSDenoisingResult:
    """
    Post-process ADS detections by removing compact noise-like features.

    Two filters are applied:

    1. Radial / kernel-size cut:
       outside `radius_cut_arcsec`, remove detected voxels whose spatial
       smoothing kernel is smaller than `min_kernel_pixels`.

    2. Connected-component filtering:
       label the 3D detection mask and remove connected regions smaller
       than `min_region_size`.
    """
    flux_path = Path(flux_path)
    mask_path = Path(mask_path)
    kernel_r_path = Path(kernel_r_path)

    with fits.open(flux_path) as hdul:
        flux_header = hdul[0].header.copy()
        flux_cube = hdul[0].data.astype(float)

    with fits.open(mask_path) as hdul:
        mask_cube = hdul[0].data.astype(float)

    with fits.open(kernel_r_path) as hdul:
        kernel_r_cube = hdul[0].data.astype(float)

    if flux_cube.shape != mask_cube.shape or flux_cube.shape != kernel_r_cube.shape:
        raise ValueError(
            "Flux cube, ADS mask cube, and spatial kernel cube must have the same shape."
        )

    nz, ny, nx = flux_cube.shape

    rr_arcsec = _radius_grid_arcsec(
        header=flux_header,
        ra_deg=center_ra_deg,
        dec_deg=center_dec_deg,
        ny=ny,
        nx=nx,
    )

    flux_filtered = flux_cube.copy()
    mask_filtered = mask_cube.copy()

    detected_initial = mask_filtered > 0
    n_detected_voxels_input = int(np.sum(detected_initial))

    small_kernel_far = (
        (mask_filtered > 0)
        & (kernel_r_cube < min_kernel_pixels)
        & (rr_arcsec[None, :, :] > radius_cut_arcsec)
    )

    flux_filtered[small_kernel_far] = 0.0
    mask_filtered[small_kernel_far] = 0.0

    n_detected_voxels_after_kernel_cut = int(np.sum(mask_filtered > 0))

    labeled = label(mask_filtered > 0, connectivity=connectivity)

    props = regionprops_table(
        labeled,
        intensity_image=flux_filtered,
        properties=("area", "label", "mean_intensity"),
    )

    region_areas = np.asarray(props["area"])
    region_labels = np.asarray(props["label"])

    keep_regions = region_areas > min_region_size

    kept_mask = np.zeros_like(mask_filtered, dtype=float)
    for reg_label in region_labels[keep_regions]:
        kept_mask[labeled == reg_label] = 1.0

    flux_denoised = flux_filtered * kept_mask
    mask_denoised = kept_mask

    n_regions_input = int(len(region_labels))
    n_regions_kept = int(np.sum(keep_regions))
    n_detected_voxels_final = int(np.sum(mask_denoised > 0))

    denoised_flux_path = _build_output_path(flux_path, ".denoise.fits")
    denoised_mask_path = _build_output_path(mask_path, ".denoise.fits")

    fits.PrimaryHDU(flux_denoised, header=flux_header).writeto(
        denoised_flux_path,
        overwrite=overwrite,
    )
    fits.PrimaryHDU(mask_denoised, header=flux_header).writeto(
        denoised_mask_path,
        overwrite=overwrite,
    )

    return PostADSDenoisingResult(
        flux_input_path=flux_path,
        mask_input_path=mask_path,
        kernel_r_input_path=kernel_r_path,
        denoised_flux_path=denoised_flux_path,
        denoised_mask_path=denoised_mask_path,
        radius_cut_arcsec=radius_cut_arcsec,
        min_kernel_pixels=min_kernel_pixels,
        min_region_size=min_region_size,
        n_detected_voxels_input=n_detected_voxels_input,
        n_detected_voxels_after_kernel_cut=n_detected_voxels_after_kernel_cut,
        n_detected_voxels_final=n_detected_voxels_final,
        n_regions_input=n_regions_input,
        n_regions_kept=n_regions_kept,
    )