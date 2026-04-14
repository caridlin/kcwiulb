from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS

from kcwiulb.wcs import wavelength_to_index


@dataclass
class SourceMaskResult:
    collapse_flux_input_path: Path
    collapse_var_input_path: Path
    apply_flux_input_path: Path
    mask_path: Path
    bg_masked_flux_path: Path
    n_auto_masked: int
    n_final_masked: int
    collapsed_flux: np.ndarray
    collapsed_snr: np.ndarray
    auto_mask: np.ndarray
    final_mask: np.ndarray


def _replace_fits_suffix(path: Path, suffix: str) -> Path:
    if path.suffix.lower() != ".fits":
        raise ValueError(f"Expected a .fits file, got {path}")
    return path.with_name(path.name[:-5] + suffix)


def _build_line_exclude_mask(
    header: fits.Header,
    n_wave: int,
    line_mask: tuple[float, float] | None,
) -> np.ndarray:
    keep = np.ones(n_wave, dtype=bool)

    if line_mask is None:
        return keep

    i0 = max(0, wavelength_to_index(line_mask[0], header))
    i1 = min(n_wave, wavelength_to_index(line_mask[1], header))
    if i1 > i0:
        keep[i0:i1] = False

    return keep


def _collapse_flux_and_var(
    flux_cube: np.ndarray,
    var_cube: np.ndarray,
    keep_wave_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flux_collapsed = np.sum(flux_cube[keep_wave_mask], axis=0)
    var_collapsed = np.sum(var_cube[keep_wave_mask], axis=0)
    return flux_collapsed, var_collapsed


def _make_snr_map(
    collapsed_flux: np.ndarray,
    collapsed_var: np.ndarray,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = collapsed_flux / np.sqrt(collapsed_var)
    snr[~np.isfinite(snr)] = 0.0
    return snr


def _auto_mask_from_sigma_clip(
    collapsed_flux: np.ndarray,
    sigma_clip_value: float,
) -> np.ndarray:
    clipped = sigma_clip(
        collapsed_flux,
        sigma=sigma_clip_value,
        maxiters=None,
        masked=True,
    )
    return np.array(clipped.mask, dtype=bool)


def _circle_region(
    ny: int,
    nx: int,
    x_center: float,
    y_center: float,
    radius_pix: float,
) -> np.ndarray:
    yy, xx = np.mgrid[:ny, :nx]
    rr2 = (xx - x_center) ** 2 + (yy - y_center) ** 2
    return rr2 <= radius_pix**2


def _sky_to_pixel(
    header: fits.Header,
    ra_deg: float,
    dec_deg: float,
) -> tuple[float, float]:
    wcs = WCS(header)
    x_pix, y_pix, _ = wcs.world_to_pixel_values(ra_deg, dec_deg, header["CRVAL3"])
    return float(x_pix), float(y_pix)


def _circles_to_region_mask(
    header: fits.Header,
    ny: int,
    nx: int,
    circles: list[tuple[str, float, float, float]] | None,
) -> np.ndarray:
    region = np.zeros((ny, nx), dtype=bool)

    if not circles:
        return region

    for mode, a, b, r in circles:
        if mode == "sky":
            x_c, y_c = _sky_to_pixel(header, a, b)
        elif mode == "pixel":
            x_c, y_c = float(a), float(b)
        else:
            raise ValueError(f"Unknown circle mode '{mode}'. Use 'sky' or 'pixel'.")

        region |= _circle_region(ny, nx, x_c, y_c, r)

    return region


def _filter_detected_mask_by_circles(
    detected_mask: np.ndarray,
    header: fits.Header,
    circles: list[tuple[str, float, float, float]] | None,
) -> np.ndarray:
    if not circles:
        return detected_mask.copy()

    ny, nx = detected_mask.shape
    keep_region = _circles_to_region_mask(
        header=header,
        ny=ny,
        nx=nx,
        circles=circles,
    )
    return detected_mask & keep_region


def run_source_mask(
    collapse_flux_path: str | Path,
    collapse_var_path: str | Path,
    apply_flux_path: str | Path,
    keep_circles: list[tuple[str, float, float, float]] | None,
    line_mask: tuple[float, float] | None,
    sigma_clip_value: float = 5.0,
    manual_filter_circles: list[tuple[str, float, float, float]] | None = None,
    masked_value: float = 0.0,
    overwrite: bool = True,
) -> SourceMaskResult:
    collapse_flux_path = Path(collapse_flux_path)
    collapse_var_path = Path(collapse_var_path)
    apply_flux_path = Path(apply_flux_path)

    with fits.open(collapse_flux_path) as hdul_flux:
        flux_cube = hdul_flux[0].data.astype(float)
        flux_header = hdul_flux[0].header.copy()

    with fits.open(collapse_var_path) as hdul_var:
        var_cube = hdul_var[0].data.astype(float)

    with fits.open(apply_flux_path) as hdul_apply:
        apply_cube = hdul_apply[0].data.astype(float)
        apply_header = hdul_apply[0].header.copy()

    if flux_cube.ndim != 3:
        raise ValueError(f"Expected 3D collapse flux cube, got shape {flux_cube.shape}")
    if flux_cube.shape != var_cube.shape:
        raise ValueError(
            f"Flux/variance shape mismatch: {flux_cube.shape} vs {var_cube.shape}"
        )
    if apply_cube.shape[1:] != flux_cube.shape[1:]:
        raise ValueError(
            f"Apply cube spatial shape {apply_cube.shape[1:]} does not match "
            f"collapse cube spatial shape {flux_cube.shape[1:]}"
        )

    n_wave, ny, nx = flux_cube.shape

    keep_wave_mask = _build_line_exclude_mask(
        header=flux_header,
        n_wave=n_wave,
        line_mask=line_mask,
    )

    collapsed_flux, collapsed_var = _collapse_flux_and_var(
        flux_cube=flux_cube,
        var_cube=var_cube,
        keep_wave_mask=keep_wave_mask,
    )

    collapsed_snr = _make_snr_map(
        collapsed_flux=collapsed_flux,
        collapsed_var=collapsed_var,
    )

    auto_mask = _auto_mask_from_sigma_clip(
        collapsed_flux=collapsed_flux,
        sigma_clip_value=sigma_clip_value,
    )

    keep_region = _circles_to_region_mask(
        header=flux_header,
        ny=ny,
        nx=nx,
        circles=keep_circles,
    )
    auto_mask[keep_region] = False

    final_mask = _filter_detected_mask_by_circles(
        detected_mask=auto_mask,
        header=flux_header,
        circles=manual_filter_circles,
    )
    final_mask[keep_region] = False

    masked_apply_cube = apply_cube.copy()
    masked_apply_cube[:, final_mask] = masked_value

    mask_path = _replace_fits_suffix(collapse_flux_path, ".mask.fits")
    bg_masked_flux_path = _replace_fits_suffix(apply_flux_path, ".mask.fits")

    fits.PrimaryHDU(
        data=final_mask.astype(np.uint8),
        header=flux_header.copy(),
    ).writeto(mask_path, overwrite=overwrite)

    fits.PrimaryHDU(
        data=masked_apply_cube,
        header=apply_header.copy(),
    ).writeto(bg_masked_flux_path, overwrite=overwrite)

    return SourceMaskResult(
        collapse_flux_input_path=collapse_flux_path,
        collapse_var_input_path=collapse_var_path,
        apply_flux_input_path=apply_flux_path,
        mask_path=mask_path,
        bg_masked_flux_path=bg_masked_flux_path,
        n_auto_masked=int(np.count_nonzero(auto_mask)),
        n_final_masked=int(np.count_nonzero(final_mask)),
        collapsed_flux=collapsed_flux,
        collapsed_snr=collapsed_snr,
        auto_mask=auto_mask,
        final_mask=final_mask,
    )