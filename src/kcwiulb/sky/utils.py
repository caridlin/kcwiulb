from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import ndimage

from kcwiulb.wcs import wavelength_to_index, index_to_wavelength


def extract_cube_id(text: str) -> str:
    return text.split("(")[0].strip()


def extract_field(text: str) -> str | None:
    if "(" in text and ")" in text:
        return text.split("(")[1].split(")")[0].strip()
    return None


def read_sky_map_iter1(path: str | Path) -> dict[str, list[dict[str, str]]]:
    path = Path(path)
    groups: dict[str, list[dict[str, str]]] = {}
    current_field = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_field = line[1:-1]
                groups[current_field] = []
                continue

            science, sky1, sky2 = [x.strip() for x in line.split("|")]
            groups[current_field].append(
                {
                    "science": extract_cube_id(science),
                    "sky1": extract_cube_id(sky1),
                    "sky1_field": extract_field(sky1),
                    "sky2": extract_cube_id(sky2),
                    "sky2_field": extract_field(sky2),
                }
            )

    return groups


def read_sky_map_iter2(path: str | Path) -> dict[str, list[dict[str, str]]]:
    path = Path(path)
    groups: dict[str, list[dict[str, str]]] = {}
    current_field = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_field = line[1:-1]
                groups[current_field] = []
                continue

            science, sky1, sky2, sky3, sky4 = [x.strip() for x in line.split("|")]
            groups[current_field].append(
                {
                    "science": extract_cube_id(science),
                    "sky1": extract_cube_id(sky1),
                    "sky1_field": extract_field(sky1),
                    "sky2": extract_cube_id(sky2),
                    "sky2_field": extract_field(sky2),
                    "sky3": extract_cube_id(sky3),
                    "sky3_field": extract_field(sky3),
                    "sky4": extract_cube_id(sky4),
                    "sky4_field": extract_field(sky4),
                }
            )

    return groups


def load_cube(path: str | Path) -> tuple[np.ndarray, fits.Header, np.ndarray]:
    path = Path(path)
    with fits.open(path) as hdul:
        cube = hdul[0].data.copy()
        header = hdul[0].header.copy()
        uncert = hdul[1].data.copy()
    return cube, header, uncert


def build_wavelength_axis(header: fits.Header, nw: int) -> np.ndarray:
    return np.array([index_to_wavelength(i, header) for i in range(nw)])


def collapse_ranges(
    cube: np.ndarray,
    header: fits.Header,
    wavelength_ranges: list[tuple[float, float]],
) -> np.ndarray:
    slabs = []
    for wl0, wl1 in wavelength_ranges:
        i0 = wavelength_to_index(wl0, header)
        i1 = wavelength_to_index(wl1, header)
        slabs.append(cube[i0:i1])
    return np.concatenate(slabs, axis=0)


def whiteband_image(
    cube: np.ndarray,
    header: fits.Header,
    wavelength_ranges: list[tuple[float, float]],
) -> np.ndarray:
    return np.sum(collapse_ranges(cube, header, wavelength_ranges), axis=0)


def sigma_clip_mask_2d(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    clipped = sigma_clip(image, sigma=sigma, maxiters=None, masked=True)
    return np.array(clipped.mask, dtype=bool)


def combine_masks_2d(*masks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= m
    return out


def mask2d_to_mask3d(mask2d: np.ndarray, cube_shape: tuple[int, int, int]) -> np.ndarray:
    nw, ny, nx = cube_shape
    return np.broadcast_to(mask2d[None, :, :], (nw, ny, nx)).copy()


def masked_median_spectrum(
    cube: np.ndarray,
    uncert: np.ndarray,
    mask2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask3d = mask2d_to_mask3d(mask2d, cube.shape)
    cm = ma.masked_array(cube, mask=mask3d)
    um = ma.masked_array(uncert, mask=mask3d)

    spec = np.ma.median(cm, axis=(1, 2)).filled(np.nan)

    n_used = np.sum(~mask2d)
    if n_used <= 1:
        raise ValueError("Too few unmasked spaxels to compute sky spectrum.")

    spec_unc = np.sqrt(
        np.nansum(np.square(um.filled(np.nan)), axis=(1, 2)) * np.pi / (2 * (n_used - 1))
    )
    return spec, spec_unc


def weighted_quantile(values, quantiles=0.5, sample_weight=None):
    values = np.array(values).flatten()
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    sample_weight = np.array(sample_weight).flatten()

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


def weighted_median_filter_1d(
    spec: np.ndarray,
    spec_unc: np.ndarray,
    width: int = 125,
    quantile: float = 0.3,
) -> np.ndarray:
    return ndimage.generic_filter(
        spec,
        weighted_quantile,
        size=width,
        mode="reflect",
        extra_keywords={
            "quantiles": quantile,
            "sample_weight": np.abs(spec_unc),
        },
    )


def write_cube(
    header: fits.Header,
    data: np.ndarray,
    uncert: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hdu0 = fits.PrimaryHDU(data=data, header=header.copy())
    hdu1 = fits.ImageHDU(data=uncert)
    fits.HDUList([hdu0, hdu1]).writeto(output_path, overwrite=True)
    return output_path


def resolve_cube_path(base: Path, channel: str, field: str, cube_id: str, suffix: str) -> Path:
    return base / channel / field / f"{cube_id}{suffix}"