from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry import Polygon as shapely_polygon

from kcwiulb.coadd.blue import (
    load_cube,
    get_header2d,
    build_common_header,
    _precompute_pixel_boxes,
    _precompute_coadd_pixel_radec,
    _normalize_input_paths,
    _print_row_progress,
)


@dataclass
class RedCosmicRayResult:
    group_name: str
    input_paths: list[Path]
    output_paths: list[Path]
    n_cubes: int
    alpha: float
    px_thresh: float
    pa: float
    n_masked_total: int
    masked_per_cube: list[int]
    common_shape: tuple[int, int]


def write_cr_masked_cube(
    path: Path,
    data: np.ndarray,
    header: fits.Header,
    uncert: np.ndarray,
    cr_mask: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(data=data, header=header),
            fits.ImageHDU(data=uncert, name="UNCERT"),
            fits.ImageHDU(data=cr_mask.astype(np.uint8), name="CRMASK"),
        ]
    )
    hdul.writeto(path, overwrite=True)
    return path


def _weighted_stats_from_samples(
    values: list[np.ndarray],
    uncerts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if len(values) == 0:
        return None, None

    arr = np.asarray(values, dtype=float)
    uarr = np.asarray(uncerts, dtype=float)

    if arr.ndim != 2:
        raise ValueError("Expected 2D stacked spectra array.")

    med = np.nanmedian(arr, axis=0)

    n = arr.shape[0]
    if n < 2:
        return med, np.full_like(med, np.inf, dtype=float)

    sig = np.sqrt(
        np.nansum(np.square(uarr), axis=0) * np.pi / (2.0 * (n - 1))
    )

    bad = ~np.isfinite(sig) | (sig <= 0)
    sig[bad] = np.inf

    return med, sig


def cosmic_ray_mask_red_group(
    input_paths: list[str | Path],
    group_name: str,
    pa: float,
    alpha: float = 5.0,
    px_thresh: float = 0.1,
    output_dir: str | Path | None = None,
    suffix: str = ".cr.fits",
) -> RedCosmicRayResult:
    """
    Identify cosmic rays in red-channel cubes using a common coadd-space comparison.

    For each pixel in a common WCS plane:
    1. Collect overlapping spectra from all input cubes.
    2. Build a median comparison spectrum.
    3. Flag voxels in each contributing input spectrum where:
           data > median + alpha * sigma_median
       using the corresponding overlapping input spaxels.

    Notes
    -----
    - This writes out a 3-HDU FITS file for each cube:
        HDU 0: original flux cube
        HDU 1: original uncertainty cube
        HDU 2: cosmic-ray mask cube (0/1)
    - The input flux is not modified here; only the mask is produced.
    - By default, outputs are written next to each input cube.
      If output_dir is given, all outputs are written there instead.
    """
    input_paths = _normalize_input_paths(input_paths)

    print(f"[cosmic_ray_mask_red_group] Starting group '{group_name}'")
    print(f"[cosmic_ray_mask_red_group] PA: {pa}")
    print(f"[cosmic_ray_mask_red_group] alpha: {alpha}")
    print(f"[cosmic_ray_mask_red_group] PX_THRESH: {px_thresh}")
    print(f"[cosmic_ray_mask_red_group] N input cubes: {len(input_paths)}")
    for i, p in enumerate(input_paths, start=1):
        print(f"  Input {i:02d}: {p}")

    cubes: list[np.ndarray] = []
    uncerts: list[np.ndarray] = []
    headers: list[fits.Header] = []
    wcs2d: list[WCS] = []
    pixel_boxes: list[list[list]] = []

    print("[cosmic_ray_mask_red_group] Loading cubes...")
    for i, path in enumerate(input_paths, start=1):
        data, header, uncert = load_cube(path)
        cubes.append(data)
        uncerts.append(uncert)
        headers.append(header)
        wcs2d.append(WCS(get_header2d(header)))
        pixel_boxes.append(_precompute_pixel_boxes(data.shape[1], data.shape[2]))

        print(
            f"  Loaded {i:02d}/{len(input_paths)} | "
            f"{path.name} | shape={data.shape}",
            flush=True,
        )

    print("[cosmic_ray_mask_red_group] Building common WCS/header...")
    common_hdr, common_wcs, _, _ = build_common_header(headers, pa=pa)
    common_size_y = common_hdr["NAXIS2"]
    common_size_x = common_hdr["NAXIS1"]

    print("[cosmic_ray_mask_red_group] Common grid:")
    print(f"  NAXIS1 = {common_size_x}")
    print(f"  NAXIS2 = {common_size_y}")

    print("[cosmic_ray_mask_red_group] Precomputing coadd pixel footprints...")
    common_pixel_radec = _precompute_coadd_pixel_radec(common_wcs, common_size_y, common_size_x)

    cr_masks = [np.zeros_like(cube, dtype=bool) for cube in cubes]

    print("[cosmic_ray_mask_red_group] Starting main loop...")
    import time
    loop_start = time.time()

    for y_j in range(common_size_y):
        _print_row_progress(y_j, common_size_y, loop_start, every=5)

        for x_k in range(common_size_x):
            pixel_common_radec = common_pixel_radec[y_j][x_k]

            contributors: list[dict] = []

            for i, cube in enumerate(cubes):
                pixel_common_proj = shapely_polygon(
                    wcs2d[i].all_world2pix(pixel_common_radec, 0)
                )
                region_bounds = list(pixel_common_proj.exterior.bounds)

                xb0, yb0, xb1, yb1 = (int(round(v)) for v in region_bounds)
                xb0 = max(xb0, 0)
                yb0 = max(yb0, 0)
                xb1 = min(xb1 + 1, cube.shape[2])
                yb1 = min(yb1 + 1, cube.shape[1])

                if not (xb0 < xb1 and yb0 < yb1):
                    continue

                overlap_tot = 0.0
                spec_sum = np.zeros(cube.shape[0], dtype=float)
                var_sum = np.zeros(cube.shape[0], dtype=float)
                overlapping_pixels: list[tuple[int, int, float]] = []

                for y_i in range(yb0, yb1):
                    for x_i in range(xb0, xb1):
                        pixel = pixel_boxes[i][y_i][x_i]
                        overlap = pixel_common_proj.intersection(pixel).area / pixel.area
                        if overlap <= 0:
                            continue

                        overlap_tot += overlap
                        overlapping_pixels.append((y_i, x_i, overlap))
                        spec_sum += overlap * cube[:, y_i, x_i]
                        var_sum += np.square(overlap * uncerts[i][:, y_i, x_i])

                if overlap_tot <= px_thresh:
                    continue

                spec = spec_sum / overlap_tot
                spec_uncert = np.sqrt(var_sum) / overlap_tot

                contributors.append(
                    {
                        "cube_index": i,
                        "spec": spec,
                        "spec_uncert": spec_uncert,
                        "pixels": overlapping_pixels,
                    }
                )

            if len(contributors) < 2:
                continue

            all_specs = [c["spec"] for c in contributors]
            all_uncerts = [c["spec_uncert"] for c in contributors]

            med_spec, med_uncert = _weighted_stats_from_samples(all_specs, all_uncerts)
            if med_spec is None:
                continue

            threshold = med_spec + alpha * med_uncert

            for contributor in contributors:
                cube_index = contributor["cube_index"]
                spec = contributor["spec"]
                bad_lambda = spec > threshold

                if not np.any(bad_lambda):
                    continue

                for y_i, x_i, _ in contributor["pixels"]:
                    cr_masks[cube_index][:, y_i, x_i] |= bad_lambda

    masked_per_cube = [int(mask.sum()) for mask in cr_masks]
    n_masked_total = int(sum(masked_per_cube))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    print("[cosmic_ray_mask_red_group] Writing outputs...")
    for path, cube, header, uncert, mask in zip(input_paths, cubes, headers, uncerts, cr_masks):
        if output_dir is None:
            out_path = path.with_name(path.stem + suffix)
        else:
            out_path = output_dir / f"{path.stem}{suffix}"

        write_cr_masked_cube(out_path, cube, header, uncert, mask)
        output_paths.append(out_path)

        print(
            f"  Wrote: {out_path} | masked voxels = {int(mask.sum())}",
            flush=True,
        )

    print("[cosmic_ray_mask_red_group] Done.")
    print(f"  Total masked voxels: {n_masked_total}")

    return RedCosmicRayResult(
        group_name=group_name,
        input_paths=input_paths,
        output_paths=output_paths,
        n_cubes=len(input_paths),
        alpha=alpha,
        px_thresh=px_thresh,
        pa=pa,
        n_masked_total=n_masked_total,
        masked_per_cube=masked_per_cube,
        common_shape=(common_size_y, common_size_x),
    )