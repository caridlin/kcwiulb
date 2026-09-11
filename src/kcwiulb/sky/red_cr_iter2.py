from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from shapely.geometry import Polygon as shapely_polygon

from kcwiulb.sky.utils import load_cube
from kcwiulb.coadd.blue import (
    get_header2d,
    build_common_header,
    _precompute_pixel_boxes,
    _precompute_coadd_pixel_radec,
    _normalize_input_paths,
    _print_row_progress,
)
from kcwiulb.sky.red_cr_iter1 import (
    RedCosmicRayResult,
    write_cr_masked_cube,
    _weighted_stats_from_samples,
)


def cosmic_ray_mask_red_group_iter2(
    input_paths: list[str | Path],
    group_name: str,
    pa: float,
    alpha: float = 5.0,
    beta: float = 5.0,
    gamma: float = 15.0,
    px_thresh: float = 0.1,
    qso_ra_deg: float | None = None,
    qso_dec_deg: float | None = None,
    qso_protect_radius_arcsec: float = 0.0,
    output_dir: str | Path | None = None,
    suffix: str = ".cr.fits",
) -> RedCosmicRayResult:
    """
    Identify cosmic rays in red-channel iter2 cubes using the original
    common-coadd-pixel comparison algorithm with bright-signal protection.

    For each common-grid pixel:

    1. Project the common pixel into every input cube.
    2. Collect every individual input spaxel with overlap > px_thresh.
    3. Compute the wavelength-by-wavelength median spectrum and its
       uncertainty from all contributing individual spaxels.
    4. For every contributing input spaxel:
         - if |data| < beta * uncertainty:
               flag when |data - median| >= alpha * sigma_median
         - if |data| >= beta * uncertainty:
               flag when |data - median| >= gamma * sigma_median
    5. If the spaxel lies within qso_protect_radius_arcsec of the supplied
       QSO RA/Dec, do not flag any wavelengths in that spaxel.

    Notes
    -----
    - Positive and negative deviations are both tested.
    - Each input spaxel is tested individually.
    - QSO protection is spatial and uses sky coordinates, so it follows
      the source across dithered / shifted cubes.
    """
    input_paths = _normalize_input_paths(input_paths)

    if (qso_ra_deg is None) != (qso_dec_deg is None):
        raise ValueError("qso_ra_deg and qso_dec_deg must either both be given or both be None.")

    use_qso_protection = (
        qso_ra_deg is not None
        and qso_dec_deg is not None
        and qso_protect_radius_arcsec > 0
    )

    qso_coord = None
    if use_qso_protection:
        qso_coord = SkyCoord(
            ra=qso_ra_deg * u.deg,
            dec=qso_dec_deg * u.deg,
            frame="icrs",
        )

    print(f"[cosmic_ray_mask_red_group_iter2] Starting group '{group_name}'")
    print(f"[cosmic_ray_mask_red_group_iter2] PA: {pa}")
    print(f"[cosmic_ray_mask_red_group_iter2] alpha: {alpha}")
    print(f"[cosmic_ray_mask_red_group_iter2] beta: {beta}")
    print(f"[cosmic_ray_mask_red_group_iter2] gamma: {gamma}")
    print(f"[cosmic_ray_mask_red_group_iter2] PX_THRESH: {px_thresh}")

    if use_qso_protection:
        print(f"[cosmic_ray_mask_red_group_iter2] QSO RA: {qso_ra_deg}")
        print(f"[cosmic_ray_mask_red_group_iter2] QSO Dec: {qso_dec_deg}")
        print(
            "[cosmic_ray_mask_red_group_iter2] "
            f"QSO protect radius: {qso_protect_radius_arcsec:.3f} arcsec"
        )
    else:
        print("[cosmic_ray_mask_red_group_iter2] QSO protection: disabled")

    print(f"[cosmic_ray_mask_red_group_iter2] N input cubes: {len(input_paths)}")
    for i, path in enumerate(input_paths, start=1):
        print(f"  Input {i:02d}: {path}")

    cubes: list[np.ndarray] = []
    uncerts: list[np.ndarray] = []
    headers: list[fits.Header] = []
    wcs2d: list[WCS] = []
    pixel_boxes: list[list[list]] = []

    print("[cosmic_ray_mask_red_group_iter2] Loading cubes...")
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

    print("[cosmic_ray_mask_red_group_iter2] Building common WCS/header...")
    common_hdr, common_wcs, _, _ = build_common_header(headers, pa=pa)
    common_size_y = common_hdr["NAXIS2"]
    common_size_x = common_hdr["NAXIS1"]

    print("[cosmic_ray_mask_red_group_iter2] Common grid:")
    print(f"  NAXIS1 = {common_size_x}")
    print(f"  NAXIS2 = {common_size_y}")

    print("[cosmic_ray_mask_red_group_iter2] Precomputing coadd pixel footprints...")
    common_pixel_radec = _precompute_coadd_pixel_radec(
        common_wcs,
        common_size_y,
        common_size_x,
    )

    cr_masks = [np.zeros_like(cube, dtype=bool) for cube in cubes]

    print("[cosmic_ray_mask_red_group_iter2] Starting main loop...")

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

                for y_i in range(yb0, yb1):
                    for x_i in range(xb0, xb1):
                        pixel = pixel_boxes[i][y_i][x_i]
                        overlap = pixel_common_proj.intersection(pixel).area / pixel.area

                        if overlap <= px_thresh:
                            continue

                        protected_qso = False

                        if use_qso_protection:
                            ra_i, dec_i = wcs2d[i].all_pix2world(
                                x_i,
                                y_i,
                                0,
                            )

                            spaxel_coord = SkyCoord(
                                ra=ra_i * u.deg,
                                dec=dec_i * u.deg,
                                frame="icrs",
                            )

                            separation = spaxel_coord.separation(qso_coord).arcsec
                            protected_qso = separation <= qso_protect_radius_arcsec

                        contributors.append(
                            {
                                "cube_index": i,
                                "y": y_i,
                                "x": x_i,
                                "spec": cube[:, y_i, x_i],
                                "uncert": uncerts[i][:, y_i, x_i],
                                "overlap": overlap,
                                "protected_qso": protected_qso,
                            }
                        )

            if len(contributors) < 2:
                continue

            all_specs = [c["spec"] for c in contributors]
            all_uncerts = [c["uncert"] for c in contributors]

            med_spec, med_uncert = _weighted_stats_from_samples(
                all_specs,
                all_uncerts,
            )

            if med_spec is None:
                continue

            for contributor in contributors:
                cube_index = contributor["cube_index"]
                y_i = contributor["y"]
                x_i = contributor["x"]
                spec = contributor["spec"]
                spec_uncert = contributor["uncert"]

                if contributor["protected_qso"]:
                    continue

                delta = np.abs(spec - med_spec)

                low_snr = np.abs(spec) < beta * spec_uncert
                high_snr = ~low_snr

                bad_lambda = (
                    (low_snr & (delta >= alpha * med_uncert))
                    | (high_snr & (delta >= gamma * med_uncert))
                )

                if not np.any(bad_lambda):
                    continue

                cr_masks[cube_index][:, y_i, x_i] |= bad_lambda

    masked_per_cube = [int(mask.sum()) for mask in cr_masks]
    n_masked_total = int(sum(masked_per_cube))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    print("[cosmic_ray_mask_red_group_iter2] Writing outputs...")

    for path, cube, header, uncert, mask in zip(
        input_paths,
        cubes,
        headers,
        uncerts,
        cr_masks,
    ):
        if output_dir is None:
            out_path = path.with_name(path.stem + suffix)
        else:
            out_path = output_dir / f"{path.stem}{suffix}"

        write_cr_masked_cube(
            out_path,
            cube,
            header,
            uncert,
            mask,
        )

        output_paths.append(out_path)

        print(
            f"  Wrote: {out_path} | masked voxels = {int(mask.sum())}",
            flush=True,
        )

    print("[cosmic_ray_mask_red_group_iter2] Done.")
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