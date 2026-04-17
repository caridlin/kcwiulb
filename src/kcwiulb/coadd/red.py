from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon as shapely_polygon

from kcwiulb.coadd.blue import (
    load_cube,
    get_header2d,
    get_pxarea_arcsec,
    build_common_header,
    write_fits_cube,
    _precompute_pixel_boxes,
    _precompute_coadd_pixel_radec,
    _normalize_input_paths,
    _print_row_progress,
)


@dataclass
class RedCoaddResult:
    group_name: str
    product: str
    input_paths: list[Path]
    mask_paths: list[Path]
    output_flux_path: Path
    output_var_path: Path
    output_cov_data_path: Path
    output_cov_coord_path: Path
    n_cubes: int
    shape: tuple[int, int, int]


def load_cr_mask(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        if len(hdul) < 3:
            raise ValueError(f"Mask file has no extension 2: {path}")
        return hdul[2].data.astype(bool)


def _normalize_mask_paths(mask_paths: list[str | Path], n_expected: int) -> list[Path]:
    paths = [Path(p) for p in mask_paths]
    if len(paths) != n_expected:
        raise ValueError(
            f"mask_paths length mismatch: got {len(paths)}, expected {n_expected}"
        )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing mask cube: {path}")
    return paths


def coadd_red_group(
    input_paths: list[str | Path],
    mask_paths: list[str | Path],
    group_name: str,
    product: str,
    pa: float,
    px_thresh: float = 0.1,
    output_dir: str | Path | None = None,
) -> RedCoaddResult:
    """
    Red coadd with notebook-faithful CR-mask handling.

    Parameters
    ----------
    input_paths
        Flux/variance cubes to coadd, e.g.
        *_icubes.wc.c.sky.cr.sky.cr.sky.fits
        or
        *_icubes.wc.c.sky.cr.sky2.cr.sky2.fits

    mask_paths
        CR-mask source cubes, e.g.
        *_icubes.wc.c.sky.cr.sky2.cr.fits
        The CR mask is always read from extension 2 of these files.
    """
    input_paths = _normalize_input_paths(input_paths)
    mask_paths = _normalize_mask_paths(mask_paths, len(input_paths))

    print(f"[coadd_red_group] Starting group '{group_name}'")
    print(f"[coadd_red_group] Product: {product}")
    print(f"[coadd_red_group] PA: {pa}")
    print(f"[coadd_red_group] PX_THRESH: {px_thresh}")
    print(f"[coadd_red_group] N input cubes: {len(input_paths)}")
    for i, (p_in, p_mask) in enumerate(zip(input_paths, mask_paths), start=1):
        print(f"  Input {i:02d}: {p_in}")
        print(f"  Mask  {i:02d}: {p_mask}")

    cubes = []
    uncerts = []
    masks = []
    headers = []
    wcs2d = []
    px_area = []
    t_exp = []
    pixel_boxes = []

    print("[coadd_red_group] Loading cubes...")
    for i, (cube_path, mask_path) in enumerate(zip(input_paths, mask_paths), start=1):
        data, header, uncert = load_cube(cube_path)
        mask = load_cr_mask(mask_path)

        if mask.shape != data.shape:
            raise ValueError(
                f"Mask shape mismatch for {mask_path}: got {mask.shape}, expected {data.shape}"
            )

        cubes.append(data)
        uncerts.append(uncert)
        masks.append(mask)
        headers.append(header)
        wcs2d.append(WCS(get_header2d(header)))
        px_area.append(get_pxarea_arcsec(header))
        t_exp.append(header["XPOSURE"])
        pixel_boxes.append(_precompute_pixel_boxes(data.shape[1], data.shape[2]))

        print(
            f"  Loaded {i:02d}/{len(input_paths)} | "
            f"{cube_path.name} | shape={data.shape} | XPOSURE={header['XPOSURE']}",
            flush=True,
        )

    print("[coadd_red_group] Building common WCS/header...")
    coadd_hdr, coadd_wcs, coadd_px_area, footprints = build_common_header(headers, pa=pa)

    coadd_size_w = coadd_hdr["NAXIS3"]
    coadd_size_y = coadd_hdr["NAXIS2"]
    coadd_size_x = coadd_hdr["NAXIS1"]

    print("[coadd_red_group] Coadd grid:")
    print(f"  NAXIS1 = {coadd_size_x}")
    print(f"  NAXIS2 = {coadd_size_y}")
    print(f"  NAXIS3 = {coadd_size_w}")
    print(f"  CRPIX1 = {coadd_hdr['CRPIX1']}")
    print(f"  CRPIX2 = {coadd_hdr['CRPIX2']}")
    print(f"  CRVAL1 = {coadd_hdr['CRVAL1']}")
    print(f"  CRVAL2 = {coadd_hdr['CRVAL2']}")
    print(f"  CD1_1  = {coadd_hdr['CD1_1']}")
    print(f"  CD1_2  = {coadd_hdr['CD1_2']}")
    print(f"  CD2_1  = {coadd_hdr['CD2_1']}")
    print(f"  CD2_2  = {coadd_hdr['CD2_2']}")
    print(f"  Coadd pixel area (arcsec^2) = {coadd_px_area}")

    print("[coadd_red_group] Input footprints (RA, Dec corners):")
    for i, fp in enumerate(footprints, start=1):
        print(f"  Footprint {i:02d}:")
        for corner in fp:
            print(f"    ({corner[0]:.8f}, {corner[1]:.8f})")

    coadd_data = np.zeros((coadd_size_w, coadd_size_y, coadd_size_x))
    coadd_var = np.zeros_like(coadd_data)

    # notebook-style wavelength-dependent effective exposure
    t_exp_tot = np.zeros_like(coadd_data)
    # notebook-style scalar exposure map used in bookkeeping
    t_exp_var_tot = np.zeros((coadd_size_y, coadd_size_x))

    px_area_ratio_list = [coadd_px_area / a for a in px_area]
    coadd_pixel_radec = _precompute_coadd_pixel_radec(coadd_wcs, coadd_size_y, coadd_size_x)

    cov_dict: dict[tuple[int, int], np.ndarray] = {}

    print("[coadd_red_group] Starting main coadd loop...")
    loop_start = time.time()

    for y_j in range(coadd_size_y):
        _print_row_progress(y_j, coadd_size_y, loop_start, every=5)

        for x_k in range(coadd_size_x):
            pixel_coadd_radec = coadd_pixel_radec[y_j][x_k]

            for i, cube in enumerate(cubes):
                pixel_coadd_proj = shapely_polygon(
                    wcs2d[i].all_world2pix(pixel_coadd_radec, 0)
                )
                region_bounds = list(pixel_coadd_proj.exterior.bounds)

                xb0, yb0, xb1, yb1 = (int(round(v)) for v in region_bounds)
                xb0 = max(xb0, 0)
                yb0 = max(yb0, 0)
                xb1 = min(xb1 + 1, cube.shape[2])
                yb1 = min(yb1 + 1, cube.shape[1])

                if not (xb0 < xb1 and yb0 < yb1):
                    continue

                px_area_ratio = px_area_ratio_list[i]
                overlap_tot = 0.0
                coadd_data_ = np.zeros(cube.shape[0])
                coadd_var_ = np.zeros(cube.shape[0])
                coadd_exp_ = np.zeros(cube.shape[0])

                for y_i in range(yb0, yb1):
                    for x_i in range(xb0, xb1):
                        pixel = pixel_boxes[i][y_i][x_i]
                        overlap = pixel_coadd_proj.intersection(pixel).area / pixel.area
                        if overlap <= 0:
                            continue

                        overlap_tot += overlap

                        weight = 1.0 - masks[i][:, y_i, x_i].astype(float)

                        # notebook-faithful CR handling:
                        # masked voxels contribute zero flux and zero effective exposure
                        coadd_data_ += overlap * t_exp[i] * cube[:, y_i, x_i] * weight
                        coadd_exp_ += overlap * t_exp[i] * weight

                        # notebook-faithful variance handling:
                        # variance is accumulated as written in the notebook
                        coadd_var_ += (overlap * t_exp[i] * uncerts[i][:, y_i, x_i]) ** 2

                if overlap_tot <= px_thresh:
                    continue

                t_exp_tot[:, y_j, x_k] += coadd_exp_ / overlap_tot
                t_exp_var_tot[y_j, x_k] += t_exp[i]
                coadd_data[:, y_j, x_k] += coadd_data_ / overlap_tot * px_area_ratio
                coadd_var[:, y_j, x_k] += coadd_var_ / overlap_tot**2 * px_area_ratio**2

                pixel_region = shapely_box(xb0, yb0, xb1, yb1)

                # notebook-style covariance neighborhood search
                for y_m in range(y_j, min(y_j + 4, coadd_size_y)):
                    for x_n in range(max(x_k - 10, 0), min(x_k + 11, coadd_size_x)):
                        if y_m == y_j and x_n < x_k:
                            continue

                        pixel_coadd_radec1 = coadd_pixel_radec[y_m][x_n]
                        pixel_coadd_proj1 = shapely_polygon(
                            wcs2d[i].all_world2pix(pixel_coadd_radec1, 0)
                        )

                        region_bounds1 = list(pixel_coadd_proj1.exterior.bounds)
                        xb2, yb2, xb3, yb3 = (int(round(v)) for v in region_bounds1)
                        xb2 = max(xb2, 0)
                        yb2 = max(yb2, 0)
                        xb3 = min(xb3 + 1, cube.shape[2])
                        yb3 = min(yb3 + 1, cube.shape[1])

                        if not (xb2 < xb3 and yb2 < yb3):
                            continue

                        pixel_region1 = shapely_box(xb2, yb2, xb3, yb3)
                        intersect_region = pixel_region1.intersection(pixel_region)

                        if intersect_region.area <= 0:
                            continue

                        xb4, yb4, xb5, yb5 = (
                            int(round(v)) for v in list(intersect_region.exterior.bounds)
                        )

                        overlap_tot1 = 0.0
                        cov_var_ = np.zeros(cube.shape[0])

                        for y_i in range(yb2, yb3):
                            for x_i in range(xb2, xb3):
                                pixel1 = pixel_boxes[i][y_i][x_i]
                                overlap1 = pixel_coadd_proj1.intersection(pixel1).area / pixel1.area
                                if overlap1 <= 0:
                                    continue

                                overlap_tot1 += overlap1

                                if y_i in range(yb4, yb5) and x_i in range(xb4, xb5):
                                    pixel2 = pixel_boxes[i][y_i][x_i]
                                    overlap21 = pixel_coadd_proj.intersection(pixel2).area / pixel2.area
                                    overlap22 = pixel_coadd_proj1.intersection(pixel2).area / pixel2.area
                                    if overlap21 > 0 and overlap22 > 0:
                                        cov_var_ += (
                                            overlap21
                                            * overlap22
                                            * t_exp[i] ** 2
                                            * uncerts[i][:, y_i, x_i] ** 2
                                        )

                        if overlap_tot1 > px_thresh:
                            coord = (coadd_size_x * y_j + x_k, coadd_size_x * y_m + x_n)
                            val = cov_var_ / overlap_tot / overlap_tot1 * px_area_ratio**2
                            if coord in cov_dict:
                                cov_dict[coord] += val
                            else:
                                cov_dict[coord] = val.copy()

    print("[coadd_red_group] Normalizing by total exposure time...")
    t_exp_tot[t_exp_tot == 0] = np.inf

    for y_j in range(coadd_data.shape[1]):
        for x_k in range(coadd_data.shape[2]):
            coadd_data[:, y_j, x_k] /= t_exp_tot[:, y_j, x_k]
            coadd_var[:, y_j, x_k] /= t_exp_tot[:, y_j, x_k] ** 2

    print("[coadd_red_group] Packing covariance arrays...")
    cov_coordinate = np.array(sorted(cov_dict.keys()), dtype=int)
    cov_data = np.array([cov_dict[tuple(coord)] for coord in cov_coordinate], dtype=float)

    for index, coo_ in enumerate(cov_coordinate):
        y1 = int(coo_[0] / coadd_size_x)
        x1 = coo_[0] % coadd_size_x
        y2 = int(coo_[1] / coadd_size_x)
        x2 = coo_[1] % coadd_size_x
        cov_data[index] = cov_data[index] / t_exp_tot[:, y1, x1] / t_exp_tot[:, y2, x2]

    if output_dir is None:
        output_dir = input_paths[0].parent.parent.parent / "coadd" / "red" / group_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flux_path = output_dir / f"coadd_red_{group_name}_{product}.fits"
    var_path = output_dir / f"coadd_red_{group_name}_{product}_var.fits"
    cov_data_path = output_dir / f"coadd_red_{group_name}_{product}_cov_data.npy"
    cov_coord_path = output_dir / f"coadd_red_{group_name}_{product}_cov_coordinate.npy"

    print("[coadd_red_group] Writing outputs...")
    write_fits_cube(flux_path, coadd_data, coadd_hdr)
    write_fits_cube(var_path, coadd_var, coadd_hdr)
    np.save(cov_data_path, cov_data)
    np.save(cov_coord_path, cov_coordinate)

    print("[coadd_red_group] Done.")
    print(f"  Flux: {flux_path}")
    print(f"  Var:  {var_path}")
    print(f"  Cov data:  {cov_data_path}")
    print(f"  Cov coord: {cov_coord_path}")

    return RedCoaddResult(
        group_name=group_name,
        product=product,
        input_paths=input_paths,
        mask_paths=mask_paths,
        output_flux_path=flux_path,
        output_var_path=var_path,
        output_cov_data_path=cov_data_path,
        output_cov_coord_path=cov_coord_path,
        n_cubes=len(input_paths),
        shape=coadd_data.shape,
    )