from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from shapely.geometry import Polygon as shapely_polygon
from shapely.geometry import box as shapely_box
from kcwiulb.sky.utils import load_cube


@dataclass
class BlueCoaddResult:
    group_name: str
    product: str
    input_paths: list[Path]
    output_flux_path: Path
    output_var_path: Path
    output_exposure_path: Path
    output_cov_data_path: Path
    output_cov_coord_path: Path
    n_cubes: int
    shape: tuple[int, int, int]


def get_header2d(header3d: fits.Header) -> fits.Header:
    """Convert a 3D cube header into a 2D spatial header."""
    header2d = header3d.copy()

    for key in [key for key in header2d.keys() if "3" in key]:
        try:
            del header2d[key]
        except KeyError:
            pass

    header2d["NAXIS"] = 2
    header2d["WCSDIM"] = 2
    return header2d


def rotate_wcs(wcs: WCS, theta_deg: float) -> WCS:
    theta = np.deg2rad(theta_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    new_wcs = wcs.deepcopy()
    new_wcs.wcs.cd = rotation @ new_wcs.wcs.cd
    new_wcs.wcs.set()
    return new_wcs


def get_pxarea_arcsec(header: fits.Header) -> float:
    if header["NAXIS"] == 3:
        header = get_header2d(header)
    elif header["NAXIS"] != 2:
        raise ValueError("Function only takes 2D or 3D input.")

    yscale, xscale = proj_plane_pixel_scales(WCS(header))
    yscale = (yscale * u.deg).to(u.arcsec).value
    xscale = (xscale * u.deg).to(u.arcsec).value
    return yscale * xscale


def build_common_header(
    headers: list[fits.Header],
    pa: float,
) -> tuple[fits.Header, WCS, float, list[np.ndarray]]:
    h0 = get_header2d(headers[0])
    wcs0 = WCS(h0)

    dx0, dy0 = proj_plane_pixel_scales(wcs0)

    if dx0 > dy0:
        wcs0.wcs.cd[:, 0] /= dx0 / dy0
    else:
        wcs0.wcs.cd[:, 1] /= dy0 / dx0

    pas = [header["ROTPOSN"] for header in headers]
    wcs0 = rotate_wcs(wcs0, pas[0] - pa)
    wcs0.wcs.set()

    footprints = [WCS(get_header2d(header)).calc_footprint() for header in headers]

    x0 = y0 = x1 = y1 = 0.0

    for footprint in footprints:
        ras, decs = footprint[:, 0], footprint[:, 1]
        x_all, y_all = wcs0.all_world2pix(ras, decs, 0)

        x0 = min(np.min(x_all), x0)
        y0 = min(np.min(y_all), y0)
        x1 = max(np.max(x_all), x1)
        y1 = max(np.max(y_all), y1)

    coadd_size_x = int(round((x1 - x0) + 1))
    coadd_size_y = int(round((y1 - y0) + 1))

    ra0, dec0 = wcs0.all_pix2world(x0, y0, 0)

    wcs0.wcs.crpix = [1, 1]
    wcs0.wcs.crval = [ra0, dec0]
    wcs0.wcs.set()

    hdr0 = wcs0.to_header()
    coadd_hdr = headers[0].copy()

    coadd_hdr["NAXIS1"] = coadd_size_x
    coadd_hdr["NAXIS2"] = coadd_size_y
    coadd_hdr["CRPIX1"] = hdr0["CRPIX1"]
    coadd_hdr["CRPIX2"] = hdr0["CRPIX2"]
    coadd_hdr["CRVAL1"] = hdr0["CRVAL1"]
    coadd_hdr["CRVAL2"] = hdr0["CRVAL2"]
    coadd_hdr["CD1_1"] = wcs0.wcs.cd[0, 0]
    coadd_hdr["CD1_2"] = wcs0.wcs.cd[0, 1]
    coadd_hdr["CD2_1"] = wcs0.wcs.cd[1, 0]
    coadd_hdr["CD2_2"] = wcs0.wcs.cd[1, 1]

    coadd_px_area = get_pxarea_arcsec(get_header2d(coadd_hdr))
    return coadd_hdr, WCS(get_header2d(coadd_hdr)), coadd_px_area, footprints


def write_fits_cube(path: Path, data: np.ndarray, header: fits.Header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
    return path


def _precompute_pixel_boxes(ny: int, nx: int) -> list[list]:
    return [
        [shapely_box(x - 0.5, y - 0.5, x + 0.5, y + 0.5) for x in range(nx)]
        for y in range(ny)
    ]


def _precompute_coadd_pixel_radec(
    coadd_wcs: WCS,
    ny: int,
    nx: int,
) -> list[list[np.ndarray]]:
    pixel_radec = []

    for y_j in range(ny):
        row = []

        for x_k in range(nx):
            vertices = np.array(
                [
                    [x_k - 0.5, y_j - 0.5],
                    [x_k - 0.5, y_j + 0.5],
                    [x_k + 0.5, y_j + 0.5],
                    [x_k + 0.5, y_j - 0.5],
                ]
            )
            row.append(coadd_wcs.all_pix2world(vertices, 0))

        pixel_radec.append(row)

    return pixel_radec


def _normalize_input_paths(input_paths: list[str | Path]) -> list[Path]:
    paths = [Path(path) for path in input_paths]

    if not paths:
        raise ValueError("input_paths is empty.")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing input cube: {path}")

    return paths


def _print_row_progress(
    y_j: int,
    total_rows: int,
    start_time: float,
    every: int = 5,
) -> None:
    if y_j % every != 0 and y_j != total_rows - 1:
        return

    elapsed = time.time() - start_time
    fraction = (y_j + 1) / total_rows
    eta = elapsed / fraction - elapsed if fraction > 0 else 0.0

    bar_len = 30
    filled = int(bar_len * fraction)
    bar = "█" * filled + "-" * (bar_len - filled)

    print(
        f"  Rows [{bar}] {100 * fraction:5.1f}% | "
        f"{y_j + 1}/{total_rows} | Elapsed: {elapsed:7.1f}s | "
        f"ETA: {eta:7.1f}s",
        flush=True,
    )


def coadd_blue_group(
    input_paths: list[str | Path],
    group_name: str,
    product: str,
    pa: float,
    px_thresh: float = 0.1,
    output_dir: str | Path | None = None,
) -> BlueCoaddResult:
    input_paths = _normalize_input_paths(input_paths)

    print(f"[coadd_blue_group] Starting group '{group_name}'")
    print(f"[coadd_blue_group] Product: {product}")
    print(f"[coadd_blue_group] PA: {pa}")
    print(f"[coadd_blue_group] PX_THRESH: {px_thresh}")
    print(f"[coadd_blue_group] N input cubes: {len(input_paths)}")

    for i, path in enumerate(input_paths, start=1):
        print(f"  Input {i:02d}: {path}")

    cubes = []
    uncerts = []
    headers = []
    wcs2d = []
    px_area = []
    t_exp = []
    pixel_boxes = []

    print("[coadd_blue_group] Loading cubes...")

    for i, path in enumerate(input_paths, start=1):
        data, header, uncert = load_cube(path)

        cubes.append(data)
        uncerts.append(uncert)
        headers.append(header)
        wcs2d.append(WCS(get_header2d(header)))
        px_area.append(get_pxarea_arcsec(header))
        t_exp.append(float(header["XPOSURE"]))
        pixel_boxes.append(_precompute_pixel_boxes(data.shape[1], data.shape[2]))

        print(
            f"  Loaded {i:02d}/{len(input_paths)} | {path.name} | "
            f"shape={data.shape} | XPOSURE={header['XPOSURE']}",
            flush=True,
        )

    print("[coadd_blue_group] Building common WCS/header...")

    coadd_hdr, coadd_wcs, coadd_px_area, footprints = build_common_header(
        headers,
        pa=pa,
    )

    coadd_size_w = coadd_hdr["NAXIS3"]
    coadd_size_y = coadd_hdr["NAXIS2"]
    coadd_size_x = coadd_hdr["NAXIS1"]

    print("[coadd_blue_group] Coadd grid:")
    print(f"  NAXIS1 = {coadd_size_x}")
    print(f"  NAXIS2 = {coadd_size_y}")
    print(f"  NAXIS3 = {coadd_size_w}")
    print(f"  CRPIX1 = {coadd_hdr['CRPIX1']}")
    print(f"  CRPIX2 = {coadd_hdr['CRPIX2']}")
    print(f"  CRVAL1 = {coadd_hdr['CRVAL1']}")
    print(f"  CRVAL2 = {coadd_hdr['CRVAL2']}")
    print(f"  Coadd pixel area (arcsec^2) = {coadd_px_area}")

    print("[coadd_blue_group] Input footprints:")

    for i, footprint in enumerate(footprints, start=1):
        print(f"  Footprint {i:02d}:")
        for corner in footprint:
            print(f"    ({corner[0]:.8f}, {corner[1]:.8f})")

    coadd_data = np.zeros(
        (coadd_size_w, coadd_size_y, coadd_size_x),
        dtype=float,
    )
    coadd_var = np.zeros_like(coadd_data)
    t_exp_tot = np.zeros((coadd_size_y, coadd_size_x), dtype=float)

    px_area_ratio_list = [coadd_px_area / area for area in px_area]
    coadd_pixel_radec = _precompute_coadd_pixel_radec(
        coadd_wcs,
        coadd_size_y,
        coadd_size_x,
    )

    cov_dict: dict[tuple[int, int], np.ndarray] = {}

    print("[coadd_blue_group] Starting main coadd loop...")
    loop_start = time.time()

    for y_j in range(coadd_size_y):
        _print_row_progress(y_j, coadd_size_y, loop_start)

        for x_k in range(coadd_size_x):
            pixel_coadd_radec = coadd_pixel_radec[y_j][x_k]

            for i, cube in enumerate(cubes):
                pixel_coadd_proj = shapely_polygon(
                    wcs2d[i].all_world2pix(pixel_coadd_radec, 0)
                )

                xb0, yb0, xb1, yb1 = (
                    int(round(value))
                    for value in pixel_coadd_proj.exterior.bounds
                )

                xb0 = max(xb0, 0)
                yb0 = max(yb0, 0)
                xb1 = min(xb1 + 1, cube.shape[2])
                yb1 = min(yb1 + 1, cube.shape[1])

                if not (xb0 < xb1 and yb0 < yb1):
                    continue

                px_area_ratio = px_area_ratio_list[i]
                overlap_tot = 0.0
                coadd_data_ = np.zeros(cube.shape[0], dtype=float)
                coadd_var_ = np.zeros(cube.shape[0], dtype=float)

                for y_i in range(yb0, yb1):
                    for x_i in range(xb0, xb1):
                        pixel = pixel_boxes[i][y_i][x_i]
                        overlap = (
                            pixel_coadd_proj.intersection(pixel).area / pixel.area
                        )

                        if overlap <= 0:
                            continue

                        overlap_tot += overlap
                        coadd_data_ += overlap * t_exp[i] * cube[:, y_i, x_i]
                        coadd_var_ += np.square(
                            overlap * t_exp[i] * uncerts[i][:, y_i, x_i]
                        )

                if overlap_tot <= px_thresh:
                    continue

                t_exp_tot[y_j, x_k] += t_exp[i]
                coadd_data[:, y_j, x_k] += (
                    coadd_data_ / overlap_tot * px_area_ratio
                )
                coadd_var[:, y_j, x_k] += (
                    coadd_var_ / overlap_tot**2 * px_area_ratio**2
                )

                pixel_region = shapely_box(xb0, yb0, xb1, yb1)

                for y_m in range(y_j, min(y_j + 4, coadd_size_y)):
                    for x_n in range(
                        max(x_k - 10, 0),
                        min(x_k + 11, coadd_size_x),
                    ):
                        if y_m == y_j and x_n < x_k:
                            continue

                        pixel_coadd_proj1 = shapely_polygon(
                            wcs2d[i].all_world2pix(
                                coadd_pixel_radec[y_m][x_n],
                                0,
                            )
                        )

                        xb2, yb2, xb3, yb3 = (
                            int(round(value))
                            for value in pixel_coadd_proj1.exterior.bounds
                        )

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
                            int(round(value))
                            for value in intersect_region.exterior.bounds
                        )

                        overlap_tot1 = 0.0
                        cov_var_ = np.zeros(cube.shape[0], dtype=float)

                        for y_i in range(yb2, yb3):
                            for x_i in range(xb2, xb3):
                                pixel1 = pixel_boxes[i][y_i][x_i]
                                overlap1 = (
                                    pixel_coadd_proj1.intersection(pixel1).area
                                    / pixel1.area
                                )

                                if overlap1 <= 0:
                                    continue

                                overlap_tot1 += overlap1

                                if y_i in range(yb4, yb5) and x_i in range(xb4, xb5):
                                    pixel2 = pixel_boxes[i][y_i][x_i]
                                    overlap21 = (
                                        pixel_coadd_proj.intersection(pixel2).area
                                        / pixel2.area
                                    )
                                    overlap22 = (
                                        pixel_coadd_proj1.intersection(pixel2).area
                                        / pixel2.area
                                    )

                                    if overlap21 > 0 and overlap22 > 0:
                                        cov_var_ += (
                                            overlap21
                                            * overlap22
                                            * t_exp[i] ** 2
                                            * np.square(
                                                uncerts[i][:, y_i, x_i]
                                            )
                                        )

                        if overlap_tot1 > px_thresh:
                            coord = (
                                coadd_size_x * y_j + x_k,
                                coadd_size_x * y_m + x_n,
                            )
                            value = (
                                cov_var_
                                / overlap_tot
                                / overlap_tot1
                                * px_area_ratio**2
                            )

                            if coord in cov_dict:
                                cov_dict[coord] += value
                            else:
                                cov_dict[coord] = value.copy()

    print("[coadd_blue_group] Normalizing by total exposure time...")

    # Preserve the true exposure map before replacing zeros with infinity.
    t_exp_map = t_exp_tot.copy()
    t_exp_tot[t_exp_tot == 0] = np.inf

    for y_j in range(coadd_size_y):
        for x_k in range(coadd_size_x):
            coadd_data[:, y_j, x_k] /= t_exp_tot[y_j, x_k]
            coadd_var[:, y_j, x_k] /= t_exp_tot[y_j, x_k] ** 2

    print("[coadd_blue_group] Packing covariance arrays...")

    cov_coordinate = np.array(sorted(cov_dict.keys()), dtype=int)
    cov_data = np.array(
        [cov_dict[tuple(coord)] for coord in cov_coordinate],
        dtype=float,
    )

    for index, coordinate in enumerate(cov_coordinate):
        y1, x1 = divmod(coordinate[0], coadd_size_x)
        y2, x2 = divmod(coordinate[1], coadd_size_x)

        cov_data[index] /= (
            t_exp_tot[y1, x1] * t_exp_tot[y2, x2]
        )

    if output_dir is None:
        output_dir = (
            input_paths[0].parent.parent.parent
            / "coadd"
            / "blue"
            / group_name
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"coadd_blue_{group_name}_{product}"

    flux_path = output_dir / f"{prefix}.fits"
    var_path = output_dir / f"{prefix}_var.fits"
    exposure_path = output_dir / f"{prefix}_exposure.fits"
    cov_data_path = output_dir / f"{prefix}_cov_data.npy"
    cov_coord_path = output_dir / f"{prefix}_cov_coordinate.npy"

    print("[coadd_blue_group] Writing outputs...")

    write_fits_cube(flux_path, coadd_data, coadd_hdr)
    write_fits_cube(var_path, coadd_var, coadd_hdr)
    write_fits_cube(exposure_path, t_exp_map, get_header2d(coadd_hdr))
    np.save(cov_data_path, cov_data)
    np.save(cov_coord_path, cov_coordinate)

    print("[coadd_blue_group] Done.")
    print(f"  Flux: {flux_path}")
    print(f"  Var: {var_path}")
    print(f"  Exposure: {exposure_path}")
    print(f"  Cov data: {cov_data_path}")
    print(f"  Cov coord: {cov_coord_path}")

    return BlueCoaddResult(
        group_name=group_name,
        product=product,
        input_paths=input_paths,
        output_flux_path=flux_path,
        output_var_path=var_path,
        output_exposure_path=exposure_path,
        output_cov_data_path=cov_data_path,
        output_cov_coord_path=cov_coord_path,
        n_cubes=len(input_paths),
        shape=coadd_data.shape,
    )