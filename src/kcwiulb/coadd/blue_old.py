from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon as shapely_polygon


@dataclass
class BlueCoaddResult:
    field: str
    product: str
    input_paths: list[Path]
    output_flux_path: Path
    output_var_path: Path
    output_cov_data_path: Path
    output_cov_coord_path: Path
    n_cubes: int
    shape: tuple[int, int, int]


def get_product_suffix(product: str) -> str:
    if product == "sky":
        return "_icubes.wc.c.sky.sky.fits"
    if product == "sky2":
        return "_icubes.wc.c.sky.sky2.fits"
    raise ValueError(f"Unsupported PRODUCT: {product}")


def find_field_cubes(
    base: Path,
    channel: str,
    field: str,
    product: str,
) -> list[Path]:
    suffix = get_product_suffix(product)
    field_dir = base / channel / field

    if not field_dir.exists():
        raise FileNotFoundError(f"Missing field directory: {field_dir}")

    paths = sorted(field_dir.glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No {product} cubes found in {field_dir}")

    return paths


def load_cube(path: Path) -> tuple[np.ndarray, fits.Header, np.ndarray]:
    with fits.open(path) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
        uncert = hdul[1].data.copy()
    return data, header, uncert


def get_header2d(header3d: fits.Header) -> fits.Header:
    header2d = header3d.copy()

    keys_to_delete = [key for key in header2d.keys() if "3" in key]
    for key in keys_to_delete:
        try:
            del header2d[key]
        except KeyError:
            pass

    header2d["NAXIS"] = 2
    header2d["WCSDIM"] = 2
    return header2d


def rotate_wcs(wcs: WCS, theta_deg: float) -> WCS:
    theta = np.deg2rad(theta_deg)
    sinq = np.sin(theta)
    cosq = np.cos(theta)
    mrot = np.array([[cosq, -sinq], [sinq, cosq]])

    new_wcs = wcs.deepcopy()
    newcd = np.dot(mrot, new_wcs.wcs.cd)
    new_wcs.wcs.cd = newcd
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


def build_common_header(headers: list[fits.Header], pa: float) -> tuple[fits.Header, WCS, float]:
    h0 = get_header2d(headers[0])
    wcs0 = WCS(h0)

    dx0, dy0 = proj_plane_pixel_scales(wcs0)

    # enforce 1:1 aspect ratio
    if dx0 > dy0:
        wcs0.wcs.cd[:, 0] = wcs0.wcs.cd[:, 0] / (dx0 / dy0)
    else:
        wcs0.wcs.cd[:, 1] = wcs0.wcs.cd[:, 1] / (dy0 / dx0)

    pas = [h["ROTPOSN"] for h in headers]
    wcs0 = rotate_wcs(wcs0, pas[0] - pa)
    wcs0.wcs.set()

    footprints = [WCS(get_header2d(h)).calc_footprint() for h in headers]

    x0, y0 = 0.0, 0.0
    x1, y1 = 0.0, 0.0
    for fp in footprints:
        ras, decs = fp[:, 0], fp[:, 1]
        x_all, y_all = wcs0.all_world2pix(ras, decs, 0)
        x0 = min(np.min(x_all), x0)
        y0 = min(np.min(y_all), y0)
        x1 = max(np.max(x_all), x1)
        y1 = max(np.max(y_all), y1)

    coadd_size_x = int(round((x1 - x0) + 1))
    coadd_size_y = int(round((y1 - y0) + 1))

    ra0, dec0 = wcs0.all_pix2world(x0, y0, 0)

    wcs0.wcs.crpix[0] = 1
    wcs0.wcs.crval[0] = ra0
    wcs0.wcs.crpix[1] = 1
    wcs0.wcs.crval[1] = dec0
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
    return coadd_hdr, WCS(get_header2d(coadd_hdr)), coadd_px_area


def write_fits_cube(path: Path, data: np.ndarray, header: fits.Header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(data=data, header=header)]).writeto(path, overwrite=True)
    return path


def coadd_blue_field(
    base: str | Path,
    channel: str,
    field: str,
    product: str,
    pa: float,
    px_thresh: float = 0.1,
    output_dir: str | Path | None = None,
) -> BlueCoaddResult:
    base = Path(base)
    channel = channel.lower()

    input_paths = find_field_cubes(base, channel, field, product)

    cubes = []
    uncerts = []
    headers = []
    wcs2d = []
    px_area = []
    t_exp = []

    for path in input_paths:
        data, header, uncert = load_cube(path)
        cubes.append(data)
        uncerts.append(uncert)
        headers.append(header)
        wcs2d.append(WCS(get_header2d(header)))
        px_area.append(get_pxarea_arcsec(header))
        t_exp.append(header["XPOSURE"])

    coadd_hdr, coadd_wcs, coadd_px_area = build_common_header(headers, pa=pa)

    coadd_size_w = coadd_hdr["NAXIS3"]
    coadd_size_y = coadd_hdr["NAXIS2"]
    coadd_size_x = coadd_hdr["NAXIS1"]

    coadd_data = np.zeros((coadd_size_w, coadd_size_y, coadd_size_x))
    coadd_var = np.zeros_like(coadd_data)
    t_exp_tot = np.zeros((coadd_size_y, coadd_size_x))

    cov_coordinate: list[list[int]] = []
    cov_data: list[np.ndarray] = []

    for y_j in range(coadd_size_y):
        for x_k in range(coadd_size_x):
            pixel_verts_coadd = np.array([
                [x_k - 0.5, y_j - 0.5],
                [x_k - 0.5, y_j + 0.5],
                [x_k + 0.5, y_j + 0.5],
                [x_k + 0.5, y_j - 0.5],
            ])

            pixel_coadd_radec = coadd_wcs.all_pix2world(pixel_verts_coadd, 0)

            for i, cube in enumerate(cubes):
                pixel_coadd_proj = shapely_polygon(wcs2d[i].all_world2pix(pixel_coadd_radec, 0))
                region_bounds = list(pixel_coadd_proj.exterior.bounds)

                xb0, yb0, xb1, yb1 = (int(round(v)) for v in region_bounds)
                xb0 = max(xb0, 0)
                yb0 = max(yb0, 0)
                xb1 = min(xb1 + 1, cube.shape[2])
                yb1 = min(yb1 + 1, cube.shape[1])

                if xb0 < xb1 and yb0 < yb1:
                    px_area_ratio = coadd_px_area / px_area[i]
                    overlap_tot = 0.0
                    coadd_data_ = np.zeros(cube.shape[0])
                    coadd_var_ = np.zeros(cube.shape[0])

                    for y_i in range(yb0, yb1):
                        for x_i in range(xb0, xb1):
                            pixel = shapely_box(x_i - 0.5, y_i - 0.5, x_i + 0.5, y_i + 0.5)
                            overlap = pixel_coadd_proj.intersection(pixel).area / pixel.area
                            overlap_tot += overlap

                            coadd_data_ += overlap * t_exp[i] * cube[:, y_i, x_i]
                            coadd_var_ += (overlap * t_exp[i] * uncerts[i][:, y_i, x_i]) ** 2

                    if overlap_tot > px_thresh:
                        t_exp_tot[y_j, x_k] += t_exp[i]
                        coadd_data[:, y_j, x_k] += coadd_data_ / overlap_tot * px_area_ratio
                        coadd_var[:, y_j, x_k] += coadd_var_ / overlap_tot**2 * px_area_ratio**2

                # covariance block from notebook
                for y_m in range(y_j, min(y_j + 4, coadd_size_y)):
                    for x_n in range(max(x_k - 10, 0), min(x_k + 11, coadd_size_x)):
                        if y_m == y_j and x_n < x_k:
                            continue

                        pixel_verts_coadd1 = np.array([
                            [x_n - 0.5, y_m - 0.5],
                            [x_n - 0.5, y_m + 0.5],
                            [x_n + 0.5, y_m + 0.5],
                            [x_n + 0.5, y_m - 0.5],
                        ])

                        pixel_coadd_radec1 = coadd_wcs.all_pix2world(pixel_verts_coadd1, 0)
                        pixel_coadd_proj1 = shapely_polygon(wcs2d[i].all_world2pix(pixel_coadd_radec1, 0))

                        region_bounds1 = list(pixel_coadd_proj1.exterior.bounds)
                        xb2, yb2, xb3, yb3 = (int(round(v)) for v in region_bounds1)
                        xb2 = max(xb2, 0)
                        yb2 = max(yb2, 0)
                        xb3 = min(xb3 + 1, cube.shape[2])
                        yb3 = min(yb3 + 1, cube.shape[1])

                        pixel_region = shapely_box(xb0, yb0, xb1, yb1)
                        pixel_region1 = shapely_box(xb2, yb2, xb3, yb3)
                        intersect_region = pixel_region1.intersection(pixel_region)

                        if intersect_region.area > 0 and xb2 < xb3 and yb2 < yb3:
                            xb4, yb4, xb5, yb5 = (
                                int(round(v)) for v in list(intersect_region.exterior.bounds)
                            )
                            overlap_tot1 = 0.0
                            cov_var_ = np.zeros(cube.shape[0])

                            for y_i in range(yb2, yb3):
                                for x_i in range(xb2, xb3):
                                    pixel1 = shapely_box(x_i - 0.5, y_i - 0.5, x_i + 0.5, y_i + 0.5)
                                    overlap1 = pixel_coadd_proj1.intersection(pixel1).area / pixel1.area
                                    overlap_tot1 += overlap1

                                    if y_i in range(yb4, yb5) and x_i in range(xb4, xb5):
                                        pixel2 = shapely_box(x_i - 0.5, y_i - 0.5, x_i + 0.5, y_i + 0.5)
                                        overlap21 = pixel_coadd_proj.intersection(pixel2).area / pixel2.area
                                        overlap22 = pixel_coadd_proj1.intersection(pixel2).area / pixel2.area
                                        cov_var_ += (
                                            overlap21
                                            * overlap22
                                            * t_exp[i] ** 2
                                            * uncerts[i][:, y_i, x_i] ** 2
                                        )

                            if overlap_tot1 > px_thresh:
                                coord = [coadd_size_x * y_j + x_k, coadd_size_x * y_m + x_n]
                                j = bisect_left(cov_coordinate, coord)
                                val = cov_var_ / overlap_tot / overlap_tot1 * px_area_ratio**2
                                if j == len(cov_coordinate) or cov_coordinate[j] != coord:
                                    cov_coordinate.insert(j, coord)
                                    cov_data.insert(j, val)
                                else:
                                    cov_data[j] += val

    t_exp_tot[t_exp_tot == 0] = np.inf

    for y_j in range(coadd_data.shape[1]):
        for x_k in range(coadd_data.shape[2]):
            coadd_data[:, y_j, x_k] /= t_exp_tot[y_j, x_k]
            coadd_var[:, y_j, x_k] /= t_exp_tot[y_j, x_k] ** 2

    for index, coo_ in enumerate(cov_coordinate):
        y1 = int(coo_[0] / coadd_size_x)
        x1 = coo_[0] % coadd_size_x
        y2 = int(coo_[1] / coadd_size_x)
        x2 = coo_[1] % coadd_size_x
        cov_data[index] = cov_data[index] / t_exp_tot[y1, x1] / t_exp_tot[y2, x2]

    if output_dir is None:
        output_dir = base / "coadd" / channel / field
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flux_path = output_dir / f"coadd_{channel}_{product}.fits"
    var_path = output_dir / f"coadd_{channel}_{product}_var.fits"
    cov_data_path = output_dir / f"coadd_{channel}_{product}_cov_data.npy"
    cov_coord_path = output_dir / f"coadd_{channel}_{product}_cov_coordinate.npy"

    write_fits_cube(flux_path, coadd_data, coadd_hdr)
    write_fits_cube(var_path, coadd_var, coadd_hdr)
    np.save(cov_data_path, np.array(cov_data, dtype=float))
    np.save(cov_coord_path, np.array(cov_coordinate, dtype=int))

    return BlueCoaddResult(
        field=field,
        product=product,
        input_paths=input_paths,
        output_flux_path=flux_path,
        output_var_path=var_path,
        output_cov_data_path=cov_data_path,
        output_cov_coord_path=cov_coord_path,
        n_cubes=len(input_paths),
        shape=coadd_data.shape,
    )