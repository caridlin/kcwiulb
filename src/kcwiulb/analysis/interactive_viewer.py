from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip

from tqdm import tqdm

from kcwiulb.wcs import index_to_wavelength, wavelength_to_index

import warnings
from astropy.utils.exceptions import AstropyUserWarning

warnings.filterwarnings(
    "ignore",
    category=AstropyUserWarning,
    message="Model is linear in parameters; consider using linear fitting methods."
)

@dataclass
class ContinuumSubtractionResult:
    flux_input_path: Path
    var_input_path: Path
    flux_bg_model_path: Path
    flux_bg_sub_path: Path
    var_bg_sub_path: Path
    wavelength_min_actual: float
    wavelength_max_actual: float
    continuum_order: int
    n_masked_channels: int


def _replace_fits_suffix(path: Path, suffix: str) -> Path:
    """
    Replace trailing '.fits' with a custom suffix, matching notebook style.

    Example
    -------
    coadd_blue_a_sky.wc.oii.fits -> coadd_blue_a_sky.wc.oii.bg.model.fits
    """
    if path.suffix.lower() != ".fits":
        raise ValueError(f"Expected a .fits file, got {path}")
    return path.with_name(path.name[:-5] + suffix)


def _build_fit_mask(
    header: fits.Header,
    n_wave: int,
    line_mask: tuple[float, float] | None,
    extra_masks: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Build a mask for channels used in continuum fitting.

    True  = include in fit
    False = exclude from fit
    """
    fit_mask = np.ones(n_wave, dtype=bool)

    masks: list[tuple[float, float]] = []
    if line_mask is not None:
        masks.append(line_mask)
    if extra_masks is not None:
        masks.extend(extra_masks)

    for wl_min, wl_max in masks:
        i0 = max(0, wavelength_to_index(wl_min, header))
        i1 = min(n_wave, wavelength_to_index(wl_max, header))
        if i1 > i0:
            fit_mask[i0:i1] = False

    return fit_mask


def _polynomial_variance_from_covariance(
    cov_matrix: np.ndarray | None,
    wave: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    Compute variance of the fitted polynomial model evaluated at each wavelength.

    For polynomial coefficients c_i and covariance matrix C_ij:

        Var[p(w)] = sum_i sum_j C_ij w^(p_i + p_j)

    where p_i are powers descending from `order` to 0 for np.polyfit-style
    ordering, but Astropy Polynomial1D stores coefficients in ascending order:
        c0 + c1*x + c2*x^2 + ...

    So the basis vector is:
        [1, w, w^2, ..., w^order]
    """
    if cov_matrix is None:
        return np.zeros_like(wave, dtype=float)

    cov = np.array(cov_matrix, dtype=float)
    n_param = order + 1

    if cov.shape != (n_param, n_param):
        return np.zeros_like(wave, dtype=float)

    basis = np.vstack([wave**k for k in range(n_param)])  # shape (n_param, n_wave)

    # Var = b^T C b for each wavelength
    var_model = np.einsum("iw,ij,jw->w", basis, cov, basis)

    # Numerical protection
    var_model = np.where(np.isfinite(var_model), var_model, 0.0)
    var_model = np.clip(var_model, 0.0, None)

    return var_model


def continuum_subtract_cube_pair(
    flux_path: str | Path,
    var_path: str | Path,
    continuum_order: int = 2,
    line_mask: tuple[float, float] | None = None,
    extra_masks: list[tuple[float, float]] | None = None,
    sigma_clip_value: float = 3.0,
    niter: int = 3,
    overwrite: bool = True,
) -> ContinuumSubtractionResult:
    """
    Fit and subtract a low-order continuum from a flux cube using iterative
    sigma-clipped polynomial fitting, then propagate the continuum-fit
    uncertainty into the variance cube.

    Outputs follow the notebook naming convention:

    Flux cube:
      ...bg.model.fits   continuum model
      ...bg.fits         continuum-subtracted flux

    Variance cube:
      ...bg.fits         updated variance after adding continuum-fit variance
    """
    flux_path = Path(flux_path)
    var_path = Path(var_path)

    if continuum_order not in (0, 1, 2):
        raise ValueError("continuum_order must be 0, 1, or 2 to match the notebook.")

    flux_bg_model_path = _replace_fits_suffix(flux_path, ".bg.model.fits")
    flux_bg_sub_path = _replace_fits_suffix(flux_path, ".bg.fits")
    var_bg_sub_path = _replace_fits_suffix(var_path, ".bg.fits")

    with fits.open(flux_path) as hdul_flux:
        flux_cube = hdul_flux[0].data.astype(float)
        header = hdul_flux[0].header.copy()

    with fits.open(var_path) as hdul_var:
        var_cube = hdul_var[0].data.astype(float)
        var_header = hdul_var[0].header.copy()

    if flux_cube.ndim != 3:
        raise ValueError(f"Expected 3D flux cube, got shape {flux_cube.shape}")
    if var_cube.shape != flux_cube.shape:
        raise ValueError(
            f"Flux/variance shape mismatch: {flux_cube.shape} vs {var_cube.shape}"
        )

    n_wave, n_y, n_x = flux_cube.shape
    wave = np.array([index_to_wavelength(i, header) for i in range(n_wave)], dtype=float)

    fit_mask = _build_fit_mask(
        header=header,
        n_wave=n_wave,
        line_mask=line_mask,
        extra_masks=extra_masks,
    )

    bg_model = np.zeros_like(flux_cube, dtype=float)
    c_sub = np.zeros_like(flux_cube, dtype=float)
    var_out = np.zeros_like(var_cube, dtype=float)

    fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
    or_fit = fitting.FittingWithOutlierRemoval(
        fit,
        sigma_clip,
        niter=niter,
        sigma=sigma_clip_value,
    )

    f_init = models.Polynomial1D(degree=continuum_order)

    # only fit on allowed channels
    wave_fit = wave[fit_mask]

    for y_ind in tqdm(range(n_y), desc="Continuum fitting"):
        for x_ind in range(n_x):
            y = flux_cube[:, y_ind, x_ind]

            # Match notebook logic: skip all-zero spectra
            if np.nanmax(np.abs(y)) == 0:
                bg_model[:, y_ind, x_ind] = 0.0
                c_sub[:, y_ind, x_ind] = 0.0
                var_out[:, y_ind, x_ind] = var_cube[:, y_ind, x_ind]
                continue

            y_var = var_cube[:, y_ind, x_ind]

            y_fit = y[fit_mask]

            finite = np.isfinite(y_fit) & np.isfinite(wave_fit)
            if np.count_nonzero(finite) < continuum_order + 1:
                bg_model[:, y_ind, x_ind] = np.nan
                c_sub[:, y_ind, x_ind] = np.nan
                var_out[:, y_ind, x_ind] = np.nan
                continue

            try:
                fitted_poly, mask = or_fit(
                    f_init,
                    wave_fit[finite],
                    y_fit[finite],
                )

                cont_full = fitted_poly(wave)
                bg_model[:, y_ind, x_ind] = cont_full
                c_sub[:, y_ind, x_ind] = y - cont_full

                cov_matrix = getattr(fitted_poly, "cov_matrix", None)

                # Astropy may return a covariance wrapper instead of ndarray
                if cov_matrix is not None and hasattr(cov_matrix, "cov_matrix"):
                    cov_matrix = cov_matrix.cov_matrix
                elif cov_matrix is not None and hasattr(cov_matrix, "__array__"):
                    cov_matrix = np.array(cov_matrix)

                var_add = _polynomial_variance_from_covariance(
                    cov_matrix=cov_matrix,
                    wave=wave,
                    order=continuum_order,
                )

                var_out[:, y_ind, x_ind] = y_var + var_add

            except Exception:
                bg_model[:, y_ind, x_ind] = np.nan
                c_sub[:, y_ind, x_ind] = np.nan
                var_out[:, y_ind, x_ind] = np.nan

    # Write continuum model
    hdu = fits.PrimaryHDU(bg_model)
    hdul = fits.HDUList([hdu])
    hdul[0].header = header.copy()
    hdul[0].header["CSORD"] = (continuum_order, "Continuum polynomial order")
    if line_mask is not None:
        hdul[0].header["CSLMIN"] = (float(line_mask[0]), "Continuum fit mask min [A]")
        hdul[0].header["CSLMAX"] = (float(line_mask[1]), "Continuum fit mask max [A]")
    hdul.writeto(flux_bg_model_path, overwrite=overwrite)

    # Write continuum-subtracted flux
    hdu = fits.PrimaryHDU(c_sub)
    hdul = fits.HDUList([hdu])
    hdul[0].header = header.copy()
    hdul[0].header["CSORD"] = (continuum_order, "Continuum polynomial order")
    if line_mask is not None:
        hdul[0].header["CSLMIN"] = (float(line_mask[0]), "Continuum fit mask min [A]")
        hdul[0].header["CSLMAX"] = (float(line_mask[1]), "Continuum fit mask max [A]")
    hdul.writeto(flux_bg_sub_path, overwrite=overwrite)

    # Write updated variance cube
    hdu = fits.PrimaryHDU(var_out)
    hdul = fits.HDUList([hdu])
    hdul[0].header = var_header.copy()
    hdul[0].header["CSORD"] = (continuum_order, "Continuum polynomial order")
    if line_mask is not None:
        hdul[0].header["CSLMIN"] = (float(line_mask[0]), "Continuum fit mask min [A]")
        hdul[0].header["CSLMAX"] = (float(line_mask[1]), "Continuum fit mask max [A]")
    hdul.writeto(var_bg_sub_path, overwrite=overwrite)

    return ContinuumSubtractionResult(
        flux_input_path=flux_path,
        var_input_path=var_path,
        flux_bg_model_path=flux_bg_model_path,
        flux_bg_sub_path=flux_bg_sub_path,
        var_bg_sub_path=var_bg_sub_path,
        wavelength_min_actual=float(wave[0]),
        wavelength_max_actual=float(wave[-1]),
        continuum_order=continuum_order,
        n_masked_channels=int(np.count_nonzero(~fit_mask)),
    )