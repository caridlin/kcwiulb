from pathlib import Path

from kcwiulb.coadd.wcs_coadd import solve_absolute_wcs_for_coadd
from kcwiulb.plot.wcs_diagnostics import plot_wcs_diagnostics


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.fits"
VAR_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.fits"

EXTRA_PATHS = [VAR_PATH]

# SDSS source coordinates
RA_DEG = 137.626253387
DEC_DEG = 10.240302230

# Continuum collapse region
WAVELENGTH_RANGES = [
    (3700, 3980),
    (4800, 5100),
]

# Cutout on collapsed coadd image
ROW_START = 23
COL_START = 65
N_ROWS = 30
N_COLS = 30

# Initial Gaussian guesses in cutout coordinates
AMPLITUDE_INIT = 2.0
X_MEAN_INIT = 14.0
Y_MEAN_INIT = 12.0
X_STDDEV_INIT = 2.0
Y_STDDEV_INIT = 2.0

# Diagnostic output
DIAG_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_wcsfit.png"


def main():
    result = solve_absolute_wcs_for_coadd(
        flux_cube_path=FLUX_PATH,
        ra_deg=RA_DEG,
        dec_deg=DEC_DEG,
        wavelength_ranges=WAVELENGTH_RANGES,
        row_start=ROW_START,
        col_start=COL_START,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        amplitude_init=AMPLITUDE_INIT,
        x_mean_init=X_MEAN_INIT,
        y_mean_init=Y_MEAN_INIT,
        x_stddev_init=X_STDDEV_INIT,
        y_stddev_init=Y_STDDEV_INIT,
        extra_paths_to_update=EXTRA_PATHS,
        pixel_origin=1,
    )

    plot_wcs_diagnostics(
        fit_result=result.fit_result,
        row_start=ROW_START,
        col_start=COL_START,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        title=f"Coadd WCS Fit: {CHANNEL} {GROUP} {PRODUCT}",
        savepath=DIAG_PATH,
        show=True,
    )

    print("[done]")
    print(f"  flux input:  {result.flux_input_path}")
    print(f"  flux output: {result.flux_output_path}")
    print(f"  RA:          {result.ra_deg:.8f}")
    print(f"  DEC:         {result.dec_deg:.8f}")
    print(f"  CRPIX1:      {result.x_ref:.4f}")
    print(f"  CRPIX2:      {result.y_ref:.4f}")
    print(f"  diagnostic:  {DIAG_PATH}")
    print("  updated files:")
    for path in result.updated_paths:
        print(f"    {path}")


if __name__ == "__main__":
    main()