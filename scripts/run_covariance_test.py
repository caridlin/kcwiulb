from pathlib import Path

from kcwiulb.coadd.covariance_test import run_covariance_test


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.fits"
VAR_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.fits"
COV_DATA_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_cov_data.npy"
COV_COORD_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_cov_coordinate.npy"

# Save the single PDF directly in the coadd directory
OUTPUT_DIR = COADD_DIR

KERNEL_SIZES = list(range(1, 4))
COLLAPSE_WAVELENGTH_RANGES = [(3700, 3980), (4150, 5200)]


def main():
    print("Running covariance test")
    print(f"  Flux:      {FLUX_PATH}")
    print(f"  Var:       {VAR_PATH}")
    print(f"  Cov data:  {COV_DATA_PATH}")
    print(f"  Cov coord: {COV_COORD_PATH}")
    print(f"  Output:    {OUTPUT_DIR}")

    for path in [FLUX_PATH, VAR_PATH, COV_DATA_PATH, COV_COORD_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    result = run_covariance_test(
        flux_path=FLUX_PATH,
        var_path=VAR_PATH,
        cov_data_path=COV_DATA_PATH,
        cov_coord_path=COV_COORD_PATH,
        kernel_sizes=KERNEL_SIZES,
        wavelength_ranges=COLLAPSE_WAVELENGTH_RANGES,
        output_dir=OUTPUT_DIR,
    )

    print("\n[done]")
    print(f"  output_pdf: {result.output_pdf_path}")
    print(f"  alpha:      {result.fitted_alpha:.4f}")
    print(f"  norm:       {result.fitted_norm:.4f}")
    print(f"  thresh:     {result.fitted_thresh:.4f}")


if __name__ == "__main__":
    main()