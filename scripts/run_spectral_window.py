from pathlib import Path

from kcwiulb.analysis.spectral_window import (
    crop_multiple_spectral_windows_group,
)


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.fits"
VAR_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.wc.fits"
COV_DATA_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_cov_data.npy"


# ==========================================================
# Spectral windows
# ==========================================================

WINDOWS = {
    "mgii": (3950, 4100),
    "oii": (5250, 5450),
}

def main():

    results = crop_multiple_spectral_windows_group(
        flux_path=FLUX_PATH,
        var_path=VAR_PATH,
        cov_data_path=COV_DATA_PATH,
        windows=WINDOWS,
    )

    print("\n================================")
    print("SPECTRAL WINDOW CROPS COMPLETE")
    print("================================")

    for label, result in results.items():

        print(f"\n[{label}]")
        print(
            f"  requested : {result.wavelength_min:.2f}"
            f"–{result.wavelength_max:.2f} Å"
        )
        print(
            f"  actual    : {result.wavelength_min_actual:.2f}"
            f"–{result.wavelength_max_actual:.2f} Å"
        )
        print(
            f"  pixels    : {result.n_spectral_pixels}"
        )
        print("  outputs:")

        for path in result.output_paths:
            print(f"    {path}")


if __name__ == "__main__":
    main()