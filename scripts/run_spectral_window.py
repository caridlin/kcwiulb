from pathlib import Path

from kcwiulb.analysis.spectral_window import crop_spectral_window_group


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.fits"
VAR_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.wc.fits"
COV_DATA_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_cov_data.npy"

# Example: OII
LABEL = "oii"
WAVELENGTH_MIN = 4100
WAVELENGTH_MAX = 4300


def main():
    result = crop_spectral_window_group(
        flux_path=FLUX_PATH,
        var_path=VAR_PATH,
        cov_data_path=COV_DATA_PATH,
        wavelength_min=WAVELENGTH_MIN,
        wavelength_max=WAVELENGTH_MAX,
        label=LABEL,
    )

    print("[done]")
    print(f"  label: {LABEL}")
    print(f"  requested: {WAVELENGTH_MIN}-{WAVELENGTH_MAX}")
    print(f"  actual:    {result.wavelength_min_actual:.2f}-{result.wavelength_max_actual:.2f}")
    print(f"  pixels:    {result.n_spectral_pixels}")
    print("  outputs:")
    for p in result.output_paths:
        print(f"    {p}")


if __name__ == "__main__":
    main()