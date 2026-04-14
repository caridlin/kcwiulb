from pathlib import Path

from kcwiulb.ads.ads import run_adaptive_smoothing


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"
LABEL = "oii"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.bg.mask.fits"
VAR_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.wc.{LABEL}.bg.fits"

COV_DATA_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_cov_data_{LABEL}.npy"
COV_COORD_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_cov_coordinate.npy"

SNR_MIN = 2.5
SNR_MAX = 2 * SNR_MIN

XY_RANGE = (1, 50)
XY_STEP = 1
XY_STEP_MIN = 1

Z_RANGE = (1, 20)
Z_STEP = 1
Z_STEP_MIN = 1

KERNEL_TYPE = "box"

DIAGNOSTIC_WAVELENGTH_RANGES = [
    (4100, 4240),
    (4275, 4300),
]

COVARIANCE_HISTOGRAM_KERNEL_SIZES = [1]
COVARIANCE_KERNEL_SIZES = list(range(1, 12))


def main():
    result = run_adaptive_smoothing(
        flux_path=FLUX_PATH,
        var_path=VAR_PATH,
        cov_data_path=COV_DATA_PATH,
        cov_coord_path=COV_COORD_PATH,
        wavelength_ranges=DIAGNOSTIC_WAVELENGTH_RANGES,
        snr_min=SNR_MIN,
        snr_max=SNR_MAX,
        xy_range=XY_RANGE,
        xy_step=XY_STEP,
        xy_step_min=XY_STEP_MIN,
        z_range=Z_RANGE,
        z_step=Z_STEP,
        z_step_min=Z_STEP_MIN,
        kernel_type=KERNEL_TYPE,
        covariance_histogram_kernel_sizes=COVARIANCE_HISTOGRAM_KERNEL_SIZES,
        covariance_kernel_sizes=COVARIANCE_KERNEL_SIZES,
    )

    print("[done]")
    print(f"  input flux:            {result.flux_input_path}")
    print(f"  input variance:        {result.var_input_path}")
    print(
        f"  fitted covariance:     alpha={result.fitted_alpha:.4f}, "
        f"norm={result.fitted_norm:.4f}, thresh={result.fitted_thresh:.4f}"
    )
    print(f"  ADS flux:              {result.ads_flux_path}")
    print(f"  ADS mask:              {result.ads_mask_path}")
    print(f"  ADS SNR:               {result.ads_snr_path}")
    print(f"  ADS spatial kernel:    {result.kernel_r_path}")
    print(f"  ADS wavelength kernel: {result.kernel_w_path}")
    print(f"  diagnostics PDF:       {result.diagnostic_pdf_path}")


if __name__ == "__main__":
    main()