from pathlib import Path

from kcwiulb.ads.post_ads_denoising import run_post_ads_denoising


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"
LABEL = "oii"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

ADS_FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.bg.mask.ads2.5.fits"
ADS_MASK_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.bg.mask.ads2.5.mask.fits"
ADS_KERNEL_R_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.bg.mask.ads2.5.kernelr.fits"

CENTER_RA_DEG = 137.631259665
CENTER_DEC_DEG = 10.247538059

RADIUS_CUT_ARCSEC = 7.0
MIN_KERNEL_PIXELS = 5.0
MIN_REGION_SIZE = 150


def main():
    result = run_post_ads_denoising(
        flux_path=ADS_FLUX_PATH,
        mask_path=ADS_MASK_PATH,
        kernel_r_path=ADS_KERNEL_R_PATH,
        center_ra_deg=CENTER_RA_DEG,
        center_dec_deg=CENTER_DEC_DEG,
        radius_cut_arcsec=RADIUS_CUT_ARCSEC,
        min_kernel_pixels=MIN_KERNEL_PIXELS,
        min_region_size=MIN_REGION_SIZE,
    )

    print("[done]")
    print(f"  input ADS flux:             {result.flux_input_path}")
    print(f"  input ADS mask:             {result.mask_input_path}")
    print(f"  input spatial kernel cube:  {result.kernel_r_input_path}")
    print(f"  denoised flux:              {result.denoised_flux_path}")
    print(f"  denoised mask:              {result.denoised_mask_path}")
    print(f"  initial detected voxels:    {result.n_detected_voxels_input}")
    print(f"  after kernel cut:           {result.n_detected_voxels_after_kernel_cut}")
    print(f"  final detected voxels:      {result.n_detected_voxels_final}")
    print(f"  input regions:              {result.n_regions_input}")
    print(f"  kept regions:               {result.n_regions_kept}")


if __name__ == "__main__":
    main()