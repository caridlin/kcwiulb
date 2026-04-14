from pathlib import Path

from astropy.io import fits

from kcwiulb.analysis.source_mask import run_source_mask
from kcwiulb.plot.source_mask_diagnostics import save_source_mask_diagnostic


BASE = Path(__file__).resolve().parent

# =========================
# Data selection
# =========================
CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"
LABEL = "oii"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

# Build the mask from the original windowed cube
COLLAPSE_FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.fits"
COLLAPSE_VAR_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.wc.{LABEL}.fits"

# Apply the mask to the bg-subtracted cube for the next step
APPLY_FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.bg.fits"

# =========================
# Keep regions (always unmasked)
# Format:
#   ("sky", ra_deg, dec_deg, radius_pix)
#   ("pixel", x_pix, y_pix, radius_pix)
# =========================
KEEP_CIRCLES = [
    ("sky", 137.631259665, 10.247538059, 20),
    ("pixel", 160, 22, 20),
]

# =========================
# Collapse / mask settings
# =========================
LINE_MASK = (4240, 4275)
SIGMA_CLIP_VALUE = 5.0
MASKED_VALUE = 0.0

# =========================
# Optional refinement:
# keep only detected mask pixels inside these circles
# Format:
#   ("sky", ra_deg, dec_deg, radius_pix)
#   ("pixel", x_pix, y_pix, radius_pix)
# Set to None to use the automatic mask directly
# =========================
MANUAL_FILTER_CIRCLES = None

# Example:
# MANUAL_FILTER_CIRCLES = [
#     ("pixel", 29, 34, 20),
#     ("pixel", 81, 35, 20),
#     ("pixel", 124, 61, 20),
#     ("pixel", 162, 62, 20),
# ]

# =========================
# Plot control
# =========================
SHOW_PLOT = True
SAVE_PLOT = True

# Plot display ranges
FLUX_VMIN = None   # e.g. -5
FLUX_VMAX = None   # e.g. 20
SNR_VMIN = -50
SNR_VMAX = 50


def main():
    result = run_source_mask(
        collapse_flux_path=COLLAPSE_FLUX_PATH,
        collapse_var_path=COLLAPSE_VAR_PATH,
        apply_flux_path=APPLY_FLUX_PATH,
        keep_circles=KEEP_CIRCLES,
        line_mask=LINE_MASK,
        sigma_clip_value=SIGMA_CLIP_VALUE,
        manual_filter_circles=MANUAL_FILTER_CIRCLES,
        masked_value=MASKED_VALUE,
    )

    diagnostic_png = COLLAPSE_FLUX_PATH.with_name(
        COLLAPSE_FLUX_PATH.name.replace(".fits", ".mask_diagnostic.png")
    )

    collapse_header = fits.getheader(COLLAPSE_FLUX_PATH)

    if SHOW_PLOT or SAVE_PLOT:
        save_source_mask_diagnostic(
            result=result,
            collapse_header=collapse_header,
            keep_circles=KEEP_CIRCLES,
            manual_filter_circles=MANUAL_FILTER_CIRCLES,
            output_path=diagnostic_png,
            show=SHOW_PLOT,
            save=SAVE_PLOT,
            flux_vmin=FLUX_VMIN,
            flux_vmax=FLUX_VMAX,
            snr_vmin=SNR_VMIN,
            snr_vmax=SNR_VMAX,
        )

    print("[done]")
    print(f"  collapse flux input:  {result.collapse_flux_input_path}")
    print(f"  collapse var input:   {result.collapse_var_input_path}")
    print(f"  apply flux input:     {result.apply_flux_input_path}")
    print(f"  mask:                 {result.mask_path}")
    print(f"  bg masked flux cube:  {result.bg_masked_flux_path}")
    if SAVE_PLOT:
        print(f"  diagnostic png:       {diagnostic_png}")
    print(f"  n auto masked:        {result.n_auto_masked}")
    print(f"  n final masked:       {result.n_final_masked}")


if __name__ == "__main__":
    main()