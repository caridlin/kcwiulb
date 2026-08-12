from pathlib import Path

from kcwiulb.analysis.continuum_subtraction import (
    continuum_subtract_multiple_cube_pairs,
)


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP


# ==========================================================
# Cropped spectral-window products
# ==========================================================

LABELS = [
    "feii2626",
    "mgii",
    "oii",
]

FLUX_PATHS = {
    label: COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{label}.fits"
    for label in LABELS
}

VAR_PATHS = {
    label: COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.wc.{label}.fits"
    for label in LABELS
}


# ==========================================================
# Continuum-subtraction settings
# z_sys = 0.434400
# ==========================================================

CONFIGS = {

    "feii2626": {
        "continuum_order": 2,
        "line_mask": (3755, 3780),
    },

    "mgii": {
        "continuum_order": 2,
        "line_mask": (3970, 4050),
    },

    "oii": {
        "continuum_order": 2,
        "line_mask": (5325, 5370),
    },

}


# ==========================================================
# Main
# ==========================================================

def main():

    results = continuum_subtract_multiple_cube_pairs(
        flux_paths=FLUX_PATHS,
        var_paths=VAR_PATHS,
        configs=CONFIGS,
    )

    print("\n================================")
    print("CONTINUUM SUBTRACTION COMPLETE")
    print("================================")

    for label, result in results.items():

        print(f"\n[{label}]")
        print(f"  range : {result.wavelength_min_actual:.2f}"
              f"-{result.wavelength_max_actual:.2f} A")
        print(f"  order : {result.continuum_order}")
        print(f"  masked: {result.n_masked_channels} channels")
        print(f"  flux  : {result.flux_bg_sub_path}")
        print(f"  var   : {result.var_bg_sub_path}")


if __name__ == "__main__":
    main()