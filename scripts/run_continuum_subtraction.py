from pathlib import Path

from kcwiulb.analysis.continuum_subtraction import continuum_subtract_cube_pair


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"
LABEL = "oii"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.fits"
VAR_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.wc.{LABEL}.fits"

CONTINUUM_ORDER = 2
LINE_MASK = (4240, 4275)


def main():
    result = continuum_subtract_cube_pair(
        flux_path=FLUX_PATH,
        var_path=VAR_PATH,
        continuum_order=CONTINUUM_ORDER,
        line_mask=LINE_MASK,
    )

    print("[done]")
    print(f"  flux input:     {result.flux_input_path}")
    print(f"  var input:      {result.var_input_path}")
    print(f"  bg model:       {result.flux_bg_model_path}")
    print(f"  flux bg-sub:    {result.flux_bg_sub_path}")
    print(f"  var bg-sub:     {result.var_bg_sub_path}")
    print(f"  wave range:     {result.wavelength_min_actual:.2f} - {result.wavelength_max_actual:.2f} A")
    print(f"  order:          {result.continuum_order}")
    print(f"  masked chans:   {result.n_masked_channels}")


if __name__ == "__main__":
    main()