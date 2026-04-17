from pathlib import Path
import subprocess
import sys

from kcwiulb.plot.sky_diagnostics import plot_red_iter3_diagnostics
from kcwiulb.sky.red_iter3 import subtract_red_iter3
from kcwiulb.sky.utils import read_sky_map_iter2, resolve_cube_path


# ============================================================
# USER SETTINGS (EDIT THIS ONLY)
# ============================================================
BASE = Path(__file__).resolve().parent
CHANNEL = "red"

SKY_MAP = BASE / "sky_map_red_iter3.txt"

FIELD = "offset2_a"
SCIENCE_ID = "kr231022_00178"

COLLAPSE_WAVELENGTH_RANGES = [(7000, 7500), (7700, 8000)]

# explicit wavelength boundaries between regions
SPLIT_WAVELENGTHS = [7200, 7700]

# Optional manual overrides (set to None to use sky map)
SKY1_OVERRIDE = None
SKY2_OVERRIDE = None
SKY3_OVERRIDE = None
SKY4_OVERRIDE = None

WRITE_OUTPUT = True
SAVE_DIAGNOSTIC = True
SHOW_DIAGNOSTIC = True
# ============================================================


def open_pdf(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform == "win32":
            import os
            os.startfile(str(path))
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"[WARNING] Could not open PDF: {e}")


def main():
    sky_map = read_sky_map_iter2(SKY_MAP)

    if FIELD not in sky_map:
        raise ValueError(f"Field {FIELD} not found in sky map")

    entry_match = None
    for entry in sky_map[FIELD]:
        if entry["science"] == SCIENCE_ID:
            entry_match = entry
            break

    if entry_match is None:
        raise ValueError(f"Science ID {SCIENCE_ID} not found in field {FIELD}")

    sky1_id = SKY1_OVERRIDE or entry_match["sky1"]
    sky2_id = SKY2_OVERRIDE or entry_match["sky2"]
    sky3_id = SKY3_OVERRIDE or entry_match["sky3"]
    sky4_id = SKY4_OVERRIDE or entry_match["sky4"]

    sky1_field = entry_match["sky1_field"]
    sky2_field = entry_match["sky2_field"]
    sky3_field = entry_match["sky3_field"]
    sky4_field = entry_match["sky4_field"]

    # --------------------------------------------------------
    # Original unsubtracted cubes
    # --------------------------------------------------------
    science_path = BASE / CHANNEL / FIELD / f"{SCIENCE_ID}_icubes.wc.c.fits"

    sky_paths = [
        resolve_cube_path(BASE, CHANNEL, sky1_field, sky1_id, "_icubes.wc.c.fits"),
        resolve_cube_path(BASE, CHANNEL, sky2_field, sky2_id, "_icubes.wc.c.fits"),
        resolve_cube_path(BASE, CHANNEL, sky3_field, sky3_id, "_icubes.wc.c.fits"),
        resolve_cube_path(BASE, CHANNEL, sky4_field, sky4_id, "_icubes.wc.c.fits"),
    ]

    # --------------------------------------------------------
    # Latest CR-masked intermediate products for mask construction
    # science mask source:
    #   *_icubes.wc.c.sky.cr.sky2.cr.fits
    # sky mask sources:
    #   *_icubes.wc.c.sky.cr.sky2.cr.fits
    # --------------------------------------------------------
    science_mask_path = (
        BASE / CHANNEL / FIELD / f"{SCIENCE_ID}_icubes.wc.c.sky.cr.sky2.cr.fits"
    )

    sky_mask_paths = [
        resolve_cube_path(
            BASE, CHANNEL, sky1_field, sky1_id, "_icubes.wc.c.sky.cr.sky2.cr.fits"
        ),
        resolve_cube_path(
            BASE, CHANNEL, sky2_field, sky2_id, "_icubes.wc.c.sky.cr.sky2.cr.fits"
        ),
        resolve_cube_path(
            BASE, CHANNEL, sky3_field, sky3_id, "_icubes.wc.c.sky.cr.sky2.cr.fits"
        ),
        resolve_cube_path(
            BASE, CHANNEL, sky4_field, sky4_id, "_icubes.wc.c.sky.cr.sky2.cr.fits"
        ),
    ]

    all_paths = [science_path, science_mask_path] + sky_paths + sky_mask_paths
    missing = [p for p in all_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing file(s):\n" + "\n".join(f"  {p}" for p in missing)
        )

    print(f"\nRunning single cube (red iter3):")
    print(f"  Field: {FIELD}")
    print(f"  Science: {SCIENCE_ID}")
    print(f"    sky1: {sky1_id} ({sky1_field})")
    print(f"    sky2: {sky2_id} ({sky2_field})")
    print(f"    sky3: {sky3_id} ({sky3_field})")
    print(f"    sky4: {sky4_id} ({sky4_field})")

    if WRITE_OUTPUT:
        output_path_sky = science_path.with_name(
            science_path.stem + ".sky.cr.sky.cr.sky.fits"
        )
        output_path_sky2 = science_path.with_name(
            science_path.stem + ".sky.cr.sky2.cr.sky2.fits"
        )
    else:
        output_path_sky = None
        output_path_sky2 = None

    result = subtract_red_iter3(
        science_path=science_path,
        science_mask_path=science_mask_path,
        sky_paths=sky_paths,
        sky_mask_paths=sky_mask_paths,
        output_path_sky=output_path_sky,
        output_path_sky2=output_path_sky2,
        collapse_wavelength_ranges=COLLAPSE_WAVELENGTH_RANGES,
        split_wavelengths=SPLIT_WAVELENGTHS,
    )

    if SAVE_DIAGNOSTIC:
        diag_path = BASE / "diagnostics" / CHANNEL / FIELD / f"{SCIENCE_ID}_sky_iter3.pdf"

        plot_red_iter3_diagnostics(
            result,
            savepath=diag_path,
            show=False,
        )

        print(f"  Saved diagnostic: {diag_path}")

        if SHOW_DIAGNOSTIC:
            open_pdf(diag_path)

    if WRITE_OUTPUT:
        print(f"  Saved spaxel-subtracted cube: {result.output_path_sky}")
        print(f"  Saved median-subtracted cube: {result.output_path_sky2}")


if __name__ == "__main__":
    main()