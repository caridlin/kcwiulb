from pathlib import Path
import subprocess
import sys

from kcwiulb.plot.sky_diagnostics import plot_blue_iter2_diagnostics
from kcwiulb.sky.blue_iter2 import subtract_blue_iter2
from kcwiulb.sky.utils import read_sky_map_iter2, resolve_cube_path


# ============================================================
# USER SETTINGS (EDIT THIS ONLY)
# ============================================================
BASE = Path(__file__).resolve().parent
CHANNEL = "blue"

SKY_MAP = BASE / "sky_map_blue_iter2.txt"

FIELD = "offset2_a"
SCIENCE_ID = "kb240208_00108"

COLLAPSE_WAVELENGTH_RANGES = [(3700, 3980), (4150, 5200)]

# Examples:
# None                   -> one fit region
# [5530]                 -> two fit regions
# [4750, 5400, 5530]     -> four fit regions
SPLIT_WAVELENGTHS = [4750, 5400, 5530]

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

    # --------------------------------------------------------
    # Find matching entry
    # --------------------------------------------------------
    entry_match = None
    for entry in sky_map[FIELD]:
        if entry["science"] == SCIENCE_ID:
            entry_match = entry
            break

    if entry_match is None:
        raise ValueError(f"Science ID {SCIENCE_ID} not found in field {FIELD}")

    sky1_id = entry_match["sky1"]
    sky2_id = entry_match["sky2"]
    sky3_id = entry_match["sky3"]
    sky4_id = entry_match["sky4"]

    sky1_field = entry_match["sky1_field"]
    sky2_field = entry_match["sky2_field"]
    sky3_field = entry_match["sky3_field"]
    sky4_field = entry_match["sky4_field"]

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------
    science_path = BASE / CHANNEL / FIELD / f"{SCIENCE_ID}_icubes.wc.c.fits"
    first_pass_path = science_path.with_name(science_path.stem + ".sky.fits")

    sky1_path = resolve_cube_path(BASE, CHANNEL, sky1_field, sky1_id, "_icubes.wc.c.fits")
    sky2_path = resolve_cube_path(BASE, CHANNEL, sky2_field, sky2_id, "_icubes.wc.c.fits")
    sky3_path = resolve_cube_path(BASE, CHANNEL, sky3_field, sky3_id, "_icubes.wc.c.fits")
    sky4_path = resolve_cube_path(BASE, CHANNEL, sky4_field, sky4_id, "_icubes.wc.c.fits")

    if not science_path.exists():
        raise FileNotFoundError(f"Missing science file: {science_path}")
    if not first_pass_path.exists():
        raise FileNotFoundError(f"Missing first-pass iter1 file: {first_pass_path}")
    if not sky1_path.exists():
        raise FileNotFoundError(f"Missing sky1 file: {sky1_path}")
    if not sky2_path.exists():
        raise FileNotFoundError(f"Missing sky2 file: {sky2_path}")
    if not sky3_path.exists():
        raise FileNotFoundError(f"Missing sky3 file: {sky3_path}")
    if not sky4_path.exists():
        raise FileNotFoundError(f"Missing sky4 file: {sky4_path}")

    print(f"\nRunning single cube iter2:")
    print(f"  Science: {SCIENCE_ID}")
    print(f"    sky1: {sky1_id} ({sky1_field})")
    print(f"    sky2: {sky2_id} ({sky2_field})")
    print(f"    sky3: {sky3_id} ({sky3_field})")
    print(f"    sky4: {sky4_id} ({sky4_field})")
    print(f"  split_wavelengths: {SPLIT_WAVELENGTHS}")

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------
    output_path_sky = (
        science_path.with_name(science_path.stem + ".sky.sky.fits")
        if WRITE_OUTPUT else None
    )
    output_path_sky2 = (
        science_path.with_name(science_path.stem + ".sky.sky2.fits")
        if WRITE_OUTPUT else None
    )

    # --------------------------------------------------------
    # Run subtraction
    # --------------------------------------------------------
    result = subtract_blue_iter2(
        science_cropped_path=science_path,
        first_pass_path=first_pass_path,
        sky_paths=[sky1_path, sky2_path, sky3_path, sky4_path],
        output_path_sky=output_path_sky,
        output_path_sky2=output_path_sky2,
        collapse_wavelength_ranges=COLLAPSE_WAVELENGTH_RANGES,
        split_wavelengths=SPLIT_WAVELENGTHS,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    if SAVE_DIAGNOSTIC:
        diag_path = BASE / "diagnostics" / CHANNEL / FIELD / f"{SCIENCE_ID}_sky_iter2.pdf"

        plot_blue_iter2_diagnostics(
            result,
            savepath=diag_path,
            show=False,
        )

        print(f"  Saved diagnostic: {diag_path}")

        if SHOW_DIAGNOSTIC:
            open_pdf(diag_path)

    if WRITE_OUTPUT:
        print(f"  Saved cube 1: {output_path_sky}")
        print(f"  Saved cube 2: {output_path_sky2}")


if __name__ == "__main__":
    main()