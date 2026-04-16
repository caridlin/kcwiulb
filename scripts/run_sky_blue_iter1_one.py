from pathlib import Path
import subprocess
import sys

from kcwiulb.plot.sky_diagnostics import plot_blue_iter1_diagnostics
from kcwiulb.sky.blue_iter1 import subtract_blue_iter1
from kcwiulb.sky.utils import read_sky_map_iter1, resolve_cube_path


# ============================================================
# USER SETTINGS (EDIT THIS ONLY)
# ============================================================
BASE = Path(__file__).resolve().parent
CHANNEL = "blue"

SKY_MAP = BASE / "sky_map_blue_iter1.txt"

FIELD = "offset2_a"
SCIENCE_ID = "kb240208_00108"

COLLAPSE_WAVELENGTH_RANGES = [(3700, 3980), (4150, 5200)]

WRITE_OUTPUT = True
SAVE_DIAGNOSTIC = True
SHOW_DIAGNOSTIC = True   # ✅ NEW
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
    sky_map = read_sky_map_iter1(SKY_MAP)

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
    sky1_field = entry_match["sky1_field"]
    sky2_field = entry_match["sky2_field"]

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------
    science_path = BASE / CHANNEL / FIELD / f"{SCIENCE_ID}_icubes.wc.c.fits"
    sky1_path = resolve_cube_path(BASE, CHANNEL, sky1_field, sky1_id, "_icubes.wc.c.fits")
    sky2_path = resolve_cube_path(BASE, CHANNEL, sky2_field, sky2_id, "_icubes.wc.c.fits")

    if not science_path.exists():
        raise FileNotFoundError(f"Missing science file: {science_path}")
    if not sky1_path.exists():
        raise FileNotFoundError(f"Missing sky1 file: {sky1_path}")
    if not sky2_path.exists():
        raise FileNotFoundError(f"Missing sky2 file: {sky2_path}")

    print(f"\nRunning single cube:")
    print(f"  Science: {SCIENCE_ID}")
    print(f"    sky1: {sky1_id} ({sky1_field})")
    print(f"    sky2: {sky2_id} ({sky2_field})")

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------
    output_path = (
        science_path.with_name(science_path.stem + ".sky.fits")
        if WRITE_OUTPUT else None
    )

    # --------------------------------------------------------
    # Run subtraction
    # --------------------------------------------------------
    result = subtract_blue_iter1(
        science_path=science_path,
        sky1_path=sky1_path,
        sky2_path=sky2_path,
        output_path=output_path,
        collapse_wavelength_ranges=COLLAPSE_WAVELENGTH_RANGES,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    if SAVE_DIAGNOSTIC:
        diag_path = BASE / "diagnostics" / CHANNEL / FIELD / f"{SCIENCE_ID}_sky_iter1.pdf"

        plot_blue_iter1_diagnostics(
            result,
            savepath=diag_path,
            show=False,
        )

        print(f"  Saved diagnostic: {diag_path}")

        if SHOW_DIAGNOSTIC:
            open_pdf(diag_path)

    if WRITE_OUTPUT:
        print(f"  Saved cube: {output_path}")


if __name__ == "__main__":
    main()