from pathlib import Path

from kcwiulb.plot.sky_diagnostics import plot_blue_iter1_diagnostics
from kcwiulb.sky.blue_iter1 import subtract_blue_iter1
from kcwiulb.sky.utils import read_sky_map_iter1, resolve_cube_path


BASE = Path(__file__).resolve().parent
CHANNEL = "blue"

SKY_MAP = BASE / "sky_map_blue_iter1.txt"

COLLAPSE_WAVELENGTH_RANGES = [(3700, 3980), (4150, 5200)]

WRITE_OUTPUT = True
SAVE_DIAGNOSTIC = True


def main():
    sky_map = read_sky_map_iter1(SKY_MAP)

    for field, entries in sky_map.items():
        print(f"\nField: {field}")

        for entry in entries:
            science_id = entry["science"]
            sky1_id = entry["sky1"]
            sky2_id = entry["sky2"]
            sky1_field = entry["sky1_field"]
            sky2_field = entry["sky2_field"]

            science_path = BASE / CHANNEL / field / f"{science_id}_icubes.wc.c.fits"
            sky1_path = resolve_cube_path(BASE, CHANNEL, sky1_field, sky1_id, "_icubes.wc.c.fits")
            sky2_path = resolve_cube_path(BASE, CHANNEL, sky2_field, sky2_id, "_icubes.wc.c.fits")

            if not science_path.exists() or not sky1_path.exists() or not sky2_path.exists():
                print(f"  [SKIP] Missing file for {science_id}")
                continue

            print(f"  Science: {science_id}")
            print(f"    sky1: {sky1_id} ({sky1_field})")
            print(f"    sky2: {sky2_id} ({sky2_field})")

            output_path = science_path.with_name(science_path.stem + ".sky.fits") if WRITE_OUTPUT else None

            result = subtract_blue_iter1(
                science_path=science_path,
                sky1_path=sky1_path,
                sky2_path=sky2_path,
                output_path=output_path,
                collapse_wavelength_ranges=COLLAPSE_WAVELENGTH_RANGES,
            )

            if SAVE_DIAGNOSTIC:
                diag_path = BASE / "diagnostics" / CHANNEL / field / f"{science_id}_sky_iter1.pdf"
                plot_blue_iter1_diagnostics(
                    result,
                    savepath=diag_path,
                    show=False,
                )


if __name__ == "__main__":
    main()