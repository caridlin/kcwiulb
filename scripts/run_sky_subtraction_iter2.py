from pathlib import Path

from kcwiulb.plot.sky_diagnostics import plot_blue_iter2_diagnostics
from kcwiulb.sky.blue_iter2 import subtract_blue_iter2
from kcwiulb.sky.utils import read_sky_map_iter2, resolve_cube_path


BASE = Path(__file__).resolve().parent
CHANNEL = "blue"

SKY_MAP = BASE / "sky_map_blue_iter2.txt"

COLLAPSE_WAVELENGTH_RANGES = [(3700, 3980), (4150, 5200)]

# Examples:
# None                   -> one fit region
# [5530]                 -> two fit regions
# [4750, 5400, 5530]     -> four fit regions
SPLIT_WAVELENGTHS = [4750, 5400, 5530]

WRITE_OUTPUT = True
SAVE_DIAGNOSTIC = True


def main():
    sky_map = read_sky_map_iter2(SKY_MAP)

    for field, entries in sky_map.items():
        print(f"\nField: {field}")

        for entry in entries:
            science_id = entry["science"]

            sky1_id = entry["sky1"]
            sky2_id = entry["sky2"]
            sky3_id = entry["sky3"]
            sky4_id = entry["sky4"]

            sky1_field = entry["sky1_field"]
            sky2_field = entry["sky2_field"]
            sky3_field = entry["sky3_field"]
            sky4_field = entry["sky4_field"]

            science_path = BASE / CHANNEL / field / f"{science_id}_icubes.wc.c.fits"
            first_pass_path = science_path.with_name(science_path.stem + ".sky.fits")

            sky1_path = resolve_cube_path(BASE, CHANNEL, sky1_field, sky1_id, "_icubes.wc.c.fits")
            sky2_path = resolve_cube_path(BASE, CHANNEL, sky2_field, sky2_id, "_icubes.wc.c.fits")
            sky3_path = resolve_cube_path(BASE, CHANNEL, sky3_field, sky3_id, "_icubes.wc.c.fits")
            sky4_path = resolve_cube_path(BASE, CHANNEL, sky4_field, sky4_id, "_icubes.wc.c.fits")

            missing = []
            if not science_path.exists():
                missing.append(str(science_path))
            if not first_pass_path.exists():
                missing.append(str(first_pass_path))
            if not sky1_path.exists():
                missing.append(str(sky1_path))
            if not sky2_path.exists():
                missing.append(str(sky2_path))
            if not sky3_path.exists():
                missing.append(str(sky3_path))
            if not sky4_path.exists():
                missing.append(str(sky4_path))

            if missing:
                print(f"  [SKIP] Missing file(s) for {science_id}")
                for path in missing:
                    print(f"    missing: {path}")
                continue

            print(f"  Science: {science_id}")
            print(f"    sky1: {sky1_id} ({sky1_field})")
            print(f"    sky2: {sky2_id} ({sky2_field})")
            print(f"    sky3: {sky3_id} ({sky3_field})")
            print(f"    sky4: {sky4_id} ({sky4_field})")
            print(f"    split_wavelengths: {SPLIT_WAVELENGTHS}")

            output_path_sky = (
                science_path.with_name(science_path.stem + ".sky.sky.fits")
                if WRITE_OUTPUT else None
            )
            output_path_sky2 = (
                science_path.with_name(science_path.stem + ".sky.sky2.fits")
                if WRITE_OUTPUT else None
            )

            result = subtract_blue_iter2(
                science_cropped_path=science_path,
                first_pass_path=first_pass_path,
                sky_paths=[sky1_path, sky2_path, sky3_path, sky4_path],
                output_path_sky=output_path_sky,
                output_path_sky2=output_path_sky2,
                collapse_wavelength_ranges=COLLAPSE_WAVELENGTH_RANGES,
                split_wavelengths=SPLIT_WAVELENGTHS,
            )

            if SAVE_DIAGNOSTIC:
                diag_path = BASE / "diagnostics" / CHANNEL / field / f"{science_id}_sky_iter2.pdf"
                plot_blue_iter2_diagnostics(
                    result,
                    savepath=diag_path,
                    show=False,
                )

            if WRITE_OUTPUT:
                print(f"    saved: {output_path_sky.name}")
                print(f"    saved: {output_path_sky2.name}")

            if SAVE_DIAGNOSTIC:
                print(f"    diagnostic: {diag_path.name}")


if __name__ == "__main__":
    main()