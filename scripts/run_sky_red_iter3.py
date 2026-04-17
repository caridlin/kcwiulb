from pathlib import Path

from kcwiulb.plot.sky_diagnostics import plot_red_iter3_diagnostics
from kcwiulb.sky.red_iter3 import subtract_red_iter3
from kcwiulb.sky.utils import read_sky_map_iter2, resolve_cube_path


BASE = Path(__file__).resolve().parent
CHANNEL = "red"

SKY_MAP = BASE / "sky_map_red_iter3.txt"

COLLAPSE_WAVELENGTH_RANGES = [(7000, 7500), (7700, 8000)]

# explicit wavelength boundaries between regions
SPLIT_WAVELENGTHS = [7200, 7700]

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

            # --------------------------------------------------------
            # Original unsubtracted cubes
            # --------------------------------------------------------
            science_path = BASE / CHANNEL / field / f"{science_id}_icubes.wc.c.fits"

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
                BASE / CHANNEL / field / f"{science_id}_icubes.wc.c.sky.cr.sky2.cr.fits"
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
                print(f"  [SKIP] Missing file(s) for {science_id}")
                for p in missing:
                    print(f"    {p}")
                continue

            print(f"  Science: {science_id}")
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
                diag_path = BASE / "diagnostics" / CHANNEL / field / f"{science_id}_sky_iter3.pdf"
                plot_red_iter3_diagnostics(
                    result,
                    savepath=diag_path,
                    show=False,
                )


if __name__ == "__main__":
    main()