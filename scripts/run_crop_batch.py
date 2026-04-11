from pathlib import Path

import numpy as np
from astropy.io import fits

from kcwiulb.crop import crop_fits_cube
from kcwiulb.plot.crop_diagnostics import plot_crop_diagnostics


# ============================================================
# USER SETTINGS
# ============================================================
BASE = Path(__file__).resolve().parent

CHANNEL = "red"   # "blue" or "red"
WRITE_OUTPUT = True
SAVE_DIAGNOSTIC = True
SHOW_DIAGNOSTIC = True
# ============================================================


# ============================================================
# CROP CONFIG
# - wav_crop: actual continuous wavelength crop for the output cube
# - diag_wav_ranges: wavelength segments used only for diagnostic collapse plots
# ============================================================
CROP_FIELDS = {
    "blue": {
        "xcrop": (2, 26),
        "wav_crop": (3652.0, 5675.0),
        "diag_wav_ranges": [(3700, 3980), (4150, 5200)],
        "ycrop": (17, 80),
    },
    "red": {
        "xcrop": (2, 26),
        "wav_crop": (6879.0, 8166.0),
        "diag_wav_ranges": [(7020, 7030), (7100, 7120)],
        "ycrop": (12, 78),
    },
}
# ============================================================


def read_master_filelist(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    current_field = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_field = line[1:-1]
                groups[current_field] = []
            else:
                if current_field is None:
                    raise ValueError(f"Found cube ID before field header in {path}")
                groups[current_field].append(line)

    return groups


def main():
    if CHANNEL not in CROP_FIELDS:
        raise ValueError(f"Unknown channel: {CHANNEL}")

    cfg = CROP_FIELDS[CHANNEL]

    master_filelist = BASE / f"master_filelist_{CHANNEL}.txt"
    if not master_filelist.exists():
        raise FileNotFoundError(f"Missing file list: {master_filelist}")

    groups = read_master_filelist(master_filelist)

    print(f"\nRunning crop batch for channel: {CHANNEL}")
    print(f"Using file list: {master_filelist}")
    print("Crop parameters:")
    print(f"  xcrop           = {cfg['xcrop']}")
    print(f"  ycrop           = {cfg['ycrop']}")
    print(f"  wav_crop        = {cfg['wav_crop']}")
    print(f"  diag_wav_ranges = {cfg['diag_wav_ranges']}")

    for field, cube_ids in groups.items():
        print(f"\nField: {field}")

        cropped_cube_paths = []
        cropped_cube_ids = []

        for cube_id in cube_ids:
            cube = BASE / CHANNEL / field / f"{cube_id}_icubes.wc.fits"

            if not cube.exists():
                print(f"  [MISSING] {cube}")
                continue

            print(f"\n  Cropping: {cube.name}")

            # --------------------------------------------------------
            # Print WAVGOOD info for user guidance
            # --------------------------------------------------------
            with fits.open(cube) as hdul:
                header = hdul[0].header

                if "WAVGOOD0" in header and "WAVGOOD1" in header:
                    w0 = header["WAVGOOD0"]
                    w1 = header["WAVGOOD1"]

                    print(f"    WAVGOOD range: {w0:.2f} – {w1:.2f}")
                    print(
                        f"    Suggested crop: "
                        f"{np.ceil(w0):.0f} – {np.floor(w1):.0f}"
                    )
                else:
                    print("    [WARNING] WAVGOOD0/1 not found")

            # --------------------------------------------------------
            # Run crop
            # --------------------------------------------------------
            if WRITE_OUTPUT:
                result = crop_fits_cube(
                    input_path=cube,
                    output_path=None,
                    xcrop=cfg["xcrop"],
                    ycrop=cfg["ycrop"],
                    wav_crop=cfg["wav_crop"],
                )

                cropped_cube_paths.append(result.output_path)
                cropped_cube_ids.append(cube_id)
            else:
                result = crop_fits_cube(
                    input_path=cube,
                    output_path=None,
                    xcrop=cfg["xcrop"],
                    ycrop=cfg["ycrop"],
                    wav_crop=cfg["wav_crop"],
                )

                cropped_cube_paths.append(cube)
                cropped_cube_ids.append(cube_id)

        # --------------------------------------------------------
        # Save one diagnostic figure per field
        # --------------------------------------------------------
        if SAVE_DIAGNOSTIC and cropped_cube_paths:
            diag_dir = BASE / "diagnostics" / CHANNEL / field
            diag_dir.mkdir(parents=True, exist_ok=True)

            diag_path = diag_dir / f"{field}_crop.png"

            plot_crop_diagnostics(
                cube_paths=cropped_cube_paths,
                cube_ids=cropped_cube_ids,
                diag_wav_ranges=cfg["diag_wav_ranges"],
                title=f"{CHANNEL} / {field} / crop diagnostics",
                savepath=diag_path,
                show=SHOW_DIAGNOSTIC,
                ncols=4,
            )

            print(f"\n  Saved crop diagnostic: {diag_path}")


if __name__ == "__main__":
    main()