from pathlib import Path

from kcwiulb.plot.sky_diagnostics import plot_red_selfsky_iter1_diagnostics
from kcwiulb.sky.red_selfsky_iter1 import subtract_red_selfsky_iter1


# ============================================================
# USER SETTINGS
# ============================================================
BASE = Path(__file__).resolve().parent
CHANNEL = "red"

MASTER_FILELIST = BASE / "master_filelist_red.txt"

FIELD = "science"

COLLAPSE_WAVELENGTH_RANGES = [(6500, 8500)]

# QSO / source position
QSO_RA_DEG = 22.673637627
QSO_DEC_DEG = 0.666761859

# Exclude this radius around the QSO from the self-sky estimate.
# Test multiple radii for extended-emission work to ensure real
# source/CGM emission is not entering the sky estimate.
SOURCE_MASK_RADIUS_ARCSEC = 7.0

# Additional white-band sigma clipping removes other bright sources
# outside the explicit QSO exclusion region.
SIGMA_CLIP = 3.0

WRITE_OUTPUT = True
SAVE_DIAGNOSTIC = True

# Diagnostic display limits
WHITEBAND_VMIN = -1
WHITEBAND_VMAX = 80
RESIDUAL_ZOOM_YLIM = (-0.002, 0.002)
# ============================================================


def read_master_filelist(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    current_field = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_field = line[1:-1]
                groups[current_field] = []
                continue

            if current_field is None:
                raise ValueError(
                    f"Found cube ID before any field header in {path}"
                )

            groups[current_field].append(line)

    return groups


def main():
    if QSO_RA_DEG is None or QSO_DEC_DEG is None:
        raise ValueError(
            "QSO_RA_DEG and QSO_DEC_DEG must be specified "
            "for self-sky subtraction."
        )

    master = read_master_filelist(MASTER_FILELIST)

    if FIELD not in master:
        raise ValueError(
            f"Field '{FIELD}' not found in {MASTER_FILELIST}. "
            f"Available fields: {list(master.keys())}"
        )

    cube_ids = master[FIELD]

    if len(cube_ids) == 0:
        raise ValueError(f"No cube IDs found for field '{FIELD}'.")

    print("Running red self-sky subtraction — iter1")
    print(f"  Master filelist: {MASTER_FILELIST}")
    print(f"  Field: {FIELD}")
    print(f"  QSO RA: {QSO_RA_DEG}")
    print(f"  QSO Dec: {QSO_DEC_DEG}")
    print(f"  Source mask radius: {SOURCE_MASK_RADIUS_ARCSEC} arcsec")
    print(f"  Sigma clip: {SIGMA_CLIP}")
    print(f"  Write output: {WRITE_OUTPUT}")
    print(f"  Save diagnostic: {SAVE_DIAGNOSTIC}")
    print(f"  N cubes: {len(cube_ids)}")
    print(f"  Cube IDs: {cube_ids}")

    for cube_id in cube_ids:
        science_path = (
            BASE
            / CHANNEL
            / FIELD
            / f"{cube_id}_icubes.wc.c.fits"
        )

        if not science_path.exists():
            print(f"\n  [SKIP] Missing: {science_path}")
            continue

        print(f"\n  Science: {cube_id}")

        output_path = (
            science_path.with_name(
                science_path.stem + ".sky.fits"
            )
            if WRITE_OUTPUT
            else None
        )

        result = subtract_red_selfsky_iter1(
            science_path=science_path,
            output_path=output_path,
            collapse_wavelength_ranges=COLLAPSE_WAVELENGTH_RANGES,
            qso_ra_deg=QSO_RA_DEG,
            qso_dec_deg=QSO_DEC_DEG,
            source_mask_radius_arcsec=SOURCE_MASK_RADIUS_ARCSEC,
            sigma_clip=SIGMA_CLIP,
        )

        print(
            f"    QSO pixel: "
            f"({result.qso_x:.2f}, {result.qso_y:.2f})"
        )
        print(
            f"    Source mask radius: "
            f"{result.source_mask_radius_arcsec:.2f} arcsec"
        )

        n_total = result.sky_mask.size
        n_sky = int((~result.sky_mask).sum())

        print(
            f"    Sky pixels: "
            f"{n_sky}/{n_total} ({n_sky / n_total:.1%})"
        )

        if WRITE_OUTPUT:
            print(f"    Output: {result.output_path}")

        if SAVE_DIAGNOSTIC:
            diag_path = (
                BASE
                / "diagnostics"
                / CHANNEL
                / FIELD
                / f"{cube_id}_sky_selfsky_iter1.pdf"
            )

            plot_red_selfsky_iter1_diagnostics(
                result,
                savepath=diag_path,
                show=False,
                whiteband_vmin=WHITEBAND_VMIN,
                whiteband_vmax=WHITEBAND_VMAX,
                residual_zoom_ylim=RESIDUAL_ZOOM_YLIM,
            )

            print(f"    Diagnostic: {diag_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()