from pathlib import Path
import time
from collections import defaultdict
import warnings

from astropy.io import fits
from astropy.wcs import FITSFixedWarning

from kcwiulb.coadd.red import coadd_red_group
from kcwiulb.plot.coadd_diagnostics import plot_coadd_diagnostics


warnings.simplefilter("ignore", FITSFixedWarning)


BASE = Path(__file__).resolve().parent
CHANNEL = "red"

MASTER_FILELIST = BASE / "master_filelist_red.txt"

PRODUCT = "sky2"    # "sky" or "sky2"
PA = 125
PX_THRESH = 0.1

# ------------------------------------------------------------
# RUN MODE
# ------------------------------------------------------------
# True  -> run coadd, then make diagnostics
# False -> skip coadd and only make diagnostics from saved outputs
RUN_COADD = True

# ------------------------------------------------------------
# GROUP SELECTION
# ------------------------------------------------------------
GROUP_SUFFIXES_TO_RUN = ["a"]

# ------------------------------------------------------------
# FIELD FILTERING
# ------------------------------------------------------------
OFFSETS_TO_RUN = ["offset2", "offset3"]
FIELDS_TO_EXCLUDE = []
FILES_TO_EXCLUDE = ["kr240208_00106", "kr231022_00188"]

# ------------------------------------------------------------
# DIAGNOSTIC OPTIONS
# ------------------------------------------------------------
SAVE_DIAGNOSTICS = True
SHOW_DIAGNOSTICS = True


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
                raise ValueError(f"Found cube ID before any field header in {path}")

            groups[current_field].append(line)

    return groups


def split_field_name(field: str) -> tuple[str, str | None]:
    if "_" not in field:
        return field, None
    base, suffix = field.rsplit("_", 1)
    return base, suffix


def cube_id_to_path(base: Path, channel: str, field: str, cube_id: str, product: str) -> Path:
    if product == "sky":
        suffix = "_icubes.wc.c.sky.cr.sky.cr.sky.fits"
    elif product == "sky2":
        suffix = "_icubes.wc.c.sky.cr.sky2.cr.sky2.fits"
    else:
        raise ValueError(f"Unsupported PRODUCT: {product}")

    return base / channel / field / f"{cube_id}{suffix}"


def cube_id_to_mask_path(base: Path, channel: str, field: str, cube_id: str) -> Path:
    suffix = "_icubes.wc.c.sky.cr.sky2.cr.fits"
    return base / channel / field / f"{cube_id}{suffix}"


def print_progress(i: int, total: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    avg_time = elapsed / i if i > 0 else 0.0
    remaining = avg_time * (total - i)

    percent = 100 * i / total
    bar_len = 30
    filled = int(bar_len * i / total)
    bar = "█" * filled + "-" * (bar_len - filled)

    print(
        f"\r[{bar}] {percent:5.1f}% | "
        f"Elapsed: {elapsed:6.1f}s | "
        f"ETA: {remaining:6.1f}s",
        end="",
        flush=True,
    )


def run_coadd_diagnostics(
    flux_path: Path,
    var_path: Path,
    output_dir: Path | None = None,
    show: bool = True,
    save: bool = True,
) -> None:
    print("\n[Diagnostics] Loading coadd outputs...")

    coadd_data = fits.getdata(flux_path)
    coadd_var = fits.getdata(var_path)

    if output_dir is None:
        output_dir = flux_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = None
    if save:
        save_path = output_dir / f"{flux_path.stem}_diagnostics.png"

    print("[Diagnostics] Generating multi-panel figure...")

    plot_coadd_diagnostics(
        coadd_data=coadd_data,
        coadd_var=coadd_var,
        t_exp_tot=None,
        save_path=save_path,
    )

    if save:
        print(f"  Saved: {save_path}")

    if show and save_path is not None:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(save_path)
        plt.figure(figsize=(15, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    elif show and save_path is None:
        plot_coadd_diagnostics(
            coadd_data=coadd_data,
            coadd_var=coadd_var,
            t_exp_tot=None,
            save_path=None,
        )


def main():
    if PRODUCT not in {"sky", "sky2"}:
        raise ValueError(f"Invalid PRODUCT: {PRODUCT}")

    master = read_master_filelist(MASTER_FILELIST)

    # --------------------------------------------------------
    # First filter at field level
    # --------------------------------------------------------
    filtered_fields: dict[str, list[str]] = {}

    for field, cube_ids in master.items():
        base_name, suffix = split_field_name(field)

        if OFFSETS_TO_RUN is not None and base_name not in OFFSETS_TO_RUN:
            continue

        if GROUP_SUFFIXES_TO_RUN is not None and suffix not in GROUP_SUFFIXES_TO_RUN:
            continue

        if field in FIELDS_TO_EXCLUDE:
            continue

        kept_cube_ids = [cid for cid in cube_ids if cid not in FILES_TO_EXCLUDE]

        if len(kept_cube_ids) == 0:
            print(f"[WARNING] No valid files left in field {field}, skipping")
            continue

        filtered_fields[field] = kept_cube_ids

    if len(filtered_fields) == 0:
        raise ValueError("No valid fields remain after filtering.")

    # --------------------------------------------------------
    # Group fields by suffix: a, b, c, ...
    # --------------------------------------------------------
    grouped_input_paths: dict[str, list[Path]] = defaultdict(list)
    grouped_mask_paths: dict[str, list[Path]] = defaultdict(list)
    grouped_fields: dict[str, list[str]] = defaultdict(list)
    grouped_cube_ids: dict[str, list[str]] = defaultdict(list)

    for field, cube_ids in filtered_fields.items():
        _, suffix = split_field_name(field)

        if suffix is None:
            group_name = field
        else:
            group_name = suffix

        grouped_fields[group_name].append(field)
        grouped_cube_ids[group_name].extend(cube_ids)

        for cube_id in cube_ids:
            grouped_input_paths[group_name].append(
                cube_id_to_path(BASE, CHANNEL, field, cube_id, PRODUCT)
            )
            grouped_mask_paths[group_name].append(
                cube_id_to_mask_path(BASE, CHANNEL, field, cube_id)
            )

    group_names = sorted(grouped_input_paths.keys())
    total = len(group_names)

    print("Running red coadd")
    print(f"  RUN_COADD: {RUN_COADD}")
    print(f"  Product: {PRODUCT}")
    print(f"  PA: {PA}")
    print(f"  PX_THRESH: {PX_THRESH}")
    print(f"  GROUP_SUFFIXES_TO_RUN: {GROUP_SUFFIXES_TO_RUN}")
    print(f"  OFFSETS_TO_RUN: {OFFSETS_TO_RUN}")
    print(f"  FIELDS_TO_EXCLUDE: {FIELDS_TO_EXCLUDE}")
    print(f"  FILES_TO_EXCLUDE: {FILES_TO_EXCLUDE}")
    print(f"  Coadd groups: {group_names}\n")

    t_start = time.time()

    for i, group_name in enumerate(group_names, start=1):
        input_paths = grouped_input_paths[group_name]
        mask_paths = grouped_mask_paths[group_name]
        fields_used = sorted(grouped_fields[group_name])
        cube_ids_used = grouped_cube_ids[group_name]

        print(f"\nGroup: {group_name}")
        print(f"  Fields: {fields_used}")
        print(f"  N cubes: {len(input_paths)}")
        print(f"  Cube IDs: {cube_ids_used}")

        missing_inputs = [p for p in input_paths if not p.exists()]
        missing_masks = [p for p in mask_paths if not p.exists()]

        if missing_inputs or missing_masks:
            print("  [SKIP] Missing files:")
            for p in missing_inputs:
                print(f"    input: {p}")
            for p in missing_masks:
                print(f"    mask : {p}")
            print_progress(i, total, t_start)
            continue

        t0 = time.time()

        if RUN_COADD:
            try:
                result = coadd_red_group(
                    input_paths=input_paths,
                    mask_paths=mask_paths,
                    group_name=group_name,
                    product=PRODUCT,
                    pa=PA,
                    px_thresh=PX_THRESH,
                )
            except Exception as e:
                print(f"  [SKIP] {e}")
                print_progress(i, total, t_start)
                continue

            dt = time.time() - t0

            print(f"  Coadded {result.n_cubes} cubes in {dt:.1f}s")
            print(f"  Flux: {result.output_flux_path}")
            print(f"  Var:  {result.output_var_path}")
            print(f"  Cov data:  {result.output_cov_data_path}")
            print(f"  Cov coord: {result.output_cov_coord_path}")

            flux_path = result.output_flux_path
            var_path = result.output_var_path
            diag_dir = flux_path.parent

        else:
            output_dir = BASE / "coadd" / CHANNEL / group_name
            flux_path = output_dir / f"coadd_red_{group_name}_{PRODUCT}.fits"
            var_path = output_dir / f"coadd_red_{group_name}_{PRODUCT}_var.fits"
            diag_dir = output_dir

            if not flux_path.exists():
                print(f"  [SKIP] Missing coadd flux file: {flux_path}")
                print_progress(i, total, t_start)
                continue

            if not var_path.exists():
                print(f"  [SKIP] Missing coadd variance file: {var_path}")
                print_progress(i, total, t_start)
                continue

            print("  Using existing coadd outputs:")
            print(f"  Flux: {flux_path}")
            print(f"  Var:  {var_path}")

        try:
            run_coadd_diagnostics(
                flux_path=flux_path,
                var_path=var_path,
                output_dir=diag_dir,
                show=SHOW_DIAGNOSTICS,
                save=SAVE_DIAGNOSTICS,
            )
        except Exception as e:
            print(f"  [WARNING] Diagnostics failed: {e}")

        print_progress(i, total, t_start)

    print("\n\nDone.")


if __name__ == "__main__":
    main()