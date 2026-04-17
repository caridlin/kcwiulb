from pathlib import Path
import time
from collections import defaultdict
import warnings

from astropy.wcs import FITSFixedWarning

from kcwiulb.sky.red_cr_iter2 import cosmic_ray_mask_red_group_iter2

warnings.simplefilter("ignore", FITSFixedWarning)

BASE = Path(__file__).resolve().parent
CHANNEL = "red"

MASTER_FILELIST = BASE / "master_filelist_red.txt"

PA = 125
ALPHA = 5.0  # sigma threshold for cosmic ray detection
BETA = 5.0   # protects bright lines / sources from being misidentified as cosmic rays
PX_THRESH = 0.1

# ------------------------------------------------------------
# GROUP SELECTION
# ------------------------------------------------------------
GROUP_SUFFIXES_TO_RUN = ["a"]

# ------------------------------------------------------------
# FIELD FILTERING
# ------------------------------------------------------------
OFFSETS_TO_RUN = ["offset2", "offset3"]
FIELDS_TO_EXCLUDE = []
FILES_TO_EXCLUDE = []

# ------------------------------------------------------------
# INPUT/OUTPUT OPTIONS
# ------------------------------------------------------------
# Iteration 2 CR masking is run on the sky2 product only
INPUT_SUFFIX = "_icubes.wc.c.sky.cr.sky2.fits"
OUTPUT_SUFFIX = ".cr.fits"


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


def cube_id_to_path(base: Path, channel: str, field: str, cube_id: str) -> Path:
    return base / channel / field / f"{cube_id}{INPUT_SUFFIX}"


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


def main():
    master = read_master_filelist(MASTER_FILELIST)

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

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    grouped_fields: dict[str, list[str]] = defaultdict(list)
    grouped_cube_ids: dict[str, list[str]] = defaultdict(list)

    for field, cube_ids in filtered_fields.items():
        _, suffix = split_field_name(field)

        group_name = field if suffix is None else suffix

        grouped_fields[group_name].append(field)
        grouped_cube_ids[group_name].extend(cube_ids)

        for cube_id in cube_ids:
            path = cube_id_to_path(BASE, CHANNEL, field, cube_id)
            grouped_paths[group_name].append(path)

    group_names = sorted(grouped_paths.keys())
    total = len(group_names)

    print("Running red cosmic-ray masking (iter2, sky2 only)")
    print(f"  PA: {PA}")
    print(f"  ALPHA: {ALPHA}")
    print(f"  BETA: {BETA}")
    print(f"  PX_THRESH: {PX_THRESH}")
    print(f"  GROUP_SUFFIXES_TO_RUN: {GROUP_SUFFIXES_TO_RUN}")
    print(f"  OFFSETS_TO_RUN: {OFFSETS_TO_RUN}")
    print(f"  FIELDS_TO_EXCLUDE: {FIELDS_TO_EXCLUDE}")
    print(f"  FILES_TO_EXCLUDE: {FILES_TO_EXCLUDE}")
    print(f"  Input suffix: {INPUT_SUFFIX}")
    print(f"  Output suffix: {OUTPUT_SUFFIX}")
    print(f"  Groups: {group_names}\n")

    t_start = time.time()

    for i, group_name in enumerate(group_names, start=1):
        input_paths = grouped_paths[group_name]
        fields_used = sorted(grouped_fields[group_name])
        cube_ids_used = grouped_cube_ids[group_name]

        print(f"\nGroup: {group_name}")
        print(f"  Fields: {fields_used}")
        print(f"  N cubes: {len(input_paths)}")
        print(f"  Cube IDs: {cube_ids_used}")

        missing = [p for p in input_paths if not p.exists()]
        if missing:
            print("  [SKIP] Missing files:")
            for p in missing:
                print(f"    {p}")
            print_progress(i, total, t_start)
            continue

        t0 = time.time()

        try:
            result = cosmic_ray_mask_red_group_iter2(
                input_paths=input_paths,
                group_name=group_name,
                pa=PA,
                alpha=ALPHA,
                beta=BETA,
                px_thresh=PX_THRESH,
                output_dir=None,
                suffix=OUTPUT_SUFFIX,
            )
        except Exception as e:
            print(f"  [SKIP] {e}")
            print_progress(i, total, t_start)
            continue

        dt = time.time() - t0

        print(f"  Masked {result.n_cubes} cubes in {dt:.1f}s")
        print(f"  Total masked voxels: {result.n_masked_total}")
        print(f"  Masked per cube: {result.masked_per_cube}")

        for out_path in result.output_paths:
            print(f"  Output: {out_path}")

        print_progress(i, total, t_start)

    print("\n\nDone.")


if __name__ == "__main__":
    main()