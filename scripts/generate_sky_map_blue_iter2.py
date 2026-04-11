import re
from pathlib import Path


BASE = Path(__file__).resolve().parent
MASTER_FILELIST = BASE / "master_filelist_blue.txt"
OUTPUT_FILE = BASE / "sky_map_blue_iter2.txt"


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


def extract_number(cube_id: str) -> int:
    m = re.search(r"_(\d+)$", cube_id)
    if not m:
        raise ValueError(f"Could not extract frame number from {cube_id}")
    return int(m.group(1))


def extract_date_code(cube_id: str) -> str:
    m = re.match(r"(kb\d+)_", cube_id)
    if not m:
        raise ValueError(f"Could not extract date code from {cube_id}")
    return m.group(1)


def split_field(field: str) -> tuple[str, str]:
    base, suffix = field.rsplit("_", 1)
    return base, suffix


def get_primary_and_fallback_sky_fields(science_field: str, all_fields: list[str]) -> list[str]:
    science_base, science_suffix = split_field(science_field)
    sky_suffix = "b" if science_suffix == "a" else "a"

    primary = f"{science_base}_{sky_suffix}"

    fallback = []
    for field in all_fields:
        base, suffix = split_field(field)
        if suffix == sky_suffix and field != primary:
            fallback.append(field)

    fallback.sort()
    return [primary] + fallback


def get_candidate_skies(
    science_field: str,
    science_id: str,
    groups: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """
    Return (cube_id, field) pairs
    """
    date_code = extract_date_code(science_id)
    all_fields = list(groups.keys())
    sky_fields = get_primary_and_fallback_sky_fields(science_field, all_fields)

    candidates: list[tuple[str, str]] = []

    for sky_field in sky_fields:
        for sid in groups[sky_field]:
            if extract_date_code(sid) == date_code:
                candidates.append((sid, sky_field))

    # deduplicate
    seen = set()
    deduped = []
    for sid, field in candidates:
        if sid not in seen:
            deduped.append((sid, field))
            seen.add(sid)

    return deduped


def get_adjacent_four_skies(
    science_field: str,
    science_id: str,
    groups: dict[str, list[str]],
) -> list[tuple[str, str]]:

    sci_num = extract_number(science_id)
    candidates = get_candidate_skies(science_field, science_id, groups)

    if len(candidates) < 4:
        raise RuntimeError(
            f"Not enough sky candidates for {science_id}: {candidates}"
        )

    sky_list = sorted(
        [(extract_number(sid), sid, field) for sid, field in candidates],
        key=lambda x: x[0]
    )

    lower = [(sid, field) for num, sid, field in sky_list if num < sci_num]
    upper = [(sid, field) for num, sid, field in sky_list if num > sci_num]

    if len(lower) >= 2 and len(upper) >= 2:
        selected = lower[-2:] + upper[:2]
    else:
        # fallback: nearest 4
        selected = sorted(
            candidates,
            key=lambda x: abs(extract_number(x[0]) - sci_num)
        )[:4]

    selected = sorted(selected, key=lambda x: extract_number(x[0]))

    if len(selected) != 4:
        raise RuntimeError(f"Failed for {science_id}: {selected}")

    return selected


def build_iter2_sky_map(groups: dict[str, list[str]]) -> dict[str, list[dict]]:
    sky_map = {}

    for science_field, science_ids in groups.items():
        entries = []

        for science_id in science_ids:
            skies = get_adjacent_four_skies(
                science_field,
                science_id,
                groups
            )

            entries.append(
                {
                    "science": science_id,
                    "skies": skies,
                }
            )

        sky_map[science_field] = entries

    return sky_map


def write_sky_map(path: Path, sky_map: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Proposed sky map for blue iteration 2\n")
        f.write("# Includes field labels for debugging\n")
        f.write("# Format:\n")
        f.write("# science | sky1(field) | sky2(field) | sky3(field) | sky4(field)\n\n")

        for field, entries in sky_map.items():
            f.write(f"[{field}]\n")

            for entry in entries:
                sky_str = " | ".join(
                    f"{sid}({fld})" for sid, fld in entry["skies"]
                )

                f.write(f"{entry['science']} | {sky_str}\n")

            f.write("\n")


def main():
    groups = read_master_filelist(MASTER_FILELIST)
    sky_map = build_iter2_sky_map(groups)
    write_sky_map(OUTPUT_FILE, sky_map)

    print(f"Wrote: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()