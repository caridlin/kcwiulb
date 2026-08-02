import re
from pathlib import Path


BASE = Path(__file__).resolve().parent
CHANNEL = "blue"

MASTER_FILELIST = BASE / "master_filelist_blue.txt"
OUTPUT_FILE = BASE / "sky_map_blue_iter2.txt"


def read_master_filelist(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    current_group: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_group = line[1:-1].strip()
                groups[current_group] = []
                continue

            if current_group is None:
                raise ValueError(
                    f"Found cube ID before a group header in {path}: {line}"
                )

            groups[current_group].append(line)

    return groups


def extract_frame_number(cube_id: str) -> int:
    match = re.search(r"_(\d+)$", cube_id)

    if match is None:
        raise ValueError(
            f"Could not extract frame number from cube ID: {cube_id}"
        )

    return int(match.group(1))


def extract_date_code(cube_id: str) -> str:
    match = re.match(r"(kb\d+)_", cube_id)

    if match is None:
        raise ValueError(
            f"Could not extract date code from cube ID: {cube_id}"
        )

    return match.group(1)


def select_adjacent_four_skies(
    science_id: str,
    candidates: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Select four sky frames.

    Parameters
    ----------
    science_id
        Science cube ID.
    candidates
        Candidate sky frames as (sky_cube_id, sky_field).

    Priority
    --------
    1. Same observing date.
    2. Prefer two frames below and two above.
    3. Otherwise, use the four nearest frames.
    """
    science_date = extract_date_code(science_id)
    science_number = extract_frame_number(science_id)

    same_date = [
        (sky_id, sky_field)
        for sky_id, sky_field in candidates
        if extract_date_code(sky_id) == science_date
    ]

    if len(same_date) < 4:
        raise RuntimeError(
            f"Not enough same-date sky candidates for {science_id}: "
            f"{same_date}"
        )

    same_date.sort(
        key=lambda item: extract_frame_number(item[0])
    )

    lower = [
        item
        for item in same_date
        if extract_frame_number(item[0]) < science_number
    ]

    upper = [
        item
        for item in same_date
        if extract_frame_number(item[0]) > science_number
    ]

    if len(lower) >= 2 and len(upper) >= 2:
        selected = lower[-2:] + upper[:2]
    else:
        selected = sorted(
            same_date,
            key=lambda item: (
                abs(
                    extract_frame_number(item[0])
                    - science_number
                ),
                extract_frame_number(item[0]),
            ),
        )[:4]

    selected = sorted(
        selected,
        key=lambda item: extract_frame_number(item[0]),
    )

    if len(selected) != 4:
        raise RuntimeError(
            f"Failed to select four skies for {science_id}: "
            f"{selected}"
        )

    return selected


# ============================================================
# Explicit [science] / [sky] layout
# ============================================================
def build_explicit_sky_map(
    groups: dict[str, list[str]],
) -> dict[str, list[dict]]:
    science_ids = groups["science"]
    sky_ids = groups["sky"]

    if not science_ids:
        raise ValueError("The [science] group is empty.")

    if len(sky_ids) < 4:
        raise ValueError(
            "The [sky] group must contain at least four frames."
        )

    candidates = [
        (sky_id, "sky")
        for sky_id in sky_ids
    ]

    entries = []

    for science_id in science_ids:
        entries.append(
            {
                "science": science_id,
                "skies": select_adjacent_four_skies(
                    science_id=science_id,
                    candidates=candidates,
                ),
            }
        )

    return {"science": entries}


# ============================================================
# Paired *_a / *_b field layout
# ============================================================
def split_field(field: str) -> tuple[str, str]:
    try:
        base, suffix = field.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(
            f"Expected paired field name such as offset2_a: "
            f"{field}"
        ) from exc

    if suffix not in {"a", "b"}:
        raise ValueError(
            f"Expected field suffix '_a' or '_b': {field}"
        )

    return base, suffix


def get_sky_fields(
    science_field: str,
    all_fields: list[str],
) -> list[str]:
    """
    Return ordered candidate sky fields.

    The directly paired field is preferred first, followed by
    fallback fields on the same a/b side.
    """
    science_base, science_suffix = split_field(science_field)
    sky_suffix = "b" if science_suffix == "a" else "a"

    primary = f"{science_base}_{sky_suffix}"

    fallback = []

    for field in all_fields:
        _, suffix = split_field(field)

        if suffix == sky_suffix and field != primary:
            fallback.append(field)

    fallback.sort()

    existing_fields = set(all_fields)

    return [
        field
        for field in [primary, *fallback]
        if field in existing_fields
    ]


def get_paired_candidates(
    science_field: str,
    groups: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """
    Build ordered candidate skies as (cube_id, field).
    """
    candidates: list[tuple[str, str]] = []

    for sky_field in get_sky_fields(
        science_field=science_field,
        all_fields=list(groups),
    ):
        for sky_id in groups[sky_field]:
            candidates.append((sky_id, sky_field))

    # Remove duplicate cube IDs while preserving field priority.
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []

    for sky_id, sky_field in candidates:
        if sky_id not in seen:
            deduped.append((sky_id, sky_field))
            seen.add(sky_id)

    return deduped


def build_paired_sky_map(
    groups: dict[str, list[str]],
) -> dict[str, list[dict]]:
    sky_map: dict[str, list[dict]] = {}

    for science_field, science_ids in groups.items():
        candidates = get_paired_candidates(
            science_field=science_field,
            groups=groups,
        )

        entries = []

        for science_id in science_ids:
            entries.append(
                {
                    "science": science_id,
                    "skies": select_adjacent_four_skies(
                        science_id=science_id,
                        candidates=candidates,
                    ),
                }
            )

        sky_map[science_field] = entries

    return sky_map


def detect_layout(
    groups: dict[str, list[str]],
) -> str:
    lower_names = {
        name.lower()
        for name in groups
    }

    has_science = "science" in lower_names
    has_sky = "sky" in lower_names

    if has_science or has_sky:
        if not (has_science and has_sky):
            raise ValueError(
                "Explicit layout requires both [science] and [sky]."
            )

        return "explicit"

    return "paired"


def normalize_explicit_group_names(
    groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        name.lower(): cube_ids
        for name, cube_ids in groups.items()
    }


def write_sky_map(
    path: Path,
    sky_map: dict[str, list[dict]],
    layout: str,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "# Proposed sky map for blue iteration 2\n"
        )

        if layout == "explicit":
            f.write("# Format:\n")
            f.write(
                "# science | sky1 | sky2 | sky3 | sky4\n\n"
            )
        else:
            f.write(
                "# Includes field labels for sky resolution\n"
            )
            f.write("# Format:\n")
            f.write(
                "# science | sky1(field) | sky2(field) | "
                "sky3(field) | sky4(field)\n\n"
            )

        for field, entries in sky_map.items():
            f.write(f"[{field}]\n")

            for entry in entries:
                if layout == "explicit":
                    sky_text = " | ".join(
                        sky_id
                        for sky_id, _ in entry["skies"]
                    )
                else:
                    sky_text = " | ".join(
                        f"{sky_id}({sky_field})"
                        for sky_id, sky_field in entry["skies"]
                    )

                f.write(
                    f"{entry['science']} | {sky_text}\n"
                )

            f.write("\n")


def main() -> None:
    groups = read_master_filelist(MASTER_FILELIST)
    layout = detect_layout(groups)

    if layout == "explicit":
        groups = normalize_explicit_group_names(groups)
        sky_map = build_explicit_sky_map(groups)
    else:
        sky_map = build_paired_sky_map(groups)

    write_sky_map(
        path=OUTPUT_FILE,
        sky_map=sky_map,
        layout=layout,
    )

    print(f"Detected layout: {layout}")
    print(f"Wrote: {OUTPUT_FILE}")

    for field, entries in sky_map.items():
        print(f"\n[{field}]")

        for entry in entries:
            sky_text = ", ".join(
                (
                    f"{sky_id} ({sky_field})"
                    if layout == "paired"
                    else sky_id
                )
                for sky_id, sky_field in entry["skies"]
            )

            print(
                f"  {entry['science']}: {sky_text}"
            )


if __name__ == "__main__":
    main()