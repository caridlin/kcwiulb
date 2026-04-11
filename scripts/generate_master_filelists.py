from pathlib import Path


# 🔧 Base directory (your project root)
BASE = Path(__file__).resolve().parent

# Channels to process
CHANNELS = ["blue", "red"]

# File pattern (change later if needed)
PATTERN = "*_icubes.fits"


def extract_id(filename: str) -> str:
    """
    Convert:
    kb231211_00145_icubes.fits -> kb231211_00145
    """
    return filename.replace("_icubes.fits", "")


def collect_files(channel: str) -> dict[str, list[str]]:
    """
    Automatically find all field folders under a channel
    and collect cube IDs.
    """
    data = {}

    channel_dir = BASE / channel

    if not channel_dir.exists():
        print(f"[WARNING] Missing channel folder: {channel_dir}")
        return data

    # 🔥 Auto-detect all subfolders (fields)
    field_dirs = sorted(
        [d for d in channel_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name
    )

    if not field_dirs:
        print(f"[WARNING] No field folders found in {channel_dir}")
        return data

    for field_dir in field_dirs:
        field = field_dir.name

        files = sorted(field_dir.glob(PATTERN))

        if not files:
            print(f"[WARNING] No matching files in {field_dir}")
            continue

        ids = [extract_id(f.name) for f in files]

        data[field] = ids

    return data


def write_filelist(data: dict[str, list[str]], output_path: Path):
    """
    Write grouped file list to txt file.
    """
    with open(output_path, "w") as f:
        for field, ids in data.items():
            f.write(f"[{field}]\n")
            for cube_id in ids:
                f.write(f"{cube_id}\n")
            f.write("\n")

    print(f"[OK] Wrote: {output_path}")


def main():
    for channel in CHANNELS:
        print(f"\nProcessing channel: {channel}")

        data = collect_files(channel)

        if not data:
            print(f"[WARNING] No data found for {channel}")
            continue

        output = BASE / f"master_filelist_{channel}.txt"
        write_filelist(data, output)


if __name__ == "__main__":
    main()