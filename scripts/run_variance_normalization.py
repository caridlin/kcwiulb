from pathlib import Path
import shutil

from kcwiulb.coadd.variance_normalization import run_variance_scaling


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.fits"
VAR_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_var.wc.fits"
COV_DATA_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_cov_data.npy"
COV_COORD_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}_cov_coordinate.npy"

WAVELENGTH_RANGES = [
    (3700, 4000),
    (4800, 5100),
]


def backup_file(path: Path) -> Path:
    backup = path.with_name(f"{path.stem}_old{path.suffix}")
    print(f"[backup] {path.name} -> {backup.name}")
    shutil.move(path, backup)
    return backup


def main():
    print("[variance normalization]")

    result = run_variance_scaling(
        flux_path=FLUX_PATH,
        var_path=VAR_PATH,
        cov_data_path=COV_DATA_PATH,
        cov_coord_path=COV_COORD_PATH,
        wavelength_ranges=WAVELENGTH_RANGES,
        output_dir=COADD_DIR,
        prefix=FLUX_PATH.stem,
    )

    var_backup = backup_file(VAR_PATH)
    cov_backup = backup_file(COV_DATA_PATH)

    print("[overwrite] replacing original files with scaled versions")
    shutil.move(result.var_output_path, VAR_PATH)
    shutil.move(result.cov_data_output_path, COV_DATA_PATH)

    print("\n[done]")
    print(f"  fitted mu:    {result.fitted_mu:.4f}")
    print(f"  fitted sigma: {result.fitted_sigma:.4f}")
    print(f"  scale factor: {result.scale_factor:.4f}")
    print(f"  backup var:   {var_backup}")
    print(f"  backup cov:   {cov_backup}")
    print(f"  plot:         {result.output_plot_path}")


if __name__ == "__main__":
    main()