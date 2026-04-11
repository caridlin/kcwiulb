from pathlib import Path

from kcwiulb.wcs import solve_absolute_wcs_from_reference, write_wcs_corrected_cube
from kcwiulb.plot.wcs_diagnostics import plot_wcs_diagnostics


# ============================================================
# USER SETTINGS (edit each run)
# ============================================================
BASE = Path(__file__).resolve().parent

# FIELD = "offset2_a"
# CUBE_ID = "kb240208_00108"
# CHANNEL = "blue"  
FIELD = "offset2_a"
CUBE_ID = "kr240208_00116"
CHANNEL = "red"  

WRITE_OUTPUT = True
SHOW_PLOT = True
SAVE_PLOT = True

OVERRIDES = {
    # "amplitude_init": 200,
    # "x_mean_init": 7,
    # "y_mean_init": 18,
}
# ============================================================


# ============================================================
# WCS CONFIG (split by channel)
# ============================================================
WCS_FIELDS = {

    "blue": {
        "offset2_a": {
            "ra_deg": 137.631259665,
            "dec_deg": 10.247538059,
            "wavelength_ranges": [(3700, 3980), (4150, 5200)],
            "row_start": 30,
            "col_start": 15,
            "n_rows": 30,
            "n_cols": 10,
            "amplitude_init": 100,
            "x_mean_init": 6,
            "y_mean_init": 16,
            "x_stddev_init": 4,
            "y_stddev_init": 3,
        },
        "offset3_a": {
            "ra_deg": 137.623977217,
            "dec_deg": 0.237107386,
            "wavelength_ranges": [(3700, 3980), (4150, 5200)],
            "row_start": 30,
            "col_start": 2,
            "n_rows": 30,
            "n_cols": 10,
            "amplitude_init": 500,
            "x_mean_init": 5,
            "y_mean_init": 16,
            "x_stddev_init": 4,
            "y_stddev_init": 2,
        },
        "offset2_b": {
            "ra_deg": 138.744238913,
            "dec_deg": 11.145876819,
            "wavelength_ranges": [(3700, 3980), (4150, 5200)],
            "row_start": 26,
            "col_start": 11,
            "n_rows": 20,
            "n_cols": 12,
            "amplitude_init": 100,
            "x_mean_init": 7,
            "y_mean_init": 11,
            "x_stddev_init": 4,
            "y_stddev_init": 2,
        },
        "offset3_b": {
            "ra_deg": 138.750174953,
            "dec_deg": 11.141044412,
            "wavelength_ranges": [(3700, 3980), (4150, 5200)],
            "row_start": 41,
            "col_start": 14,
            "n_rows": 20,
            "n_cols": 12,
            "amplitude_init": 50,
            "x_mean_init": 9,
            "y_mean_init": 11,
            "x_stddev_init": 4,
            "y_stddev_init": 2,
        },
    },

    "red": {
        "offset2_a": {
            "ra_deg": 137.631259665,
            "dec_deg": 10.247538059,
            "wavelength_ranges": [(7010, 7030), (7120, 7120)],
            "row_start": 28,
            "col_start": 15,
            "n_rows": 30,
            "n_cols": 10,
            "amplitude_init": 3,
            "x_mean_init": 8,
            "y_mean_init": 10,
            "x_stddev_init": 4,
            "y_stddev_init": 3,
        },
        "offset3_a": {
            "ra_deg": 137.623977217,
            "dec_deg": 10.237107386,
            "wavelength_ranges": [(7000, 7030), (7100, 7120)],
            "row_start": 25,
            "col_start": 2,
            "n_rows": 30,
            "n_cols": 10,
            "amplitude_init": 30,
            "x_mean_init": 3,
            "y_mean_init": 17,
            "x_stddev_init": 2,
            "y_stddev_init": 2,
        },
        "offset2_b": {
            "ra_deg": 138.744238913,
            "dec_deg": 11.145876819,
            "wavelength_ranges": [(7000, 7030), (7100, 7120)],
            "row_start": 23,
            "col_start": 10,
            "n_rows": 20,
            "n_cols": 12,
            "amplitude_init": 10,
            "x_mean_init": 6,
            "y_mean_init": 7,
            "x_stddev_init": 2,
            "y_stddev_init": 2,
        },
        "offset3_b": {
            "ra_deg": 138.750174953,
            "dec_deg": 11.141044412,
            "wavelength_ranges": [(7000, 7030), (7100, 7120)],
            "row_start": 40,
            "col_start": 14,
            "n_rows": 20,
            "n_cols": 12,
            "amplitude_init": 2,
            "x_mean_init": 10,
            "y_mean_init": 10,
            "x_stddev_init": 2,
            "y_stddev_init": 2,
        },
    },
}
# ============================================================


def main():

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    if CHANNEL not in WCS_FIELDS:
        raise ValueError(f"Unknown channel: {CHANNEL}")

    if FIELD not in WCS_FIELDS[CHANNEL]:
        raise ValueError(f"Unknown field: {FIELD}")

    cfg = {**WCS_FIELDS[CHANNEL][FIELD], **OVERRIDES}

    cube = BASE / CHANNEL / FIELD / f"{CUBE_ID}_icubes.fits"
    if not cube.exists():
        raise FileNotFoundError(f"Missing file: {cube}")

    # --------------------------------------------------------
    # Diagnostics output
    # --------------------------------------------------------
    diag_dir = BASE / "diagnostics" / CHANNEL / FIELD
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_path = diag_dir / f"{CUBE_ID}_wcsfit.png"

    # --------------------------------------------------------
    # Run WCS solve
    # --------------------------------------------------------
    print(f"\nRunning WCS for: {cube}")
    print(f"Channel: {CHANNEL}")
    print(f"Field: {FIELD}")

    print("\nParameters:")
    for k, v in cfg.items():
        if k != "wavelength_ranges":
            print(f"  {k}: {v}")

    result = solve_absolute_wcs_from_reference(
        cube_path=cube,
        **cfg,
    )

    # --------------------------------------------------------
    # Fit results
    # --------------------------------------------------------
    fit = result.fit_result

    print("\nFit Results:")
    print(f"  x_mean    = {fit.x_mean:.3f}")
    print(f"  y_mean    = {fit.y_mean:.3f}")
    print(f"  x_stddev  = {fit.x_stddev:.3f}")
    print(f"  y_stddev  = {fit.y_stddev:.3f}")
    print(f"  amplitude = {fit.amplitude:.3f}")
    print(f"  constant  = {fit.constant:.3f}")
    print(f"  x_ref     = {result.x_ref:.3f}")
    print(f"  y_ref     = {result.y_ref:.3f}")

    # --------------------------------------------------------
    # Diagnostics plot
    # --------------------------------------------------------
    plot_wcs_diagnostics(
        fit,
        row_start=cfg["row_start"],
        col_start=cfg["col_start"],
        n_rows=cfg["n_rows"],
        n_cols=cfg["n_cols"],
        title=f"{CHANNEL} / {FIELD} / {CUBE_ID}",
        savepath=plot_path if SAVE_PLOT else None,
        show=SHOW_PLOT,
    )

    print(f"\nSaved diagnostic: {plot_path}")

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------
    if WRITE_OUTPUT:
        out = write_wcs_corrected_cube(
            input_path=cube,
            output_path=None,
            ra_deg=result.ra_deg,
            dec_deg=result.dec_deg,
            x_ref=result.x_ref,
            y_ref=result.y_ref,
        )
        print(f"\nSaved WCS cube: {out.output_path}")
    else:
        print("\nWRITE_OUTPUT = False → no file written")


if __name__ == "__main__":
    main()