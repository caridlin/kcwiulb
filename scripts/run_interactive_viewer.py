from pathlib import Path

from bokeh.io import curdoc

from kcwiulb.analysis.interactive_viewer import create_interactive_viewer


BASE = Path(__file__).resolve().parent

CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"
LABEL = "oii"

COADD_DIR = BASE / "coadd" / CHANNEL / GROUP

FLUX_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.fits"
RESIDUAL_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.bg.fits"
MODEL_PATH = COADD_DIR / f"coadd_{CHANNEL}_{GROUP}_{PRODUCT}.wc.{LABEL}.bg.model.fits"

# Set to None if you do not want the comparison panel
COMPARISON_PATH = FLUX_PATH
# COMPARISON_PATH = None

COLLAPSE_EXCLUDE = (4240, 4275)
SPECTRUM_X_RANGE = (4100, 4300)

IMAGE_LOW = -0.001
IMAGE_HIGH = 0.001
COLLAPSE_LOW = -0.1
COLLAPSE_HIGH = 0.3

# Image-panel height (width will be derived automatically from nx/ny)
HEIGHT1 = 300

# Spectrum-panel height
HEIGHT2 = 300

# Spectrum-panel width (free parameter)
WIDTH2 = 800

create_interactive_viewer(
    flux_path=FLUX_PATH,
    residual_path=RESIDUAL_PATH,
    model_path=MODEL_PATH,
    comparison_path=COMPARISON_PATH,
    collapse_exclude=COLLAPSE_EXCLUDE,
    spectrum_x_range=SPECTRUM_X_RANGE,
    image_low=IMAGE_LOW,
    image_high=IMAGE_HIGH,
    collapse_low=COLLAPSE_LOW,
    collapse_high=COLLAPSE_HIGH,
    height1=HEIGHT1,
    height2=HEIGHT2,
    width2=WIDTH2,
    doc=curdoc(),
)