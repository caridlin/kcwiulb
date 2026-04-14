## Repository Structure

The `kcwiulb` repository is organized as follows:

```text
kcwiulb/
│
├── src/kcwiulb/         # core library code
│   ├── ads/             # adaptive smoothing and detection
│   ├── coadd/           # coaddition and covariance handling
│   ├── sky/             # sky subtraction routines
│   ├── analysis/        # analysis utilities
│   ├── plot/            # plotting utilities
│   ├── wcs.py           # WCS-related functions
│   ├── crop.py          # cube cropping
│   └── cli.py           # command-line entry point
│
├── scripts/             # wrapper scripts for each pipeline step
│   ├── run_ads.py
│   ├── run_coadd_blue.py
│   ├── run_sky_subtraction_iter*.py
│   ├── run_source_mask.py
│   └── ...
│
├── docs/                # step-by-step documentation
├── examples/            # example configs and diagnostic figures
├── tests/               # basic tests (optional)
│
├── environment.yml      # conda environment
├── pyproject.toml       # package configuration
└── README.md
```

### How to Use This Structure

- The **core functionality** lives in `src/kcwiulb/`  
- The **scripts/** directory provides ready-to-use wrappers for each pipeline step  
- Users typically:
  1. copy a script from `scripts/` into their data directory  
  2. modify parameters (paths, thresholds, ranges, etc.)  
  3. run the script locally  

This design keeps the pipeline flexible while allowing users to customize each step for their dataset.