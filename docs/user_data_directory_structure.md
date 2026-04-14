## User Data Directory Structure

The pipeline is designed to operate on a user-defined data directory containing KCWI data cubes and intermediate products.

A typical working directory may look like:

```text
project_root/
│
├── blue/                         # KCWI-blue data cubes and intermediate products
│   ├── offset2_a/
│   ├── offset3_a/
│   ├── offset2_b/
│   └── offset3_b/
│
├── red/                          # KCWI-red data cubes and intermediate products
│   ├── offset2_a/
│   ├── offset3_a/
│   ├── offset2_b/
│   └── offset3_b/
│
├── coadd/                        # coadded cubes and covariance products
│   ├── blue/
│   └── red/
│
├── diagnostics/                  # diagnostic plots produced during processing
│   ├── blue/
│   └── red/
│
├── master_filelist_blue.txt      # generated list of blue cubes
└── master_filelist_red.txt       # generated list of red cubes
```

In practice, users will often copy selected wrapper scripts (e.g., `run_ads.py`, `run_coadd_blue.py`) from the repository `scripts/` directory into `project_root/` and modify the parameters there. These working copies are omitted from the schematic above for clarity.

---

## Input Data

Each field directory should contain KCWI data cubes produced by the standard KCWI Data Reduction Pipeline (DRP):

https://kcwi-drp.readthedocs.io/en/latest/

```text
Blue channel: kbXXXXXX_XXXXX_icubes.fits
Red channel:  krXXXXXX_XXXXX_icubes.fits
```

These `icubes.fits` files are flux-calibrated three-dimensional data cubes that include:

- bias subtraction  
- flat-fielding  
- wavelength calibration  
- slice reconstruction  

---

## Important Notes

- Sky subtraction within the KCWI DRP is **not recommended** for this workflow.  
- The `kcwiulb` pipeline performs its own sky subtraction optimized for ultra–low surface brightness analysis.  

These DRP products therefore serve as the input data for the `kcwiulb` post-processing pipeline.

---

## Usage

Users should place their data in a directory structure similar to the above and then:

1. Copy the relevant wrapper scripts into this directory  
2. Update file paths and parameters  
3. Run each pipeline step sequentially  

The exact structure can be adapted as needed, but maintaining a consistent organization is strongly recommended.