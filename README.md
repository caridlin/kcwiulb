# kcwiulb

A Python package for ultra–low surface brightness IFU emission mapping with KCWI, developed during my PhD.

This pipeline has been used in several published works, including:

- Extensive diffuse Lyman-alpha emission correlated with cosmic structure  
  D. C. Martin et al. 2023, *Nature Astronomy*, 7, 1390  

- Kinematically Complex Circumgalactic Gas Around a Low-mass Galaxy: Filamentary Inflow and Counterrotation in J0910b  
  Z. Lin et al. 2025, *The Astrophysical Journal*, 995, 12  

A detailed description of the methodology is presented in  
*A Framework for Ultra--Low Surface Brightness IFU Emission Mapping with KCWI* (submitted to PASP).

---

## Installation

We recommend using a dedicated conda environment.

```bash
conda env create -f environment.yml
conda activate kcwiulb
pip install -e .
```

---

## Repository Structure and Usage

The pipeline is designed to separate stable library code from user-specific workflows.

The package is organized into:

- `src/kcwiulb/` → core pipeline implementation (library code)  
- `scripts/` → example wrapper scripts for each processing step  

### How to use

The recommended workflow is:

1. Copy a script from `scripts/` into your data directory  
2. Modify file paths and parameters  
3. Run locally, e.g.:

```bash
python run_ads.py
```

These scripts are lightweight templates designed to be adapted to each dataset.

For more details, see:
- [User Data Directory Structure](docs/user_data_directory_structure.md)
- [Repository Structure](docs/repository_structure.md)

---

## Pipeline Overview

The kcwiulb pipeline processes KCWI data cubes through the following stages:

### Pre-processing
1. Generate file lists  
2. WCS correction  
3. Cube cropping  

---

### Sky Subtraction (Step 4)

Different workflows are used depending on the observing mode and channel:

![Sky Subtraction Workflow](examples/figures/pipeline_flowchart.png)

- **Blue channel**
  - Iteration 1  
  - Iteration 2 (multi-sky residual modeling)  

- **Red channel**
  - Iterative sky subtraction with cosmic-ray (CR) removal  
  - Alternating sky subtraction and CR masking  

- **Nod-and-shuffle data**
  - Dedicated sky subtraction workflow (under active development)

---

### Coaddition (Step 5–7)
5. Coaddition (flux, variance, covariance products)  
6. WCS refinement *(optional, on coadds)*  
7. Variance normalization *(optional, on coadds)*  

---

### Post-processing / Analysis (Step 8+)
8. Spectral window selection (e.g., Hα region)  
9. Background subtraction  
10. Source masking  
11. Adaptive smoothing / signal extraction (ADS)  
12. Post-ADS processing (e.g., connected-component denoising)

Additional analysis steps (e.g., sky-line masking, stellar continuum removal, PSF subtraction) are available and may be applied depending on the science case.

---

## Documentation

- [Step 1: File Lists](docs/step1_master_filelists.md)
- [Step 2: WCS Correction](docs/step2_wcs.md)
- [Step 3: Cube Cropping](docs/step3_crop.md)

- **Step 4: Sky Subtraction**
  - Blue: [Iter 1](docs/step4_sky_blue_iter1.md), [Iter 2](docs/step4_sky_blue_iter2.md)
  - Red: [Iter 1](docs/step4_sky_red_iter1.md), [CR Masking 1](docs/step4_cr_iter1.md), [Iter 2](docs/step4_sky_red_iter2.md)

- **Step 5: Coadd**
  - [Blue](docs/step5_coadd_blue.md), [Red](docs/step5_coadd_red.md)  
  ↳ *(Optional but highly recommended)*: [Covariance Test](docs/covariance_test.md)

- [Step 6: WCS Correction (Coadd)](docs/step6_coadd_wcs.md)

- [Step 7: Variance Normalization](docs/step7_variance_normalization.md)

**Example Post-processing / Analysis Flow**

This represents one example analysis workflow; the exact sequence may vary depending on the science case and data quality.

- [Step 8: Spectral Window Selection](docs/step8_spectral_window.md)

- [Step 9: Low-Order Continuum Subtraction](docs/step9_continuum_subtraction.md)  
  ↳ *(Optional but highly recommended)*: [Interactive Viewer](docs/interactive_viewer.md)

- [Step 10: Source Masking](docs/step10_source_mask.md)

- [Step 11: Adaptive Smoothing](docs/step11_ads.md)  
  ↳ *(Optional but highly recommended)*: [Post-ADS Denoising](docs/post_ads_denoising.md)



---

## License

This project is licensed under the MIT License.

If you use this code in your work, please cite the corresponding paper:  
*A Framework for Ultra--Low Surface Brightness IFU Emission Mapping with KCWI* (submitted to PASP).

---

## Future Development

Several additional methods have already been developed and tested in standalone workflows and will be incorporated into the pipeline in future releases:

1. **Batch WCS processing**  
   Batch processing will be implemented for efficiency. However, we strongly recommend inspecting each cube individually, as WCS solutions can vary significantly between exposures.

2. **WCS correction using KCWI guider images**  
   In fields without strong continuum sources, WCS alignment will be extended to use guider images. This will support both pre- and post-KCWI-red observations, as the KCWI guider systems differ significantly between these configurations.

3. **Residual sky subtraction for nod-and-shuffle data**  
   Additional refinement of sky subtraction for pre-KCWI-red nod-and-shuffle observations.

4. **Flexible cropping per exposure**  
   Allow per-cube cropping parameters to account for small shifts in detector alignment between observing runs. In practice, these shifts are typically at the level of ~1 pixel, but accommodating them improves consistency across nights.

5. **Wavelength solution refinement**  
   The KCWI DRP wavelength solution can exhibit small offsets relative to known sky lines. A correction step will be added during coaddition to improve wavelength calibration.

6. **Alternative coaddition with Monte Carlo error propagation**  
   An additional coadd mode will be implemented using Monte Carlo error propagation, avoiding explicit covariance matrix construction. This approach is particularly useful when:
   - wavelength axes differ significantly between cubes  
   - interpolation effects are complex  
   - a computationally lighter uncertainty estimate is desired