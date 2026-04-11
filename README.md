# kcwiulb

A Python package for ultra-low surface brightness IFU emission mapping with KCWI.

A detailed description is presented in:
*A Framework for Ultra--Low Surface Brightness IFU Emission Mapping with KCWI* (submitted to PASP).

---

## Installation

See: [docs/installation.md](docs/installation.md)

---

## Pipeline Overview

1. Generate file lists  
2. WCS correction  
3. Cube cropping  
4. Sky subtraction (blue, iter1)  
5. Sky subtraction (blue, iter2)  
6. Coaddition  
7. Adaptive signal extraction  

---

## Documentation

- [Folder Structure](docs/folder_structure.md)
- [Step 1: File Lists](docs/step1_master_filelists.md)
- [Step 2: WCS Correction](docs/step2_wcs.md)
- [Step 3: Cube Cropping](docs/step3_crop.md)
- [Step 4: Sky Subtraction (Iter 1)](docs/step4_sky_blue_iter1.md)
- [Step 5: Sky Subtraction (Iter 2)](docs/step5_sky_blue_iter2.md)
- [Step 6: Coadd](docs/step6_coadd_blue.md)
- [Step 7: Adaptive Signal Extraction](docs/step7_ads.md)


## License

This project is licensed under the MIT License.

If you use this code in your work, please cite the corresponding paper:
*A Framework for Ultra--Low Surface Brightness IFU Emission Mapping with KCWI* (submitted to PASP).

---

## Status

Version 1 includes:
- WCS anchoring with per-cube validation  
- cube cropping and wavelength trimming  
- two-stage sky subtraction for KCWI-blue (iter1 and iter2)  
- covariance-aware coaddition on a common WCS grid  
- diagnostic visualization at each major processing step  

--- 

## Future Development

Several additional methods have already been developed and tested in standalone workflows, and will be incorporated into the pipeline in future releases:

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