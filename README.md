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
- WCS anchoring  
- single-cube workflow  
- diagnostic visualization  

Future:
- batch processing  
- auto quality checks  
- multi-channel alignment  
