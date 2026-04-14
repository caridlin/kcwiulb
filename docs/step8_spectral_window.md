# Spectral Window Selection

## Overview

This step extracts a narrow spectral window from the coadded cube around a target emission line.

It is primarily used for downstream analysis where working in a limited wavelength range simplifies sky subtraction, continuum treatment, and line fitting.

---

## Principle

Given a wavelength range:

$$
[\lambda_{\min}, \lambda_{\max}]
$$

the cube is cropped along the spectral axis using the WCS solution:

$$
i = \frac{\lambda - \mathrm{CRVAL3}}{\mathrm{CD3\_3}} + \mathrm{CRPIX3}
$$

The spectral WCS is updated accordingly so that the cropped cube retains a consistent wavelength solution.

---

## Procedure

- Select a target wavelength range around an emission line  
- Convert wavelengths to spectral indices using WCS  
- Extract the corresponding spectral slab  
- Update the spectral WCS (CRPIX3)  
- Apply the same operation to:
  - flux cube  
  - variance cube  
  - covariance data (renamed for consistency)  

---

## Running the Step

```bash
python run_spectral_window.py
```

---
`
## Configuration

Users can define:

```python
CHANNEL = "blue"
GROUP = "a"
PRODUCT = "sky"

LABEL = "oii"   # e.g., "ha", "oii", "oiii"

WAVELENGTH_MIN = 4100  # redshifted [O II] λλ3727,3729 (z ≈ 0.096)
WAVELENGTH_MAX = 4300
```

- LABEL defines the output file suffix
- WAVELENGTH_MIN/MAX should bracket the emission line of interest

---
## Output


- Cropped cubes:
  - `coadd_*.wc.<label>.fits`
  - `coadd_*_var.wc.<label>.fits`
- Covariance:
  - `coadd_*_cov_data_<label>.npy`

---
## Example

For:

`LABEL = "ha"`

the output files are:

```
coadd_blue_a_sky.wc.ha.fits
coadd_blue_a_sky_var.wc.ha.fits
coadd_blue_a_sky_cov_data_ha.npy
```

---

### Notes

- Users are **strongly recommended** to inspect the output cubes in DS9 (or a similar viewer) to verify that the selected wavelength window correctly captures the full emission-line profile while avoiding strong sky emission features.

- The window should also include some **continuum-dominated sky** on either side of the line to support the next continuum-subtraction step.