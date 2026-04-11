# Continuum Subtraction

## Overview

This step models and removes the continuum from the spectrally-windowed cube using a low-order polynomial fit.

It is applied after spectral window selection and prepares the data for emission-line analysis.

---

## Principle

For each spatial pixel, the spectrum is fit with a low-order polynomial (degree ≤ 2) while masking the emission line region.

The fitted continuum is then subtracted:

    Flux_sub = Flux − Continuum

The uncertainty from the continuum fit is propagated into the variance cube.

---

## Procedure

- Select wavelength window (from previous step)  
- Mask the emission-line region  
- Fit a polynomial continuum (order ≤ 2) with sigma clipping  
- Subtract the continuum from the flux cube  
- Propagate continuum-fit uncertainty into the variance cube  

---

## Running the Step

```
python run_continuum_subtraction.py
```

---

## Configuration

Users can define:

```
CHANNEL = “blue”
GROUP = “a”
PRODUCT = “sky”
LABEL = “oii”

CONTINUUM_ORDER = 2

LINE_MASK = (4240, 4275)
```

- `CONTINUUM_ORDER` should be ≤ 2  
- `LINE_MASK` excludes the emission line from the fit  

---

## Output

- Continuum model:
  - `coadd_*.wc.<label>.bg.model.fits`

- Continuum-subtracted flux:
  - `coadd_*.wc.<label>.bg.fits`

- Updated variance:
  - `coadd_*_var.wc.<label>.bg.fits`

---

## Notes

- The fit uses **iterative sigma clipping** to reject outliers  
- Only low-order polynomials (≤ 2) are used to avoid overfitting  
- Variance includes continuum-fit uncertainty  
- Off-diagonal covariance terms are **unchanged** during this step  
- Users are **strongly recommended** to inspect the outputs in DS9:
  - verify that the continuum is smooth  
  - confirm that emission lines are preserved  
  - ensure no over-subtraction occurs  
  