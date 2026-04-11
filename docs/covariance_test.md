# Covariance Calibration

## Overview

This step quantifies the impact of spatial covariance introduced during the coaddition process.

Although optional, it is **highly recommended** as a validation step to ensure that the noise properties of the coadded data are well understood and properly characterized.

---

## Motivation

During coaddition, interpolation and resampling introduce **correlations between neighboring pixels**. As a result:

- The diagonal variance alone **underestimates the true noise**
- The measured signal-to-noise ratio (SNR) becomes **biased**
- Downstream analysis (e.g., smoothing, detection thresholds) can be affected

This step measures and calibrates that effect.

---

## Method

We compare two estimates of noise:

- **With covariance**: full variance including covariance terms  
- **Diagonal only**: variance assuming no pixel correlations  

We compute the ratio:

$$\sigma_{\mathrm{measured}} / \sigma_{\mathrm{nocov}}$$

as a function of spatial binning size.

---

## Procedure

1. **Spatial rebinning**

   The cube is rebinned using kernel sizes:

$$N = 1 \times 1,\ 2 \times 2,\ \ldots,\ 11 \times 11$$

2. **Wavelength selection**

   Only selected wavelength ranges are used to avoid strong emission lines.

3. **Blank-sky masking**

   A sigma-clipped collapsed image is used to identify and mask non-background regions.

4. **SNR measurement**

   For each kernel size:
   - Compute SNR using full covariance
   - Compute SNR using diagonal-only variance
   - Fit a Gaussian to the SNR distribution

5. **Noise ratio computation**

   For each pixel:

$$\frac{\sigma_{\mathrm{measured}}}{\sigma_{\mathrm{nocov}}}
= \frac{\sqrt{\mathrm{Var}_{\mathrm{full}}}}{\sqrt{\mathrm{Var}_{\mathrm{diag}}}}$$

6. **Model fitting**

   The noise scaling is modeled as:

$$\frac{\sigma_{\mathrm{measured}}}{\sigma_{\mathrm{nocov}}}
= \mathrm{norm}\,(1 + \alpha \log N_{\mathrm{kernel}})$$

   with a plateau beyond a threshold kernel size.

---

## Output

The pipeline produces a PDF containing:

### 1. SNR Distributions

For each kernel size:
- Left: diagonal-only SNR  
- Right: full covariance SNR  

These should be approximately Gaussian with:

σ ≈ 1

for a well-behaved noise model.

---

### 2. Covariance Calibration Curve

- Black points: individual pixel measurements  
- Black line: mean trend  
- Grey region: ±1σ scatter  
- Red curve: fitted model  

This curve quantifies how much the noise is boosted by covariance.

---

## Example Result

![Covariance Calibration](../examples/figures/coadd_blue_a_sky_covariance_test.png)

---

## Interpretation

- If the ratio is close to 1 → covariance is negligible  
- If the ratio increases with kernel size → covariance is significant  
- A smooth logarithmic trend indicates a **well-behaved coadd**

---

## Notes

- Only **one off-diagonal covariance band** is stored for efficiency  
- The full covariance is reconstructed assuming symmetry  
- Off-diagonal terms are counted twice during rebinning

---

## When to Use

This step is recommended when:

- Validating a new coaddition pipeline  
- Comparing different reduction strategies  
- Preparing results for publication  
- Performing quantitative SNR-based analysis  

---

## Summary

Covariance calibration provides a **quantitative validation of noise properties** in the coadded data and ensures that measurements based on SNR are reliable.

It is strongly recommended for scientific analyses involving faint signals.