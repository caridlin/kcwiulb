# Covariance Test

## Overview

This step validates the noise properties of the coadded cube by testing whether spatial covariance has been correctly propagated.

Although optional, it is **highly recommended** as a diagnostic step to verify that the signal-to-noise ratio (SNR) is properly normalized and that the coaddition pipeline produces statistically consistent results.

---

## Motivation

Interpolation and resampling during coaddition introduce **correlations between neighboring pixels**. If these correlations are not properly accounted for:

- The diagonal variance alone **underestimates the true noise**
- The measured SNR becomes **artificially inflated**
- Downstream analyses (e.g., detection thresholds, smoothing) can be biased

Modern IFU pipelines explicitly account for these effects (e.g., Husemann et al. 2013; Danforth et al. 2016; Law et al. 2016; O’Sullivan et al. 2020). This step provides a **direct empirical test** that the propagated covariance correctly captures these correlations.

---

## Principle

For a correctly propagated noise model, the SNR distribution in blank sky regions should follow:

$\sigma \approx 1$

We compare two cases:

- **Full covariance** → expected to be correct  
- **Diagonal-only variance** → expected to underestimate noise  

The difference between the two provides a direct validation of covariance handling.

---

## Procedure

1. **Spatial rebinning**

   The cube is rebinned with increasing kernel sizes:

   N = 1×1, 2×2, ..., 11×11

2. **Wavelength selection**

   Selected wavelength ranges are used to avoid strong emission lines.

3. **Blank-sky masking**

   A sigma-clipped collapsed image identifies background-dominated regions.

4. **SNR measurement**

   For each kernel size:
   - Compute SNR using full covariance
   - Compute SNR using diagonal-only variance
   - Fit a Gaussian to each SNR distribution

---

## Output

The pipeline produces a PDF containing:

### 1. SNR Distributions (Primary Test)

For each kernel size:
- Left: diagonal-only SNR  
- Right: full covariance SNR  

Expected behavior:

- **Full covariance** → Gaussian with σ ≈ 1  
- **Diagonal-only** → σ < 1 (noise underestimated)

This directly demonstrates whether covariance has been correctly propagated.

---

### 2. Covariance Scaling (Secondary Diagnostic)

We also compute the ratio:

$\frac{\sigma_{\mathrm{measured}}}{\sigma_{\mathrm{nocov}}}$

and measure how it evolves with kernel size.

This provides a **quantitative description** of how covariance inflates the noise.

A simple model is fit:

$\frac{\sigma_{\mathrm{measured}}}{\sigma_{\mathrm{nocov}}}
= \mathrm{norm}\,(1 + \alpha \log N_{\mathrm{kernel}})$

This is useful for characterizing noise behavior but is **not required** for validation.

---

## Example Result

![Covariance Calibration](../examples/figures/coadd_blue_a_sky_covariance_test.png)

---

## Interpretation

- If full covariance gives σ ≈ 1 → ✅ noise is correctly propagated  
- If diagonal-only gives σ < 1 → expected behavior  
- If full covariance deviates from σ ≈ 1 → ❗ potential issue in coaddition  

A smooth increase in the noise ratio with kernel size indicates well-behaved covariance structure.

---

## Notes

- Only **one off-diagonal covariance band** is stored for efficiency  
- Full covariance is reconstructed assuming symmetry  
- Off-diagonal terms are counted twice during rebinning  

---

## When to Use

This test is recommended when:

- Validating a new coaddition pipeline  
- Debugging noise properties  
- Comparing reduction strategies  
- Preparing results for publication  

---

## Summary

This step provides a **direct validation of noise propagation** in the coadded data.

It ensures that spatial covariance is correctly accounted for and that SNR-based measurements are statistically reliable. The covariance scaling curve offers an additional quantitative diagnostic but is secondary to the SNR validation.