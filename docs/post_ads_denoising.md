## Post-ADS Denoising

### Overview

After adaptive smoothing, a small number of spurious detections may remain, primarily caused by photon shot noise. These typically appear as compact, isolated regions with no spatial coherence.

An optional denoising step can be applied to remove these features based on simple morphological criteria.

---

### Motivation

Adaptive smoothing is designed to recover faint emission by increasing the effective SNR. However:

- random noise fluctuations can occasionally exceed the detection threshold  
- these appear as small, disconnected regions  
- they are not physically meaningful structures  

This step removes such artifacts while preserving extended emission.

---

### Method

The denoising is performed on the detected cube (`*.ads.fits`) using connected-component analysis.

Two criteria are applied:

1. **Minimum spatial scale (optional, radial-dependent)**

   For regions far from the central galaxy (e.g., > 7″), remove detections with small smoothing scales:

   ```
   spatial kernel < 5 × 5 pixels
   ```

   This reflects the approximate seeing scale (~1.5″ × 1.5″).

2. **Minimum connected-region size**

   Segment the detected cube into connected components and remove regions smaller than a threshold:

   ```
   N_pixels < 150
   ```

   This ensures that only spatially coherent structures are retained.

---

### Interpretation

- Small regions are most likely noise-driven detections  
- Real CGM emission is expected to be spatially extended and coherent  
- The thresholds are chosen relative to the seeing scale and typical structure size  

These values can be adjusted depending on:

- spatial resolution  
- data depth  
- science goals  

---

### Notes

- This step is **not part of the core ADS algorithm**, but a post-processing refinement  
- It is particularly useful for visualization and presentation  
- For quantitative analysis, users may choose to keep the full ADS output  

---

### Summary

This optional denoising step improves the robustness of detected emission by removing small, noise-dominated regions, while preserving extended CGM structures.