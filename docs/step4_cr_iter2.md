## Cosmic Ray Identification and Masking (Red, after Iteration 2)

This step performs a refined round of cosmic ray identification using the improved sky-subtracted products from Iteration 2.

---

### Purpose

The goal of this step is to generate an improved cosmic ray mask for each cube:

```text
{cube_id}_icubes.wc.c.sky.cr.sky2.cr.fits
```

Each output file contains:

- the Iteration 2 sky-subtracted flux cube (`sky2`)  
  *(we use `sky2` because the spaxel-wise `sky` subtraction can retain cosmic ray contamination present in the sky field itself, whereas `sky2` provides a more stable, median-based estimate for cosmic ray identification)*  
- the corresponding uncertainty cube  
- an updated cosmic ray mask cube (`CRMASK`)  

As before, the flux is not modified; only the mask is produced.

---

### Key Differences from Iteration 1

- Cosmic ray identification is performed **only on the `sky2` product**:

```text
{cube_id}_icubes.wc.c.sky.cr.sky2.fits
```

- An additional upper threshold (`BETA`) is introduced to prevent masking real emission features.

---

### Method Update

The same common-grid comparison method from Iteration 1 is used, with an additional constraint:

```text
median + ALPHA × sigma < data < median + BETA × sigma
```

- `ALPHA` identifies outliers (cosmic rays)  
- `BETA` protects bright emission lines and sources  

---

### Run

```bash
python run_cr_red_iter2.py
```

---

### Inputs

```text
{cube_id}_icubes.wc.c.sky.cr.sky2.fits
```

---

### Main Parameters

```python
PA = 125
ALPHA = 3.0
BETA = 5.0
PX_THRESH = 0.1
```

---

### Output

```text
red/{field}/{cube_id}_icubes.wc.c.sky.cr.sky2.cr.fits
```

---

### Notes

- This step performs cosmic ray identification and masking only  
- The updated masks are used to further improve continuum masking in subsequent steps  
- Each iteration operates on sky-subtracted products derived from the original data (not chained residuals)  
- Inspect the first (flux) and third (CRMASK) extensions in DS9 to verify results  

Example:

![Cosmic Ray Mask Example](../examples/figures/cr_mask_iter2.png)

In this example, a specific wavelength slice shows a background gradient where Iteration 2 more cleanly identifies cosmic rays compared to Iteration 1.

For comparison, see:  
**[Cosmic Ray Identification and Masking (Iteration 1)](step4_cr_iter1.md)**