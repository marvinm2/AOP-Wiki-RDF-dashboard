# Colorblind Accessibility Findings (Deuteranopia)

**Date:** 2026-03-27
**Methodology:** Vienot, Brettel & Mollon 1999 deuteranopia simulation matrix + CIEDE2000 delta E (Sharma, Wu, Dalal 2005)
**Confusability threshold:** delta E < 10 (conservative for data visualization context where colors appear as small markers, thin lines, and legend swatches)
**Scope:** Deuteranopia only (green-blind, the most common color vision deficiency type)

---

## 1. Palette Simulation Table

The 11 VHP4Safety palette colors from `BRAND_COLORS['palette']` simulated under deuteranopia:

| Color Name | Original Hex | Simulated Hex | L* | a* | b* |
|------------|-------------|---------------|------|------|------|
| primary | #29235C | #25255c | 17.9 | 18.0 | -33.1 |
| magenta | #E6007E | #848479 | 54.9 | -2.1 | 5.9 |
| blue | #307BBF | #6c6cc0 | 49.2 | 21.7 | -44.2 |
| light_blue | #009FE3 | #8888e4 | 60.3 | 22.4 | -46.8 |
| orange | #EB5B25 | #999908 | 61.3 | -14.6 | 63.4 |
| sky_blue | #93D5F6 | #c5c5f7 | 81.0 | 10.2 | -24.6 |
| deep_magenta | #9A1C57 | #5a5a54 | 38.1 | -1.2 | 3.5 |
| teal | #45A6B2 | #9292b3 | 61.6 | 7.0 | -17.2 |
| warm_pink | #B81178 | #6a6a75 | 45.2 | 2.4 | -6.1 |
| dark_teal | #005A6C | #4c4c6d | 33.6 | 8.3 | -19.0 |
| violet | #64358C | #47478b | 33.4 | 19.3 | -38.0 |

**Key observations:**
- Magenta (#E6007E) shifts to a neutral grey-brown (#848479) -- loses all chromatic identity
- Orange (#EB5B25) shifts to olive-yellow (#999908) -- significant hue shift
- Deep magenta (#9A1C57) and warm pink (#B81178) both collapse to neutral greys
- Blues and teals shift toward purple/violet tones but remain distinguishable from each other

## 2. Pairwise Distinguishability

All 55 pairwise comparisons of the 11 simulated palette colors, sorted by delta E (ascending):

| Color A | Color B | Delta E | Confusable |
|---------|---------|---------|------------|
| dark_teal | violet | 8.51 | Yes |
| blue | light_blue | 10.64 | No |
| primary | violet | 11.63 | No |
| deep_magenta | warm_pink | 12.02 | No |
| light_blue | teal | 12.22 | No |
| primary | dark_teal | 13.46 | No |
| warm_pink | dark_teal | 13.72 | No |
| blue | violet | 14.34 | No |
| sky_blue | teal | 15.34 | No |
| magenta | warm_pink | 15.98 | No |
| blue | teal | 16.33 | No |
| magenta | deep_magenta | 16.41 | No |
| blue | dark_teal | 17.50 | No |
| teal | warm_pink | 17.86 | No |
| light_blue | sky_blue | 17.98 | No |
| blue | warm_pink | 18.91 | No |
| warm_pink | violet | 19.83 | No |
| deep_magenta | dark_teal | 20.30 | No |
| magenta | teal | 22.36 | No |
| magenta | orange | 23.34 | No |
| light_blue | warm_pink | 24.19 | No |
| primary | blue | 25.69 | No |
| light_blue | violet | 26.42 | No |
| primary | warm_pink | 26.66 | No |
| blue | sky_blue | 27.26 | No |
| teal | dark_teal | 27.56 | No |
| deep_magenta | violet | 28.06 | No |
| light_blue | dark_teal | 28.40 | No |
| teal | violet | 29.27 | No |
| deep_magenta | teal | 30.02 | No |
| magenta | dark_teal | 30.31 | No |
| primary | deep_magenta | 30.75 | No |
| blue | deep_magenta | 30.92 | No |
| sky_blue | warm_pink | 32.50 | No |
| magenta | blue | 32.81 | No |
| magenta | sky_blue | 32.91 | No |
| magenta | light_blue | 33.23 | No |
| orange | deep_magenta | 33.49 | No |
| magenta | violet | 36.83 | No |
| light_blue | deep_magenta | 37.04 | No |
| primary | light_blue | 37.28 | No |
| orange | warm_pink | 38.30 | No |
| primary | teal | 39.12 | No |
| primary | magenta | 42.95 | No |
| sky_blue | dark_teal | 43.49 | No |
| sky_blue | violet | 44.00 | No |
| sky_blue | deep_magenta | 44.22 | No |
| orange | teal | 45.61 | No |
| orange | sky_blue | 53.02 | No |
| orange | dark_teal | 54.56 | No |
| light_blue | orange | 62.59 | No |
| blue | orange | 62.82 | No |
| primary | sky_blue | 63.23 | No |
| orange | violet | 65.36 | No |
| primary | orange | 68.87 | No |

## 3. Confusable Pairs Summary

### Palette Confusable Pairs (delta E < 10)

| Color A | Color B | Original A | Original B | Delta E | Group |
|---------|---------|-----------|-----------|---------|-------|
| dark_teal | violet | #005A6C | #64358C | 8.51 | palette |

Under deuteranopia, dark_teal (#005A6C) simulates as #4c4c6d and violet (#64358C) simulates as #47478b. Both collapse toward similar dark blue-grey tones, making them difficult to distinguish in chart context (small markers, thin lines, adjacent legend items).

### OECD Status Confusable Pairs (delta E < 10)

| Color A | Color B | Original A | Original B | Delta E | Group |
|---------|---------|-----------|-----------|---------|-------|
| WNT Endorsed | No Status | #E6007E | #999999 | 9.47 | oecd_status |

Under deuteranopia, WNT Endorsed (#E6007E, magenta) simulates as #848479 (grey-brown) and No Status (#999999, grey) simulates as #999999 (unchanged). Both appear as similar neutral grey tones, making them confusable in OECD status visualizations.

### Type Colors Confusable Pairs (delta E < 10)

No confusable pairs found. All 15 pairwise comparisons have delta E >= 15.34, well above the threshold.

## 4. OECD Status Simulation Reference

For completeness, the 8 OECD status colors under deuteranopia:

| Status | Original Hex | Simulated Hex | L* | a* | b* |
|--------|-------------|---------------|------|------|------|
| EAGMST Under Review | #307BBF | #6c6cc0 | 49.2 | 21.7 | -44.2 |
| Under Development | #009FE3 | #8888e4 | 60.3 | 22.4 | -46.8 |
| TFHA/WNT Endorsed | #29235C | #25255c | 17.9 | 18.0 | -33.1 |
| WNT Endorsed | #E6007E | #848479 | 54.9 | -2.1 | 5.9 |
| Approved | #EB5B25 | #999908 | 61.3 | -14.6 | 63.4 |
| No Status | #999999 | #999999 | 63.2 | -0.0 | 0.0 |
| EAGMST Under Development | #45A6B2 | #9292b3 | 61.6 | 7.0 | -17.2 |
| Not OECD | #93D5F6 | #c5c5f7 | 81.0 | 10.2 | -24.6 |

## 5. Type Colors Simulation Reference

The 6 type colors under deuteranopia:

| Type | Original Hex | Simulated Hex | L* | a* | b* |
|------|-------------|---------------|------|------|------|
| Essential | #29235C | #25255c | 17.9 | 18.0 | -33.1 |
| Metadata | #E6007E | #848479 | 54.9 | -2.1 | 5.9 |
| Content | #EB5B25 | #999908 | 61.3 | -14.6 | 63.4 |
| Context | #93D5F6 | #c5c5f7 | 81.0 | 10.2 | -24.6 |
| Assessment | #307BBF | #6c6cc0 | 49.2 | 21.7 | -44.2 |
| Structure | #45A6B2 | #9292b3 | 61.6 | 7.0 | -17.2 |

## 6. Affected Color Groups Preview

The following color groups contain confusable pairs under deuteranopia simulation:

- **palette** -- 1 confusable pair (dark_teal / violet, delta E = 8.51)
- **oecd_status** -- 1 confusable pair (WNT Endorsed / No Status, delta E = 9.47)
- **type_colors** -- No confusable pairs

Plan 02 will cross-reference these confusable pairs against each plot's specific color usage to identify which plots are affected.

### Near-threshold pairs (10 <= delta E < 15)

These pairs are technically above the confusability threshold but warrant attention in plots with small color areas:

| Color A | Color B | Delta E | Group |
|---------|---------|---------|-------|
| blue | light_blue | 10.64 | palette |
| primary | violet | 11.63 | palette |
| deep_magenta | warm_pink | 12.02 | palette |
| light_blue | teal | 12.22 | palette |
| EAGMST Under Review | Under Development | 10.64 | oecd_status |
| Under Development | EAGMST Under Development | 12.22 | oecd_status |
| No Status | EAGMST Under Development | 13.91 | oecd_status |

## 7. Methodology Note

**Simulation algorithm:** Vienot, Brettel & Mollon (1999) "Digital Video Colourmaps for Checking the Legibility of Displays by Dichromats." The deuteranopia simulation uses a 3x3 matrix applied to linearized sRGB values, with standard sRGB transfer function (IEC 61966-2-1) for gamma handling.

**Color difference metric:** CIEDE2000 (Sharma, Wu, Dalal 2005) "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations." Colors are converted from sRGB to CIELAB via XYZ (D65 illuminant) before computing delta E.

**Scope limitation:** Only deuteranopia (the most common color vision deficiency, affecting ~6% of males) was evaluated, per decision D-09. Protanopia and tritanopia were not simulated.

**Document-only findings:** This analysis documents confusable pairs without proposing specific changes to the color palette, per decision D-11. Phase 8 will use these findings to determine per-plot color adjustments.

**Reproducibility:** All results can be reproduced by running `python .planning/phases/07-plot-audit/colorblind_analysis.py`. The script requires only numpy.

---
*Generated from colorblind_analysis.py output on 2026-03-27*
