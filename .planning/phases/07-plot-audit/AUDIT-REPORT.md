# AOP-Wiki RDF Dashboard - Plot Audit Report

**Date:** 2026-03-27
**Scope:** All plot functions in plots/latest_plots.py (20) and plots/trends_plots.py (19)
**Method:** Code analysis only (per D-06)
**Rubric:** 8-dimension binary pass/fail (per D-03)

---

## Executive Summary

- **Total functions audited:** 39
- **Total plot outputs audited:** 55 (latest: 20 single outputs; trends: 35 outputs from tuple-returning functions)
- **FIX NOW:** 19 (plot_latest_entity_counts, plot_latest_database_summary, plot_latest_avg_per_aop, plot_latest_ke_components, plot_latest_ke_annotation_depth, plot_latest_network_density, plot_latest_ontology_usage, plot_latest_aop_completeness_unique_colors, plot_latest_ke_by_bio_level, plot_latest_taxonomic_groups, plot_latest_ke_reuse, plot_latest_ke_reuse_distribution, plot_latest_ontology_diversity, plot_avg_per_aop, plot_author_counts, plot_aop_lifetime, plot_ke_components, plot_ke_components_percentage, plot_unique_ke_components)
- **FIX LATER:** 4 (plot_latest_process_usage, plot_latest_object_usage, plot_kes_by_kec_count, plot_ontology_term_growth)
- **SKIP:** 16 (plot_latest_aop_completeness, plot_latest_aop_completeness_by_status, plot_latest_ke_completeness_by_status, plot_latest_ker_completeness_by_status, plot_latest_entity_by_oecd_status, plot_main_graph, plot_network_density, plot_bio_processes, plot_bio_objects, plot_aop_property_presence, plot_ke_property_presence, plot_ker_property_presence, plot_stressor_property_presence, plot_entity_completeness_trends, plot_aop_completeness_boxplot, plot_oecd_completeness_trend)
- **Colorblind confusable pairs:** 2 pairs with delta E < 10 (dark_teal/violet in palette at 8.51; WNT Endorsed/No Status in oecd_status at 9.47)

### Key Findings

**Dominant FIX NOW pattern (latest_plots.py):** Continuous color gradient antipattern (encoding count as color when count is already on an axis) and unnecessary per-bar categorical colors when legend is hidden and axis labels suffice. The fix is mechanically uniform: replace with `marker_color='#307BBF'` or `color_discrete_sequence=[BRAND_COLORS['blue']]`.

**Dominant FIX NOW pattern (trends_plots.py):** Legacy alias color lists — functions use `[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]` instead of `BRAND_COLORS['palette']`. Secondary/CTA magenta (#E6007E) used inappropriately for data series instead of navigation blue (#307BBF).

**Chart type issue (latest_plots.py):** 5 functions use `px.pie` for data that would be clearer as bar charts (distributions, compositions, rankings).

**Wiring issues found:** 3 functions with wiring defects — plot_latest_database_summary (registered in app.py, no template slot), plot_latest_ontology_usage (same), plot_latest_aop_completeness_unique_colors (template slot but not registered in app.py).

---

## Classification Summary

### FIX NOW

| Function | Module | Failed Dimensions | Color Decision |
|----------|--------|-------------------|----------------|
| plot_latest_entity_counts | latest_plots | Color correctness | single #307BBF |
| plot_latest_database_summary | latest_plots | Color correctness, data table | single #307BBF |
| plot_latest_avg_per_aop | latest_plots | Color correctness, axis labels | single #307BBF |
| plot_latest_ke_components | latest_plots | Color correctness, chart type, axis labels | categorical palette |
| plot_latest_ke_annotation_depth | latest_plots | Color correctness, chart type, axis labels, legend clarity | single #307BBF |
| plot_latest_network_density | latest_plots | Color correctness, chart type, axis labels | categorical palette |
| plot_latest_ontology_usage | latest_plots | Color correctness, chart type, data table, axis labels | categorical palette |
| plot_latest_aop_completeness_unique_colors | latest_plots | Color correctness | single #307BBF |
| plot_latest_ke_by_bio_level | latest_plots | Color correctness | single #307BBF |
| plot_latest_taxonomic_groups | latest_plots | Color correctness | single #307BBF |
| plot_latest_ke_reuse | latest_plots | Color correctness | single #307BBF |
| plot_latest_ke_reuse_distribution | latest_plots | Color correctness | single #307BBF |
| plot_latest_ontology_diversity | latest_plots | Color correctness | single #307BBF |
| plot_avg_per_aop | trends_plots | Color correctness | categorical palette |
| plot_author_counts | trends_plots | Color correctness, axis labels | single #307BBF |
| plot_aop_lifetime | trends_plots | Color correctness, layout | single #307BBF (all 3 outputs) |
| plot_ke_components | trends_plots | Color correctness | categorical palette |
| plot_ke_components_percentage | trends_plots | Color correctness | categorical palette |
| plot_unique_ke_components | trends_plots | Color correctness | categorical palette |

### FIX LATER

| Function | Module | Failed Dimensions | Color Decision |
|----------|--------|-------------------|----------------|
| plot_latest_process_usage | latest_plots | Chart type, title, axis labels | categorical palette |
| plot_latest_object_usage | latest_plots | Chart type, title, axis labels | categorical palette |
| plot_kes_by_kec_count | trends_plots | Chart type (delta output), legend clarity, title | categorical palette |
| plot_ontology_term_growth | trends_plots | Color correctness (minor: brand colors but wrong choices) | single #307BBF |

### SKIP

| Function | Module | Color Decision |
|----------|--------|----------------|
| plot_latest_aop_completeness | latest_plots | categorical palette (type_colors) |
| plot_latest_aop_completeness_by_status | latest_plots | categorical palette (type_colors) |
| plot_latest_ke_completeness_by_status | latest_plots | categorical palette (type_colors) |
| plot_latest_ker_completeness_by_status | latest_plots | categorical palette (type_colors) |
| plot_latest_entity_by_oecd_status | latest_plots | categorical palette (oecd_status) |
| plot_main_graph | trends_plots | categorical palette |
| plot_network_density | trends_plots | single #307BBF |
| plot_bio_processes | trends_plots | categorical palette |
| plot_bio_objects | trends_plots | categorical palette |
| plot_aop_property_presence | trends_plots | categorical palette |
| plot_ke_property_presence | trends_plots | categorical palette |
| plot_ker_property_presence | trends_plots | categorical palette |
| plot_stressor_property_presence | trends_plots | categorical palette |
| plot_entity_completeness_trends | trends_plots | categorical palette |
| plot_aop_completeness_boxplot | trends_plots | single #29235C (primary) |
| plot_oecd_completeness_trend | trends_plots | oecd_status color mapping |

---

## All Color Decisions

| # | Function | Module | Color Decision | Rationale |
|---|----------|--------|----------------|-----------|
| 1 | plot_latest_aop_completeness | latest | categorical palette (type_colors) | Property types are semantic categories with established type_colors mapping |
| 2 | plot_latest_aop_completeness_by_status | latest | categorical palette (type_colors) | Same as above |
| 3 | plot_latest_aop_completeness_unique_colors | latest | single #307BBF | x-axis labels identify properties; legend hidden; single color removes noise |
| 4 | plot_latest_avg_per_aop | latest | single #307BBF | Two averages of the same concept; x-axis differentiation sufficient |
| 5 | plot_latest_database_summary | latest | single #307BBF | Entity names on x-axis; legend hidden; single color cleaner |
| 6 | plot_latest_entity_by_oecd_status | latest | categorical palette (oecd_status) | OECD status is semantic; oecd_status mapping is canonical |
| 7 | plot_latest_entity_counts | latest | single #307BBF | Entity names on x-axis; legend hidden; 5 colors add noise |
| 8 | plot_latest_ke_annotation_depth | latest | single #307BBF | Ordered depth bins (0, 1, 2...) — single color for ordinal distribution |
| 9 | plot_latest_ke_by_bio_level | latest | single #307BBF | Count on x-axis; gradient adds no information |
| 10 | plot_latest_ke_completeness_by_status | latest | categorical palette (type_colors) | Property types are semantic; type_colors mapping is correct |
| 11 | plot_latest_ke_reuse | latest | single #307BBF | Count on x-axis; gradient redundant |
| 12 | plot_latest_ke_reuse_distribution | latest | single #307BBF | Count on y-axis; gradient redundant; bins are ordinal not categorical |
| 13 | plot_latest_ke_components | latest | categorical palette | Process, Object, Action are distinct semantic categories |
| 14 | plot_latest_ker_completeness_by_status | latest | categorical palette (type_colors) | Property types are semantic |
| 15 | plot_latest_network_density | latest | categorical palette | Connected vs Isolated are distinct categories |
| 16 | plot_latest_object_usage | latest | categorical palette | Distinct ontology namespaces require differentiation |
| 17 | plot_latest_ontology_diversity | latest | single #307BBF | Ontologies on x-axis; legend hidden; single color removes noise |
| 18 | plot_latest_ontology_usage | latest | categorical palette | Distinct ontology sources require differentiation |
| 19 | plot_latest_process_usage | latest | categorical palette | Distinct ontology namespaces require differentiation |
| 20 | plot_latest_taxonomic_groups | latest | single #307BBF | Group names on y-axis; legend hidden; 25 cycling colors add noise |
| 21 | plot_aop_completeness_boxplot | trends | single #29235C (primary) | Single distribution per version; no categorical distinction needed |
| 22 | plot_aop_lifetime | trends | single #307BBF (all outputs) | Single entity type (AOPs) across 3 related views |
| 23 | plot_aop_property_presence | trends | categorical palette | Multiple AOP properties need distinction; marker shapes add accessibility |
| 24 | plot_author_counts | trends | single #307BBF | Single metric (author count); both absolute and delta outputs |
| 25 | plot_avg_per_aop | trends | categorical palette | 2 metrics (avg KEs/AOP, avg KERs/AOP) require categorical distinction |
| 26 | plot_bio_objects | trends | categorical palette | Multiple distinct ontology namespaces (GO, CHEBI, PR, etc.) |
| 27 | plot_bio_processes | trends | categorical palette | Multiple distinct ontology namespaces (GO, MP, NBO, etc.) |
| 28 | plot_entity_completeness_trends | trends | categorical palette | 4 entity types (AOPs, KEs, KERs, Stressors) require distinction |
| 29 | plot_ke_components | trends | categorical palette | 3 component types (Process, Object, Action) require distinction |
| 30 | plot_ke_components_percentage | trends | categorical palette | Same as plot_ke_components |
| 31 | plot_ke_property_presence | trends | categorical palette | Multiple KE properties need distinction; marker shapes add accessibility |
| 32 | plot_ker_property_presence | trends | categorical palette | Multiple KER properties need distinction; marker shapes add accessibility |
| 33 | plot_kes_by_kec_count | trends | categorical palette | 7 ordered groups (0, 1, 2, 3, 4, 5, 6+) require distinction in stacked area |
| 34 | plot_main_graph | trends | categorical palette | 4 entity types (AOPs, KEs, KERs, Stressors) require distinction |
| 35 | plot_network_density | trends | single #307BBF | Single metric (network density); no categorical distinction needed |
| 36 | plot_oecd_completeness_trend | trends | oecd_status color mapping | OECD status is semantic; oecd_status mapping is canonical |
| 37 | plot_ontology_term_growth | trends | single #307BBF | Single metric (unique term count); absolute and delta outputs |
| 38 | plot_stressor_property_presence | trends | categorical palette | Multiple Stressor properties need distinction; marker shapes add accessibility |
| 39 | plot_unique_ke_components | trends | categorical palette | 3 component types (Process, Object, Action) require distinction |

---

## Colorblind Accessibility (Deuteranopia)

### Background

Deuteranopia (green-blind, most common CVD type) affects ~6% of males. This section documents which VHP4Safety palette colors become confusable under deuteranopia simulation, and which specific plot functions are affected.

**Simulation methodology:** Vienot, Brettel & Mollon (1999) deuteranopia matrix applied to linearized sRGB values. Color difference metric: CIEDE2000 (Sharma, Wu, Dalal 2005). Confusability threshold: delta E < 10. Source: COLORBLIND-FINDINGS.md.

### Palette Simulation Results

| Color Name | Original Hex | Simulated Hex | Confusable With |
|------------|-------------|---------------|-----------------|
| primary | #29235C | #25255c | — |
| magenta | #E6007E | #848479 (grey-brown) | — |
| blue | #307BBF | #6c6cc0 | — (near-threshold with light_blue at 10.64) |
| light_blue | #009FE3 | #8888e4 | — (near-threshold with blue at 10.64) |
| orange | #EB5B25 | #999908 (olive-yellow) | — |
| sky_blue | #93D5F6 | #c5c5f7 | — |
| deep_magenta | #9A1C57 | #5a5a54 | — |
| teal | #45A6B2 | #9292b3 | — |
| warm_pink | #B81178 | #6a6a75 | — |
| dark_teal | #005A6C | #4c4c6d | **violet** (delta E = 8.51) |
| violet | #64358C | #47478b | **dark_teal** (delta E = 8.51) |

### Confusable Pairs (delta E < 10)

**Pair 1: dark_teal / violet (palette group, delta E = 8.51)**

Under deuteranopia, dark_teal (#005A6C) simulates as #4c4c6d and violet (#64358C) simulates as #47478b. Both collapse to similar dark blue-grey tones, indistinguishable at small marker/line sizes.

**Pair 2: WNT Endorsed / No Status (oecd_status group, delta E = 9.47)**

Under deuteranopia, WNT Endorsed (#E6007E, magenta) simulates as #848479 (grey-brown) and No Status (#999999, grey) is unchanged. Both appear as similar neutral grey tones.

**Near-threshold pairs (10 <= delta E < 15):**
- blue / light_blue: delta E = 10.64 (palette and oecd_status: EAGMST Under Review / Under Development)
- primary / violet: delta E = 11.63
- deep_magenta / warm_pink: delta E = 12.02
- light_blue / teal: delta E = 12.22 (oecd_status: Under Development / EAGMST Under Development)

### Affected Plots

**Plots affected by dark_teal / violet confusable pair (palette positions 9 and 10):**

These plots use `BRAND_COLORS['palette']` with enough categories that both dark_teal (position 9) and violet (position 10) may be assigned simultaneously:

| Plot Function | Module | Risk Level | Notes |
|---------------|--------|------------|-------|
| plot_bio_processes | trends | Medium | Up to 8 ontologies (GO, MP, NBO, MI, VT, MESH, HP, OTHER). If all 8 present, positions 9+ reached. Risk depends on ontology count in data. |
| plot_bio_objects | trends | Medium-High | Up to 11+ ontologies. Positions 9-10 more likely to be reached. |
| plot_latest_process_usage | latest | Medium | 7 ontology types. Positions 9-10 unlikely but possible with OTHER category. |
| plot_latest_object_usage | latest | Medium-High | 8+ ontology types. Positions 9-10 reachable. |
| plot_latest_ontology_usage | latest | Medium | 7+ categories. |
| plot_kes_by_kec_count | trends | Low | 7 groups (0-5, 6+). Palette positions used: 1-7. dark_teal (9) and violet (10) not reached. |
| plot_main_graph | trends | None | Only 4 categories (AOPs, KEs, KERs, Stressors). Palette positions 1-4 only. |
| plot_entity_completeness_trends | trends | None | Only 4 entity types. Palette positions 1-4. |
| plot_aop_property_presence (and ke, ker, stressor variants) | trends | Low-Medium | Number of properties varies. Marker shapes provide secondary encoding — this mitigates the colorblind concern significantly. |

**Plots affected by WNT Endorsed / No Status confusable pair (oecd_status group):**

| Plot Function | Module | Risk Level | Notes |
|---------------|--------|------------|-------|
| plot_latest_entity_by_oecd_status | latest | Medium | Uses oecd_status color map. Both WNT Endorsed and No Status may appear. If both present, colorblind users may confuse them. |
| plot_oecd_completeness_trend | trends | Medium | Uses oecd_status color map + marker shapes. Both statuses may appear. Marker shapes mitigate confusion. |

**Document-only findings** (per D-11): No fix recommendations are made here. Phase 8 will determine remediation based on these findings.

---

## Per-Plot Audit: latest_plots.py

### 1. plot_latest_entity_counts

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Uses `color="Entity"` with `color_discrete_sequence=BRAND_COLORS['palette']`. Five entity types mapped to 5 distinct palette colors. `showlegend=False` — colors serve only as visual variety since x-axis labels differentiate. Single #307BBF appropriate. |
| Chart type | Pass | `px.bar` horizontal x-axis per entity type — correct for categorical count comparison. |
| Data table | Pass | Writes to `_plot_data_cache['latest_entity_counts']` with Entity, Count, Version. |
| Title quality | Fail | No `title=` set in `update_layout`. |
| Axis labels | Pass | `yaxis=dict(title="Count")` and `xaxis=dict(title="Entity Type")` set. |
| Legend clarity | Pass | `showlegend=False` correct since entity names are on x-axis. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | Pass | `text="Count"` provides values. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** wired (latest.html, index.html) | **Colorblind note:** N/A after fix.

---

### 2. plot_latest_database_summary

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]` with `showlegend=False`. Legend hidden; x-axis differentiates. Single #307BBF appropriate. |
| Chart type | Pass | `px.bar` for entity count comparison. |
| Data table | Fail | No `_plot_data_cache` write. CSV export will fail. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Count")` set. |
| Legend clarity | Pass | `showlegend=False` correct. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | Pass | `text="Count"` with `textposition='outside'`. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** unwired (registered in app.py but absent from templates) | **Colorblind note:** N/A after fix.

---

### 3. plot_latest_avg_per_aop

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]` (deep purple, magenta) for 2 bars. `showlegend=False`. Same concept (components per AOP) — single #307BBF appropriate. |
| Chart type | Pass | `px.bar` comparing two scalar averages. |
| Data table | Pass | Writes to `_plot_data_cache['latest_avg_per_aop']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Fail | No `xaxis_title` or `yaxis_title` set. |
| Legend clarity | Pass | `showlegend=False` correct. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | Pass | `texttemplate='%{text:.1f}'` with `textposition='outside'`. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** wired (latest.html, index.html) | **Colorblind note:** N/A after fix.

---

### 4. plot_latest_ke_components

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]` for `px.pie`. After chart type fix (bar), categorical palette appropriate for Process/Object/Action. |
| Chart type | Fail | `px.pie` for 3 nearly-equal slices. Horizontal bar chart enables direct comparison. |
| Data table | Pass | Writes to `_plot_data_cache['latest_ke_components']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Fail | `px.pie` has no axis labels. Needed after fix. |
| Legend clarity | Pass | `textposition='inside', textinfo='percent+label'`. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | Pass | Default pie hover. |

**Classification:** FIX NOW | **Color decision:** categorical palette | **Wiring:** wired (latest.html, index.html) | **Colorblind note:** Process=#29235C, Object=#E6007E, Action=#307BBF — no confusable pairs.

---

### 5. plot_latest_ke_annotation_depth

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `color_discrete_sequence=BRAND_COLORS['palette']` for `px.pie` of ordered depth bins. After chart type fix (bar), single color appropriate for ordinal distribution. |
| Chart type | Fail | `px.pie` for ordinal distribution (0, 1, 2, 3... components). Bar chart for ordered bins is correct. |
| Data table | Pass | Writes to `_plot_data_cache['latest_ke_annotation_depth']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Fail | `px.pie` has no axis labels. |
| Legend clarity | Fail | Inside labels inadequate for ordinal progression. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | Pass | Default pie hover. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** wired (latest.html, index.html) | **Colorblind note:** N/A after fix.

---

### 6. plot_latest_network_density

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]` for `px.pie` binary split (Connected/Isolated). Unnecessary color for 2-slice pie. After chart type fix (bar), categorical palette for 2 distinct categories. |
| Chart type | Fail | `px.pie` for binary (Connected/Isolated) split. Stacked horizontal bar or two-bar chart more precise. |
| Data table | Pass | Writes to `_plot_data_cache['latest_aop_connectivity']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Fail | `px.pie` has no axis labels. |
| Legend clarity | Pass | `textposition='inside', textinfo='percent+label'` readable for 2 slices. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)` with annotation. |
| Tooltip quality | Pass | Default pie hover. |

**Classification:** FIX NOW | **Color decision:** categorical palette | **Wiring:** wired (latest.html as "latest_aop_connectivity", index.html) | **Colorblind note:** palette[0]/palette[1] delta E = 42.95. Safe.

---

### 7. plot_latest_ontology_usage

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | No explicit `color_discrete_sequence` — implicit template colorway. After chart type fix (bar), explicit `BRAND_COLORS['palette']` needed. |
| Chart type | Fail | `px.pie` for 7+ ontology categories. Horizontal bar sorted by count more readable. |
| Data table | Fail | No `_plot_data_cache` write. CSV export will fail. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Fail | `px.pie` has no axis labels. |
| Legend clarity | Pass | `textposition='inside', textinfo='percent+label'` acceptable for pie but slices may be too small. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | Pass | Default pie hover. |

**Classification:** FIX NOW | **Color decision:** categorical palette | **Wiring:** unwired (registered in app.py but absent from templates) | **Colorblind note:** dark_teal/violet confusable pair may appear if 10+ ontologies present.

---

### 8. plot_latest_process_usage

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=BRAND_COLORS['palette']` for categorical ontology sources. Correct mechanism. |
| Chart type | Fail | `px.pie` for multiple ontology source categories. Bar chart preferable for 5+ categories. |
| Data table | Pass | Writes to `_plot_data_cache['latest_process_usage']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Fail | `px.pie` has no axis labels. |
| Legend clarity | Pass | `textposition='inside', textinfo='percent+label'`. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | Pass | Default pie hover. |

**Classification:** FIX LATER | **Color decision:** categorical palette | **Wiring:** wired (latest.html, index.html) | **Colorblind note:** dark_teal/violet confusable pair if 9+ ontology categories present.

---

### 9. plot_latest_object_usage

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=BRAND_COLORS['palette']` for categorical ontology sources. Correct mechanism. |
| Chart type | Fail | `px.pie` for 8+ ontology categories. Bar chart preferable. |
| Data table | Pass | Writes to `_plot_data_cache['latest_object_usage']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Fail | `px.pie` has no axis labels. |
| Legend clarity | Pass | `textposition='inside', textinfo='percent+label'`. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | Pass | Default pie hover. |

**Classification:** FIX LATER | **Color decision:** categorical palette | **Wiring:** wired (latest.html, index.html) | **Colorblind note:** dark_teal/violet confusable pair if 9+ categories present.

---

### 10. plot_latest_aop_completeness

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color="Type"` with `color_discrete_map=BRAND_COLORS['type_colors']`. Correct semantic mapping for property types. |
| Chart type | Pass | `px.bar` grouped by property type — correct for completeness comparison. |
| Data table | Pass | Writes to `_plot_data_cache['latest_aop_completeness']`. Rich dataset. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="AOP Properties")`. |
| Legend clarity | Pass | `legend=dict(title="Property Type")`. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=100)`, `yaxis range=[0, 105]`, `tickangle=45`. |
| Tooltip quality | Pass | `texttemplate='%{text:.1f}%'` with `textposition='outside'`. |

**Classification:** SKIP | **Color decision:** categorical palette (type_colors) | **Wiring:** wired (latest.html, index.html) | **Colorblind note:** type_colors group has no confusable pairs (min delta E = 15.34). Safe.

---

### 11. plot_latest_aop_completeness_unique_colors

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Per-property unique colors via `color_discrete_map` cycling. `showlegend=False` — colors add no information. Single #307BBF appropriate. |
| Chart type | Pass | `px.bar` for completeness percentages — correct. |
| Data table | Pass | Writes to `_plot_data_cache['latest_aop_completeness_unique']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="AOP Properties")`. |
| Legend clarity | Pass | `showlegend=False` correct since colors convey nothing. |
| Layout & spacing | Pass | `height=600`, `margin=dict(l=50, r=20, t=70, b=120)`. |
| Tooltip quality | Pass | `texttemplate='%{text:.1f}%'` with `textposition='outside'`. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** broken (template slot in index.html but no app.py handler) | **Colorblind note:** N/A after fix.

---

### 12. plot_latest_aop_completeness_by_status

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color="Property Type"` with `color_discrete_map=BRAND_COLORS['type_colors']`. Correct. |
| Chart type | Pass | `px.bar` with `barmode="group"` — correct for cross-status property type comparison. |
| Data table | Pass | Writes to `_plot_data_cache['latest_aop_completeness_by_status']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="OECD Status")`. |
| Legend clarity | Pass | `legend=dict(title="Property Type")`. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=100)`, range=[0, 105], tickangle=45. |
| Tooltip quality | Pass | `texttemplate='%{text:.1f}%'`. |

**Classification:** SKIP | **Color decision:** categorical palette (type_colors) | **Wiring:** wired (latest.html) | **Colorblind note:** type_colors has no confusable pairs. Safe.

---

### 13. plot_latest_ke_completeness_by_status

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color="Property Type"` with `color_discrete_map=BRAND_COLORS['type_colors']`. Correct. |
| Chart type | Pass | `px.bar` with `barmode="group"`. Correct. |
| Data table | Pass | Writes to `_plot_data_cache['latest_ke_completeness_by_status']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="OECD Status of Parent AOPs")`. |
| Legend clarity | Pass | `legend=dict(title="Property Type")`. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=100)`, range=[0, 105]. |
| Tooltip quality | Pass | `texttemplate='%{text:.1f}%'`. |

**Classification:** SKIP | **Color decision:** categorical palette (type_colors) | **Wiring:** wired (latest.html) | **Colorblind note:** type_colors has no confusable pairs. Safe.

---

### 14. plot_latest_ker_completeness_by_status

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color="Property Type"` with `color_discrete_map=BRAND_COLORS['type_colors']`. Correct. |
| Chart type | Pass | `px.bar` with `barmode="group"`. Correct. |
| Data table | Pass | Writes to `_plot_data_cache['latest_ker_completeness_by_status']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="OECD Status of Parent AOPs")`. |
| Legend clarity | Pass | `legend=dict(title="Property Type")`. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=50, b=100)`, range=[0, 105]. |
| Tooltip quality | Pass | `texttemplate='%{text:.1f}%'`. |

**Classification:** SKIP | **Color decision:** categorical palette (type_colors) | **Wiring:** wired (latest.html) | **Colorblind note:** type_colors has no confusable pairs. Safe.

---

### 15. plot_latest_ke_by_bio_level

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `color="KE Count"` with `color_continuous_scale=[BRAND_COLORS['light'], BRAND_COLORS['primary']]`. Count-to-color gradient duplicates x-axis. `coloraxis_showscale=False` suppresses legend. Single #307BBF appropriate. |
| Chart type | Pass | `px.bar` with `orientation='h'` — correct for categorical level distribution. |
| Data table | Pass | Writes to `_plot_data_cache[f'latest_ke_by_bio_level_{version_key}']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `xaxis=dict(title="Number of Key Events")`. Level names serve as y-axis labels. |
| Legend clarity | Pass | `showlegend=False`, `coloraxis_showscale=False`. Correct. |
| Layout & spacing | Pass | `margin=dict(l=150, r=30, t=80, b=60)` — wide left margin for long names. |
| Tooltip quality | Pass | `textposition='outside'` shows count. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** wired (latest.html) | **Colorblind note:** N/A after fix.

---

### 16. plot_latest_taxonomic_groups

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `color="Taxonomic Group"` with `color_discrete_sequence=BRAND_COLORS['palette']` cycling for up to 25 groups. `showlegend=False` — y-axis labels identify groups. Single #307BBF appropriate. |
| Chart type | Pass | `px.bar` with `orientation='h'` sorted by count — correct for ranking. |
| Data table | Pass | Writes to `_plot_data_cache[f'latest_taxonomic_groups_{version_key}']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `xaxis=dict(title="Number of AOPs")`. |
| Legend clarity | Pass | `showlegend=False` correct. |
| Layout & spacing | Pass | `margin=dict(l=180, r=30, t=60, b=60)` — wide left margin. |
| Tooltip quality | Pass | `textposition='outside'` shows count. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** wired (latest.html) | **Colorblind note:** Currently uses dark_teal/violet pair (confusable at 8.51). Fix eliminates concern.

---

### 17. plot_latest_entity_by_oecd_status

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color="OECD Status"` with `color_discrete_map=BRAND_COLORS['oecd_status']`. Canonical semantic mapping. |
| Chart type | Pass | `px.bar` with `barmode="group"` — correct for cross-dimensional comparison (entity type × OECD status). |
| Data table | Pass | Writes to `_plot_data_cache[f'latest_entity_by_oecd_status_{version_key}']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Count")` and `xaxis=dict(title="Entity Type")`. |
| Legend clarity | Pass | `legend=dict(title="OECD Status")`. |
| Layout & spacing | Pass | `margin=dict(l=60, r=30, t=60, b=60)`. |
| Tooltip quality | Pass | `textposition='outside'` shows count. |

**Classification:** SKIP | **Color decision:** categorical palette (oecd_status) | **Wiring:** wired (latest.html) | **Colorblind note:** WNT Endorsed/No Status confusable pair (delta E = 9.47). Both statuses may appear.

---

### 18. plot_latest_ke_reuse

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `color="AOP Count"` with `color_continuous_scale=[BRAND_COLORS['light'], BRAND_COLORS['primary']]`. Count on x-axis makes gradient redundant. `coloraxis_showscale=False`. Single #307BBF appropriate. |
| Chart type | Pass | `px.bar` with `orientation='h'` — correct for top-30 KE ranking. |
| Data table | Pass | Writes to `_plot_data_cache[f'latest_ke_reuse_{version_key}']` including wiki_url. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `xaxis=dict(title="Number of AOPs")`. |
| Legend clarity | Pass | `showlegend=False`, `coloraxis_showscale=False`. |
| Layout & spacing | Pass | Dynamic `height=max(400, len(data)*25+100)`. `margin=dict(l=300, r=30, t=60, b=60)`. |
| Tooltip quality | Pass | `custom_data=['wiki_url']` for click-to-open. `textposition='outside'`. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** wired (latest.html with id="ke-reuse-plot") | **Colorblind note:** N/A after fix.

---

### 19. plot_latest_ke_reuse_distribution

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `color="Number of KEs"` with `color_continuous_scale=[BRAND_COLORS['light'], BRAND_COLORS['primary']]`. Count on y-axis makes gradient redundant. `coloraxis_showscale=False`. Single #307BBF appropriate. |
| Chart type | Pass | `px.bar` with discrete bins — appropriate for distribution histogram. |
| Data table | Pass | Writes to `_plot_data_cache[f'latest_ke_reuse_distribution_{version_key}']`. |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Number of Key Events")` and `xaxis=dict(title="Number of AOPs a KE Belongs To")`. |
| Legend clarity | Pass | `showlegend=False`, `coloraxis_showscale=False`. |
| Layout & spacing | Pass | `margin=dict(l=60, r=30, t=60, b=60)`. |
| Tooltip quality | Pass | `textposition='outside'`. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** wired (latest.html) | **Colorblind note:** N/A after fix.

---

### 20. plot_latest_ontology_diversity

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `color="Ontology"` with `color_discrete_map` cycling through `BRAND_COLORS['palette']`. `showlegend=False` — ontology names on x-axis. Single #307BBF appropriate. |
| Chart type | Pass | `px.bar` with ontologies on x-axis — correct for count comparison. |
| Data table | Pass | Writes to `_plot_data_cache[cache_key]` (`latest_ontology_diversity_{version_key}`). |
| Title quality | Fail | No `title=` set. |
| Axis labels | Pass | `yaxis=dict(title="Number of Unique Terms")` and `xaxis=dict(title="Ontology Source")`. |
| Legend clarity | Pass | `showlegend=False` correct. |
| Layout & spacing | Pass | `margin=dict(l=50, r=20, t=60, b=50)`. |
| Tooltip quality | Pass | `textposition='outside'`. |

**Classification:** FIX NOW | **Color decision:** single #307BBF | **Wiring:** wired (latest.html) | **Colorblind note:** Currently uses dark_teal/violet cycling pair. Fix eliminates concern.

---

## Per-Plot Audit: trends_plots.py

### 1. plot_main_graph

**Returns:** `tuple[str, str, pd.DataFrame]` — (absolute_html, delta_html, dataframe)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=BRAND_COLORS['palette']` with `color="Entity"` (4 types: AOPs, KEs, KERs, Stressors). Both outputs use same palette. Correct. |
| Chart type | Pass | Absolute: px.line with markers — correct for entity count time series. Delta: px.line — correct for directional change. |
| Data table | Pass | Caches `main_graph_absolute` (version, Entity, Count) and `main_graph_delta`. |
| Title quality | Fail | No figure title. Template h3 "AOP Entity Counts" provides context. Light dimension. |
| Axis labels | Pass | x="version" with explicit tickvals/ticktext/-45 angle. y="Count". |
| Legend clarity | Pass | color="Entity" — legend labels AOPs/KEs/KERs/Stressors. Clear. |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=50). |
| Tooltip quality | Pass | Template hovermode="x unified". |

**Classification:** SKIP | **Color decision:** categorical palette — 4 entity types require categorical distinction. | **Wiring:** wired (`aop_entity_counts_absolute`, `aop_entity_counts_delta`) | **Colorblind note:** blue/light_blue near-threshold (delta E 10.64) among the 4 colors used.

---

### 2. plot_avg_per_aop

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]` = `['#29235C', '#E6007E']`. Hardcoded 2-color list via legacy aliases instead of `BRAND_COLORS['palette']`. Secondary alias = magenta (CTA color) used for data series. |
| Chart type | Pass | px.line with markers — correct for ratio time series. |
| Data table | Pass | Caches `average_components_per_aop_absolute` and `average_components_per_aop_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Average" / "Δ Average". Metrics labeled. Tick rotation. |
| Legend clarity | Pass | Renamed metric labels: "Average KEs per AOP" / "Average KERs per AOP". Clear. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX NOW | **Color decision:** categorical palette — 2 metrics require distinction. Use `BRAND_COLORS['palette']` (first 2 colors). | **Wiring:** wired (`average_components_per_aop_absolute`, `average_components_per_aop_delta`) | **Colorblind note:** palette[0]/palette[1] delta E 42.95. Safe.

---

### 3. plot_network_density

**Returns:** `str` — single HTML string

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `[BRAND_COLORS['accent']]` = #307BBF via legacy alias. Single-metric plot. Correct. |
| Chart type | Pass | px.line with markers — correct for scalar density over time. |
| Data table | Pass | Caches `aop_network_density` (version, nodes, edges, density). |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y labeled "Graph Density" via labels param. Tick rotation. |
| Legend clarity | Pass | Single series, no legend needed. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** single #307BBF | **Wiring:** wired (`aop_network_density`) | **Colorblind note:** N/A — single color.

---

### 4. plot_author_counts

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Absolute: `[BRAND_COLORS['secondary']]` = magenta (#E6007E) — CTA color inappropriate for data. Delta: `[BRAND_COLORS['light']]` = sky_blue (#93D5F6) — too faint. Different colors for abs/delta of same metric. Both should be #307BBF. |
| Chart type | Pass | px.line with markers — correct for author count time series. |
| Data table | Pass | Caches `aop_authors_absolute` and `aop_authors_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Fail | y="author_count" (raw column name). Should be "Number of Unique Authors". Light dimension. |
| Legend clarity | Pass | Single series, no legend. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX NOW | **Color decision:** single #307BBF for both outputs | **Wiring:** wired (`aop_authors_absolute`, `aop_authors_delta`) | **Colorblind note:** N/A after fix.

---

### 5. plot_aop_lifetime

**Returns:** `tuple[str, str, str]` — (created_html, modified_html, scatter_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Created: primary (#29235C). Modified: secondary = magenta (#E6007E) — CTA color. Scatter: accent = #307BBF. Three different colors for 3 related views of the same entity. All should be #307BBF. |
| Chart type | Pass | Created/Modified: px.histogram by year — correct. Scatter: px.scatter (created vs modified dates) — correct. |
| Data table | Pass | Caches `aops_created_over_time`, `aops_modified_over_time`, `aop_creation_vs_modification_timeline`. |
| Title quality | Fail | No figure titles. Light dimension. |
| Axis labels | Pass | Year/created/modified labels set. |
| Legend clarity | Pass | Single series each; hover_name="aop" on scatter. |
| Layout & spacing | Fail | Created/Modified use `height=400`, scatter `height=500` but no margin dict (missing `l=50, r=20, t=50, b=50`). Inconsistent. Light dimension. |
| Tooltip quality | Pass | hover_name="aop" for scatter provides per-AOP identification. |

**Classification:** FIX NOW | **Color decision:** single #307BBF for all 3 outputs | **Wiring:** wired (`aops_created_over_time`, `aops_modified_over_time`, `aop_creation_vs_modification_timeline`) | **Colorblind note:** N/A after fix.

---

### 6. plot_ke_components

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | `[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]` — hardcoded 3-color alias list instead of `BRAND_COLORS['palette']`. Same anti-pattern as #7 and #8. |
| Chart type | Pass | px.line with markers — correct for annotation count time series. |
| Data table | Pass | Caches `ke_component_annotations_absolute` and `ke_component_annotations_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Count" / "Change". color="Component" (Process/Object/Action). |
| Legend clarity | Pass | "Process", "Object", "Action" labels. Clear. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX NOW | **Color decision:** categorical palette — 3 component types. Use `BRAND_COLORS['palette']`. | **Wiring:** wired (`ke_component_annotations_absolute`, `ke_component_annotations_delta`) | **Colorblind note:** palette[0/1/2] — no confusable pairs.

---

### 7. plot_ke_components_percentage

**Returns:** `tuple[str, str]` — (percentage_html, delta_percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Same hardcoded 3-alias list: `[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]`. |
| Chart type | Pass | px.line — correct for percentage trends. |
| Data table | Pass | Caches `ke_components_percentage_absolute` and `ke_components_percentage_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Percentage (%)" / "Percentage Change (%)" — units included. |
| Legend clarity | Pass | "Process", "Object", "Action". |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX NOW | **Color decision:** categorical palette — 3 component types. Use `BRAND_COLORS['palette']`. | **Wiring:** wired (`ke_components_percentage_absolute`, `ke_components_percentage_delta`) | **Colorblind note:** Same as #6.

---

### 8. plot_unique_ke_components

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Same hardcoded 3-alias list. Identical to #6 and #7. |
| Chart type | Pass | px.line — correct for unique component count time series. |
| Data table | Pass | Caches `unique_ke_components_absolute` and `unique_ke_components_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Unique Count" / "Change". |
| Legend clarity | Pass | "Process", "Object", "Action". |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX NOW | **Color decision:** categorical palette — 3 component types. Use `BRAND_COLORS['palette']`. | **Wiring:** wired (`unique_ke_components_absolute`, `unique_ke_components_delta`) | **Colorblind note:** Same as #6.

---

### 9. plot_bio_processes

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=BRAND_COLORS['palette']` with `color="ontology"`. Correct for multiple ontology types. |
| Chart type | Pass | px.bar with barmode="group" — acceptable for ontology distribution over time (borderline with stacked bar, but not a clear failure). |
| Data table | Pass | Caches `biological_process_annotations_absolute` and `biological_process_annotations_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | count → "Annotated KEs", ontology → "Ontology". Tick rotation. |
| Legend clarity | Pass | Ontology name labels (GO, MP, etc.) — meaningful. |
| Layout & spacing | Pass | Template defaults. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** categorical palette — multiple ontology types require distinct colors. | **Wiring:** wired (`biological_process_annotations_absolute`, `biological_process_annotations_delta`) | **Colorblind note:** dark_teal/violet confusable pair if 9+ ontologies present in data.

---

### 10. plot_bio_objects

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Same as #9: `color_discrete_sequence=BRAND_COLORS['palette']` with `color="ontology"`. Correct. |
| Chart type | Pass | px.bar with barmode="group" — same borderline reasoning as #9. |
| Data table | Pass | Caches `biological_object_annotations_absolute` and `biological_object_annotations_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | Same labeling as #9. |
| Legend clarity | Pass | Ontology names. |
| Layout & spacing | Pass | Template defaults. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** categorical palette — multiple ontology types (11+) require distinction. | **Wiring:** wired (`biological_object_annotations_absolute`, `biological_object_annotations_delta`) | **Colorblind note:** dark_teal/violet confusable pair — higher risk than #9 due to more ontologies.

---

### 11. plot_aop_property_presence

**Returns:** `tuple[str, str]` — (absolute_html, percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=BRAND_COLORS['palette']` + marker shapes per trace. Correct categorical palette with accessibility enhancement. |
| Chart type | Pass | px.line with markers — correct for property presence % over time. |
| Data table | Pass | Caches `aop_property_presence_absolute` and `aop_property_presence_percentage`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | "Number of AOPs" / "Percentage (%)". color label "Property". |
| Legend clarity | Pass | Display labels from property_labels.csv (human-readable). |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=50). |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** categorical palette — multiple AOP properties. Marker shapes add secondary encoding. | **Wiring:** wired (`aop_property_presence_absolute`, `aop_property_presence_percentage`) | **Colorblind note:** dark_teal/violet risk if 10+ properties. Marker shapes mitigate.

---

### 12. plot_ke_property_presence

**Returns:** `tuple[str, str]` — (absolute_html, percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Same pattern as #11. |
| Chart type | Pass | px.line — correct. |
| Data table | Pass | Caches `ke_property_presence_absolute` and `ke_property_presence_percentage`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | "Number of KEs" / "Percentage (%)". |
| Legend clarity | Pass | Display labels from CSV. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** categorical palette — multiple KE properties. Marker shapes. | **Wiring:** wired (`ke_property_presence_absolute`, `ke_property_presence_percentage`) | **Colorblind note:** Marker shapes mitigate dark_teal/violet risk.

---

### 13. plot_ker_property_presence

**Returns:** `tuple[str, str]` — (absolute_html, percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Same pattern as #11/#12. |
| Chart type | Pass | px.line — correct. |
| Data table | Pass | Caches `ker_property_presence_absolute` and `ker_property_presence_percentage`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | "Number of KERs" / "Percentage (%)". |
| Legend clarity | Pass | Display labels from CSV. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** categorical palette — multiple KER properties. Marker shapes. | **Wiring:** wired (`ker_property_presence_absolute`, `ker_property_presence_percentage`) | **Colorblind note:** Marker shapes mitigate.

---

### 14. plot_stressor_property_presence

**Returns:** `tuple[str, str]` — (absolute_html, percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Same pattern as #11/#12/#13. |
| Chart type | Pass | px.line — correct. |
| Data table | Pass | Caches `stressor_property_presence_absolute` and `stressor_property_presence_percentage`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | "Number of Stressors" / "Percentage (%)". |
| Legend clarity | Pass | Display labels from CSV. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** categorical palette — multiple Stressor properties. Marker shapes. | **Wiring:** wired (`stressor_property_presence_absolute`, `stressor_property_presence_percentage`) | **Colorblind note:** Marker shapes mitigate.

---

### 15. plot_kes_by_kec_count

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=BRAND_COLORS['palette']` with `color="bioevent_count_group"` (7 ordered groups). Categorical palette correct for named groups. |
| Chart type | Fail | Both use `px.area`. Delta output with stacked area can produce overlapping negative areas — bar chart cleaner for delta. Borderline classification → FIX LATER. |
| Data table | Pass | Caches `kes_by_kec_count_absolute` and `kes_by_kec_count_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Number of KEs" / "Change in KEs". color="Number of Components". |
| Legend clarity | Fail | Group labels "0", "1", ..., "6+" lack unit context. Should be "0 components" etc. Light dimension. |
| Layout & spacing | Pass | margin consistent. xaxis tick arrays set. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX LATER | **Color decision:** categorical palette — 7 ordered groups. | **Wiring:** wired (`kes_by_kec_count_absolute`, `kes_by_kec_count_delta`) | **Colorblind note:** 7 palette colors — blue/light_blue near-threshold (10.64). Not formally confusable.

---

### 16. plot_entity_completeness_trends

**Returns:** `str` — single HTML string

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=BRAND_COLORS['palette']` with `color="entity_type"` (4 types). Correct. Marker shapes applied. |
| Chart type | Pass | px.line with markers — correct for completeness % over time. Line width 3. |
| Data table | Pass | Caches `entity_completeness_trends`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Average Completeness (%)". yaxis range [0, 105]. |
| Legend clarity | Pass | Human-readable entity type names. |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=50). yaxis range prevents crowding. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** categorical palette — 4 entity types. Marker shapes. | **Wiring:** wired (`entity_completeness_trends`) | **Colorblind note:** blue/light_blue near-threshold. Marker shapes mitigate.

---

### 17. plot_aop_completeness_boxplot

**Returns:** `str` — single HTML string

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `marker=dict(color=BRAND_COLORS["primary"])` = #29235C. Single distribution. Acceptable. |
| Chart type | Pass | px.box — correct for distribution of completeness scores per version. |
| Data table | Pass | Caches `aop_completeness_boxplot`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Composite Completeness (%)". yaxis range [0, 105]. x="Version". |
| Legend clarity | Pass | showlegend=False — correct for single-series boxplot. |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=100). Extra bottom for labels. |
| Tooltip quality | Pass | Boxplot tooltips (Q1, median, Q3, min, max) by default. |

**Classification:** SKIP | **Color decision:** single #29235C (primary) — acceptable for single distribution. | **Wiring:** wired (`aop_completeness_boxplot`) | **Colorblind note:** N/A — single color.

---

### 18. plot_oecd_completeness_trend

**Returns:** `str` — single HTML string

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_map=status_color_map` built from `BRAND_COLORS['oecd_status']`. Canonical OECD mapping. Fallback to palette cycling. |
| Chart type | Pass | px.line with markers + marker shapes. Correct for mean completeness trend by OECD status. |
| Data table | Pass | Caches `oecd_completeness_trend`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Mean Completeness (%)". yaxis range [0, 105]. x="Version". |
| Legend clarity | Pass | legend title="OECD Status". |
| Layout & spacing | Pass | margin(l=50, r=150, t=50, b=50). Wide right for legend. Line width 2.5. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP | **Color decision:** oecd_status color mapping | **Wiring:** wired (`oecd_completeness_trend`) | **Colorblind note:** WNT Endorsed/No Status confusable pair (delta E 9.47). Marker shapes mitigate.

---

### 19. plot_ontology_term_growth

**Returns:** `tuple[str, str, pd.DataFrame]` — (absolute_html, delta_html, dataframe)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Absolute: `update_traces(line_color=BRAND_COLORS['primary'])` = #29235C (deep purple). Delta: `update_traces(marker_color=BRAND_COLORS['secondary'])` = #E6007E (magenta — CTA). Both should be #307BBF. Inconsistent colors across related outputs. |
| Chart type | Pass | Absolute: px.line — correct for cumulative count. Delta: px.bar — excellent choice for per-period additions. |
| Data table | Pass | Caches `ontology_term_growth_absolute` and `ontology_term_growth_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | Absolute: "Unique Ontology Terms". Delta: "New Terms Added". x="version" with tickangle=-45. |
| Legend clarity | Pass | Single series each, no legend (appropriate). |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=80). Extra bottom for labels. |
| Tooltip quality | Pass | Template hover and bar hover. |

**Classification:** FIX LATER | **Color decision:** single #307BBF for both outputs | **Wiring:** wired (`ontology_term_growth_absolute`, `ontology_term_growth_delta`) | **Colorblind note:** N/A after fix.

---

## Methodology

- **Rubric:** 8 dimensions per D-03: color correctness, chart type appropriateness, data table usefulness, title quality, axis labels & units, legend clarity, layout & spacing, tooltip quality
- **Classification:** Per D-04: FIX NOW (color or chart type fails), FIX LATER (only light dims fail), SKIP (all pass)
- **Color decisions:** Per D-05: "single #307BBF" or "categorical palette" with rationale for each function
- **Colorblind:** Deuteranopia only per D-09. Vienot 1999 simulation + CIEDE2000 delta E < 10 threshold per D-10. Document-only, no fix recommendations per D-11.
- **Scope:** Code analysis only, no running instance required per D-06.
- **Wiring check:** Cross-referenced against templates/latest.html, templates/trends.html, and app.py plot registrations.

---

*Consolidated from: AUDIT-LATEST.md (20 functions), AUDIT-TRENDS.md (19 functions), COLORBLIND-FINDINGS.md*
*Audit conducted: 2026-03-27*
