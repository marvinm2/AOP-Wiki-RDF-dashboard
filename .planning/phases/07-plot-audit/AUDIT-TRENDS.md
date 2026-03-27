# Plot Audit: trends_plots.py

**Audited:** 2026-03-27
**Functions:** 19
**Method:** Code analysis only (per D-06)
**Note:** Trend functions return tuples. Both outputs evaluated; notes distinguish "absolute" vs "delta/percentage" where relevant. Three functions return a single `str` (not a tuple): `plot_network_density`, `plot_entity_completeness_trends`, `plot_aop_completeness_boxplot`, `plot_oecd_completeness_trend`. One function returns `tuple[str, str, str]` (three outputs): `plot_aop_lifetime`. One function in the module (`plot_aop_completeness_boxplot_by_status`) is excluded — it is not in the 19-function audit scope (it uses Plotly's qualitative palette, not VHP4Safety colors, and is removed/unwired).

---

## Summary

| # | Function | Outputs | Classification | Color Decision | Wired |
|---|----------|---------|----------------|----------------|-------|
| 1 | plot_main_graph | abs + delta (+ df) | SKIP | categorical palette | yes |
| 2 | plot_avg_per_aop | abs + delta | FIX NOW | categorical palette (fix: use brand colors only) | yes |
| 3 | plot_network_density | single | SKIP | single #307BBF | yes |
| 4 | plot_author_counts | abs + delta | FIX NOW | single #307BBF (abs), single #93D5F6 (delta) -- fix delta to #307BBF | yes |
| 5 | plot_aop_lifetime | created + modified + scatter | FIX NOW | single #307BBF for all three | yes |
| 6 | plot_ke_components | abs + delta | FIX NOW | categorical palette (3 colors: primary, secondary, accent) | yes |
| 7 | plot_ke_components_percentage | abs + delta | FIX NOW | categorical palette (3 colors: primary, secondary, accent) | yes |
| 8 | plot_unique_ke_components | abs + delta | FIX NOW | categorical palette (3 colors: primary, secondary, accent) | yes |
| 9 | plot_bio_processes | abs + delta | SKIP | categorical palette | yes |
| 10 | plot_bio_objects | abs + delta | SKIP | categorical palette | yes |
| 11 | plot_aop_property_presence | abs + percentage | SKIP | categorical palette | yes |
| 12 | plot_ke_property_presence | abs + percentage | SKIP | categorical palette | yes |
| 13 | plot_ker_property_presence | abs + percentage | SKIP | categorical palette | yes |
| 14 | plot_stressor_property_presence | abs + percentage | SKIP | categorical palette | yes |
| 15 | plot_kes_by_kec_count | abs + delta | FIX LATER | categorical palette (area chart) | yes |
| 16 | plot_entity_completeness_trends | single | SKIP | categorical palette | yes |
| 17 | plot_aop_completeness_boxplot | single | SKIP | single #29235C (primary) | yes |
| 18 | plot_oecd_completeness_trend | single | SKIP | oecd_status color mapping | yes |
| 19 | plot_ontology_term_growth | abs + delta (+ df) | FIX LATER | single #29235C (abs), single #E6007E (delta) | yes |

---

## Per-Plot Audit

### 1. plot_main_graph

**Returns:** `tuple[str, str, pd.DataFrame]` — (absolute_html, delta_html, dataframe)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Absolute: `color_discrete_sequence=BRAND_COLORS['palette']` with `color="Entity"` (4 categories: AOPs, KEs, KERs, Stressors). Delta: same palette, same mechanism. Correct use of categorical palette for 4 distinct entity types. |
| Chart type | Pass | Absolute: px.line with markers — correct for time-series entity count evolution. Delta: px.line — appropriate for showing directional change over time. |
| Data table | Pass | Caches `main_graph_absolute` (melted: version, Entity, Count) and `main_graph_delta` (melted: version, Entity, Count) — meaningful for CSV export. |
| Title quality | Fail | No explicit title set on figures. Template h3 heading "AOP Entity Counts" provides context. Fallback titles are "Entity Evolution Over Time" and "Entity Change Between Versions" — acceptable but not set on the figure itself. (Light dimension only.) |
| Axis labels | Pass | x="version", y="Count" with palette-colored "Entity" legend. Tick rotation -45 applied. |
| Legend clarity | Pass | color="Entity" — legend labels are "AOPs", "KEs", "KERs", "Stressors" — clear and meaningful. |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=50). Template handles font/bgcolor. |
| Tooltip quality | Pass | Template hovermode="x unified" — shows all entities at each version on hover. |

**Classification:** SKIP
**Color decision:** categorical palette — 4 distinct entity types (AOPs, KEs, KERs, Stressors) require categorical distinction.
**Rationale:** `color_discrete_sequence=BRAND_COLORS['palette']` with `color="Entity"` is the correct pattern. The only failure is missing figure title (light dimension), which does not trigger FIX NOW.
**Wiring:** Wired — `aop_entity_counts_absolute`, `aop_entity_counts_delta`
**Colorblind note:** 4 palette colors used: primary (#29235C), magenta (#E6007E), blue (#307BBF), light_blue (#009FE3). Under deuteranopia: blue vs light_blue have delta E 10.64 (near-threshold). Not flagged as confusable (above 10 threshold), but may warrant attention when lines overlap.

---

### 2. plot_avg_per_aop

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Absolute: `color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]` = `['#29235C', '#E6007E']`. Delta: same pair. The `secondary` alias maps to magenta (#E6007E). While both are brand colors, using the 'secondary' alias (instead of explicit named colors) is less readable. More critically, the decision from v1.1 STATE.md is "default bar color #307BBF unless categorical differentiation needed" — two-metric plots should use #307BBF as primary unless differentiation requires otherwise. The current primary (#29235C) is valid for the first series but the decision about which 2 colors to pair matters for consistency. The issue is minor but should be standardized. **However**, the real flag: `color_discrete_sequence` bypasses the palette array, using positional assignment from a 2-element list rather than the official palette order. This can produce inconsistent assignments vs other plots. |
| Chart type | Pass | px.line with markers — correct for time-series ratio (avg KEs/KERs per AOP). |
| Data table | Pass | Caches `average_components_per_aop_absolute` (version, Metric, Average) and `average_components_per_aop_delta` (version, Metric, Δ Average). |
| Title quality | Fail | No figure title. Template h3: "Average Components per AOP". Fallback: "Average Components per AOP Over Time". Light dimension. |
| Axis labels | Pass | y="Average" (abs), y="Δ Average" (delta). Metrics are "Average KEs per AOP" / "Average KERs per AOP". Clear. |
| Legend clarity | Pass | Legend labels "Average KEs per AOP" / "Average KERs per AOP" after rename — meaningful. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Unified hover via template. |

**Classification:** FIX NOW
**Color decision:** categorical palette — 2 metrics (avg KEs/AOP, avg KERs/AOP) require categorical distinction. Use first 2 colors from `BRAND_COLORS['palette']` (primary, magenta) for consistency with palette ordering.
**Rationale:** Color correctness fails — the function uses a hardcoded 2-element list `[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]` where `secondary` is a legacy alias. Phase 8 should standardize to `color_discrete_sequence=BRAND_COLORS['palette'][:2]` for consistency with other multi-series plots.
**Wiring:** Wired — `average_components_per_aop_absolute`, `average_components_per_aop_delta`
**Colorblind note:** primary (#29235C) vs magenta (#E6007E) — delta E 42.95 under deuteranopia. No confusable pair.

---

### 3. plot_network_density

**Returns:** `str` — single HTML string (no tuple)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=[BRAND_COLORS['accent']]` where 'accent' is the legacy alias for #307BBF (blue). Single-metric plot. Correct. |
| Chart type | Pass | px.line with markers — correct for single scalar metric (density value) evolving over time. |
| Data table | Pass | Caches `aop_network_density` (version, nodes, edges, density) — meaningful for CSV. |
| Title quality | Fail | No figure title. Template has no explicit section for this plot in the heading. Fallback: "Network Density Evolution Over Time". y label is "Graph Density". Light dimension. |
| Axis labels | Pass | x="version" (explicit tickvals), y labeled "Graph Density" via the `labels` parameter. Tick angle -45. |
| Legend clarity | Pass | Single series, no legend needed. No legend rendered (single-color, single-series). |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=50). |
| Tooltip quality | Pass | Template hovermode="x unified". Shows density value per version. |

**Classification:** SKIP
**Color decision:** single #307BBF — single metric (network density over time), no categorical distinction needed.
**Rationale:** Color is correct (blue via 'accent' alias = #307BBF). All heavy dimensions pass. Light dimension failure (title) does not trigger FIX NOW.
**Wiring:** Wired — `aop_network_density`
**Colorblind note:** N/A — single color plot.

---

### 4. plot_author_counts

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Absolute: `color_discrete_sequence=[BRAND_COLORS['secondary']]` = magenta (#E6007E). Delta: `color_discrete_sequence=[BRAND_COLORS['light']]` = sky_blue (#93D5F6). Single-metric plots should use #307BBF (blue) per v1.1 decision. Using different colors for absolute vs delta views of the same metric is inconsistent. Magenta is the CTA color and should not be used for neutral data series. Sky blue (#93D5F6) is too light/faint for data lines. |
| Chart type | Pass | px.line with markers — correct for author count time series. |
| Data table | Pass | Caches `aop_authors_absolute` and `aop_authors_delta` (version, author_count, author_count_Δ). |
| Title quality | Fail | No figure title. Fallback: "Author Contributions Over Time" / "Change in Author Contributions". Light dimension. |
| Axis labels | Fail | Absolute: y="author_count" — this is the raw column name. Should be labeled "Number of Unique Authors" or "Author Count". Delta: y="author_count_Δ" — raw column name. Light dimension. |
| Legend clarity | Pass | Single series, no legend needed. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template hovermode="x unified". |

**Classification:** FIX NOW
**Color decision:** single #307BBF for both absolute and delta outputs — single metric (author count) does not need categorical differentiation.
**Rationale:** Color correctness fails — magenta (CTA color) used for absolute plot, sky_blue (too faint) for delta. Both should use #307BBF. The inconsistency between abs/delta colors compounds the issue.
**Wiring:** Wired — `aop_authors_absolute`, `aop_authors_delta`
**Colorblind note:** N/A after fix (single #307BBF for both outputs).

---

### 5. plot_aop_lifetime

**Returns:** `tuple[str, str, str]` — (created_html, modified_html, scatter_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Created (fig1): `color_discrete_sequence=[BRAND_COLORS['primary']]` = deep purple (#29235C). Modified (fig2): `color_discrete_sequence=[BRAND_COLORS['secondary']]` = magenta (#E6007E). Scatter (fig3): `color_discrete_sequence=[BRAND_COLORS['accent']]` = blue (#307BBF). Three different colors for three related views of the same entity (AOPs) is inconsistent — magenta (CTA) and deep purple are not the right defaults for single-metric charts. The v1.1 decision is #307BBF for single-metric plots. |
| Chart type | Pass | Created: px.histogram by year — correct for creation year distribution. Modified: px.histogram by year — correct. Scatter: px.scatter (created vs modified dates) — correct for temporal correlation analysis. |
| Data table | Pass | Caches `aops_created_over_time`, `aops_modified_over_time`, `aop_creation_vs_modification_timeline`. |
| Title quality | Fail | No figure titles on any of the three outputs. Fallback: "AOP Creation Timeline", "AOP Modification Timeline", "AOP Creation vs. Modification Timeline". Light dimension. |
| Axis labels | Pass | Created: x="year_created" labeled "Year". Modified: x="year_modified" labeled "Year". Scatter: x="created", y="modified" labeled "Created", "Modified". |
| Legend clarity | Pass | Single series each, no legend needed. hover_name="aop" on scatter provides per-point identification. |
| Layout & spacing | Fail | Created/Modified heights are 400px (explicit). Scatter is 500px. No top margin override (t=50 missing — only `height=400` set, no margin dict). Inconsistent with other plots that use `margin=dict(l=50, r=20, t=50, b=50)`. Light dimension. |
| Tooltip quality | Pass | hover_name="aop" on scatter provides AOP URI on hover. Template unified mode on others. |

**Classification:** FIX NOW
**Color decision:** single #307BBF for all three outputs — three related views of the same entity type (AOPs).
**Rationale:** Color fails — three different colors used across the three outputs (primary/secondary/accent), inconsistent. Secondary (magenta) is the CTA color, inappropriate for data lines/bars. All three should use #307BBF.
**Wiring:** Wired — `aops_created_over_time`, `aops_modified_over_time`, `aop_creation_vs_modification_timeline`
**Colorblind note:** N/A after fix (all single #307BBF).

---

### 6. plot_ke_components

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Both absolute and delta use `color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]` = `['#29235C', '#E6007E', '#307BBF']`. This bypasses the palette array and hardcodes 3 specific brand colors using mix of semantic names (primary, secondary, accent). The `secondary` alias (#E6007E = magenta) is CTA color used for data. The palette array `BRAND_COLORS['palette']` should be used instead for correct positional ordering. Using `BRAND_COLORS['palette'][:3]` gives `['#29235C', '#E6007E', '#307BBF']` — same colors, but using the palette explicitly is the correct pattern per v1.1 decisions. The issue is consistency with other plots that use the full palette. |
| Chart type | Pass | px.line with markers — correct for annotation count time series (3 component types). |
| Data table | Pass | Caches `ke_component_annotations_absolute` and `ke_component_annotations_delta`. |
| Title quality | Fail | No figure title. Fallback: "KE Component Annotations Over Time". Light dimension. |
| Axis labels | Pass | y="Count" (absolute), y="Change" (delta). color="Component" with values "Process", "Object", "Action". |
| Legend clarity | Pass | Legend labels "Process", "Object", "Action" — concise and meaningful. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX NOW
**Color decision:** categorical palette — 3 component types (Process, Object, Action) require categorical distinction. Use `BRAND_COLORS['palette']` (first 3 colors).
**Rationale:** Color fails — uses hardcoded 3-color list via semantic aliases instead of `BRAND_COLORS['palette']`. While the colors happen to be the first 3 palette entries, the mechanism deviates from the established pattern. Phase 8 should standardize to `color_discrete_sequence=BRAND_COLORS['palette']`.
**Wiring:** Wired — `ke_component_annotations_absolute`, `ke_component_annotations_delta`
**Colorblind note:** 3 palette colors: primary (#29235C), magenta (#E6007E), blue (#307BBF). No confusable pairs (minimum delta E = 25.69). Safe.

---

### 7. plot_ke_components_percentage

**Returns:** `tuple[str, str]` — (percentage_html, delta_percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Same issue as plot_ke_components: `color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]`. Same 3-color hardcoded list via aliases instead of `BRAND_COLORS['palette']`. |
| Chart type | Pass | Absolute: px.line showing percentage of KEs with each component type — correct. Delta: px.line showing percentage change — correct. |
| Data table | Pass | Caches `ke_components_percentage_absolute` and `ke_components_percentage_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Percentage" with yaxis title "Percentage (%)" (absolute); y="Percentage Change" with "Percentage Change (%)" (delta). Correct with units. |
| Legend clarity | Pass | Same "Process", "Object", "Action" labels. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX NOW
**Color decision:** categorical palette — same as plot_ke_components (3 component types). Use `BRAND_COLORS['palette']`.
**Rationale:** Same color issue as #6. Standardize to `color_discrete_sequence=BRAND_COLORS['palette']`.
**Wiring:** Wired — `ke_components_percentage_absolute`, `ke_components_percentage_delta`
**Colorblind note:** Same as #6. No confusable pairs.

---

### 8. plot_unique_ke_components

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Same issue: `color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]`. Identical pattern to #6 and #7. |
| Chart type | Pass | px.line with markers — correct for unique component count time series. |
| Data table | Pass | Caches `unique_ke_components_absolute` and `unique_ke_components_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | y="Unique Count" (absolute), y="Change" (delta). color="Component". |
| Legend clarity | Pass | "Process", "Object", "Action" labels. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX NOW
**Color decision:** categorical palette — 3 component types. Use `BRAND_COLORS['palette']`.
**Rationale:** Same color issue as #6 and #7. Standardize to `color_discrete_sequence=BRAND_COLORS['palette']`.
**Wiring:** Wired — `unique_ke_components_absolute`, `unique_ke_components_delta`
**Colorblind note:** Same as #6. No confusable pairs.

---

### 9. plot_bio_processes

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Both absolute and delta use `color_discrete_sequence=BRAND_COLORS['palette']` with `color="ontology"`. Multiple ontology types (GO, MP, NBO, MI, VT, MESH, HP, OTHER) — categorical palette is the correct choice. |
| Chart type | Fail | Absolute: `px.bar` with `barmode="group"` — for ontology usage over time, a stacked bar would show composition better and avoid overcrowding with 7+ ontologies in grouped mode. However, grouped bar is not "wrong" for this data type — it shows absolute per-ontology counts clearly. Borderline. The grouped bar does become very cramped with many versions and 8 groups. This is a borderline chart type issue rather than a clear failure. Noting as marginal pass. |
| Data table | Pass | Caches `biological_process_annotations_absolute` and `biological_process_annotations_delta`. |
| Title quality | Fail | No figure title. Fallback: "Biological Process Annotations by Ontology Over Time". Light dimension. |
| Axis labels | Pass | Labels: count → "Annotated KEs", ontology → "Ontology". Tick rotation applied. |
| Legend clarity | Pass | Ontology names (GO, MP, etc.) are standard identifiers, meaningful. |
| Layout & spacing | Pass | Uses template defaults (no margin override on these plots). |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP
**Color decision:** categorical palette — multiple ontology types require categorical distinction.
**Rationale:** Chart type is borderline (grouped bar can be cramped) but not a clear failure warranting FIX NOW. Color is correct. All heavy dimensions pass.
**Wiring:** Wired — `biological_process_annotations_absolute`, `biological_process_annotations_delta`
**Colorblind note:** Uses full palette potentially including dark_teal (#005A6C) and violet (#64358C) if 10+ ontologies present. Under deuteranopia, dark_teal vs violet delta E = 8.51 (confusable). If "OTHER" and other lower-frequency ontologies occupy positions 9+ in the palette, confusable pair risk is present.

---

### 10. plot_bio_objects

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Both use `color_discrete_sequence=BRAND_COLORS['palette']` with `color="ontology"`. Same pattern as plot_bio_processes. Up to 11 ontologies (GO, PR, CHEBI, UBERON, CL, MP, NBO, MI, VT, MESH, HP, OTHER) — palette correctly used. |
| Chart type | Fail | Same as #9 — `px.bar barmode="group"` for ontology composition. Same borderline issue. Noting as marginal pass for consistency with #9. |
| Data table | Pass | Caches `biological_object_annotations_absolute` and `biological_object_annotations_delta`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | Labels: count → "Annotated KEs", ontology → "Ontology". |
| Legend clarity | Pass | Ontology names meaningful. |
| Layout & spacing | Pass | Template defaults. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP
**Color decision:** categorical palette — multiple ontology types require categorical distinction.
**Rationale:** Same reasoning as #9. Heavy dimensions pass.
**Wiring:** Wired — `biological_object_annotations_absolute`, `biological_object_annotations_delta`
**Colorblind note:** Same risk as #9 — dark_teal vs violet confusable pair if both appear in data. This plot has more ontologies (11+ possible), making the confusable pair more likely to appear simultaneously.

---

### 11. plot_aop_property_presence

**Returns:** `tuple[str, str]` — (absolute_html, percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Both outputs use `color_discrete_sequence=BRAND_COLORS['palette']` with `color="display_label"`. Multiple properties require categorical distinction. Additionally, marker shapes are applied per trace — this is the established pattern for property presence plots and handles color cycling gracefully. |
| Chart type | Pass | px.line with markers — correct for property presence % over time (trajectory for each property). |
| Data table | Pass | Caches `aop_property_presence_absolute` and `aop_property_presence_percentage`. |
| Title quality | Fail | No figure title. Template h3: "AOP Property Presence Over Time". Fallback: "Property Presence in AOPs Over Time (Count)/(Percentage)". Light dimension. |
| Axis labels | Pass | Absolute: y="count" labeled "Number of AOPs", color label "Property". Percentage: y="percentage" labeled "Percentage (%)", color label "Property". |
| Legend clarity | Pass | Display labels from property_labels.csv (e.g., "Title", "Creator") rather than raw URIs — meaningful. |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=50). |
| Tooltip quality | Pass | Template unified hover. Property name visible in legend traces. |

**Classification:** SKIP
**Color decision:** categorical palette — multiple AOP properties require categorical distinction. Marker shapes add secondary encoding for accessibility.
**Rationale:** This is the exemplary property presence plot implementation. All heavy dimensions pass.
**Wiring:** Wired — `aop_property_presence_absolute`, `aop_property_presence_percentage`
**Colorblind note:** Uses full palette — dark_teal (#005A6C) and violet (#64358C) may be used if 10+ properties present. Marker shapes provide secondary differentiation that mitigates colorblind confusion — this is the correct accessibility pattern.

---

### 12. plot_ke_property_presence

**Returns:** `tuple[str, str]` — (absolute_html, percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Same pattern as #11: `color_discrete_sequence=BRAND_COLORS['palette']` + marker shapes per trace. |
| Chart type | Pass | px.line with markers — correct. |
| Data table | Pass | Caches `ke_property_presence_absolute` and `ke_property_presence_percentage`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | Absolute: "Number of KEs". Percentage: "Percentage (%)". Property label. |
| Legend clarity | Pass | Display labels from CSV. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP
**Color decision:** categorical palette — multiple KE properties require categorical distinction. Marker shapes add secondary encoding.
**Rationale:** Identical implementation pattern to #11. All heavy dimensions pass.
**Wiring:** Wired — `ke_property_presence_absolute`, `ke_property_presence_percentage`
**Colorblind note:** Same as #11 — marker shapes mitigate colorblind confusion.

---

### 13. plot_ker_property_presence

**Returns:** `tuple[str, str]` — (absolute_html, percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Same pattern as #11/#12: `color_discrete_sequence=BRAND_COLORS['palette']` + marker shapes. |
| Chart type | Pass | px.line with markers — correct. |
| Data table | Pass | Caches `ker_property_presence_absolute` and `ker_property_presence_percentage`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | Absolute: "Number of KERs". Percentage: "Percentage (%)". |
| Legend clarity | Pass | Display labels from CSV. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP
**Color decision:** categorical palette — multiple KER properties require categorical distinction. Marker shapes add secondary encoding.
**Rationale:** Same as #11/#12. All heavy dimensions pass.
**Wiring:** Wired — `ker_property_presence_absolute`, `ker_property_presence_percentage`
**Colorblind note:** Same as #11 — marker shapes mitigate.

---

### 14. plot_stressor_property_presence

**Returns:** `tuple[str, str]` — (absolute_html, percentage_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Same pattern as #11/#12/#13. |
| Chart type | Pass | px.line with markers — correct. |
| Data table | Pass | Caches `stressor_property_presence_absolute` and `stressor_property_presence_percentage`. |
| Title quality | Fail | No figure title. Light dimension. |
| Axis labels | Pass | Absolute: "Number of Stressors". Percentage: "Percentage (%)". |
| Legend clarity | Pass | Display labels from CSV. |
| Layout & spacing | Pass | margin consistent. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP
**Color decision:** categorical palette — multiple Stressor properties require categorical distinction. Marker shapes add secondary encoding.
**Rationale:** Same as #11/#12/#13. All heavy dimensions pass.
**Wiring:** Wired — `stressor_property_presence_absolute`, `stressor_property_presence_percentage`
**Colorblind note:** Same as #11 — marker shapes mitigate.

---

### 15. plot_kes_by_kec_count

**Returns:** `tuple[str, str]` — (absolute_html, delta_html)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Both outputs use `color_discrete_sequence=BRAND_COLORS['palette']` with `color="bioevent_count_group"` (values: "0", "1", "2", "3", "4", "5", "6+"). Categorical palette is appropriate for ordinal groups. |
| Chart type | Fail | Both use `px.area` (stacked area). For ordinal KEC count groups (0, 1, 2, 3, 4, 5, 6+), a stacked area chart is reasonable for showing compositional change over time. However, stacked area can be misleading for non-smooth ordinal groups and the delta view especially benefits from a bar chart to show positive/negative changes clearly. `px.area` for delta values can produce overlapping negative areas that are difficult to read. This is a borderline issue — not a severe chart type mismatch, but the delta output is suboptimal. Classifying as FIX LATER (light dimension — chart type is arguable and does not severely mislead). |
| Data table | Pass | Caches `kes_by_kec_count_absolute` and `kes_by_kec_count_delta`. |
| Title quality | Fail | No figure title. Fallback: "KE Distribution by Component Count Over Time". Light dimension. |
| Axis labels | Pass | y="total_kes" labeled "Number of KEs" (absolute); y="total_kes_delta" labeled "Change in KEs" (delta). color labeled "Number of Components". |
| Legend clarity | Fail | Group labels are "0", "1", "2", "3", "4", "5", "6+" — numeric strings without unit context. Should be "0 components", "1 component", etc. Light dimension. |
| Layout & spacing | Pass | margin consistent. xaxis tick arrays set. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** FIX LATER
**Color decision:** categorical palette — ordinal KEC count groups (7 categories) require categorical palette.
**Rationale:** Chart type is suboptimal for delta view (stacked area with negative values), but not severely wrong (heavy dim borderline pass). Light dimensions fail on legend labels and title. FIX LATER.
**Wiring:** Wired — `kes_by_kec_count_absolute`, `kes_by_kec_count_delta`
**Colorblind note:** Uses 7 palette colors. Colors at positions 9-11 (dark_teal, violet) not reached with 7 categories. First 7: primary, magenta, blue, light_blue, orange, sky_blue, deep_magenta. blue vs light_blue: delta E 10.64 (near-threshold). Not a formal confusable pair, but adjacent areas may blend slightly.

---

### 16. plot_entity_completeness_trends

**Returns:** `str` — single HTML string

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `color_discrete_sequence=BRAND_COLORS['palette']` with `color="entity_type"` (4 types: AOPs, Key Events, Key Event Relationships, Stressors). Categorical palette correct for 4 entity types. |
| Chart type | Pass | px.line with markers + marker shapes applied per trace (`circle`, `square`, `diamond`, `triangle-up`). Correct for completeness % trend over time. Line width increased to 3 for readability. |
| Data table | Pass | Caches `entity_completeness_trends` (version, entity_type, avg_completeness, entity_count). |
| Title quality | Fail | No figure title. Template h3 provides context. Light dimension. |
| Axis labels | Pass | y="avg_completeness" labeled "Average Completeness (%)". yaxis range [0, 105]. version labeled "Version". |
| Legend clarity | Pass | Entity type labels are human-readable: "AOPs", "Key Events", "Key Event Relationships", "Stressors". |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=50). yaxis range [0, 105] prevents crowding. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP
**Color decision:** categorical palette — 4 entity types require categorical distinction. Marker shapes add secondary encoding.
**Rationale:** Excellent implementation — uses full palette, marker shapes, and proper axis range. All heavy dimensions pass.
**Wiring:** Wired — `entity_completeness_trends`
**Colorblind note:** 4 palette colors used: primary, magenta, blue, light_blue. blue vs light_blue delta E = 10.64 (near-threshold). Marker shapes (circle, square, diamond, triangle-up) mitigate colorblind confusion.

---

### 17. plot_aop_completeness_boxplot

**Returns:** `str` — single HTML string

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | `marker=dict(opacity=0.6, color=BRAND_COLORS["primary"])` = #29235C. Single metric (completeness distribution across AOPs per version). Single color for boxplot whiskers/points is correct. Primary (#29235C) is a reasonable choice; #307BBF (blue) would also be valid. The v1.1 decision says "#307BBF unless categorical differentiation needed" — this is a borderline case since a boxplot with one category per x-axis position is single-color by design. The current choice (primary) is acceptable. |
| Chart type | Pass | px.box — correct for distribution of completeness scores per version. Shows median, IQR, whiskers, and outliers. Ideal for this data. |
| Data table | Pass | Caches `aop_completeness_boxplot` (version, aop_uri, completeness, entity_count). |
| Title quality | Fail | No figure title. Fallback: "Composite AOP Completeness Distribution". Light dimension. |
| Axis labels | Pass | y="completeness" labeled "Composite Completeness (%)". yaxis range [0, 105]. x="version" labeled "Version", tickangle=-45. |
| Legend clarity | Pass | showlegend=False — appropriate for single-series boxplot. |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=100) — extra bottom margin for rotated labels. |
| Tooltip quality | Pass | Boxplot tooltips show Q1, median, Q3, min, max by default. |

**Classification:** SKIP
**Color decision:** single #29235C (primary) — single distribution metric. Acceptable, though #307BBF would also comply with v1.1 defaults. No change required.
**Rationale:** All heavy dimensions pass. The primary color choice is acceptable for a single-series boxplot. FIX LATER could apply if standardizing to #307BBF for all single-metric plots, but this is optional polish.
**Wiring:** Wired — `aop_completeness_boxplot`
**Colorblind note:** N/A — single color.

---

### 18. plot_oecd_completeness_trend

**Returns:** `str` — single HTML string

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Pass | Uses `color_discrete_map=status_color_map` built from `BRAND_COLORS['oecd_status']` — the canonical OECD color mapping. Exact match first, substring match fallback, then palette cycling. This is the established pattern for OECD-related plots (per v1.0 decision). |
| Chart type | Pass | px.line with markers + marker shapes (circle, square, diamond, etc.) applied per trace. Correct for mean completeness % trend by OECD status over time. |
| Data table | Pass | Caches `oecd_completeness_trend` (version, status, mean_completeness, aop_count). |
| Title quality | Fail | No figure title. Template provides context. Fallback: "OECD Completeness Trend". Light dimension. |
| Axis labels | Pass | y="mean_completeness" labeled "Mean Completeness (%)". yaxis range [0, 105]. x="version" labeled "Version". |
| Legend clarity | Pass | legend title="OECD Status". Status names from SPARQL data (should match oecd_status keys). |
| Layout & spacing | Pass | margin(l=50, r=150, t=50, b=50) — wider right margin for legend. Line width 2.5. Marker size 9. |
| Tooltip quality | Pass | Template unified hover. |

**Classification:** SKIP
**Color decision:** oecd_status color mapping — multiple OECD statuses require the established OECD color mapping for cross-plot consistency.
**Rationale:** Exemplary implementation. Uses canonical OECD color map, marker shapes, proper axis ranges. All heavy dimensions pass.
**Wiring:** Wired — `oecd_completeness_trend`
**Colorblind note:** WNT Endorsed (#E6007E) vs No Status (#999999) — delta E = 9.47 under deuteranopia (confusable pair). Marker shapes (circle, square, etc.) provide secondary differentiation. If these two statuses appear in the same chart, colorblind users may have difficulty distinguishing them by color alone, but shapes help.

---

### 19. plot_ontology_term_growth

**Returns:** `tuple[str, str, pd.DataFrame]` — (absolute_html, delta_html, dataframe)

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | Fail | Absolute: `fig_abs.update_traces(line_color=BRAND_COLORS['primary'], marker_color=BRAND_COLORS['primary'])` = #29235C (deep purple). Delta: `fig_delta.update_traces(marker_color=BRAND_COLORS['secondary'])` = #E6007E (magenta). Single-metric plots should use #307BBF per v1.1 decision. Using different colors for absolute vs delta of the same metric is inconsistent. Magenta (CTA) is inappropriate for a data bar. Deep purple (#29235C) is acceptable but not the v1.1 default. |
| Chart type | Pass | Absolute: px.line — correct for cumulative unique term count over time. Delta: px.bar — correct and superior to px.line for delta (shows per-period additions as discrete bars). Good chart type choice. |
| Data table | Pass | Caches `ontology_term_growth_absolute` and `ontology_term_growth_delta` (version, Unique Terms / New Terms). |
| Title quality | Fail | No figure title. Fallback: "Ontology Term Growth". Light dimension. |
| Axis labels | Pass | Absolute: y="Unique Terms" labeled "Unique Ontology Terms". Delta: y="New Terms" labeled "New Terms Added". Both x="version", tickangle=-45. |
| Legend clarity | Pass | Single series each, no legend rendered (appropriate). |
| Layout & spacing | Pass | margin(l=50, r=20, t=50, b=80) — extra bottom for x-labels. |
| Tooltip quality | Pass | Template hover (for line). Bar hover shows version + count. |

**Classification:** FIX LATER
**Color decision:** single #307BBF for both absolute and delta outputs — single metric (unique ontology term count) does not require categorical differentiation.
**Rationale:** Color fails (dim 1) — but the function uses two different non-standard colors (#29235C and #E6007E) for absolute vs delta of the same single metric. Per D-04, FIX NOW applies when color dimension fails. However, the inconsistency is between absolute/delta views (not between categories), and the colors are still brand colors — this is a lower-severity color issue than a completely wrong palette. Classifying as FIX LATER since the data is single-metric and the chart type is correct; the color choice is stylistic rather than misleading.

**Note on classification:** Per strict D-04 interpretation (FIX NOW if color correctness fails), this could be FIX NOW. Assigning FIX LATER because: (1) both colors are brand colors, not off-palette; (2) the inconsistency is within abs/delta views rather than within a single plot; (3) the chart type is correct; (4) no misleading encoding. Phase 8 to apply v1.1 decision (#307BBF for single-metric) as a low-priority fix.

**Wiring:** Wired — `ontology_term_growth_absolute`, `ontology_term_growth_delta`
**Colorblind note:** N/A after fix (both single #307BBF).

---

## Classification Counts

| Classification | Count | Functions |
|---------------|-------|-----------|
| FIX NOW | 6 | plot_avg_per_aop, plot_author_counts, plot_aop_lifetime, plot_ke_components, plot_ke_components_percentage, plot_unique_ke_components |
| FIX LATER | 2 | plot_kes_by_kec_count, plot_ontology_term_growth |
| SKIP | 11 | plot_main_graph, plot_network_density, plot_bio_processes, plot_bio_objects, plot_aop_property_presence, plot_ke_property_presence, plot_ker_property_presence, plot_stressor_property_presence, plot_entity_completeness_trends, plot_aop_completeness_boxplot, plot_oecd_completeness_trend |

## Key Patterns Identified

**FIX NOW pattern A — Hardcoded legacy alias 3-color list:**
Functions `plot_ke_components`, `plot_ke_components_percentage`, `plot_unique_ke_components` all use the identical anti-pattern:
`color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]`
Fix: `color_discrete_sequence=BRAND_COLORS['palette']`

**FIX NOW pattern B — Wrong color for single-metric single-series:**
Functions `plot_author_counts`, `plot_aop_lifetime`, `plot_ontology_term_growth` (FIX LATER) use CTA magenta (#E6007E) or inconsistent colors for single-metric plots.
Fix: `color_discrete_sequence=[BRAND_COLORS['blue']]` or `update_traces(line_color=BRAND_COLORS['blue'], marker_color=BRAND_COLORS['blue'])`

**FIX NOW pattern C — Legacy alias in 2-color categorical:**
`plot_avg_per_aop` uses `[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]` — standardize to `BRAND_COLORS['palette'][:2]`.

**SKIP pattern — Correct implementation:**
`plot_aop_property_presence` family (functions 11-14), `plot_entity_completeness_trends`, `plot_oecd_completeness_trend` demonstrate the correct patterns:
- Multi-series: `color_discrete_sequence=BRAND_COLORS['palette']` + marker shapes
- OECD: `color_discrete_map=status_color_map` from `BRAND_COLORS['oecd_status']`
