# Plot Audit: latest_plots.py

**Audited:** 2026-03-27
**Functions:** 20
**Method:** Code analysis only (per D-06)

---

## Summary

| # | Function | Classification | Color Decision | Wired |
|---|----------|----------------|----------------|-------|
| 1 | plot_latest_entity_counts | FIX NOW | single #307BBF | yes (latest.html, index.html) |
| 2 | plot_latest_database_summary | FIX NOW | single #307BBF | no (app.py only) |
| 3 | plot_latest_avg_per_aop | FIX NOW | single #307BBF | yes (latest.html, index.html) |
| 4 | plot_latest_ke_components | FIX NOW | categorical palette | yes (latest.html, index.html) |
| 5 | plot_latest_ke_annotation_depth | FIX NOW | categorical palette | yes (latest.html, index.html) |
| 6 | plot_latest_network_density | FIX NOW | categorical palette | yes (latest.html as aop_connectivity, index.html) |
| 7 | plot_latest_ontology_usage | FIX NOW | categorical palette | no (app.py only) |
| 8 | plot_latest_process_usage | FIX LATER | categorical palette | yes (latest.html, index.html) |
| 9 | plot_latest_object_usage | FIX LATER | categorical palette | yes (latest.html, index.html) |
| 10 | plot_latest_aop_completeness | SKIP | categorical palette | yes (latest.html, index.html) |
| 11 | plot_latest_aop_completeness_unique_colors | FIX NOW | single #307BBF | no (function unwired, template broken) |
| 12 | plot_latest_aop_completeness_by_status | SKIP | categorical palette | yes (latest.html) |
| 13 | plot_latest_ke_completeness_by_status | SKIP | categorical palette | yes (latest.html) |
| 14 | plot_latest_ker_completeness_by_status | SKIP | categorical palette | yes (latest.html) |
| 15 | plot_latest_ke_by_bio_level | FIX NOW | single #307BBF | yes (latest.html) |
| 16 | plot_latest_taxonomic_groups | FIX NOW | single #307BBF | yes (latest.html) |
| 17 | plot_latest_entity_by_oecd_status | SKIP | categorical palette (oecd_status) | yes (latest.html) |
| 18 | plot_latest_ke_reuse | FIX NOW | single #307BBF | yes (latest.html) |
| 19 | plot_latest_ke_reuse_distribution | FIX NOW | single #307BBF | yes (latest.html) |
| 20 | plot_latest_ontology_diversity | FIX NOW | single #307BBF | yes (latest.html) |

---

## Per-Plot Audit

### 1. plot_latest_entity_counts

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color="Entity"` with `color_discrete_sequence=BRAND_COLORS['palette']`. Five entity types (AOPs, KEs, KERs, Stressors, Authors) mapped to 5 distinct palette colors. Since `showlegend=False` is set, the colors have no legend anchor — they serve only as visual differentiation on an x-axis bar chart where the axis labels already differentiate. Single `#307BBF` (marker_color) is appropriate and removes spurious categorical encoding. |
| Chart type | PASS | `px.bar` horizontal x-axis per entity type — correct for categorical count comparison. |
| Data table | PASS | Writes to `_plot_data_cache['latest_entity_counts']` with Entity, Count, Version columns. |
| Title quality | FAIL | No `title=` set in `update_layout`. Title must come from template heading or is absent from figure. |
| Axis labels | PASS | `yaxis=dict(title="Count")` and `xaxis=dict(title="Entity Type")` both set. |
| Legend clarity | PASS | `showlegend=False` is correct since entity names are on x-axis. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)`. Height uses template default (autosize). Reasonable. |
| Tooltip quality | PASS | Default Plotly tooltips will show Entity and Count via `text="Count"`. Adequate for this chart. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Five entity types are already differentiated by x-axis labels; legend is hidden; uniform blue removes spurious color encoding.
**Wiring:** wired to latest.html (data-plot-name="latest_entity_counts") and index.html
**Colorblind note:** N/A — fix to single color eliminates colorblind concern.

---

### 2. plot_latest_database_summary

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color="Entity"` with `color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]` (three explicit colors: deep purple, magenta, blue). Legend is hidden (`showlegend=False`). Three entity categories (AOPs, Key Events, KE Relationships) differentiated by color when x-axis labels are sufficient. Should use single `#307BBF`. |
| Chart type | PASS | `px.bar` for entity count comparison is appropriate. |
| Data table | FAIL | No `_plot_data_cache` write. The DataFrame `df` is created but never cached. CSV export would fail. |
| Title quality | FAIL | No `title=` set in `update_layout`. |
| Axis labels | PASS | `yaxis=dict(title="Count")` set. No explicit xaxis title but entity names provide context. |
| Legend clarity | PASS | `showlegend=False` correct since entity names are on x-axis. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)`. Reasonable. |
| Tooltip quality | PASS | `text="Count"` and `textposition='outside'` provide values. Plotly default hover adequate. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Entity type differentiation belongs on the axis, not in color; single color removes confusion.
**Wiring:** unwired — registered in app.py `latest_plots_with_version` map but absent from all HTML templates (latest.html, index.html). Plot is unreachable from any page.
**Colorblind note:** N/A — fix to single color eliminates concern.

---

### 3. plot_latest_avg_per_aop

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color="Metric"` with `color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]` (deep purple and magenta) for two bars. `showlegend=False` is set. Both bars show averages of the same concept (components per AOP) — single `#307BBF` is appropriate. |
| Chart type | PASS | `px.bar` with two metric bars — appropriate for comparing two scalar averages. |
| Data table | PASS | Writes to `_plot_data_cache['latest_avg_per_aop']` with rich context (AOP/KE/KER raw counts, version). |
| Title quality | FAIL | No `title=` set in `update_layout`. |
| Axis labels | FAIL | No `xaxis_title` or `yaxis_title` set in `update_layout`. Metric names on x-axis provide some context but no y-axis label. |
| Legend clarity | PASS | `showlegend=False` correct since metric names are on x-axis. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)`. Height defaults to autosize. |
| Tooltip quality | PASS | `texttemplate='%{text:.1f}'` with `textposition='outside'` provides formatted values. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Two averages of the same concept; differentiation is by x-axis label; uniform color is cleaner.
**Wiring:** wired to latest.html (data-plot-name="latest_avg_per_aop") and index.html
**Colorblind note:** N/A — fix to single color eliminates concern.

---

### 4. plot_latest_ke_components

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]` (deep purple, magenta, blue) for 3 categorical slices. Color usage is technically acceptable for a categorical pie, but `px.pie` itself is a chart type problem (see below). After chart type fix (replace with bar), categorical palette is appropriate since Process/Object/Action are distinct categories. |
| Chart type | FAIL | `px.pie` for three part-whole components. Pie charts are poor for 3 nearly-equal slices (hard to judge relative sizes). A horizontal bar chart enables direct size comparison. Pie is wrong chart type here. |
| Data table | PASS | Writes to `_plot_data_cache['latest_ke_components']` with Component, Count, Version. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | FAIL | `px.pie` has no axis labels by design. After chart type fix, axis labels will need to be added. |
| Legend clarity | PASS | `textposition='inside', textinfo='percent+label'` provides readable labels in slices. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)`. Standard. |
| Tooltip quality | PASS | Default pie tooltip shows name and value. Adequate. |

**Classification:** FIX NOW
**Color decision:** categorical palette
**Rationale:** Process, Object, Action are distinct semantic categories; different colors help distinguish them.
**Wiring:** wired to latest.html (data-plot-name="latest_ke_components") and index.html
**Colorblind note:** After palette assignment: Process=#29235C (primary), Object=#E6007E (magenta), Action=#307BBF (blue). No confusable pairs from the 3 used (primary/magenta delta E = 42.95, primary/blue = 25.69, magenta/blue = 32.81). Safe.

---

### 5. plot_latest_ke_annotation_depth

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color_discrete_sequence=BRAND_COLORS['palette']` for a `px.pie`. This is a distribution chart showing how many KEs have 0, 1, 2, 3... annotation components. The pie chart problem (see chart type below) aside, the discrete annotation depth levels (0 components, 1 component, ...) are ordered categories, not independent classes — single color on a bar chart is more appropriate. |
| Chart type | FAIL | `px.pie` for an ordered distribution (annotation depth bins). Distribution data should use a bar chart for clear comparison across ordered bins. The depth values form a natural ordinal sequence that pie slices obscure. |
| Data table | PASS | Writes to `_plot_data_cache['latest_ke_annotation_depth']` with Depth, KE Count, Sort, Version, Numeric_Depth. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | FAIL | `px.pie` has no axis labels. After chart type fix, axis labels needed. |
| Legend clarity | FAIL | `textposition='inside', textinfo='percent+label'` — useful for pie but inadequate for showing ordinal progression. Legend after fix should show depth bins clearly. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | PASS | Default pie hover shows slice name and value. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Annotation depth bins are ordered (0, 1, 2, 3, ...) — uniform color on a bar chart communicates a single distribution without spurious categorical encoding.
**Wiring:** wired to latest.html (data-plot-name="latest_ke_annotation_depth") and index.html
**Colorblind note:** N/A — fix to single color eliminates concern.

---

### 6. plot_latest_network_density

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]` (deep purple, magenta) for a `px.pie` with two slices: Connected AOPs and Isolated AOPs. A pie for a binary split is reasonable as a part-whole view, but the colors used (primary and secondary) are unnecessarily distinct — a single-color bar chart or a donut with one brand color would be cleaner. |
| Chart type | FAIL | `px.pie` for a binary (Connected/Isolated) split. A stacked horizontal bar or two-bar chart with explicit percentage annotation communicates the binary split more precisely than a pie. |
| Data table | PASS | Writes to `_plot_data_cache['latest_aop_connectivity']` with Connected/Isolated counts, Total_AOPs, Version. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | FAIL | `px.pie` has no axis labels. The annotation dict provides some textual context (`Total AOPs: X`, etc.) but this is supplementary. |
| Legend clarity | PASS | `textposition='inside', textinfo='percent+label'` — for two slices this is readable. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)` with annotation dict. |
| Tooltip quality | PASS | Default pie hover shows slice and percentage. Adequate for binary split. |

**Classification:** FIX NOW
**Color decision:** categorical palette
**Rationale:** Connected vs. Isolated are two semantically distinct categories needing differentiation; two-color categorical palette is appropriate.
**Wiring:** wired to latest.html (data-plot-name="latest_aop_connectivity") and index.html
**Colorblind note:** If two slices use palette[0] (primary #29235C) and palette[1] (magenta #E6007E), delta E under deuteranopia = 42.95. Safe.

---

### 7. plot_latest_ontology_usage

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | No `color_discrete_sequence` is set in `px.pie(df, values="Terms", names="Ontology")`. Plotly will fall back to the template colorway (BRAND_COLORS['palette'] via the vhp4safety template). This is technically acceptable but implicit — should be explicit `color_discrete_sequence=BRAND_COLORS['palette']`. |
| Chart type | FAIL | `px.pie` for 7+ ontology categories (GO, MP, NBO, MI, VT, MESH, HP, OTHER). Pie with many slices is hard to read. A horizontal bar chart sorted by count is more appropriate for multi-category ontology distribution. |
| Data table | FAIL | No `_plot_data_cache` write anywhere in the function. CSV export will fail with a KeyError or empty result. This is the only latest_plots.py function missing cache write. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | FAIL | `px.pie` has no axis labels. |
| Legend clarity | PASS | `textposition='inside', textinfo='percent+label'` — acceptable for a pie, but with 7+ categories some slices will be too small for inside labels. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | PASS | Default pie hover shows ontology name and count. |

**Classification:** FIX NOW
**Color decision:** categorical palette
**Rationale:** Each ontology source (GO, CHEBI, etc.) is a distinct semantic category; categorical colors help distinguish them on a bar chart.
**Wiring:** unwired — registered in app.py `latest_plots_with_version` map (line 1625) but absent from all HTML templates. Plot is unreachable from the dashboard UI.
**Colorblind note:** With 7+ palette colors, dark_teal (#005A6C) and violet (#64358C) may co-appear (delta E = 8.51 under deuteranopia, confusable). This pair should be avoided in the final color assignment.

---

### 8. plot_latest_process_usage

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | PASS | Uses `color_discrete_sequence=BRAND_COLORS['palette']` for a pie of ontology sources. Categorical palette is appropriate for distinct ontology names (GO, MP, NBO, etc.). |
| Chart type | FAIL | `px.pie` for multiple ontology source categories. Same problem as plot_latest_ontology_usage — bar chart would be more readable for 5+ categories. |
| Data table | PASS | Writes to `_plot_data_cache['latest_process_usage']` with Ontology, Count, Version, Component_Type columns. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | FAIL | `px.pie` has no axis labels. After chart type fix, axis labels needed. |
| Legend clarity | PASS | `textposition='inside', textinfo='percent+label'`. Legend implicit from slice labels. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | PASS | Default pie hover shows ontology and percentage. |

**Classification:** FIX LATER
**Color decision:** categorical palette
**Rationale:** GO, MP, NBO, etc. are distinct ontology namespaces; categorical colors are appropriate.
**Wiring:** wired to latest.html (data-plot-name="latest_process_usage") and index.html
**Colorblind note:** With 7+ palette colors, dark_teal/violet confusable pair (delta E 8.51) may appear if both ontologies present in data. Flag for Phase 8 color assignment.

---

### 9. plot_latest_object_usage

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | PASS | Uses `color_discrete_sequence=BRAND_COLORS['palette']` for a pie of ontology sources. Categorical palette appropriate for distinct ontology names (GO, CHEBI, PR, CL, UBERON, NCBITaxon, DOID, HP, OTHER). |
| Chart type | FAIL | `px.pie` for multiple categories. Bar chart preferable for 5+ slices. |
| Data table | PASS | Writes to `_plot_data_cache['latest_object_usage']` with Ontology, Count, Version, Component_Type. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | FAIL | No axis labels (pie chart). After chart type fix, labels needed. |
| Legend clarity | PASS | `textposition='inside', textinfo='percent+label'`. Readable for reasonable number of slices. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=50)`. |
| Tooltip quality | PASS | Default pie hover shows ontology and count. |

**Classification:** FIX LATER
**Color decision:** categorical palette
**Rationale:** GO, CHEBI, PR, etc. are distinct ontology namespaces; categorical palette appropriate.
**Wiring:** wired to latest.html (data-plot-name="latest_object_usage") and index.html
**Colorblind note:** With 8+ palette colors used, dark_teal/violet confusable pair (delta E 8.51) may appear. Flag for Phase 8.

---

### 10. plot_latest_aop_completeness

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | PASS | Uses `color="Type"` with `color_discrete_map=BRAND_COLORS['type_colors']` (Essential=#29235C, Metadata=#E6007E, Content=#EB5B25, Context=#93D5F6, Assessment=#307BBF, Structure=#45A6B2). Correct: type_colors is the appropriate mapping for property types. |
| Chart type | PASS | `px.bar` grouped by property type — correct for comparing completeness percentages across properties. |
| Data table | PASS | Writes to `_plot_data_cache['latest_aop_completeness']` with Property, Completeness, Type, URI, Count, Version, Total_AOPs. Rich dataset. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="AOP Properties")` set. Units included in y-axis. |
| Legend clarity | PASS | `legend=dict(title="Property Type")` set. Legend title descriptive. |
| Layout & spacing | PASS | `height` not set (autosize), `margin=dict(l=50, r=20, t=50, b=100)`, `yaxis range=[0, 105]`, `tickangle=45`. Appropriate for rotated tick labels. |
| Tooltip quality | PASS | `texttemplate='%{text:.1f}%'` with `textposition='outside'` provides formatted values on bars. |

**Classification:** SKIP
**Color decision:** categorical palette
**Rationale:** Property types (Essential, Metadata, Content, etc.) are meaningful semantic categories; type_colors mapping is the correct and intended coloring.
**Wiring:** wired to latest.html (data-plot-name="latest_aop_completeness") and index.html
**Colorblind note:** type_colors group has no confusable pairs (minimum delta E = 15.34 across all 15 pairwise comparisons). Safe.

---

### 11. plot_latest_aop_completeness_unique_colors

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Assigns a unique palette color to each individual property name (`color="Property"` with `color_discrete_map` cycling through `BRAND_COLORS['palette']`). Each property gets a unique color purely for visual variety — there is no semantic meaning to the color differences since `showlegend=False`. This is rainbow coloring without purpose. Single `#307BBF` is appropriate since x-axis labels already differentiate properties. |
| Chart type | PASS | `px.bar` for completeness percentages — correct chart type. |
| Data table | PASS | Writes to `_plot_data_cache['latest_aop_completeness_unique']` with same structure as `plot_latest_aop_completeness`. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="AOP Properties")` set. |
| Legend clarity | PASS | `showlegend=False` — appropriate since colors are per-property and legend would be redundant. |
| Layout & spacing | PASS | `height=600`, `margin=dict(l=50, r=20, t=70, b=120)`. Explicit height and generous bottom margin for rotated labels. |
| Tooltip quality | PASS | `texttemplate='%{text:.1f}%'` with `textposition='outside'` provides values. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Per-property unique colors add no information since legend is hidden and x-axis labels already identify each bar; uniform blue is cleaner.
**Wiring:** unwired — function is not registered in app.py `latest_plots_with_version` map. Template has `data-plot-name="latest_aop_completeness_unique"` in index.html (line 158) but will receive 404 from `/api/plot/latest_aop_completeness_unique` since no handler exists.
**Colorblind note:** N/A — fix to single color eliminates concern. Wiring defect is a separate issue.

---

### 12. plot_latest_aop_completeness_by_status

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | PASS | Uses `color="Property Type"` with `color_discrete_map=BRAND_COLORS['type_colors']`. Correct use of type_colors semantic mapping for property type differentiation. |
| Chart type | PASS | `px.bar` with `barmode="group"` — grouped bar chart is correct for comparing property type completeness across OECD status categories. |
| Data table | PASS | Writes to `_plot_data_cache['latest_aop_completeness_by_status']` with OECD Status, Property Type, Completeness, Count, Total, Version. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="OECD Status")` set. |
| Legend clarity | PASS | `legend=dict(title="Property Type")` set. Descriptive legend title. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=100)`, `yaxis range=[0, 105]`, `tickangle=45`. Appropriate. |
| Tooltip quality | PASS | `texttemplate='%{text:.1f}%'` with `textposition='outside'`. |

**Classification:** SKIP
**Color decision:** categorical palette
**Rationale:** Property types (Essential, Metadata, Content, etc.) are semantically distinct categories; type_colors is the correct mapping.
**Wiring:** wired to latest.html (data-plot-name="latest_aop_completeness_by_status")
**Colorblind note:** type_colors group has no confusable pairs. Safe.

---

### 13. plot_latest_ke_completeness_by_status

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | PASS | Uses `color="Property Type"` with `color_discrete_map=BRAND_COLORS['type_colors']`. Correct semantic mapping. |
| Chart type | PASS | `px.bar` with `barmode="group"` — appropriate for cross-status property type comparison. |
| Data table | PASS | Writes to `_plot_data_cache['latest_ke_completeness_by_status']` with full context. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="OECD Status of Parent AOPs")` set. |
| Legend clarity | PASS | `legend=dict(title="Property Type")`. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=100)`, range=[0, 105], tickangle=45. |
| Tooltip quality | PASS | `texttemplate='%{text:.1f}%'`. |

**Classification:** SKIP
**Color decision:** categorical palette
**Rationale:** Property types are semantically distinct; type_colors mapping is correct.
**Wiring:** wired to latest.html (data-plot-name="latest_ke_completeness_by_status")
**Colorblind note:** type_colors group has no confusable pairs. Safe.

---

### 14. plot_latest_ker_completeness_by_status

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | PASS | Uses `color="Property Type"` with `color_discrete_map=BRAND_COLORS['type_colors']`. Correct. |
| Chart type | PASS | `px.bar` with `barmode="group"` — correct for property type completeness by OECD status. |
| Data table | PASS | Writes to `_plot_data_cache['latest_ker_completeness_by_status']` with full context. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="Completeness (%)")` and `xaxis=dict(title="OECD Status of Parent AOPs")` set. |
| Legend clarity | PASS | `legend=dict(title="Property Type")`. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=50, b=100)`, range=[0, 105], tickangle=45. |
| Tooltip quality | PASS | `texttemplate='%{text:.1f}%'`. |

**Classification:** SKIP
**Color decision:** categorical palette
**Rationale:** Property types are semantically distinct categories; type_colors mapping is correct.
**Wiring:** wired to latest.html (data-plot-name="latest_ker_completeness_by_status")
**Colorblind note:** type_colors group has no confusable pairs. Safe.

---

### 15. plot_latest_ke_by_bio_level

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color="KE Count"` with `color_continuous_scale=[BRAND_COLORS['light'], BRAND_COLORS['primary']]` (sky blue to deep purple gradient). This maps the numeric count to a continuous color scale — but the count is already encoded on the x-axis. The gradient color adds no additional information and creates a misleading suggestion that bars with higher counts are in a "different category" from lower-count bars. Single `#307BBF` (marker_color) is appropriate. `coloraxis_showscale=False` suppresses the legend, confirming color adds nothing here. |
| Chart type | PASS | `px.bar` with `orientation='h'` — horizontal bar sorted by count is appropriate for categorical level distribution. |
| Data table | PASS | Writes to `_plot_data_cache[f'latest_ke_by_bio_level_{version_key}']` with Biological Level, KE Count, Sort, Version, Numeric_Depth. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="")` (biological level names serve as labels), `xaxis=dict(title="Number of Key Events")` set. |
| Legend clarity | PASS | `showlegend=False` and `coloraxis_showscale=False` — correct since count-based gradient adds no information. |
| Layout & spacing | PASS | `margin=dict(l=150, r=30, t=80, b=60)` — wide left margin for long level names. |
| Tooltip quality | PASS | `textposition='outside'` shows count values. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Continuous color gradient on a count-encoded axis provides no additional information; uniform blue removes redundant visual encoding.
**Wiring:** wired to latest.html (data-plot-name="latest_ke_by_bio_level")
**Colorblind note:** N/A — fix to single color eliminates concern.

---

### 16. plot_latest_taxonomic_groups

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color="Taxonomic Group"` with `color_discrete_sequence=BRAND_COLORS['palette']` for a horizontal bar chart. Each taxonomic group (up to 25) gets a unique color from the 11-color palette (cycling). `showlegend=False` is set — so the colors serve only as bar fill differentiation where y-axis labels already identify each group. No semantic meaning is conveyed by the color differences. Single `#307BBF` is appropriate. |
| Chart type | PASS | `px.bar` with `orientation='h'` sorted by count — correct for ranking 25 taxonomic groups. |
| Data table | PASS | Writes to `_plot_data_cache[f'latest_taxonomic_groups_{version_key}']` with Taxonomic Group, AOP Count, Version. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="")` (group names on axis), `xaxis=dict(title="Number of AOPs")` set. |
| Legend clarity | PASS | `showlegend=False` — correct since bar labels already identify groups. |
| Layout & spacing | PASS | `margin=dict(l=180, r=30, t=60, b=60)` — wide left margin for long taxonomic names. |
| Tooltip quality | PASS | `textposition='outside'` shows AOP count values. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** 25 bars with unique colors from an 11-color cycling palette; legend is hidden; y-axis labels differentiate bars; single color removes visual noise.
**Wiring:** wired to latest.html (data-plot-name="latest_taxonomic_groups")
**Colorblind note:** Currently uses up to 11 cycling colors including the confusable dark_teal/violet pair (delta E 8.51). Fix to single color eliminates this concern.

---

### 17. plot_latest_entity_by_oecd_status

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | PASS | Uses `color="OECD Status"` with `color_discrete_map=BRAND_COLORS['oecd_status']`. This is the correct mapping — OECD status colors are the canonical semantic color assignment for status categories. |
| Chart type | PASS | `px.bar` with `barmode="group"` — grouped bars by entity type (AOPs, KEs, KERs) with OECD status as color groups. Correct for this cross-dimensional comparison. |
| Data table | PASS | Writes to `_plot_data_cache[f'latest_entity_by_oecd_status_{version_key}']` with Entity Type, OECD Status, Count, Version. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="Count")` and `xaxis=dict(title="Entity Type")` set. |
| Legend clarity | PASS | `legend=dict(title="OECD Status")` set. Descriptive. |
| Layout & spacing | PASS | `margin=dict(l=60, r=30, t=60, b=60)`. Appropriate. |
| Tooltip quality | PASS | `textposition='outside'` shows count values. Default hover provides status and count. |

**Classification:** SKIP
**Color decision:** categorical palette (oecd_status)
**Rationale:** OECD status is a semantic classification; oecd_status color mapping is the correct and consistent assignment across all OECD-related plots.
**Wiring:** wired to latest.html (data-plot-name="latest_entity_by_oecd_status")
**Colorblind note:** OECD status palette has 1 confusable pair: WNT Endorsed (#E6007E) / No Status (#999999), delta E = 9.47 under deuteranopia. Both statuses may appear in this plot. Flag for Phase 8.

---

### 18. plot_latest_ke_reuse

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color="AOP Count"` with `color_continuous_scale=[BRAND_COLORS['light'], BRAND_COLORS['primary']]` (sky blue to deep purple gradient). AOP Count is already encoded on the x-axis. The gradient communicates redundant information and misleads readers into thinking high-count KEs are categorically different. `coloraxis_showscale=False` confirms the legend is suppressed — confirming color adds nothing. Single `#307BBF` is appropriate. |
| Chart type | PASS | `px.bar` with `orientation='h'` — horizontal bar ranked by AOP count is the correct chart type for a top-30 ranking. |
| Data table | PASS | Writes to `_plot_data_cache[f'latest_ke_reuse_{version_key}']` with KE, KE ID, Title, AOP Count, wiki_url, Version. Includes wiki_url for entity links. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="")` (KE labels on axis), `xaxis=dict(title="Number of AOPs")` set. |
| Legend clarity | PASS | `showlegend=False`, `coloraxis_showscale=False` — correct. |
| Layout & spacing | PASS | `height=max(400, len(data)*25+100)` — dynamic height based on data rows. `margin=dict(l=300, r=30, t=60, b=60)` — very wide left margin for KE title labels. |
| Tooltip quality | PASS | `custom_data=['wiki_url']` included in trace for click-to-open functionality. `textposition='outside'` shows count. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Count-to-color gradient duplicates x-axis encoding; legend is suppressed; single color is cleaner and removes misleading categorical suggestion.
**Wiring:** wired to latest.html (data-plot-name="latest_ke_reuse" with id="ke-reuse-plot")
**Colorblind note:** N/A — fix to single color eliminates concern.

---

### 19. plot_latest_ke_reuse_distribution

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color="Number of KEs"` with `color_continuous_scale=[BRAND_COLORS['light'], BRAND_COLORS['primary']]`. Same pattern as plot_latest_ke_reuse — count-to-color gradient duplicates the y-axis encoding. `coloraxis_showscale=False` suppresses the legend. Single `#307BBF` is appropriate. |
| Chart type | PASS | `px.bar` with discrete bins (1, 2, 3, 4, 5, 6-10, 11+) on x-axis — appropriate for a distribution histogram with manual bins. `type='category'` on x-axis ensures correct bin order. |
| Data table | PASS | Writes to `_plot_data_cache[f'latest_ke_reuse_distribution_{version_key}']` with AOPs per KE, Number of KEs, Version. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="Number of Key Events")` and `xaxis=dict(title="Number of AOPs a KE Belongs To")` set. Clear labels with units implied. |
| Legend clarity | PASS | `showlegend=False`, `coloraxis_showscale=False`. Correct. |
| Layout & spacing | PASS | `margin=dict(l=60, r=30, t=60, b=60)`. Standard. |
| Tooltip quality | PASS | `textposition='outside'` shows count values. Default hover shows bin and count. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Count gradient duplicates y-axis; bins are ordered distribution buckets, not categories; uniform color for distribution bars is standard practice.
**Wiring:** wired to latest.html (data-plot-name="latest_ke_reuse_distribution")
**Colorblind note:** N/A — fix to single color eliminates concern.

---

### 20. plot_latest_ontology_diversity

| Dimension | Pass/Fail | Notes |
|-----------|-----------|-------|
| Color correctness | FAIL | Uses `color="Ontology"` with `color_discrete_map` cycling through `BRAND_COLORS['palette']` per ontology. `showlegend=False` is set — colors serve only as visual variety since ontology names are already on the x-axis. Since no meaningful semantic distinction exists between ontologies (they are all database sources), unique per-ontology colors add visual noise without conveying information. Single `#307BBF` is the correct choice. Note: This differs from `plot_latest_process_usage` and `plot_latest_object_usage` only in chart type (bar vs. pie) — on a bar chart with labeled x-axis, single color is strongly preferable. |
| Chart type | PASS | `px.bar` with ontologies on x-axis sorted by count — correct chart type for comparing unique term counts per source. |
| Data table | PASS | Writes to `_plot_data_cache[cache_key]` (`latest_ontology_diversity_{version_key}`) with Ontology, Unique Terms, Version. |
| Title quality | FAIL | No `title=` set. |
| Axis labels | PASS | `yaxis=dict(title="Number of Unique Terms")` and `xaxis=dict(title="Ontology Source")` set. |
| Legend clarity | PASS | `showlegend=False` — correct since ontology names are on x-axis. |
| Layout & spacing | PASS | `margin=dict(l=50, r=20, t=60, b=50)`. Standard. |
| Tooltip quality | PASS | `textposition='outside'` shows count values. |

**Classification:** FIX NOW
**Color decision:** single #307BBF
**Rationale:** Ontologies are distinguished by x-axis labels; legend is hidden; per-ontology cycling colors add visual noise without semantic content; uniform blue is cleaner.
**Wiring:** wired to latest.html (data-plot-name="latest_ontology_diversity")
**Colorblind note:** Currently the cycling palette will assign dark_teal to one ontology and violet to another if both appear (delta E 8.51 — confusable). Fix to single color eliminates this concern.

---

## Wiring Issues Summary

| Function | Template Slot | App.py Registered | Status |
|----------|--------------|-------------------|--------|
| plot_latest_database_summary | Not present | Yes (line 1626) | Unreachable — registered but not displayed |
| plot_latest_ontology_usage | Not present | Yes (line 1625) | Unreachable — registered but not displayed |
| plot_latest_aop_completeness_unique_colors | index.html slot `latest_aop_completeness_unique` | No | Broken — template slot returns 404 |

## Classification Statistics

| Classification | Count | Functions |
|----------------|-------|-----------|
| FIX NOW | 13 | 1, 2, 3, 4, 5, 6, 7, 11, 15, 16, 18, 19, 20 |
| FIX LATER | 2 | 8, 9 |
| SKIP | 4 | 10, 12, 13, 14, 17 |

## Color Decision Statistics

| Color Decision | Count |
|----------------|-------|
| single #307BBF | 11 |
| categorical palette | 9 |

## Key Patterns Identified

1. **Continuous color scale antipattern (3 plots):** `plot_latest_ke_by_bio_level`, `plot_latest_ke_reuse`, `plot_latest_ke_reuse_distribution` all use `color_continuous_scale` with `coloraxis_showscale=False` — encoding a quantity as color that is already on an axis. Fix: replace with `marker_color='#307BBF'`.

2. **Categorical pie chart antipattern (5 plots):** `plot_latest_ke_components`, `plot_latest_ke_annotation_depth`, `plot_latest_network_density`, `plot_latest_process_usage`, `plot_latest_object_usage` all use `px.pie` for distributions or compositions that would be clearer as bar charts.

3. **Unnecessary per-bar color antipattern (3 plots):** `plot_latest_entity_counts`, `plot_latest_taxonomic_groups`, `plot_latest_ontology_diversity` assign unique colors to bars where `showlegend=False` — color adds no information.

4. **Missing title pattern (all 20 plots):** No function sets `title=` in `update_layout`. HTML headings in templates provide context, but figure-level titles are absent for standalone export scenarios.

5. **Correct patterns (5 plots):** `plot_latest_aop_completeness`, `plot_latest_aop_completeness_by_status`, `plot_latest_ke_completeness_by_status`, `plot_latest_ker_completeness_by_status`, `plot_latest_entity_by_oecd_status` all use `BRAND_COLORS['type_colors']` or `BRAND_COLORS['oecd_status']` for semantic categorical color mapping.

---

*Audit conducted: 2026-03-27*
*Auditor: Code analysis per D-06*
