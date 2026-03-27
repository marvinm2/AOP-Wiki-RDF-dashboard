---
phase: 07-plot-audit
plan: 02
subsystem: documentation
tags: [plot-audit, color-correctness, chart-type, accessibility, latest-plots]

# Dependency graph
requires:
  - "07-01: COLORBLIND-FINDINGS.md (confusable pairs for cross-reference)"
provides:
  - "Per-plot audit results for all 20 latest_plots.py functions"
  - "Color decision (single #307BBF vs categorical palette) for each plot"
  - "Wiring status for each function"
  - "Classification (FIX NOW / FIX LATER / SKIP) for each function"
affects: [07-03-PLAN, 08-color-consistency]

# Tech tracking
tech-stack:
  added: []
  patterns: ["8-dimension binary rubric: color correctness, chart type, data table, title, axes, legend, layout, tooltips"]

key-files:
  created:
    - ".planning/phases/07-plot-audit/AUDIT-LATEST.md"
  modified: []

key-decisions:
  - "13 of 20 latest plots are FIX NOW — primary drivers are categorical pie antipattern (5 plots) and continuous color scale antipattern (3 plots)"
  - "Single #307BBF assigned to 11 plots where bars/slices have no semantic color differentiation need"
  - "3 wiring defects documented: database_summary and ontology_usage are registered but not in templates; aop_completeness_unique_colors is in template but not registered in app.py"
  - "plot_latest_aop_completeness (type_colors), aop_completeness_by_status, ke_completeness_by_status, ker_completeness_by_status, entity_by_oecd_status are SKIP — already use correct semantic color mappings"

patterns-established:
  - "Continuous color scale with coloraxis_showscale=False is a code smell — it encodes a quantity as color that is already on an axis"
  - "px.pie for multi-category distributions should be replaced with horizontal bar charts"

requirements-completed: [AUDIT-01, AUDIT-02]

# Metrics
duration: 15min
completed: 2026-03-27
---

# Phase 7 Plan 02: Latest Plots Audit Summary

**Code analysis of all 20 latest_plots.py functions against an 8-dimension binary rubric, identifying 13 FIX NOW plots (color or chart type failures), 2 FIX LATER, and 4 SKIP, with explicit per-plot color decisions and wiring status.**

## Performance

- **Duration:** 15 min
- **Completed:** 2026-03-27
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Audited all 20 `plot_latest_*` functions in `plots/latest_plots.py` against the 8-dimension binary rubric
- Applied COLORBLIND-FINDINGS.md confusable pairs to each categorical plot's color usage
- Documented wiring status for all 20 functions against `latest.html`, `index.html`, and `app.py`
- Identified 3 recurring antipatterns across the codebase
- Produced AUDIT-LATEST.md with per-plot rubric tables, classifications, color decisions, and wiring status

## Task Commits

1. **Task 1: Audit all 20 latest_plots.py functions** — `58753e7` (docs)

## Files Created/Modified

- `.planning/phases/07-plot-audit/AUDIT-LATEST.md` — Complete audit report with 20 per-plot sections, summary table, wiring issues table, antipattern catalog

## Audit Results Summary

| Classification | Count | Key Functions |
|----------------|-------|---------------|
| FIX NOW | 13 | entity_counts, database_summary, avg_per_aop, ke_components, ke_annotation_depth, network_density, ontology_usage, aop_completeness_unique_colors, ke_by_bio_level, taxonomic_groups, ke_reuse, ke_reuse_distribution, ontology_diversity |
| FIX LATER | 2 | process_usage, object_usage (chart type only — color is correct) |
| SKIP | 5 | aop_completeness, aop_completeness_by_status, ke_completeness_by_status, ker_completeness_by_status, entity_by_oecd_status |

| Color Decision | Count |
|----------------|-------|
| single #307BBF | 11 |
| categorical palette | 9 |

## Antipatterns Identified

Three patterns account for all 13 FIX NOW classifications:

1. **Continuous color scale antipattern** (3 plots): `ke_by_bio_level`, `ke_reuse`, `ke_reuse_distribution` — use `color_continuous_scale` with `coloraxis_showscale=False`, encoding the same quantity already on an axis.

2. **Categorical pie chart antipattern** (5 plots): `ke_components`, `ke_annotation_depth`, `network_density`, `process_usage`, `object_usage` — use `px.pie` for distributions or compositions better shown as bar charts. Note: `process_usage` and `object_usage` are FIX LATER because only chart type fails.

3. **Unnecessary per-bar color antipattern** (4 plots, overlapping): `entity_counts`, `database_summary`, `avg_per_aop`, `taxonomic_groups`, `ontology_diversity` — assign unique colors to bars where `showlegend=False`, adding visual noise.

## Wiring Defects

| Function | Issue |
|----------|-------|
| `plot_latest_database_summary` | Registered in app.py but absent from all templates — unreachable |
| `plot_latest_ontology_usage` | Registered in app.py but absent from all templates — unreachable |
| `plot_latest_aop_completeness_unique_colors` | Template slot `latest_aop_completeness_unique` exists in index.html but function not in app.py plot_map — returns 404 |

## Decisions Made

- Count-to-color gradient (`color_continuous_scale` + `coloraxis_showscale=False`) classified as color correctness failure since it encodes redundant information.
- Pie charts for 3+ categories classified as chart type failure; 2-category pies (network_density) also classified as chart type failure since a stacked bar communicates the binary split with more precision.
- `process_usage` and `object_usage` classified FIX LATER (not FIX NOW) because their color is correct (explicit `BRAND_COLORS['palette']` for ontology categories) — only chart type fails.
- All 5 "SKIP" plots correctly use `BRAND_COLORS['type_colors']` or `BRAND_COLORS['oecd_status']` semantic mappings.

## Deviations from Plan

None — plan executed exactly as written. The 20th function (`plot_latest_ontology_diversity`) exists as expected; `plot_latest_ontology_usage` is a separate, distinct function (different query pattern and different wiring status).

## Known Stubs

None — this plan produces an analysis document, not UI components.

## Self-Check: PASSED

- AUDIT-LATEST.md exists at `.planning/phases/07-plot-audit/AUDIT-LATEST.md`
- File contains exactly 20 per-plot sections (`grep -c "^### [0-9]"` returns 20)
- All 20 sections contain Color correctness, Classification, Color decision, Wiring entries
- Commit `58753e7` verified

---

*Phase: 07-plot-audit*
*Completed: 2026-03-27*
