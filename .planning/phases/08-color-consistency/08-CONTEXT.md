# Phase 8: Color Consistency - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Enforce the audit's per-plot color decisions across all 23 flagged plots (19 FIX NOW + 4 FIX LATER) so bar color encodes meaning: single color for one-series plots, multiple colors for categorical distinction. Also fix 3 wiring defects found during audit, address colorblind-confusable pairs where safe, document the color decision framework, and replace legacy alias usage.

</domain>

<decisions>
## Implementation Decisions

### Scope
- **D-01:** Fix all 23 audit-flagged plots (19 FIX NOW + 4 FIX LATER). SKIP plots (16) remain untouched.
- **D-02:** Fix the 3 wiring defects found in audit (database_summary and ontology_usage missing template slots; aop_completeness_unique_colors missing app.py registration).
- **D-03:** Address the 2 colorblind-confusable pairs (dark_teal/violet delta E 8.51, WNT Endorsed/No Status delta E 9.47) only if the fix doesn't require modifying `BRAND_COLORS['palette']` — use plot-level color assignment overrides where those pairs co-occur.

### Color Framework Document
- **D-04:** Extend `.claude/colors.md` with a "Color Decision Framework" section — keep all color guidance in one file.
- **D-05:** Decision tree format (~10-15 lines): "Is data categorical? → palette. Otherwise → `BRAND_COLORS['blue']`." Plus quick-reference table of which plot types get which treatment. Satisfies COLOR-02.

### Fix Approach
- **D-06:** Replace legacy aliases (`BRAND_COLORS['secondary']`, `BRAND_COLORS['accent']`, `BRAND_COLORS['light']`) with explicit named keys (`BRAND_COLORS['magenta']`, `BRAND_COLORS['blue']`, `BRAND_COLORS['sky_blue']`). Clearer intent, no functional change for correctly-colored categorical plots.
- **D-07:** Single-color plots use `BRAND_COLORS['blue']` (not direct hex `'#307BBF'`) for maintainability.
- **D-08:** Fix colors on current chart types — pie charts flagged for chart type issues get their color corrected as-is. Chart type conversions are out of scope for v1.1.
- **D-09:** Color palette array remains frozen — never modify `BRAND_COLORS['palette']`. All changes are per-plot `marker_color` or `color_discrete_sequence` overrides.

### Verification
- **D-10:** Automated grep for known color antipatterns (continuous color scales on bar plots, legacy alias usage in fixed plots) plus manual checklist matching each audit entry to its fix.
- **D-11:** Regression check on SKIP plots — git diff filtered to those 16 functions confirms no unintended changes.

### Claude's Discretion
- Order in which plots are fixed (by module, by priority, or by similarity of fix)
- Exact grep patterns for antipattern detection
- Whether to batch similar fixes into single commits or one commit per plot

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Audit Report (primary input)
- `.planning/phases/07-plot-audit/AUDIT-REPORT.md` — Per-plot color decisions, FIX NOW/FIX LATER/SKIP classifications, wiring defects, colorblind findings. This is THE source of truth for what to change.

### Brand Colors & Palette
- `.claude/colors.md` — VHP4Safety brand color reference (will be extended with decision framework)
- `plots/shared.py` §BRAND_COLORS — Dict with palette array, named colors, oecd_status mapping, legacy aliases, type_colors

### Plot Functions (fix targets)
- `plots/latest_plots.py` — 13 FIX NOW + 2 FIX LATER functions to modify
- `plots/trends_plots.py` — 6 FIX NOW + 2 FIX LATER functions to modify

### Dashboard Templates (wiring fixes)
- `templates/latest.html` — Needs template slots for database_summary and ontology_usage
- `app.py` — Needs registration for aop_completeness_unique_colors

### Requirements
- `.planning/REQUIREMENTS.md` §Color Consistency — COLOR-01, COLOR-02, COLOR-03 acceptance criteria

### Phase 7 Context
- `.planning/phases/07-plot-audit/07-CONTEXT.md` — Audit methodology decisions, rubric dimensions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BRAND_COLORS` dict in `plots/shared.py` — Named color keys (`'blue'`, `'magenta'`, `'primary'`) plus `'palette'` array for categorical plots
- `VHP4SAFETY_TEMPLATE` in `plots/shared.py` — Global Plotly template with `colorway=BRAND_COLORS['palette']`
- `render_plot_html()` in `plots/shared.py` — All plots use this for consistent rendering

### Established Patterns
- Single-color fix is mechanical: add `marker_color=BRAND_COLORS['blue']` to `px.bar()` calls or set `color_discrete_sequence=[BRAND_COLORS['blue']]`
- Categorical palette fix: ensure `color_discrete_sequence=BRAND_COLORS['palette']` or rely on template `colorway`
- Legacy aliases (`'secondary'`, `'accent'`, `'light'`) map to `'magenta'`, `'blue'`, `'sky_blue'` respectively

### Integration Points
- Plot functions registered in `app.py` `plot_map` dict for lazy loading
- Template containers use `data-plot-name` attribute for JS lazy-loading
- 3 wiring defects need both app.py and template changes

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

- Chart type conversions (pie → bar for 5 plots) — out of scope for v1.1 per REQUIREMENTS.md
- Removal of legacy alias keys from BRAND_COLORS dict — only replacing usages, not deleting the keys
- Plot title rewrites — out of scope for v1.1 per REQUIREMENTS.md

</deferred>

---

*Phase: 08-color-consistency*
*Context gathered: 2026-03-27*
