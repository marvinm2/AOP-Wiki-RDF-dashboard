---
phase: 05-polish-consistency
plan: 03
subsystem: ui
tags: [plotly, template, render-helper, brand-consistency, vhp4safety]

# Dependency graph
requires:
  - phase: 05-01
    provides: VHP4Safety Plotly template (plotly_white+vhp4safety) and render_plot_html helper
provides:
  - All 58 plot functions standardized on VHP4Safety template and render_plot_html helper
  - Zero explicit template overrides across latest_plots.py and trends_plots.py
  - Consistent right-side vertical legends (Phase 5 decision)
affects: [05-04, visual-consistency]

# Tech tracking
tech-stack:
  added: []
  patterns: [render_plot_html for all to_html calls, template defaults eliminate per-plot style duplication]

key-files:
  modified:
    - plots/latest_plots.py
    - plots/trends_plots.py

key-decisions:
  - "Removed all Plotly figure titles since HTML headings provide context"
  - "Removed horizontal legend overrides — Phase 5 right-side vertical decision supersedes Phase 02-06"
  - "Kept per-plot margin overrides (l=300 for horizontal bars, r=150 for OECD legend space)"
  - "Kept OECD legend title='OECD Status' as intentional per-plot customization"
  - "Removed dead local config/plotly_config variables after pio.to_html replacement"

patterns-established:
  - "All plot rendering via render_plot_html(fig) — no direct pio.to_html or fig.to_html"
  - "Template provides defaults; update_layout only for intentional per-plot overrides"

requirements-completed: [Visual consistency]

# Metrics
duration: 13min
completed: 2026-02-23
---

# Phase 5 Plan 3: Plot Standardization Summary

**All 58 plot functions across latest_plots.py and trends_plots.py standardized on VHP4Safety template with render_plot_html helper, eliminating 713 lines of redundant styling code**

## Performance

- **Duration:** 13 min
- **Started:** 2026-02-23T18:59:59Z
- **Completed:** 2026-02-23T19:13:22Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Replaced all 58 `pio.to_html()` / `fig.to_html()` calls with `render_plot_html()` helper (21 in latest_plots.py, 37 in trends_plots.py)
- Removed all `template="plotly_white"` overrides (21 in latest, 36 in trends) — default template `plotly_white+vhp4safety` now applies automatically
- Removed redundant `update_layout()` parameters (hovermode, autosize, bgcolor, font) that match template defaults
- Removed all Plotly figure titles (HTML `<h3>` headings provide context)
- Enforced right-side vertical legends per Phase 5 decision (removed horizontal legend overrides)
- Net reduction: 713 lines deleted, 143 lines added (570 lines of style duplication eliminated)

## Task Commits

Each task was committed atomically:

1. **Task 1: Standardize latest_plots.py** - `a9dbadc` (feat)
2. **Task 2: Standardize trends_plots.py** - `4dfd3cc` (feat)

## Files Created/Modified
- `plots/latest_plots.py` - 21 plot functions standardized: removed template overrides, replaced to_html with render_plot_html, removed unused plotly.io and plotly.graph_objects imports
- `plots/trends_plots.py` - 37 plot renderings standardized: same transformations plus removed dead config variables, cleaned up blank lines, simplified OECD legend overrides

## Decisions Made
- Removed all Plotly figure titles since HTML headings provide context for every plot
- Phase 5 right-side vertical legend decision supersedes Phase 02-06 centered horizontal legend — removed all `orientation="h"` overrides
- Kept OECD functions' `legend=dict(title="OECD Status")` as intentional customization (position from template)
- Kept per-plot margin overrides where needed: `l=300` for horizontal bar charts, `r=150` for OECD legend space, `b=100` for rotated tick labels
- Removed unused `import plotly.io as pio` from both files (local import in `plot_aop_completeness_boxplot_by_status` also removed)
- Removed 8 dead local `config` / `plotly_config` variable assignments that were only used in replaced `pio.to_html()` calls

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All plot functions now use the VHP4Safety template and render_plot_html helper
- Ready for Plan 04 (final polish plan) which can build on consistent styling foundation
- Any future plots should follow the established pattern: use `render_plot_html(fig)` and avoid setting `template=` explicitly

---
*Phase: 05-polish-consistency*
*Completed: 2026-02-23*
