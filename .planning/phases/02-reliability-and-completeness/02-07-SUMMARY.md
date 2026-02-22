---
phase: 02-reliability-and-completeness
plan: 07
subsystem: ui, api
tags: [plotly, legend-layout, png-export, methodology-notes, uat-closure]

# Dependency graph
requires:
  - phase: 02-reliability-and-completeness
    provides: "UAT testing results identifying 3 cosmetic/minor issues"
provides:
  - "OECD completeness trend plot with right-side vertical legend (no overlap)"
  - "PNG/SVG export for all 3 completeness-by-status latest plots"
  - "Data scope caveat in all 17 trend methodology note limitations"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Right-side vertical legends for multi-category trend plots"
    - "Data scope caveats in all trend methodology notes"

key-files:
  created: []
  modified:
    - plots/trends_plots.py
    - app.py
    - static/data/methodology_notes.json

key-decisions:
  - "OECD legend moved to right-side vertical matching sibling boxplot pattern per user request"
  - "Right margin increased to r=150 and top margin reduced to t=50 to accommodate vertical legend"
  - "Completeness-by-status plots added to startup computation for cache population, not template rendering"

patterns-established:
  - "Multi-status OECD plots use right-side vertical legends (orientation=v, x=1.02)"

requirements-completed: [RELY-02, RELY-04, EXPL-07]

# Metrics
duration: 2min
completed: 2026-02-22
---

# Phase 2 Plan 7: UAT Gap Closure Summary

**OECD legend fixed to right-side vertical, 3 completeness-by-status plots added to startup for PNG/SVG export, and data scope caveat appended to all 17 trend methodology notes**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-22T14:56:29Z
- **Completed:** 2026-02-22T14:58:54Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- OECD completeness trend plot legend moved from horizontal centered (above plot, causing overlap) to right-side vertical matching the sibling boxplot pattern
- All 3 latest completeness-by-status plots (AOP, KE, KER) now computed at startup, populating `_plot_figure_cache` so PNG/SVG export works without prior page visit
- All 17 trend methodology notes now include data scope caveat explaining that data covers only published RDF release versions

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix OECD legend to right-side vertical and add 3 missing plots to startup computation** - `f470edc` (fix)
2. **Task 2: Add data scope caveat to all trend methodology note limitations** - `3c6b351` (docs)

**Plan metadata:** (pending final commit)

## Files Created/Modified
- `plots/trends_plots.py` - Changed OECD legend from horizontal (orientation="h") to vertical right-side (orientation="v", x=1.02), adjusted margins
- `app.py` - Added 3 completeness-by-status plots to compute_plots_parallel plot_tasks list and result extraction
- `static/data/methodology_notes.json` - Appended RDF release version coverage caveat to limitations of all 17 trend entries

## Decisions Made
- OECD legend moved to right-side vertical per user request ("Maybe add the legend on the right side just like other plots?")
- Right margin increased from 20 to 150 and top margin reduced from 120 to 50 to accommodate the new vertical legend position
- The 3 completeness-by-status result variables are extracted but serve only to populate the figure cache; template rendering uses the dynamic version-selector map

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 2 is now complete with all UAT issues resolved
- All plots render correctly with proper legends
- All PNG/SVG exports functional
- All methodology notes include appropriate data scope caveats
- Ready for Phase 3 (Network Analysis) or any subsequent phases

## Self-Check: PASSED

All files verified present. All commit hashes verified in git log.

---
*Phase: 02-reliability-and-completeness*
*Completed: 2026-02-22*
