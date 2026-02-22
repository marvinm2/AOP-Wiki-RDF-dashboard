---
phase: 02-reliability-and-completeness
plan: 06
subsystem: ui, data-quality
tags: [plotly, methodology-notes, layout, json, jinja2]

requires:
  - phase: 02-03
    provides: "Methodology notes system with JSON data and Jinja2 macro"
provides:
  - "Fixed OECD and entity completeness plot layouts without legend/title overlap"
  - "Complete methodology note coverage on all AOP Lifetime sub-plots (19 total calls)"
  - "Cleaned methodology note limitations text across all 28 entries"
affects: []

tech-stack:
  added: []
  patterns:
    - "Horizontal legend with increased top margin (t=100-120) and y=1.10-1.15 for multi-trace plots"
    - "Methodology notes shared across related sub-plots using same JSON key"

key-files:
  created: []
  modified:
    - "plots/trends_plots.py"
    - "templates/trends.html"
    - "static/data/methodology_notes.json"

key-decisions:
  - "Centered horizontal legends (xanchor=center, x=0.5) instead of right-aligned for better visual balance"
  - "Varied presence-only limitation text per entity type for contextual clarity instead of identical text"
  - "Removed implementation details (SPARQL generation, marker shape rendering) from limitations as non-researcher-relevant"

patterns-established:
  - "Methodology note limitations should contain only methodological/data-quality caveats, no performance or implementation details"

requirements-completed: [RELY-02, EXPL-07]

duration: 3min
completed: 2026-02-22
---

# Phase 2 Plan 06: Gap Closure Summary

**Fixed OECD plot legend overlap, added 2 missing methodology notes to AOP Lifetime section, and cleaned all 28 methodology note limitations to remove performance/implementation jargon**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-22T10:16:45Z
- **Completed:** 2026-02-22T10:19:55Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- OECD completeness trend plot and entity completeness trends plot now have adequate spacing between title and horizontal legend (top margin increased, legend repositioned)
- All three AOP Lifetime sub-plot boxes (Created, Modified, Creation vs Modification) now have expandable methodology notes, bringing total methodology_note calls in trends.html to 19
- All 28 methodology note entries cleaned: removed query performance text, implementation details about SPARQL generation, and visual rendering details from limitations fields

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix OECD plot layout and add missing methodology notes** - `5c84ae2` (fix)
2. **Task 2: Clean up methodology note limitations across all entries** - `250a8f3` (fix)

## Files Created/Modified
- `plots/trends_plots.py` - Increased top margins and repositioned legends for OECD and entity completeness plots
- `templates/trends.html` - Added methodology_note macro calls to AOPs Modified and AOP Creation vs Modification plot boxes
- `static/data/methodology_notes.json` - Cleaned limitations text across 6 entries (removed performance and implementation details)

## Decisions Made
- Centered horizontal legends (xanchor="center", x=0.5) rather than right-aligned for better visual balance with longer legend items
- Varied the "measures presence only" limitation phrasing per entity type (KE: "presence of each property, not quality or accuracy", KER: "presence of each property, not the strength or quality of evidence") for more contextual relevance
- Removed developer-facing optimization notes from aop_completeness_boxplot data_source field ("reduced from 10 to 4 queries")

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 2 gap closure complete: all verification warnings addressed
- EXPL-07 requirement (every plot has methodology notes) fully satisfied on trends page
- RELY-02 requirement (methodology quality) satisfied with cleaned limitations
- Ready for Phase 3 (Network Analysis) or remaining Phase 2 plans

## Self-Check: PASSED

All files verified present, all commit hashes found in git log.

---
*Phase: 02-reliability-and-completeness*
*Completed: 2026-02-22*
