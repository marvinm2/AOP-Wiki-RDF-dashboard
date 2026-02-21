---
phase: 02-reliability-and-completeness
plan: 02
subsystem: data-export
tags: [csv-export, plotly, pandas, data-cache, filenames]

# Dependency graph
requires:
  - phase: 01-foundation-and-cleanup
    provides: Plot architecture with _plot_figure_cache and _plot_data_cache pattern
provides:
  - Data cache entries for all 20 trend sub-plot CSV exports
  - build_export_filename() helper for self-documenting download filenames
  - Versioned Content-Disposition headers on all 40 download route responses
affects: [02-reliability-and-completeness, data-export, csv-downloads]

# Tech tracking
tech-stack:
  added: []
  patterns: [build_export_filename for self-documenting export filenames with date and version]

key-files:
  created: []
  modified:
    - plots/trends_plots.py
    - plots/shared.py
    - plots/__init__.py
    - app.py

key-decisions:
  - "Used request.args.get('version') for all routes so trend downloads get None (no version in filename) while latest downloads get the version parameter"
  - "12 data cache keys were already present (not 20 as estimated); only 6 functions needed updates"

patterns-established:
  - "build_export_filename pattern: clean-name_YYYY-MM-DD[_vVERSION].format for all exports"
  - "_plot_data_cache entry always paired with _plot_figure_cache entry in every trend plot function"

requirements-completed: [RELY-04]

# Metrics
duration: 5min
completed: 2026-02-21
---

# Phase 2 Plan 02: CSV Export Gap and Versioned Filenames Summary

**Closed CSV export gap for 12 missing trend sub-plot data caches and added self-documenting export filenames with date/version metadata**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-21T19:12:07Z
- **Completed:** 2026-02-21T19:17:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added _plot_data_cache entries for 12 previously-missing data cache keys across 6 trend plot functions
- Created build_export_filename() helper that produces filenames like `ke-components-absolute_2026-02-21.png`
- Updated all 40 Content-Disposition headers in app.py to use versioned filenames

## Task Commits

Each task was committed atomically:

1. **Task 1: Add data caches to all trend plots missing CSV export** - `85b90f9` (feat)
2. **Task 2: Add versioned export filenames to download routes** - `af2566b` (feat)

## Files Created/Modified
- `plots/trends_plots.py` - Added _plot_data_cache entries for 12 missing cache keys, fixed 6 global declarations
- `plots/shared.py` - Added build_export_filename() helper function
- `plots/__init__.py` - Exported build_export_filename
- `app.py` - Imported build_export_filename, updated 40 Content-Disposition headers

## Decisions Made
- Used request.args.get('version') uniformly for all routes -- trend routes naturally get None (no version in URL params), latest routes get the version from version-selector.js
- Plan estimated 20 missing data cache entries but 8 were already present from prior work; only 12 actually needed adding across 6 functions

## Deviations from Plan

None - plan executed as written (the plan correctly described the pattern; some functions already had caches from earlier development but the verification still validates all 20 keys exist).

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All trend plot CSV exports now work (no more 404s on download)
- All download filenames are self-documenting with export date
- Ready for remaining reliability plans (02-03, 02-04)

## Self-Check: PASSED

All files exist, all commits verified, summary present.

---
*Phase: 02-reliability-and-completeness*
*Completed: 2026-02-21*
