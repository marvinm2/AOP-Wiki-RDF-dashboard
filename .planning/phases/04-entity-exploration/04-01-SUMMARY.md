---
phase: 04-entity-exploration
plan: 01
subsystem: ui
tags: [flask, jinja2, javascript, css, api, data-tables, url-state]

# Dependency graph
requires:
  - phase: 02-reliability-and-completeness
    provides: Jinja2 macro pattern, methodology notes, _plot_data_cache infrastructure
provides:
  - /api/plot-data/<plot_name> endpoint for raw data as JSON
  - data_table_toggle Jinja2 macro for reusable toggle buttons
  - raw-data-tables.js toggle component with on-demand fetch
  - Shareable URL state via ?version= parameter
  - CSS styling for raw data tables matching VHP4Safety brand
affects: [04-02, 04-03, future plots]

# Tech tracking
tech-stack:
  added: []
  patterns: [data-table-toggle macro pattern, URL state encoding with URLSearchParams, version-changed custom event for cross-component communication]

key-files:
  created:
    - static/js/raw-data-tables.js
    - templates/macros/data_table.html
  modified:
    - app.py
    - static/css/main.css
    - templates/latest.html
    - templates/trends.html
    - templates/trends_page.html
    - static/js/version-selector.js

key-decisions:
  - "Used absolute plot name as data source for dual-view (abs/delta) plot-boxes since underlying data is same"
  - "Data table toggle placed after lazy-plot div inside each plot-box for consistent layout"
  - "URL state uses history.replaceState to avoid creating browser history entries on version change"
  - "version-changed custom event pattern for decoupled communication between version-selector and raw-data-tables"

patterns-established:
  - "data_table_toggle macro: reusable Jinja2 macro for adding raw data inspection to any plot"
  - "version-changed event: custom DOM event dispatched when version changes, listened by data table cache invalidation"
  - "URL state encoding: ?version= parameter for shareable versioned views"

requirements-completed: [EXPL-05, EXPL-06]

# Metrics
duration: 6min
completed: 2026-02-23
---

# Phase 4 Plan 1: Raw Data Tables and Shareable URLs Summary

**Raw data table toggle infrastructure on all 31 plots with /api/plot-data endpoint and shareable ?version= URL state encoding**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-23T09:21:20Z
- **Completed:** 2026-02-23T09:27:31Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Created /api/plot-data/<plot_name> API endpoint serving cached DataFrame as JSON with 100-row truncation
- Applied "Show Raw Data" toggle button to all 11 latest.html and 20 trends.html plot containers
- Implemented shareable URL state encoding that persists and restores version selection via ?version= parameter
- Added version-changed custom event for cross-component cache invalidation between version selector and data tables

## Task Commits

Each task was committed atomically:

1. **Task 1: Create raw data table API endpoint, JS component, and CSS** - `bde56c6` (feat)
2. **Task 2: Apply data table toggles to all existing plots and add shareable URLs** - `f8ebd89` (feat)

## Files Created/Modified
- `app.py` - Added /api/plot-data/<plot_name> endpoint with version-aware cache lookup and NaN handling
- `static/js/raw-data-tables.js` - Toggle component with on-demand fetch, HTML table builder, and version-changed listener
- `templates/macros/data_table.html` - Reusable Jinja2 macro with button and content container
- `static/css/main.css` - VHP4Safety-branded data table styles with sticky headers and hover effects
- `templates/latest.html` - Added macro import, 11 data_table_toggle calls, and script tag
- `templates/trends.html` - Added 20 data_table_toggle calls for all trend plot-boxes
- `templates/trends_page.html` - Added macro import and raw-data-tables.js script tag
- `static/js/version-selector.js` - Added updateURLState(), restoreVersionFromURL(), and version-changed event dispatch

## Decisions Made
- Used absolute plot name as data source for dual-view (abs/delta) plot-boxes since the underlying data is the same
- Placed data table toggle after the lazy-plot div inside each plot-box for consistent layout
- Used history.replaceState() for URL updates to avoid polluting browser history with every version change
- Used version-changed custom DOM event pattern for decoupled communication between version-selector.js and raw-data-tables.js

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Raw data table infrastructure is in place for all existing and future plots
- The data_table_toggle macro can be added to any new plot container following the established pattern
- URL state encoding works on both snapshot and trends pages
- Ready for plans 04-02 and 04-03 to build on this foundation

## Self-Check: PASSED

All 8 files verified present. Both task commits (bde56c6, f8ebd89) verified in git log.

---
*Phase: 04-entity-exploration*
*Completed: 2026-02-23*
