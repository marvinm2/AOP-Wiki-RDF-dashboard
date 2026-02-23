---
phase: 04-entity-exploration
plan: 02
subsystem: ui
tags: [flask, plotly, sparql, jinja2, javascript, entity-links]

# Dependency graph
requires:
  - phase: 04-entity-exploration
    provides: Raw data table toggle macro, URL state encoding, /api/plot-data endpoint
  - phase: 02-reliability-and-completeness
    provides: _plot_data_cache infrastructure, methodology notes JSON, brand colors
provides:
  - 5 new latest-data plot functions (bio level, taxonomy, OECD status, KE reuse, reuse distribution)
  - Generic /download/latest/<plot_name> route for any latest plot CSV/PNG/SVG export
  - EXPL-04 entity links via Plotly customdata + click-to-AOP-Wiki handler
  - Breakdown Analysis and KE Reuse Analysis sections on snapshot page
affects: [04-03, future-plots]

# Tech tracking
tech-stack:
  added: []
  patterns: [generic download route for latest plots, MutationObserver for Plotly click handler on lazy-loaded plots, SPARQL OPTIONAL+BIND pattern for unannotated entities]

key-files:
  created: []
  modified:
    - plots/latest_plots.py
    - plots/__init__.py
    - app.py
    - templates/latest.html
    - static/js/version-selector.js
    - static/data/methodology_notes.json

key-decisions:
  - "Added generic /download/latest/<plot_name> route instead of 5 individual routes to reduce route proliferation"
  - "Used MutationObserver on KE reuse plot container to attach Plotly click handler after lazy load completes"
  - "Used OPTIONAL+BIND pattern for bio level and OECD status queries to include unannotated entities as 'Not Annotated' or 'No Status'"
  - "KE reuse plot is the only entity-link candidate since bio level, taxonomy, and OECD status show aggregate groupings, not individual entities"
  - "Discrete bins (1,2,3,4,5,6-10,11+) for KE reuse distribution instead of raw histogram for readability"

patterns-established:
  - "Generic latest download route: /download/latest/<plot_name> with version-keyed cache fallback"
  - "MutationObserver pattern for attaching event handlers to lazy-loaded Plotly plots"
  - "SPARQL BIND(IF(BOUND(...))) pattern for labeling unannotated entities"

requirements-completed: [EXPL-04]

# Metrics
duration: 6min
completed: 2026-02-23
---

# Phase 4 Plan 2: Latest-Data Breakdown Plots and Entity Links Summary

**5 new breakdown plots (bio level, taxonomy, OECD status, KE reuse top-30, reuse distribution) with click-to-AOP-Wiki entity links on the KE reuse chart**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-23T09:30:14Z
- **Completed:** 2026-02-23T09:36:30Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Implemented 5 new SPARQL-powered plot functions covering biological level, taxonomic groups, OECD status breakdown, KE reuse ranking, and KE reuse distribution
- Added EXPL-04 entity links: clicking bars in the KE reuse chart opens the corresponding AOP-Wiki page via Plotly customdata + MutationObserver click handler
- Created generic /download/latest/<plot_name> route to serve CSV/PNG/SVG exports for any latest-data plot without individual route boilerplate
- Added two new sections (Breakdown Analysis, KE Reuse Analysis) to the snapshot page with full methodology notes, data table toggles, and download buttons

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement 5 new latest-data plot functions with SPARQL queries** - `68eeab9` (feat)
2. **Task 2: Register new plots in app.py, templates, version selector, and methodology notes** - `2dffa9e` (feat)

## Files Created/Modified
- `plots/latest_plots.py` - Added 5 new plot functions: plot_latest_ke_by_bio_level, plot_latest_taxonomic_groups, plot_latest_entity_by_oecd_status, plot_latest_ke_reuse (with wiki_url customdata), plot_latest_ke_reuse_distribution
- `plots/__init__.py` - Exported all 5 new functions in import block, __all__, and get_available_functions()
- `app.py` - Added imports, registered 5 plots in latest_plots_with_version dict, added generic /download/latest/<plot_name> route, updated bulk download categories
- `templates/latest.html` - Added Breakdown Analysis section (3 plot containers), KE Reuse Analysis section (2 plot containers), navigation buttons, and EXPL-04 click handler script
- `static/js/version-selector.js` - Added 5 new plot names to versionedPlots array
- `static/data/methodology_notes.json` - Added 5 new methodology entries with descriptions, data sources, limitations, and SPARQL queries

## Decisions Made
- Added generic /download/latest/<plot_name> route instead of 5 individual routes to reduce route proliferation; uses version-keyed cache lookup with fallback
- Used MutationObserver to attach Plotly click handler on KE reuse plot after lazy loading completes, since the plot div does not exist until fetched
- Used OPTIONAL+BIND pattern in SPARQL for bio level and OECD status to include unannotated entities rather than silently dropping them
- Only KE reuse bar chart gets entity links (EXPL-04) since it shows individual named entities; other plots show aggregate groupings where entity links do not apply
- Used discrete bins (1,2,3,4,5,6-10,11+) for KE reuse distribution instead of raw histogram for better readability

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed BRAND_COLORS key mismatch**
- **Found during:** Task 1
- **Issue:** Plan referenced `BRAND_COLORS['primary_dark']` but actual key in shared.py is `BRAND_COLORS['primary']`
- **Fix:** Changed all references to `BRAND_COLORS['primary']` in the 5 new plot functions
- **Files modified:** plots/latest_plots.py
- **Committed in:** 68eeab9 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor key name correction. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 5 new plots follow established patterns and are ready for version selector, CSV export, and raw data table toggle
- The generic /download/latest/<plot_name> route can serve any future latest-data plots without adding individual routes
- EXPL-04 entity links pattern (MutationObserver + Plotly customdata) is reusable for any future plots showing individual entities
- Ready for plan 04-03 to continue Phase 4 execution

## Self-Check: PASSED

All 6 modified files verified present. Both task commits (68eeab9, 2dffa9e) verified in git log.

---
*Phase: 04-entity-exploration*
*Completed: 2026-02-23*
