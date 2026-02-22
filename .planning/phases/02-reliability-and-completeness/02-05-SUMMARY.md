---
phase: 02-reliability-and-completeness
plan: 05
subsystem: sparql-optimization
tags: [sparql, virtuoso, parallel-queries, threadpoolexecutor, boxplot, ke-components]

# Dependency graph
requires:
  - phase: 02-01
    provides: Per-version parallel SPARQL query pattern with ThreadPoolExecutor
provides:
  - Working KE component annotation trend plots (3 functions)
  - Working latest KE component pie chart
  - Working latest ontology usage chart
  - Complete boxplot with all 30 versions (2018-2025)
affects: [trends-page, latest-page, csv-export]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-version parallel SPARQL queries for any triple-OPTIONAL pattern that exceeds Virtuoso limits"
    - "Direct GRAPH <uri> targeting instead of GRAPH ?graph + FILTER for latest-data queries"

key-files:
  created: []
  modified:
    - plots/trends_plots.py
    - plots/latest_plots.py

key-decisions:
  - "Root cause was Virtuoso 400s execution time limit on triple-OPTIONAL cross-product, not wrong predicates"
  - "hasBiologicalEvent intermediate pattern IS correct (27568 results), flat predicates return 0"
  - "Fixed by per-version parallel queries (ThreadPoolExecutor, 4 workers) matching 02-01 established pattern"
  - "Latest KE component and ontology usage queries switched to direct GRAPH <uri> for single-version reliability"

patterns-established:
  - "Per-version parallel queries: mandatory for any SPARQL with 2+ OPTIONALs spanning all graphs"
  - "Direct graph targeting: prefer GRAPH <uri> over GRAPH ?graph + FILTER for latest-data plots"

requirements-completed: [RELY-02]

# Metrics
duration: 9min
completed: 2026-02-22
---

# Phase 2 Plan 5: Gap Closure - KE Components and Boxplot Summary

**Per-version parallel SPARQL queries fix 4 broken KE component plots and restore full 30-version boxplot data**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-22T10:16:42Z
- **Completed:** 2026-02-22T10:25:49Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Diagnosed root cause: Virtuoso estimated execution time (864s) exceeds 400s limit when triple OPTIONALs create cross-product across all 30 graphs
- Fixed 3 trend plot functions (plot_ke_components, plot_ke_components_percentage, plot_unique_ke_components) using per-version parallel queries
- Fixed 2 latest plot functions (plot_latest_ke_components, plot_latest_ontology_usage) using direct graph URI targeting
- Restored boxplot from ~7 pre-2020 versions to all 30 versions (2018-04-01 through 2025-07-01) in ~8 seconds

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix KE component annotation SPARQL queries** - `cb3ff28` (fix)
2. **Task 2: Split boxplot query to avoid Virtuoso MaxResultRows truncation** - `09400b0` (fix)

## Files Created/Modified
- `plots/trends_plots.py` - Added _query_ke_components_version() helper, _query_ke_components_percentage_version() helper, _query_boxplot_version() and _query_boxplot_entity_props() helpers; refactored plot_ke_components, plot_ke_components_percentage, plot_unique_ke_components, and plot_aop_completeness_boxplot to use per-version parallel queries
- `plots/latest_plots.py` - Refactored plot_latest_ke_components and plot_latest_ontology_usage to use direct GRAPH <uri> targeting instead of GRAPH ?graph with FILTER

## Decisions Made
- Root cause was NOT wrong predicates (the plan's initial hypothesis). The `aopo:hasBiologicalEvent` intermediate pattern with `hasProcess/hasObject/hasAction` sub-predicates is correct and returns data. The real issue is that 3 OPTIONALs create a cross-product that exceeds Virtuoso's 400-second execution time estimate when scanning all graphs.
- Confirmed predicates: `hasBiologicalEvent` returns 27,568 results; flat predicates (`has_biological_process`) return 0. The intermediate node pattern is the correct RDF structure.
- Used same ThreadPoolExecutor(max_workers=4) pattern established in 02-01 for consistency.
- For latest-data plots, switched to direct `GRAPH <uri>` targeting because even single-version queries with `GRAPH ?graph + FILTER + ORDER BY DESC LIMIT 1` hit the Virtuoso execution time estimator.
- Added shared `_query_ke_components_version()` with `use_distinct` parameter to serve both `plot_ke_components()` (total counts) and `plot_unique_ke_components()` (distinct counts).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed plan's incorrect predicate diagnosis**
- **Found during:** Task 1 (diagnostic phase)
- **Issue:** Plan assumed predicates might be wrong (flat vs intermediate). Diagnostic showed intermediate pattern IS correct; the issue is Virtuoso execution time limits on triple-OPTIONAL cross-product.
- **Fix:** Applied per-version parallel query strategy instead of predicate replacement
- **Files modified:** plots/trends_plots.py, plots/latest_plots.py
- **Verification:** All KE component functions return chart data, not fallback error cards
- **Committed in:** cb3ff28 (Task 1 commit)

**2. [Rule 2 - Missing Critical] Updated plot_latest_ontology_usage to direct graph targeting**
- **Found during:** Task 1
- **Issue:** While fixing KE components, identified that plot_latest_ontology_usage used same fragile GRAPH ?graph + FILTER pattern. Currently works due to UNION (not OPTIONAL) but could break with data growth.
- **Fix:** Switched to direct GRAPH <uri> targeting with version detection query
- **Files modified:** plots/latest_plots.py
- **Verification:** Ontology usage chart renders correctly
- **Committed in:** cb3ff28 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 bug diagnosis correction, 1 preventive fix)
**Impact on plan:** Diagnosis deviated from plan's hypothesis but achieved the same goal (working KE component plots). The preventive ontology fix was minimal scope addition for robustness.

## Issues Encountered
- The cross-graph triple-OPTIONAL pattern `GRAPH ?graph { ... OPTIONAL { ?bioevent aopo:hasProcess ?process } OPTIONAL { ?bioevent aopo:hasObject ?object } OPTIONAL { ?bioevent aopo:hasAction ?action } }` causes Virtuoso to estimate 864 seconds execution time (exceeds 400s limit). This estimate is applied BEFORE execution, so even LIMIT 1 doesn't help.
- The boxplot had the same cross-graph issue with its UNION queries scanning all entity types across all versions, silently truncating results.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All KE component plots (3 trends + 1 latest) now render chart data
- Boxplot includes all 30 versions
- Ready for plan 02-06 (remaining gap closure items)
- Verification items 2 and 4 from Phase 2 are resolved

## Self-Check: PASSED

- All modified files exist on disk
- Both task commits verified (cb3ff28, 09400b0)
- KE component functions return chart data (verified via Python import tests)
- Boxplot includes all 30 versions including post-2020 (verified via cache inspection)

---
*Phase: 02-reliability-and-completeness*
*Completed: 2026-02-22*
