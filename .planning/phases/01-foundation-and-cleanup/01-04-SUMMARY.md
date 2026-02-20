---
phase: 01-foundation-and-cleanup
plan: 04
subsystem: api
tags: [flask, error-handling, api-validation, plotly, fallback]

# Dependency graph
requires:
  - phase: 01-foundation-and-cleanup
    provides: "Production hardening with safe_plot_execution fallback infrastructure (01-02)"
provides:
  - "API content validation returning success:false for empty/fallback plot HTML"
  - "Generalized safe_plot_execution tuple fallback supporting 3-element string tuples"
  - "Return type annotation on plot_aop_lifetime ensuring correct fallback tuple generation"
affects: [api, error-handling, lazy-loading]

# Tech tracking
tech-stack:
  added: []
  patterns: ["API content validation with sentinel detection before returning success:true"]

key-files:
  created: []
  modified:
    - app.py
    - plots/shared.py
    - plots/trends_plots.py

key-decisions:
  - "Generalized tuple fallback by counting str elements in annotation rather than hardcoding function names"
  - "Detect 'Data Unavailable' sentinel string in fallback HTML to distinguish from legitimate content"

patterns-established:
  - "API content validation: always check for empty strings and fallback sentinels before returning success:true"
  - "Return type annotations on all multi-return plot functions to enable correct fallback tuple generation"

requirements-completed: [RELY-03]

# Metrics
duration: 1min
completed: 2026-02-20
---

# Phase 1 Plan 4: Error Card Content Validation Summary

**API handler validates plot content (empty strings, fallback sentinels, callable exceptions) and returns success:false to trigger differentiated error cards with retry buttons**

## Performance

- **Duration:** 1 min
- **Started:** 2026-02-20T13:41:00Z
- **Completed:** 2026-02-20T13:42:05Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- API handler now returns success:false with descriptive error messages for all three failure modes: empty HTML, fallback sentinel, and callable exceptions
- safe_plot_execution generalized to produce correct fallback tuples for any N-element string tuple annotation (not just plot_main_graph)
- plot_aop_lifetime annotated with return type tuple[str, str, str] so safe_plot_execution generates 3-element fallback on failure

## Task Commits

Each task was committed atomically:

1. **Task 1: Add return type annotation to plot_aop_lifetime and fix API content validation** - `8bf067e` (fix)

**Plan metadata:** [pending] (docs: complete plan)

## Files Created/Modified
- `app.py` - Added content validation in plot_map API branch: try/except for callables, empty string check, "Data Unavailable" sentinel detection
- `plots/shared.py` - Generalized tuple fallback logic in safe_plot_execution to count str elements from annotation instead of hardcoding function names
- `plots/trends_plots.py` - Added return type annotation `-> tuple[str, str, str]` to plot_aop_lifetime

## Decisions Made
- Used str element counting from annotation string rather than hardcoding function names, making the fallback logic extensible for any future multi-return plot functions
- Added DataFrame check guard (`'DataFrame' not in type_str`) before the str-count branch to preserve existing plot_main_graph behavior

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 (Foundation and Cleanup) is now complete with all 4 plans executed
- Error card UI now fires for all plot failure modes, closing UAT gap from test 5
- Ready for Phase 2 (data quality) and Phase 3 (network analysis) which can run in parallel

## Self-Check: PASSED

- FOUND: app.py
- FOUND: plots/shared.py
- FOUND: plots/trends_plots.py
- FOUND: 8bf067e (task 1 commit)
- FOUND: 01-04-SUMMARY.md

---
*Phase: 01-foundation-and-cleanup*
*Completed: 2026-02-20*
