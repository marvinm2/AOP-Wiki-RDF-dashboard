---
phase: 01-foundation-and-cleanup
plan: 02
subsystem: infra
tags: [gunicorn, cache-eviction, json-logging, health-endpoint, error-cards, production-hardening]

# Dependency graph
requires:
  - phase: 01-01
    provides: Clean codebase with pinned dependencies and no dead imports
provides:
  - Gunicorn production configuration with preload, gthread workers, and Docker settings
  - VersionedPlotCache with TTL expiry, LRU eviction, and pinned latest version
  - Centralized structured JSON logging via python-json-logger
  - Honest health endpoint returning 503 when SPARQL is down
  - Differentiated error cards with retry buttons for timeout, unreachable, and generic errors
affects: [01-03, 02, 03, 06]

# Tech tracking
tech-stack:
  added: []
  patterns: [versioned-cache-with-pinning, centralized-json-logging, honest-health-checks, differentiated-error-cards]

key-files:
  created:
    - gunicorn.conf.py
  modified:
    - app.py
    - config.py
    - plots/shared.py
    - plots/__init__.py
    - Dockerfile
    - static/js/lazy-loading.js
    - static/css/lazy-loading.css

key-decisions:
  - "VersionedPlotCache uses OrderedDict with TTL=1800s and max_versions=5 for bounded memory growth"
  - "Health endpoint returns 503 (not 200 with degraded) when SPARQL is down, per user decision for honest reporting"
  - "JSON logging configured centrally in config.py, called before any other imports in app.py"
  - "Error cards use unicode symbols instead of emoji per project conventions"

patterns-established:
  - "VersionedPlotCache: dict-like interface with __getitem__/__setitem__/__contains__ for backward compatibility"
  - "Cache pinning: call pin_version() at startup after determining latest version"
  - "Centralized logging: configure_logging() in config.py, imported and called first in app.py"

requirements-completed: [INFR-03, INFR-04, RELY-03]

# Metrics
duration: 8min
completed: 2026-02-20
---

# Phase 1 Plan 2: Production Hardening Summary

**Gunicorn with preloaded gthread workers, VersionedPlotCache replacing unbounded dicts (TTL+LRU+pinning), centralized JSON logging, honest 503 health endpoint, and differentiated error cards**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-20T12:12:05Z
- **Completed:** 2026-02-20T12:20:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Replaced plain dict caches with VersionedPlotCache class providing TTL expiry (30 min), max version cap (5), and pinned latest version that is never evicted
- Created gunicorn.conf.py with preload_app=True (shared startup computation via COW fork), gthread workers, 120s timeout, and /dev/shm heartbeat
- Updated Dockerfile CMD from `flask run` to `gunicorn -c gunicorn.conf.py app:app`
- Centralized structured JSON logging in config.py using python-json-logger, removing duplicate basicConfig calls
- Fixed health endpoint to return HTTP 503 with "unhealthy" status when SPARQL endpoint is down
- Enhanced error cards with differentiated messages for timeout, unreachable, and generic errors with styled retry buttons
- Wired cache pinning at startup so latest version data is never evicted

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement VersionedPlotCache, JSON logging, and Gunicorn configuration** - `ef37a4b` (feat)
2. **Task 2: Fix health endpoint, enhance error cards, and wire cache pinning** - `c23d5df` (feat)

**Plan metadata:** (pending)

## Files Created/Modified
- `gunicorn.conf.py` - Production Gunicorn config with preload_app, gthread, 2 workers, 120s timeout
- `plots/shared.py` - VersionedPlotCache class replacing _plot_data_cache and _plot_figure_cache dicts
- `plots/__init__.py` - Export VersionedPlotCache class
- `config.py` - Added configure_logging() function with python-json-logger JsonFormatter
- `app.py` - Removed basicConfig, added configure_logging() call, fixed health endpoint to 503, wired cache pinning
- `Dockerfile` - Changed CMD from flask run to gunicorn, removed FLASK_APP env var
- `static/js/lazy-loading.js` - Enhanced showErrorState() with timeout/unreachable/generic differentiation
- `static/css/lazy-loading.css` - Styled error-icon, error-title, error-suggestion, error-retry-btn classes

## Decisions Made
- VersionedPlotCache uses OrderedDict with TTL=1800s (30 min) and max_versions=5 -- based on research estimating ~15 MB per version, keeping worst case under 75 MB
- Health endpoint returns 503 when SPARQL is completely down, per user's explicit "honest reporting" preference even if it triggers container restarts
- JSON logging configured before any other imports in app.py to ensure all modules use the JSON formatter from the start
- Error cards use unicode text symbols (stopwatch, warning sign) instead of emoji per project conventions (no emoji unless requested)
- Retry button styled with brand navigation color (#307BBF) for consistency with VHP4Safety design

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing runtime error: `aop_lifetime` plot returns more values than expected during unpacking (ValueError in app.py line 239). This is NOT caused by our changes and existed before plan 01-01. Logged in 01-01 summary. Out of scope.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Application is production-ready with Gunicorn serving, bounded caches, and honest health reporting
- Developer documentation (plan 01-03) can proceed -- all infrastructure changes are in place
- Phase 2 and Phase 3 can build on top of this stable foundation

## Self-Check: PASSED

- All 8 files verified present on disk
- Commit ef37a4b (Task 1) found in git log
- Commit c23d5df (Task 2) found in git log

---
*Phase: 01-foundation-and-cleanup*
*Completed: 2026-02-20*
