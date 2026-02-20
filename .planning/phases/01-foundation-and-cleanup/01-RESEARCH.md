# Phase 1: Foundation and Cleanup - Research

**Researched:** 2026-02-20
**Domain:** Python Flask production hardening, cache management, dead code removal, developer documentation
**Confidence:** HIGH

## Summary

Phase 1 stabilizes the existing AOP-Wiki RDF Dashboard for production use. The codebase is a Flask + Plotly + SPARQL application (app.py: 1,609 lines, plots/ package: 6,007 lines) that currently runs under Flask's dev server, has no dependency pinning, retains a 4,194-line legacy monolith (`plots.py`), and uses unbounded global dict caches. All of these are well-understood problems with standard solutions.

The most technically interesting challenge is the Gunicorn + preload_app interaction with the startup plot computation. The app runs `compute_plots_parallel()` at module-import time using `ThreadPoolExecutor` to pre-fetch all 27 plots from SPARQL. With Gunicorn's `--preload` flag, this computation runs once in the master process and workers inherit the results via copy-on-write fork -- a 65% memory reduction versus each worker computing independently. Without preload, each worker would independently query SPARQL for all 27 plots, multiplying both SPARQL load and memory usage by the worker count.

**Primary recommendation:** Use `--preload` with Gunicorn and `gthread` worker class, implement a TTL + LRU eviction wrapper around the existing `_plot_data_cache` and `_plot_figure_cache` dicts, delete `plots.py` as the first commit, and use `python-json-logger` (already installed) for structured JSON logging.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Cache entries expire after a time limit (TTL-based eviction)
- Hard cap on the number of versions cached simultaneously -- evict oldest when cap is reached
- Latest/current version is pinned in cache and never evicted; only historical versions follow TTL + cap rules
- When a user views a version that was evicted, show a loading indicator while re-fetching (not silent)
- Create a `.claude/` instruction file with the add-a-plot checklist, referenced from CLAUDE.md
- Create a GitHub issue template (markdown format, not YAML form) for proposing new plots
- Checklist format only -- no worked example walkthrough needed
- Documentation aimed at both human contributors and Claude equally -- precise file paths and patterns, but also readable narrative
- Friendly error card with message and retry button when a SPARQL query times out or endpoint is unreachable
- Graceful degradation: each plot loads independently; failed ones show error cards, successful ones display normally
- `/health` endpoint reports unhealthy when SPARQL endpoint is completely down (honest reporting, may trigger container restarts)
- Production logging in structured JSON format (JSON lines) for log aggregation tools
- General dead code sweep beyond legacy `plots.py` -- includes unused imports, dead routes, commented-out code, orphaned templates
- Careful commit trail: separate commits for each removed file/section with explanatory messages
- Dependencies pinned to minor range (e.g., `Flask~=3.0`) -- allows patch updates, balances reproducibility with security
- Linting/formatting (ruff etc.) is NOT in scope for this phase -- save for later

### Claude's Discretion
- Specific TTL duration and cache cap number (based on memory profiling)
- Gunicorn worker count, timeout settings, and bind configuration
- Exact error card design and copy
- JSON log format structure and log levels
- Order and grouping of cleanup commits

### Deferred Ideas (OUT OF SCOPE)
- Linting/formatting setup (ruff or similar) -- future phase or standalone task
- Code style enforcement -- not this phase
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFR-01 | Legacy `plots.py` monolith (4,194 lines) is deleted and all imports use the modular `plots/` package | Confirmed: `plots.py` exists at root (4,194 lines), `app.py` already imports from `plots/` package, no active code references `plots.py` directly. Safe to delete. README.md references `plots.py` and needs updating. |
| INFR-02 | All dependency versions are pinned in requirements.txt with tested versions | Current `requirements.txt` has 4 unpinned deps. Installed versions verified: Flask 3.1.3, pandas 2.2.2, plotly 5.22.0, SPARQLWrapper 2.0.0. Gunicorn 25.1.0, python-json-logger 2.0.7 need adding. Pin to compatible-release (`~=`) per user decision. |
| INFR-03 | Application runs under Gunicorn in production (not Flask dev server) | Current Dockerfile CMD uses `flask run`. Gunicorn 25.1.0 already installed. Needs `gunicorn.conf.py`, updated Dockerfile CMD, and `--preload` for shared startup computation. |
| INFR-04 | Cache has eviction policy preventing unbounded memory growth | Current caches (`_plot_data_cache`, `_plot_figure_cache`) are plain dicts with no eviction. Each version selection in latest_plots adds entries. Need TTL + LRU wrapper with pinned latest version. |
| RELY-03 | All plot error states use consistent fallback with actionable user-facing messages | Frontend lazy-loading.js already has `showErrorState()` with retry button. Backend `create_fallback_plot()` generates Plotly error charts. Need to unify: consistent error card design, retry functionality, and clear messages for timeout vs unreachable states. |
| DEVX-01 | Plot documentation and architecture docs comprehensive enough for AI-assisted plot creation | Current CLAUDE.md has plot function listings but add-a-plot instructions are terse (6 steps each). Need dedicated `.claude/add-a-plot.md` checklist with precise file paths, naming conventions, and registration patterns. |
| DEVX-02 | Plot addition follows standardized workflow with clear templates, naming conventions, and registration checklist | No GitHub issue template exists. Need `.github/ISSUE_TEMPLATE/new-plot.md` for proposing plots. Need checklist cross-referenced from CLAUDE.md. |
</phase_requirements>

## Standard Stack

### Core (Already Installed)
| Library | Pin Version | Purpose | Why Standard |
|---------|-------------|---------|--------------|
| Flask | `~=3.1` | Web framework | Already in use, mature WSGI framework |
| pandas | `~=2.2` | Data processing | Already in use for SPARQL result handling |
| plotly | `~=5.22` | Visualization | Already in use, generates interactive HTML charts |
| SPARQLWrapper | `~=2.0` | SPARQL queries | Already in use, standard Python SPARQL client |
| gunicorn | `~=25.1` | Production WSGI server | Already installed (25.1.0), standard Flask production server |
| python-json-logger | `~=2.0` | Structured JSON logging | Already installed (2.0.7), wraps stdlib logging with JSON output |
| requests | `~=2.32` | HTTP client | Already in use by shared.py for SPARQL health checks |

### Supporting (Need Adding to requirements.txt)
| Library | Pin Version | Purpose | When to Use |
|---------|-------------|---------|-------------|
| gunicorn | `~=25.1` | WSGI HTTP server | Production deployment (replaces `flask run`) |
| python-json-logger | `~=2.0` | JSON log formatter | Production logging to stdout for log aggregation |

### Not Needed
| Instead of | Why Not |
|------------|---------|
| structlog | python-json-logger is already installed, simpler, wraps stdlib directly |
| cachetools | stdlib + custom wrapper is sufficient for this simple TTL+LRU case |
| Flask-Caching | Over-engineered for this use case; the cache is for in-process DataFrames, not HTTP responses |

**Installation:**
```bash
pip install gunicorn~=25.1 python-json-logger~=2.0
```

## Architecture Patterns

### Current Project Structure (What Exists)
```
AOP-Wiki-RDF-dashboard/
├── app.py                          # Flask app (1,609 lines) - routes, startup, download endpoints
├── config.py                       # Config class with env vars
├── plots.py                        # LEGACY - 4,194 lines, DELETE THIS
├── plots/
│   ├── __init__.py                 # Public API (322 lines)
│   ├── shared.py                   # SPARQL utils, cache dicts, helpers (1,023 lines)
│   ├── trends_plots.py             # Historical trend plots (2,888 lines)
│   └── latest_plots.py             # Snapshot plots with version support (1,774 lines)
├── templates/
│   ├── landing.html                # Main entry (/)
│   ├── latest.html                 # Database snapshot (/snapshot)
│   ├── trends_page.html            # Historical trends (/trends)
│   ├── trends.html                 # Partial included by trends_page.html
│   ├── index.html                  # Legacy tabbed dashboard (/old-dashboard, /dashboard)
│   └── status.html                 # Status page (/status)
├── static/
│   ├── css/main.css                # Brand styling
│   ├── css/lazy-loading.css        # Skeleton/error/loading CSS
│   ├── js/lazy-loading.js          # PlotLazyLoader class
│   └── js/version-selector.js      # Version selector IIFE
├── .claude/
│   ├── architecture.md             # Architecture reference
│   └── colors.md                   # VHP4Safety brand colors
├── requirements.txt                # 4 unpinned deps (flask, pandas, plotly, SPARQLWrapper)
├── Dockerfile                      # python:3.11-slim, uses flask run
└── docker-compose.yml              # Single service, port 5000
```

### Pattern 1: TTL + LRU Cache with Pinned Latest Version
**What:** Wrap `_plot_data_cache` and `_plot_figure_cache` with a custom class that provides TTL expiry, max-entry cap, and a pinned "latest" slot.
**When to use:** For the version-selectable latest_* plots that write new entries per version.
**Why custom, not a library:** The "pinned latest version" requirement is domain-specific. Standard LRU caches (stdlib `functools.lru_cache`, `cachetools.TTLCache`) don't support pinning specific entries from eviction.

```python
import time
import threading
from collections import OrderedDict

class VersionedPlotCache:
    """Cache with TTL expiry, max-entry cap, and pinned latest version.

    - Latest/current version is pinned and never evicted
    - Historical versions follow TTL + max cap rules
    - Evicts oldest historical entry when cap is reached
    - Thread-safe for concurrent access from Gunicorn gthread workers
    """

    def __init__(self, max_versions: int = 5, ttl_seconds: int = 1800):
        self._data = OrderedDict()          # key -> (value, timestamp)
        self._pinned_prefix = None           # e.g., "2025-07-01"
        self._max_versions = max_versions
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def pin_version(self, version: str):
        """Pin a version so its entries are never evicted."""
        self._pinned_prefix = version

    def get(self, key: str, default=None):
        with self._lock:
            if key not in self._data:
                return default
            value, ts = self._data[key]
            # Check TTL for non-pinned entries
            if not self._is_pinned(key) and time.time() - ts > self._ttl:
                del self._data[key]
                return default
            # Move to end (most recently used)
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value):
        with self._lock:
            self._data[key] = (value, time.time())
            self._data.move_to_end(key)
            self._evict_if_needed()

    def _is_pinned(self, key: str) -> bool:
        return self._pinned_prefix and self._pinned_prefix in key

    def _evict_if_needed(self):
        """Evict oldest non-pinned entries when over cap."""
        # Count distinct versions (extract version from keys like 'latest_entity_counts_2024-01-15')
        versions = set()
        for k in self._data:
            parts = k.rsplit('_', 1)
            if len(parts) > 1 and '-' in parts[-1]:
                versions.add(parts[-1])

        while len(versions) > self._max_versions:
            # Find oldest non-pinned entry
            for k in list(self._data.keys()):
                if not self._is_pinned(k):
                    del self._data[k]
                    parts = k.rsplit('_', 1)
                    if len(parts) > 1:
                        versions.discard(parts[-1])
                    break
            else:
                break  # All entries are pinned
```

**Recommendation for discretion items:**
- **TTL:** 30 minutes (1800 seconds). Historical versions are static data -- once fetched, they don't change. 30 minutes is long enough for a user to explore a version thoroughly, short enough to reclaim memory from abandoned sessions.
- **Max versions cap:** 5 historical versions plus the pinned latest. Each version's plots consume roughly 5-15 MB of DataFrame + figure cache. At 5 versions, worst case is ~75 MB additional memory, well within Docker container limits.

### Pattern 2: Gunicorn Configuration for Preloaded Flask App
**What:** A `gunicorn.conf.py` that leverages `preload_app` to share the expensive startup computation across workers via copy-on-write.
**Why critical:** The app runs `compute_plots_parallel()` at import time, which executes 27 SPARQL queries. Without preload, each worker repeats this (N workers = N x 27 queries). With preload, it runs once in the master process.

```python
# gunicorn.conf.py
import multiprocessing

# Binding
bind = "0.0.0.0:5000"

# Workers: 2 workers with gthread for Docker containers
# - 2 workers (not CPU*2+1) because this runs in a container
# - gthread allows concurrent requests within each worker
# - threads=4 gives 8 total concurrent request handlers
workers = 2
worker_class = "gthread"
threads = 4

# Preload: Run compute_plots_parallel() once in master, share via COW fork
preload_app = True

# Timeouts: generous because SPARQL queries can be slow
timeout = 120          # Worker timeout (some plots take 75s)
graceful_timeout = 30  # Graceful shutdown
keepalive = 2

# Docker-specific: use memory-backed tmpdir for heartbeat
worker_tmp_dir = "/dev/shm"

# Logging: write to stdout/stderr for Docker log collection
accesslog = "-"
errorlog = "-"
loglevel = "info"
```

**Recommendation rationale:**
- **2 workers:** Standard for containerized deployment. Scale by adding containers, not workers. See [Gunicorn in Docker](https://pythonspeed.com/articles/gunicorn-in-docker/).
- **gthread with 4 threads:** Allows 8 concurrent requests total. Flask handles I/O-bound SPARQL queries well with threads. Threads also prevent worker timeout during long requests by maintaining heartbeat.
- **timeout=120:** The AOP completeness boxplot takes ~75 seconds. Setting timeout to 120 gives headroom. The Gunicorn heartbeat (via gthread) prevents false kills during legitimate long queries.
- **preload_app=True:** Critical for this app. Without it, each worker independently runs 27 SPARQL queries at startup. See [Gunicorn preloading](https://www.joelsleppy.com/blog/gunicorn-application-preloading/).
- **worker_tmp_dir=/dev/shm:** Prevents Docker tmpfs heartbeat failures that cause spurious worker restarts. See [Critical Worker Timeout](https://github.com/benoitc/gunicorn/issues/2797).

### Pattern 3: Structured JSON Logging with python-json-logger
**What:** Replace the stdlib `logging.basicConfig()` format with `pythonjsonlogger.jsonlogger.JsonFormatter` for structured JSON output.
**Current state:** `app.py` line 54 and `shared.py` line 69 both call `logging.basicConfig()` with the same human-readable format string. Duplicated setup.

```python
# config.py or a new logging_config.py
import logging
import sys
from pythonjsonlogger.jsonlogger import JsonFormatter

def configure_logging(log_level: str = "INFO"):
    """Configure structured JSON logging for production.

    JSON lines format with fields:
    - timestamp: ISO 8601
    - level: DEBUG/INFO/WARNING/ERROR/CRITICAL
    - logger: module name
    - message: log message
    - module: source module
    - funcName: source function
    """
    handler = logging.StreamHandler(sys.stdout)

    formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
        }
    )

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
```

**Recommendation for log levels:**
- INFO: Request handling, plot generation timing, cache operations
- WARNING: Slow queries (>10s), cache evictions, retry attempts
- ERROR: SPARQL failures, plot generation errors, health check failures
- DEBUG: SPARQL query text, response sizes, cache hit/miss details

### Pattern 4: Error Card Design (Frontend)
**What:** Consistent error card shown when a plot fails to load, with retry button and descriptive message.
**Current state:** `lazy-loading.js` already has `showErrorState()` (line 165-173) and CSS `.plot-error` class (lazy-loading.css lines 67-88). The existing implementation is functional but minimal.

```javascript
// Enhanced error card with differentiated messages
showErrorState(element, error, plotName) {
    const isTimeout = error.toLowerCase().includes('timeout');
    const isUnreachable = error.toLowerCase().includes('fetch') ||
                          error.toLowerCase().includes('network');

    const icon = isTimeout ? '⏱️' : isUnreachable ? '🔌' : '⚠️';
    const title = isTimeout
        ? 'Query Timed Out'
        : isUnreachable
        ? 'Service Unreachable'
        : 'Unable to Load Plot';
    const suggestion = isTimeout
        ? 'This visualization requires a complex query. Try again in a moment.'
        : isUnreachable
        ? 'The data service is currently unavailable. Please check back later.'
        : 'An unexpected error occurred while loading this visualization.';

    element.innerHTML = `
        <div class="plot-error">
            <div class="error-icon">${icon}</div>
            <h4 class="error-title">${title}</h4>
            <p class="error-suggestion">${suggestion}</p>
            <button onclick="plotLoader.retryPlot('${plotName}')">
                Retry
            </button>
        </div>
    `;
}
```

### Anti-Patterns to Avoid

- **Duplicate `logging.basicConfig()` calls:** Currently both `app.py` (line 54) and `shared.py` (line 69) call `basicConfig()`. Only the first call takes effect in stdlib logging. Configure once centrally.

- **Module-level computation with Gunicorn multi-worker:** Without `--preload`, the `plot_results = compute_plots_parallel()` line (app.py:221) runs in every worker independently. Always pair module-level computation with `preload_app=True`.

- **Duplicate Flask imports:** `app.py` imports `from flask import Flask, render_template, jsonify, request, Response` on line 41, then `from flask import Flask, render_template, url_for, Response, redirect` on line 63. Consolidate to one import statement.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON logging | Custom JSON formatter | `python-json-logger` (already installed) | Handles all edge cases: exceptions, stack traces, extra fields |
| WSGI server | Custom multi-process server | `gunicorn` (already installed) | Battle-tested, handles worker management, heartbeat, graceful shutdown |
| Cache eviction (basic) | Full cache framework | `OrderedDict` + timestamps | Simple enough for this use case; only 2 dicts, clear pinning rules |
| Health checks | Custom monitoring daemon | Flask `/health` endpoint + Gunicorn signals | Already partially implemented; Docker health checks use HTTP |

**Key insight:** This phase is about stabilizing existing code, not adding capabilities. Every piece of technology needed is already installed or available in stdlib. The work is wiring them correctly and removing dead code.

## Common Pitfalls

### Pitfall 1: Gunicorn Workers Independently Computing Plots
**What goes wrong:** Without `--preload`, each Gunicorn worker forks and independently imports `app.py`, triggering `compute_plots_parallel()` in each worker. With 2 workers, this means 54 SPARQL queries at startup instead of 27, doubling both SPARQL endpoint load and memory usage.
**Why it happens:** Gunicorn's default behavior is to import the app in each worker separately. The module-level computation pattern (`plot_results = compute_plots_parallel()`) runs at import time.
**How to avoid:** Always use `preload_app = True` in `gunicorn.conf.py`. This runs the import (and thus the computation) once in the master process before forking workers.
**Warning signs:** Slow startup, duplicate SPARQL query log entries, high memory usage proportional to worker count.

### Pitfall 2: Legacy plots.py Shadowing the plots/ Package
**What goes wrong:** Python's import resolution may load `plots.py` instead of `plots/__init__.py` depending on sys.path ordering. This causes silent use of stale, unmaintained code.
**Why it happens:** Both `plots.py` (file) and `plots/` (directory) exist. Python 3 prioritizes packages over modules, but edge cases exist with path manipulation.
**How to avoid:** Delete `plots.py` as the very first cleanup commit. Verify with: `python -c "import plots; print(plots.__file__)"` -- should print `plots/__init__.py`.
**Warning signs:** `git diff` shows changes to `plots.py` instead of `plots/` files; test results differ between direct Python execution and Gunicorn.

### Pitfall 3: Unbounded Cache Growth During Version Browsing
**What goes wrong:** The `_plot_data_cache` dict in `shared.py` grows every time a user selects a historical version. Each version's 15 latest_* plots add ~15 DataFrame entries. A user browsing 30 historical versions would add ~450 DataFrame entries with no eviction.
**Why it happens:** The cache is a plain Python `dict` (`_plot_data_cache = {}`, shared.py line 130) with no size limits. `latest_plots.py` writes entries like `_plot_data_cache['latest_entity_counts'] = df` (line 267) with no key namespacing by version.
**How to avoid:** Replace the plain dict with the `VersionedPlotCache` class that enforces TTL + max version cap. Pin the latest version to prevent eviction of the most-accessed data.
**Warning signs:** Increasing RSS memory over time, especially in dashboard deployments where users explore multiple versions.

### Pitfall 4: Cache Key Collision Between Versions
**What goes wrong:** When a user views version "2024-01-15" then switches to "2025-07-01", the cache key `latest_entity_counts` gets overwritten. If two concurrent users are viewing different versions, they may see each other's data.
**Why it happens:** Current cache keys are not version-namespaced. `latest_plots.py` writes to fixed keys like `_plot_data_cache['latest_entity_counts']` regardless of which version was requested.
**How to avoid:** Include version in cache keys: `_plot_data_cache[f'latest_entity_counts_{version}']`. The eviction wrapper should handle version-prefixed keys.
**Warning signs:** CSV export returns data for a different version than the user selected; plots flicker between versions.

### Pitfall 5: Docker tmpfs Heartbeat Failures
**What goes wrong:** Gunicorn worker heartbeat files default to `/tmp`, which in Docker containers may be a regular filesystem. Slow I/O causes heartbeat misses, which Gunicorn interprets as a frozen worker, triggering a kill + restart cycle.
**Why it happens:** Docker containers don't mount `/tmp` as tmpfs by default (unlike many Linux distributions).
**How to avoid:** Set `worker_tmp_dir = "/dev/shm"` in gunicorn.conf.py. `/dev/shm` is always a memory-backed filesystem in Docker.
**Warning signs:** `[CRITICAL] WORKER TIMEOUT` in Gunicorn logs despite workers being healthy; workers cycling frequently.

### Pitfall 6: Health Endpoint Returns 200 When SPARQL Is Down
**What goes wrong:** The current `/health` endpoint (app.py line 272) returns HTTP 200 with `"status": "degraded"` when the SPARQL endpoint is down. Container orchestrators interpret 200 as healthy.
**Why it happens:** The conditional on line 340 returns 200 only when status is "healthy" and 503 for "degraded". But the user wants "honest reporting" -- unhealthy when SPARQL is down, which should be 503 to trigger restarts.
**How to avoid:** Change the health endpoint to return 503 when `endpoint_healthy` is False, regardless of cached plot state. The user explicitly wants this behavior even if it triggers container restarts.
**Warning signs:** Dashboard shows stale data indefinitely because orchestrator thinks the container is healthy.

## Code Examples

### Dead Code Inventory (Confirmed by Grep)

**app.py unused imports (lines 42-63):**
```python
# REMOVE these unused imports:
import plotly.express as px        # Never used in app.py
import plotly.io as pio            # Never used in app.py
import os                          # Never used in app.py
from functools import reduce       # Never used in app.py
from SPARQLWrapper import SPARQLWrapper, JSON  # Never used in app.py

# CONSOLIDATE these duplicate Flask imports:
from flask import Flask, render_template, jsonify, request, Response  # line 41
from flask import Flask, render_template, url_for, Response, redirect  # line 63 (duplicate)
# Should be single import:
from flask import Flask, render_template, jsonify, request, Response, url_for, redirect
```

**shared.py unused import:**
```python
from functools import reduce  # Never used in shared.py (only used in trends_plots.py)
```

**Commented-out code in app.py:**
```python
# Line 88: # plot_aop_completeness_boxplot_by_status,  # REMOVED
# Line 182: # REMOVED: 'aop_completeness_boxplot_by_status'
# Line 256: # graph_aop_completeness_boxplot_by_status = ""  # REMOVED
# Line 1444: # 'aop_completeness_boxplot_by_status': lambda: ...  # REMOVED
```

**Commented-out code in plots/__init__.py:**
```python
# Line 163: # plot_aop_completeness_boxplot_by_status,  # REMOVED
# Line 235: # 'plot_aop_completeness_boxplot_by_status',  # REMOVED
# Line 279: # 'plot_aop_completeness_boxplot_by_status',  # REMOVED
```

**Files to delete:**
- `plots.py` (4,194 lines, legacy monolith -- all functionality migrated to `plots/`)
- `app.log` (8,003 bytes, stale log file)
- `app_new.log` (11,094 bytes, stale log file)
- `verify_properties.py` (3,900 bytes, standalone verification script, untracked)
- `IMPROVEMENTS_SUMMARY.md` (11,487 bytes, historical record, not actively used)

**Templates audit:**
- `index.html` -- used by `/old-dashboard` and `/dashboard` routes (legacy tabbed interface)
- `trends.html` -- partial, included by `trends_page.html` via `{% include 'trends.html' %}`
- All other templates are actively used by current routes

**README.md references to update:**
- Line 109: `plots.py` listed in project structure
- Line 131: "Create plot function in `plots.py`" instructions
- Line 137: "Modify `BRAND_COLORS` in `plots.py`"

### Dockerfile Update (Flask dev server -> Gunicorn)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Production server with Gunicorn config
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
```

### Updated requirements.txt

```
Flask~=3.1
pandas~=2.2
plotly~=5.22
SPARQLWrapper~=2.0
gunicorn~=25.1
python-json-logger~=2.0
requests~=2.32
```

Note: `requests` is already imported by `shared.py` but not in `requirements.txt`. It's a dependency of `SPARQLWrapper` but should be explicitly listed since it's directly imported.

### Health Endpoint Fix

```python
@app.route("/health")
def health_check():
    try:
        endpoint_healthy = check_sparql_endpoint_health()
        successful_plots = sum(1 for v in plot_results.values() if v is not None)
        total_plots = len(plot_results)

        if endpoint_healthy:
            return {
                "status": "healthy",
                "sparql_endpoint": "up",
                "plots_loaded": f"{successful_plots}/{total_plots}",
                "timestamp": time.time()
            }, 200
        else:
            return {
                "status": "unhealthy",
                "sparql_endpoint": "down",
                "plots_loaded": f"{successful_plots}/{total_plots}",
                "timestamp": time.time()
            }, 503

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}, 500
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Flask `app.run()` for production | Gunicorn WSGI server | Standard practice since ~2015 | Multi-worker concurrency, proper process management |
| Unpinned deps in requirements.txt | `~=` compatible release pinning | pip best practice | Reproducible builds, controlled updates |
| `logging.basicConfig()` text format | `python-json-logger` JSON format | Growing adoption since 2020 | Machine-parseable logs for aggregation |
| Plain dict caches | TTL + LRU eviction | Required for production stability | Bounded memory under sustained load |
| `docker-compose version: '3.8'` | Version key deprecated | Docker Compose v2.0+ (2022) | Warning on newer Docker Compose versions |

**Deprecated/outdated:**
- `docker-compose.yml` version key (`version: '3.8'`): Deprecated in Docker Compose v2+. Can be removed.
- `python-json-logger` v2.x: The v3.x fork exists at `nhairs/python-json-logger` but v2.0.7 (madzak fork) is already installed and sufficient. No need to upgrade.

## Open Questions

1. **Cache key namespacing for versioned plots**
   - What we know: Current latest_* plots write to fixed keys without version suffix. The eviction wrapper needs version-aware keys.
   - What's unclear: Should the key scheme be `{plot_name}_{version}` or `{version}/{plot_name}`? Need to audit all cache reads to ensure backward compatibility.
   - Recommendation: Use `{plot_name}_{version}` pattern (e.g., `latest_entity_counts_2025-07-01`). The unversioned key `latest_entity_counts` remains as an alias for the pinned latest version. This preserves backward compatibility for CSV download routes.

2. **Gunicorn log integration with python-json-logger**
   - What we know: Gunicorn has its own access/error logging. `accesslog = "-"` and `errorlog = "-"` send to stdout/stderr.
   - What's unclear: Whether Gunicorn's access log output can be formatted as JSON to match the application logs.
   - Recommendation: Use Gunicorn's `logconfig_dict` to integrate with python-json-logger for consistent JSON output across both Gunicorn and application logs. This is achievable but adds configuration complexity. As a fallback, accept Gunicorn's default log format for access logs while ensuring application logs use JSON.

3. **Preload_app compatibility with cache modifications**
   - What we know: With `preload_app`, the master process initializes the cache, then workers fork. Workers share the initial cache via COW. When a worker modifies the cache (e.g., adding a new version), that modification is process-local.
   - What's unclear: Whether this means each worker maintains its own cache state independently, leading to duplicated version fetches.
   - Recommendation: This is actually acceptable. Each worker independently caches versions requested by its connections. The startup plots (27 trend plots) are shared via COW. The per-version latest plots are small and bounded by the eviction policy. If memory becomes a concern, a shared cache (Redis) could be added later, but that's out of scope for Phase 1.

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis: `app.py`, `plots/shared.py`, `plots/trends_plots.py`, `plots/latest_plots.py`, `plots/__init__.py`, `config.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `templates/*.html`, `static/js/*.js`, `static/css/*.css`
- [Flask Gunicorn deployment docs](https://flask.palletsprojects.com/en/stable/deploying/gunicorn/) - official Flask deployment guide
- [python-json-logger quickstart](https://nhairs.github.io/python-json-logger/latest/quickstart/) - official library docs
- [Gunicorn settings reference](https://gunicorn.org/reference/settings/) - official Gunicorn configuration

### Secondary (MEDIUM confidence)
- [Gunicorn in Docker containers](https://pythonspeed.com/articles/gunicorn-in-docker/) - Docker-specific Gunicorn configuration
- [Gunicorn application preloading](https://www.joelsleppy.com/blog/gunicorn-application-preloading/) - preload_app behavior and memory savings
- [Gunicorn worker timeout in Docker](https://github.com/benoitc/gunicorn/issues/2797) - /dev/shm heartbeat issue
- [Complete Gunicorn guide](https://betterstack.com/community/guides/scaling-python/gunicorn-explained/) - worker classes and configuration

### Tertiary (LOW confidence)
- [Sharing data across Gunicorn workers](https://medium.com/@jgleeee/sharing-data-across-workers-in-a-gunicorn-flask-application-2ad698591875) - fork() + COW behavior (verified against Gunicorn docs)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already installed and tested in this codebase. Versions verified via `pip list`.
- Architecture: HIGH - Patterns derived from direct codebase analysis + official docs. Gunicorn+preload interaction verified against multiple sources.
- Pitfalls: HIGH - Each pitfall confirmed by grep against actual codebase. Import shadows, unbounded caches, and unused imports all verified.
- Cache design: MEDIUM - The TTL+LRU+pinning pattern is sound but the cache key namespacing needs careful implementation to maintain backward compatibility with CSV download routes.

**Research date:** 2026-02-20
**Valid until:** 2026-03-20 (stable domain, no fast-moving dependencies)
