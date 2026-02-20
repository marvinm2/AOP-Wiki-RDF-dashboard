# Architecture

**Analysis Date:** 2026-02-20

## Pattern Overview

**Overall:** Layered monolithic Flask application with parallel plot computation and lazy loading

**Key Characteristics:**
- Multi-tier architecture: Flask web server → Plot generation layer → SPARQL data layer
- Parallel startup computation using ThreadPoolExecutor (22 plots precomputed at boot)
- Lazy loading for performance (99.98% faster page loads)
- Global data caching system for CSV exports and figure objects
- Dual-page design: Database Snapshot (versioned current state) and Historical Trends (time-series)

## Layers

**Web Layer (Flask Routes):**
- Purpose: HTTP request handling, response formatting, file serving
- Location: `app.py` (lines 272-1609)
- Contains: Route decorators (@app.route), health checks, download endpoints, API routes
- Depends on: Plots module for visualization data, Config for settings
- Used by: Browser clients, API consumers, monitoring systems

**Plot Generation Layer:**
- Purpose: Create interactive Plotly visualizations from SPARQL data
- Location: `plots/` package (latest_plots.py, trends_plots.py, shared.py)
- Contains: 30+ visualization functions, data processing, caching logic
- Depends on: SPARQL queries via shared.py, Plotly rendering, pandas for data manipulation
- Used by: Flask routes for template injection, lazy-loading API endpoints

**Data/Query Layer:**
- Purpose: SPARQL endpoint communication with retry logic and health monitoring
- Location: `plots/shared.py` (functions: run_sparql_query_with_retry, check_sparql_endpoint_health, extract_counts)
- Contains: SPARQL query execution, connection pooling, error handling, retry logic
- Depends on: SPARQLWrapper library, Config for endpoint URL/timeout settings
- Used by: All plot generation functions

**Configuration Layer:**
- Purpose: Centralized environment variable management and validation
- Location: `config.py`
- Contains: Class Config with 13 configurable parameters (SPARQL endpoint, timeouts, parallelism, Flask settings)
- Depends on: Python os.environ
- Used by: app.py at startup, shared.py initialization

## Data Flow

**Startup Flow:**
1. Flask app initializes with Config validation (config.py)
2. Import plots module, triggering plots/__init__.py exports
3. Call compute_plots_parallel() which:
   - Checks SPARQL endpoint health
   - Submits 22 plot generation tasks to ThreadPoolExecutor
   - Each task calls safe_plot_execution wrapper around a plot function
   - Returns dict mapping plot names to HTML strings (or None on failure)
4. Extract plot results into global variables for template context
5. Flask starts serving on configured host:port

**Request Flow (Latest Data / Snapshot Page):**
1. GET /snapshot → render_template("latest.html")
2. Template loads version-selector.js which calls /api/latest-version
3. User selects version → version-selector.js calls /api/plot/<plot_name>?version=YYYY-MM-DD
4. Flask /api/plot/<plot_name> route:
   - If plot_name in cache and no version specified: return cached HTML
   - Else: Call plot function with version parameter, cache result, return HTML
5. Client-side JavaScript injects returned HTML into lazy-plot divs

**Request Flow (Historical Trends):**
1. GET /trends → render_template("trends_page.html")
2. Page loads lazy-loading.js which scans for data-plot-name attributes
3. For each lazy plot div, fetch /api/plot/<plot_name> with no version param
4. Returns precomputed HTML from startup (e.g., main_graph returns tuple unpacked: abs, delta, data)
5. JavaScript displays both absolute and delta plots side-by-side

**CSV Export Flow:**
1. User clicks "Download CSV" button (e.g., /download/latest_entity_counts)
2. Flask route calls get_csv_with_metadata(plot_name, include_metadata=True)
3. Retrieves cached DataFrame from _plot_data_cache[plot_name]
4. Converts to CSV string with optional metadata header (generation time, version, record count)
5. Returns Response with text/csv MIME type and attachment header

**State Management:**
- **Global Variables:** plot_results dict (startup computed plots), individual plot HTML variables (graph_main_abs, graph_avg_delta, etc.)
- **Data Caching:** _plot_data_cache (DataFrames), _plot_figure_cache (Plotly figure objects)
- **Version Detection:** Latest version determined automatically via "ORDER BY DESC(?graph) LIMIT 1" in SPARQL queries
- **Thread Safety:** ThreadPoolExecutor handles concurrent plot generation at startup; post-startup is single-threaded

## Key Abstractions

**Plot Function Contracts:**
- **Latest plots:** Return single HTML string (e.g., plot_latest_entity_counts() → str)
- **Trend plots (diff):** Return 2-tuple (absolute_html, delta_html) or 3-tuple with data (e.g., plot_main_graph() → (str, str, DataFrame))
- **Trend plots (property presence):** Return 2-tuple (absolute, percentage) with marker shape differentiation
- **All functions accept optional version: str parameter for historical snapshots**

**SPARQL Query Pattern:**
- Base filter: `_build_graph_filter(version)` returns (where_filter, order_limit)
- If version specified: WHERE FILTER(?graph = <http://aopwiki.org/graph/YYYY-MM-DD>)
- If no version: ORDER BY DESC(?graph) LIMIT 1 after GROUP BY

**Plot Execution Wrapper:**
- safe_plot_execution(plot_function) wraps any plot function
- Catches exceptions, logs errors, returns None on failure (graceful degradation)
- Used by compute_plots_parallel() to prevent single plot failure from blocking others

**Safe CSV Reading:**
- safe_read_csv(filename, default_data) returns DataFrame
- Falls back to provided default_data list or empty DataFrame if file missing/corrupted
- Used for property_labels.csv loading with sensible defaults

## Entry Points

**Web Entry Points:**

**GET /:**
- Location: `app.py`, line 1555
- Triggers: Landing page with navigation buttons
- Responsibilities: Render landing.html intro page

**GET /snapshot:**
- Location: `app.py`, line 1566
- Triggers: Database snapshot view with version selector
- Responsibilities: Render latest.html with all "latest" plots + version selector UI

**GET /trends:**
- Location: `app.py`, line 1582
- Triggers: Historical trends analysis page
- Responsibilities: Render trends_page.html with all historical plots (absolute + delta views)

**GET /api/plot/<plot_name>:**
- Location: `app.py`, line 1387
- Triggers: Lazy loading of individual plots via JavaScript
- Responsibilities: Return cached or newly-generated plot HTML; support optional ?version parameter

**API Entry Points:**

**GET /api/latest-version:**
- Returns: {"version": "2024-10-01"} from latest RDF graph
- Used by: version-selector.js on snapshot page

**GET /api/versions:**
- Returns: {"versions": ["2024-10-01", "2024-07-01", ...]} sorted descending
- Used by: version-selector dropdown population

**GET /api/properties/<entity_type>:**
- Returns: Grouped properties from property_labels.csv filtered by entity type
- Used by: Property display panels in UI

**Monitoring Entry Points:**

**GET /health:**
- Returns: JSON with status, SPARQL endpoint health, plots_loaded ratio
- HTTP 200 (healthy), 503 (degraded), or 500 (error)
- Used by: Load balancers, monitoring systems

**GET /status:**
- Triggers: Status monitoring dashboard page
- Responsibilities: Render status.html with real-time health metrics

**Application Entry Point:**

**if __name__ == "__main__":**
- Location: `app.py`, line 1602
- Triggers: Direct script execution
- Responsibilities: Log startup info, call app.run() with Config settings

## Error Handling

**Strategy:** Graceful degradation with fallback visualizations

**Patterns:**

**SPARQL Query Errors:**
- Retry logic in run_sparql_query_with_retry(): exponential backoff up to Config.SPARQL_MAX_RETRIES (default 3)
- Classify errors: syntax errors (no retry), endpoint errors (no retry), network errors (retry)
- Timeout: Config.SPARQL_TIMEOUT (default 30s) per query
- Failure path: Log error, return empty list, plot generation fails gracefully

**Plot Generation Errors:**
- Wrapped in safe_plot_execution() which catches all exceptions
- Failed plots return None instead of HTML
- Fallback visualization via create_fallback_plot() generates placeholder chart
- Logged to application stderr with function name and error details

**Data Processing Errors:**
- CSV reading: safe_read_csv() with fallback defaults (empty DataFrame or provided default_data)
- Missing files: Log warning, return empty or default data
- Parser errors: Log error, return empty DataFrame, continue

**Health Check Errors:**
- /health endpoint catches all exceptions, returns HTTP 500 with "error" status
- Endpoint health check tries lightweight query; logs failures but doesn't crash application

## Cross-Cutting Concerns

**Logging:**
- Framework: Python logging module configured in config.py and plots/shared.py
- Level: Config.LOG_LEVEL (default INFO)
- Format: %(asctime)s - %(levelname)s - %(message)s
- Key events: Startup, plot computation timing, SPARQL query failures, health checks

**Validation:**
- Config validation on startup via Config.validate_config()
- SPARQL endpoint URL validation using urllib.parse.urlparse
- Numeric range validation for ports, timeouts, worker counts
- Logical consistency checks between related parameters (e.g., PLOT_TIMEOUT > 0)

**Authentication:**
- Not implemented; application assumes SPARQL endpoint is on trusted network
- No API key or bearer token validation in routes
- Health check uses unauthenticated GET on SPARQL endpoint

**Performance Monitoring:**
- Execution time tracking in run_sparql_query_with_retry() logs slow queries (>10s)
- Plot computation timing in compute_plots_parallel() logs total time and success rate
- Per-plot timeout enforcement via ThreadPoolExecutor.result(timeout=Config.PLOT_TIMEOUT)
- Data caching prevents recomputation of expensive queries

**Caching Strategy:**
- _plot_data_cache: Global dict mapping plot_name → DataFrame
- _plot_figure_cache: Global dict mapping plot_name → Plotly figure objects
- Populated during plot generation, accessed by CSV export routes
- Cleared via plots.clear_plot_cache() function if needed
- Enables fast CSV exports without re-querying SPARQL

---

*Architecture analysis: 2026-02-20*
