"""AOP-Wiki RDF Dashboard Flask Application.

This is the main Flask application that serves the AOP-Wiki RDF Dashboard. It provides
a web-based interface for visualizing AOP (Adverse Outcome Pathway) data evolution
over time, with comprehensive CSV export capabilities for all visualizations.

The application features:
    - Interactive web dashboard with tabbed navigation
    - Real-time visualization of RDF data evolution
    - Parallel plot computation for optimal performance
    - Comprehensive CSV data export system
    - Health monitoring and status endpoints
    - Professional branding and responsive design

Key Components:
    - Parallel plot computation system using ThreadPoolExecutor
    - Global data caching for CSV exports
    - Health monitoring with SPARQL endpoint checking
    - Professional web interface with brand-consistent styling
    - Comprehensive error handling and fallback mechanisms

Web Endpoints:
    /: Main dashboard with interactive visualizations
    /health: Health check endpoint for monitoring
    /status: Real-time status monitoring page
    /download/*: CSV download endpoints for all plots

Performance Features:
    - Parallel plot generation at startup
    - Global data caching to prevent recomputation
    - Configurable timeout and retry mechanisms
    - Real-time performance monitoring

The application integrates with a SPARQL endpoint to query RDF data and generates
interactive Plotly visualizations with consistent VHP4Safety branding.

Author:
    Marvin Martens

"""
from config import Config, configure_logging

# Configure structured JSON logging before any other imports that might log
configure_logging()

from flask import Flask, render_template, jsonify, request, Response, url_for, redirect
from flask_cors import CORS
import pandas as pd
import time
import os
import re
import json
import logging

logger = logging.getLogger(__name__)

# Validate configuration
if not Config.validate_config():
    logger.error("Invalid configuration detected, using defaults")

app = Flask(
    __name__,
    static_folder="static",     # This is default, but good to be explicit
    template_folder="templates"
)
CORS(app)  # Wildcard CORS for VHP4Safety cross-origin access

# --- Static assets (#175) ------------------------------------------------------
# Static files are cached for a year; every url_for('static', ...) carries a hash
# of the file's content, so a changed file gets a new URL and is fetched again.
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000
_static_hashes: dict = {}


@app.url_defaults
def _static_cache_buster(endpoint, values):
    if endpoint == 'static' and 'filename' in values and 'v' not in values:
        filename = values['filename']
        if filename not in _static_hashes:
            import hashlib
            path = os.path.join(app.static_folder, filename)
            try:
                with open(path, 'rb') as fh:
                    _static_hashes[filename] = hashlib.md5(fh.read()).hexdigest()[:10]
            except OSError:
                _static_hashes[filename] = None
        if _static_hashes[filename]:
            values['v'] = _static_hashes[filename]


def _load_vendor_manifest() -> dict:
    """Third-party JS served from static/vendor with SRI (scripts/vendor_js.py)."""
    try:
        with open(os.path.join(app.static_folder, 'vendor', 'manifest.json')) as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        logger.error(f"Vendor manifest unreadable: {e}")
        return {}


_VENDOR_MANIFEST = _load_vendor_manifest()


@app.context_processor
def _inject_vendor_manifest():
    return {'vendor': _VENDOR_MANIFEST}


# Load methodology notes for plot documentation
methodology_notes_path = os.path.join(app.static_folder, 'data', 'methodology_notes.json')
with open(methodology_notes_path, 'r') as f:
    methodology_notes = json.load(f)

import plots
from plots import registry
from plots import (
    available_ontology_branches,
    available_domains,
    check_sparql_endpoint_health,
    wait_for_sparql_ready,
    safe_plot_execution,
    get_latest_version,
    get_all_versions,
    get_properties_for_entity,
    _plot_data_cache,
    _plot_figure_cache,
    build_export_filename,
    export_figure_as_image,
    register_cache_rewarm,
    plot_cache_key,
    rewarm_cache_key,
    get_csv_with_metadata,
    create_bulk_download,
    get_or_compute_network
)
from plots.recovery import TrendPlotRecovery, result_has_usable_html, run_tasks_with_deadline

from usage_analytics import record_event, get_summary, init_db


# Pages worth counting as visits. Plot views are deliberately NOT tracked:
# plots lazy-load top-to-bottom, so /api/plot/<name> fires automatically for the
# top plots regardless of intent — that measures scroll position, not interest.
PAGE_LABELS = {
    '/': 'home',
    '/snapshot': 'snapshot',
    '/trends': 'trends',
    '/network': 'network',
    '/about': 'about',
}


def _classify_request(path: str):
    """Map a request path to an (event, target) pair for usage analytics.

    Tracks page visits ('page', <page label>) and downloads ('download', <plot>).
    Returns (None, None) for everything else. No PII is derived — only the page
    or plot identifier already present in the URL.
    """
    if path in PAGE_LABELS:
        return 'page', PAGE_LABELS[path]
    if path.startswith('/download/'):
        seg = path[len('/download/'):]
        if seg == 'bulk':
            return 'download', None
        if seg.startswith('latest/'):
            return 'download', 'latest_' + seg.split('/', 1)[1]
        if seg.startswith('trend/'):
            return 'download', seg.split('/', 1)[1]
        if seg.startswith('network/'):
            return 'download', 'network_' + seg.split('/', 1)[1]
        return 'download', seg or None
    return None, None


@app.after_request
def _track_usage(response):
    """Record a usage event for successful page visits and downloads.

    Centralised here so individual routes stay untouched. Swallows all errors
    so analytics can never affect the served response.
    """
    try:
        if response.status_code == 200:
            event, plot = _classify_request(request.path)
            if event:
                record_event(
                    event,
                    plot=plot,
                    version=request.args.get('version'),
                    fmt=request.args.get('format') or request.args.get('formats'),
                    route=request.path,
                )
    except Exception:
        pass
    return response


try:
    init_db()
except Exception as _usage_exc:  # pragma: no cover - non-fatal
    logger.warning(f"Usage analytics DB init failed: {_usage_exc}")

# Plots precomputed at startup, keyed by task name, from the plot registry
# (#178). Module-level so a trend plot whose startup computation failed can be
# recomputed on demand (#157).
STARTUP_PLOT_FUNCTIONS = {
    task: getattr(plots, producer)
    for task, producer in registry.startup_tasks().items()
}

def compute_plots_parallel() -> dict:
    """Compute all visualization plots in parallel for optimal startup performance.
    
    This function orchestrates the parallel generation of all dashboard plots using
    ThreadPoolExecutor. It manages the execution of 22 different plot functions,
    handles timeouts, and provides comprehensive error handling for individual
    plot failures.
    
    The function first performs a SPARQL endpoint health check, then executes all
    plot generation tasks concurrently with configurable parallelism and timeout
    settings. Each plot function is wrapped in safe execution to prevent individual
    failures from affecting the overall system.
    
    Plot Categories:
        - Historical trends: Main graph, averages, network density, components
        - Latest data snapshots: Entity counts, component analysis, completeness
        - Specialized analysis: Author counts, AOP lifetime, property presence
        - Ontology usage: Biological processes and objects analysis
    
    Returns:
        dict: Dictionary mapping plot names to their generated HTML content.
            Successful plots contain HTML strings, failed plots contain None.
            Keys include: 'main_graph', 'avg_per_aop', 'network_density',
            'latest_entity_counts', 'latest_ke_components', etc.
    
    Performance:
        - Uses configurable parallel workers (Config.PARALLEL_WORKERS)
        - One overall deadline (what is left of Config.STARTUP_BUDGET); stragglers recover on demand
        - Comprehensive timing and success rate logging
        - Graceful degradation for individual plot failures
    
    Example:
        >>> results = compute_plots_parallel()
        >>> successful_plots = sum(1 for v in results.values() if v is not None)
        >>> print(f"Generated {successful_plots}/{len(results)} plots successfully")
    
    Note:
        This function is called once at application startup to precompute all
        visualizations. The results are stored in global variables for serving
        to web requests.
    """
    logger.info("Starting parallel plot computation...")
    start_time = time.time()
    
    # Wait for the endpoint to serve real data, not just HTTP (#157). A Virtuoso
    # started alongside us can answer while still empty, and whatever we compute
    # now is served for the life of the process.
    if not wait_for_sparql_ready():
        logger.error("SPARQL endpoint is not ready, proceeding with degraded service")
    
    plot_tasks = [
        (name, lambda func=func: safe_plot_execution(func))
        for name, func in STARTUP_PLOT_FUNCTIONS.items()
    ]
    
    # One deadline for the whole precompute (#170). The old per-future
    # `future.result(timeout=...)` sat inside as_completed(), which only yields
    # finished futures, so it could never fire.
    # The precompute gets what the readiness wait left of STARTUP_BUDGET, so the
    # whole startup stays inside the swarm healthcheck's start_period.
    precompute_timeout = max(Config.STARTUP_MIN_PRECOMPUTE,
                             Config.STARTUP_BUDGET - (time.time() - start_time))
    results = run_tasks_with_deadline(
        dict(plot_tasks),
        max_workers=Config.PARALLEL_WORKERS,
        timeout=precompute_timeout,
    )
    
    total_time = time.time() - start_time
    # Fallback HTML is not None, so count real HTML only (#157).
    successful_plots = sum(1 for v in results.values() if result_has_usable_html(v))
    logger.info(f"Plot computation completed in {total_time:.2f}s. {successful_plots}/{len(plot_tasks)} plots successful")
    
    return results

# --- Precompute plots at startup ---
_startup_complete = False
plot_results = compute_plots_parallel()
_startup_complete = True

# Pre-rendered HTML for every snapshot plot the startup already computed (#177).
# /api/plot serves these for a request with no version and no tuning parameters,
# instead of re-running the plot's SPARQL on every page view; any ?version= or
# setting still renders on demand. Only while the endpoint's latest version is
# still the one these were built from (see _serve_precomputed).
_latest_precomputed_html: dict[str, str] = {
    k: v for k, v in plot_results.items()
    # A fallback from a failed startup run must not be pinned here (#157);
    # leaving it out makes /api/plot render the plot per request instead.
    if k.startswith('latest_') and isinstance(v, str) and result_has_usable_html(v)
}


def _serve_precomputed(plot_name: str) -> bool:
    """True if the startup HTML for plot_name still shows the latest version."""
    return plot_name in _latest_precomputed_html and get_latest_version() == _latest_version

# Snapshot plot functions, keyed by plot name. Module-level (rather than local
# to /api/plot) so the export cache-rewarm hook can resolve a cache key back to
# the function that produces it.
LATEST_PLOT_FUNCTIONS = {
    spec.id: getattr(plots, spec.producer) for spec in registry.snapshot_specs()
}


_CACHE_KEY_VERSION_RE = re.compile(r'^(?P<name>.+)_(?P<version>\d{4}-\d{2}-\d{2}|latest)$')


def _rewarm_plot_cache(cache_key: str) -> bool:
    """Recompute the latest_* plot behind `cache_key` so its export can proceed.

    Plot data/figure caches are per-process. With 2 gunicorn workers, a plot
    rendered lazily in one worker is absent from the other, so its download used
    to 404 about half the time; the same applies to any ?version= render, which
    is never part of the startup precompute. Re-running the plot function
    repopulates both caches as a side effect.

    Args:
        cache_key: e.g. 'latest_ke_reuse_2026-07-01', 'latest_ke_reuse_latest',
            or a bare 'latest_ke_reuse'.

    Returns:
        bool: True if a plot function was found and executed.
    """
    name, version = cache_key, None
    match = _CACHE_KEY_VERSION_RE.match(cache_key)
    if match:
        name = match.group('name')
        raw_version = match.group('version')
        version = None if raw_version == 'latest' else raw_version

    plot_function = LATEST_PLOT_FUNCTIONS.get(name)
    if plot_function is None:
        # Trend plots are computed pre-fork and pinned, so they are shared by
        # copy-on-write. They only miss when their startup run failed (#157).
        if cache_key in trend_recovery:
            return trend_recovery.recover(cache_key) is not None
        logger.debug(f"No rewarm available for cache key {cache_key}")
        return False

    plot_function(version) if version else plot_function()
    return True


register_cache_rewarm(_rewarm_plot_cache)

# Pin latest version in caches so it is never evicted
_latest_version = get_latest_version()


def _current_latest_graph_uri() -> str:
    """Graph URI for the methodology "Run on Endpoint" links (#169).

    Snapshot plots pick the newest graph per request, so the disclosed query
    should too. This comes from the memoized version registry, unlike
    ``_latest_version``, which is fixed at startup to match the startup caches.
    """
    version = get_latest_version()
    return f"http://aopwiki.org/graph/{version}" if version != "Unknown" else ""


if _latest_version and _latest_version != "Unknown":
    _plot_data_cache.pin_version(_latest_version)
    _plot_figure_cache.pin_version(_latest_version)
    logger.info(f"Pinned version {_latest_version} in data and figure caches")

# Trend plots served by /api/plot -> (startup task, position of their HTML in
# the task's result tuple, or None when the task returns the HTML itself).
TREND_PLOT_OUTPUTS = registry.trend_outputs()


def _startup_trend_html(plot_name: str) -> str:
    """The startup HTML for a trend plot, or "" if its task failed or timed out."""
    task, index = TREND_PLOT_OUTPUTS[plot_name]
    result = plot_results.get(task)
    try:
        html = result if index is None else result[index]
    except (TypeError, IndexError, KeyError):
        html = None
    return html or ""


# Re-runs the startup task behind a trend plot whose startup run failed (#157).
trend_recovery = TrendPlotRecovery(
    STARTUP_PLOT_FUNCTIONS, TREND_PLOT_OUTPUTS,
    run=safe_plot_execution, cooldown=Config.PLOT_RECOVERY_COOLDOWN,
)




def failed_startup_plots() -> list:
    """Startup tasks whose failed result is still being served.

    Only trend plots are served from their startup result. latest_* plots are
    rendered per request (startup is just a warm-up, and a failed one is never
    pinned in _latest_precomputed_html), so they cannot be stuck. A trend task
    stops counting as failed once every plot it produces has been recovered.
    """
    failed = []
    for task, result in plot_results.items():
        if task in LATEST_PLOT_FUNCTIONS or result_has_usable_html(result):
            continue
        outputs = [name for name, (t, _) in TREND_PLOT_OUTPUTS.items() if t == task]
        if outputs and all(trend_recovery.is_recovered(name) for name in outputs):
            continue
        failed.append(task)
    return sorted(failed)


def _version_freshness() -> dict:
    """Latest version on the endpoint and how old it is (#177).

    ``latest_version_age_days`` counts from the snapshot's date, so a value well
    past ~110 days means a quarterly snapshot has not been loaded.
    ``startup_version`` is what the startup caches (trend plots, landing counts)
    were built from; it differs from ``latest_version`` until the next restart.
    """
    from datetime import date
    latest = get_latest_version()
    info = {"latest_version": latest, "startup_version": _latest_version}
    try:
        info["latest_version_age_days"] = (date.today() - date.fromisoformat(latest)).days
    except ValueError:
        pass
    return info


@app.route("/health")
def health_check():
    """Health check endpoint for application and service monitoring.
    
    Provides a RESTful health check endpoint that monitors both the application
    state and the underlying SPARQL endpoint connectivity. Returns structured
    JSON with health status information suitable for monitoring systems,
    load balancers, and health checks.
    
    Health Checks:
        - SPARQL endpoint connectivity and responsiveness
        - Plot generation success rate from startup
        - Overall application health assessment
        - Timestamp for monitoring freshness
    
    Returns:
        tuple: (dict, int) containing:
            - dict: JSON response with health status information including:
                - status: "healthy", "degraded", "unhealthy", or "error"
                - sparql_endpoint: "up" or "down"
                - plots_loaded: "X/Y" startup plots not stuck on a fallback
                  (trend plots recovered on demand count as loaded; latest_*
                  plots render per request and are never stuck)
                - plots_failed: names of the failed startup plots (only when
                  there are any)
                - timestamp: Unix timestamp of health check
            - int: HTTP status code

    HTTP Status Codes:
        200: Endpoint up. "healthy" if every startup plot is usable, otherwise
             "degraded" with plots_failed. Degraded stays 200 on purpose: failed
             trend plots recover on demand, and a 503 would make swarm replace
             the task (#157).
        503: Still starting, or the SPARQL endpoint is down ("unhealthy").
        500: Health check itself failed (unexpected error)

    Example Response:
        >>> # Healthy response
        {
            "status": "healthy",
            "sparql_endpoint": "up",
            "plots_loaded": "41/41",
            "timestamp": 1640995200.0
        }

        >>> # Degraded response: some startup plots are fallbacks
        {
            "status": "degraded",
            "sparql_endpoint": "up",
            "plots_loaded": "39/41",
            "plots_failed": ["main_graph", "network_density"],
            "timestamp": 1640995200.0
        }
    
    Usage:
        This endpoint is designed for automated monitoring systems:
        
        >>> import requests
        >>> response = requests.get("http://localhost:5000/health")
        >>> if response.status_code == 200:
        ...     print("Application is healthy")
        ... else:
        ...     print(f"Application is degraded: {response.json()}")
    """
    try:
        if not _startup_complete:
            return {
                "status": "starting",
                "message": "Precomputing visualizations",
                "timestamp": time.time()
            }, 503

        endpoint_healthy = check_sparql_endpoint_health()
        failed_plots = failed_startup_plots()
        total_plots = len(plot_results)
        successful_plots = total_plots - len(failed_plots)

        if endpoint_healthy:
            # Still 200 when plots failed: they recover on demand, and a 503
            # would make swarm replace the task (and race Virtuoso again).
            health_status = {
                "status": "degraded" if failed_plots else "healthy",
                "sparql_endpoint": "up",
                "plots_loaded": f"{successful_plots}/{total_plots}",
                "timestamp": time.time()
            }
            health_status.update(_version_freshness())
            if failed_plots:
                health_status["plots_failed"] = failed_plots
            return health_status, 200
        else:
            health_status = {
                "status": "unhealthy",
                "sparql_endpoint": "down",
                "plots_loaded": f"{successful_plots}/{total_plots}",
                "timestamp": time.time()
            }
            return health_status, 503

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}, 500

@app.route("/status")  
def status_page():
    """Serve the real-time status monitoring dashboard page.
    
    Renders the status.html template that provides a web-based interface for
    monitoring application health, performance metrics, and system status.
    This page complements the /health endpoint by providing human-readable
    monitoring information.
    
    The status page includes:
        - Real-time health status indicators
        - SPARQL endpoint connectivity status  
        - Plot generation success rates
        - Performance metrics and timing information
        - System resource utilization
        - Interactive monitoring dashboard
    
    Returns:
        str: Rendered HTML template for the status monitoring page
    
    Template:
        Uses templates/status.html for the monitoring interface
        
    Example:
        Access via browser: http://localhost:5000/status
        
    Note:
        This endpoint is intended for human monitoring and debugging,
        while /health is designed for automated systems.
    """
    return render_template("status.html", active_page='status')

def _export_response(plot_name, *, filename_base=None, version=None, start=None, end=None):
    """CSV/PNG/SVG download of one plot, as every /download route serves it.

    Snapshot plots resolve to their per-version cache key inside the export
    helpers (#166) and are recomputed if this worker hasn't rendered that
    version (#148); there is no fallback to another version's data. ``start``
    and ``end`` limit trend plots to a snapshot range (#44).
    """
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'
    filename_base = filename_base or plot_name
    # The filename names the version the request asked for, as before (#59).
    filename_version = request.args.get('version')

    try:
        if export_format == 'csv':
            body = get_csv_with_metadata(plot_name, include_metadata, start=start, end=end, version=version)
            if not body:
                return "No data available for download", 404
            mimetype = 'text/csv'
        elif export_format in ('png', 'svg'):
            body = export_figure_as_image(plot_name, export_format, start=start, end=end, version=version)
            if not body:
                return "No figure available for export", 404
            mimetype = f'image/{export_format}'
        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

        filename = build_export_filename(filename_base, export_format, filename_version)
        return Response(body, mimetype=mimetype,
                        headers={'Content-Disposition': f'attachment; filename={filename}'})

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500


@app.route("/download/latest/<plot_name>")
def download_latest_generic(plot_name):
    """Snapshot plot download by suffix: /download/latest/ke_reuse -> latest_ke_reuse."""
    return _export_response(f'latest_{plot_name}', version=request.args.get('version'))


@app.route("/download/trend/<plot_name>")
def download_trend_plot(plot_name):
    """Trend plot download, optionally limited to ?start=&end= snapshots."""
    return _export_response(plot_name, start=request.args.get('start'), end=request.args.get('end'))


@app.route("/download/<plot_name>")
def download_generic(plot_name):
    """Download any plot by id (#42), or by a legacy alias (registry.DOWNLOAD_ALIASES).

    Replaces the per-plot download routes (#178); their URLs resolve here.
    """
    target = registry.DOWNLOAD_ALIASES.get(plot_name, plot_name)
    return _export_response(target, version=request.args.get('version'),
                            start=request.args.get('start'), end=request.args.get('end'))


@app.route("/download/bulk")
def download_bulk():
    """Bulk download multiple plots in a ZIP archive.

    Query Parameters:
        plots (str): Comma-separated list of plot names (e.g., "latest_entity_counts,latest_ke_components")
        formats (str): Comma-separated list of formats (default: "csv,png,svg")
        category (str): Predefined category: "all", "database-state", "network-analysis", "ke-analysis", "data-quality"

    Returns:
        Response: ZIP file containing all requested plots in all requested formats

    Example Usage:
        /download/bulk?plots=latest_entity_counts,latest_ke_components&formats=csv,png
        /download/bulk?category=database-state&formats=png,svg
        /download/bulk?category=all&formats=csv
    """
    try:
        # Predefined plot categories, from the plot registry (#178)
        categories = registry.bulk_categories()

        # Get plot names from query params
        category = request.args.get('category', '').lower()
        plots_param = request.args.get('plots', '')

        if category and category in categories:
            plot_names = categories[category]
        elif plots_param:
            plot_names = [p.strip() for p in plots_param.split(',')]
        else:
            return "Please specify either 'category' or 'plots' parameter", 400

        # Get formats from query params
        formats_param = request.args.get('formats', 'csv,png,svg')
        formats = [f.strip().lower() for f in formats_param.split(',')]

        # Validate formats
        valid_formats = {'csv', 'png', 'svg'}
        formats = [f for f in formats if f in valid_formats]
        if not formats:
            return "No valid formats specified. Use csv, png, or svg.", 400

        # Create ZIP file, honouring optional snapshot-range filter (#44)
        start = request.args.get('start')
        end = request.args.get('end')
        version = request.args.get('version')
        zip_bytes = create_bulk_download(plot_names, formats, start=start, end=end, version=version)
        if not zip_bytes:
            return "Failed to create bulk download", 500

        # Generate filename, embedding the snapshot range when one is active (#59)
        suffix = f"_{start or ''}_to_{end or ''}" if (start or end) else ""
        if version:
            suffix += f"_v{version}"
        if category:
            filename = f"aopwiki_{category}{suffix}_plots.zip"
        else:
            filename = f"aopwiki_{len(plot_names)}{suffix}_plots.zip"

        return Response(
            zip_bytes,
            mimetype='application/zip',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )

    except Exception as e:
        logger.error(f"Bulk download failed: {str(e)}")
        return f"Bulk download failed: {str(e)}", 500


# Individual plot endpoints for lazy loading
@app.route("/api/latest-version")
def api_get_latest_version():
    """API endpoint to get the latest AOP-Wiki database version."""
    try:
        version = get_latest_version()
        return jsonify({"version": version})
    except Exception as e:
        logger.error(f"Error getting latest version: {e}")
        return jsonify({"version": "Data available on dashboard", "error": str(e)}), 500


@app.route("/api/organ-system-buckets")
def api_organ_system_buckets():
    """Serve the curated organ-system bucket dictionary that backs the
    AOP organ-coverage plots.

    Returns the full Signal A/A'/B/C mapping so reviewers can audit which
    UBERON / CL / GO BP terms and which keyword regexes feed each bucket.
    Linked from the methodology note for those plots.
    """
    from plots.organ_systems import serialise_for_methodology
    return jsonify(serialise_for_methodology())


@app.route("/api/ontology-branches")
def api_ontology_branches():
    """Serve the offline ontology-branch cache behind the coverage-holes plot.

    Returns provenance plus, per organ system, every candidate sub-branch with
    its label, subtree size and depth — so a reviewer can audit which parts of
    the ontology the plot considers, and how the used/unused split was reached,
    without re-running the Ubergraph build. Linked from the methodology note.
    """
    from plots.latest_plots import _load_ontology_branch_cache
    return jsonify(_load_ontology_branch_cache())


@app.route("/api/domain-lens")
def api_domain_lens():
    """Serve the offline domain vocabulary behind the domain-lens plots (#149).

    Returns provenance, the ontology roots per domain and every in-use term
    with its label and originating root — so a reviewer can audit exactly which
    terms put an AOP in a domain, without re-running the Ubergraph build.
    Linked from the methodology notes.
    """
    from plots.domain_plots import serialise_domain_lens
    return jsonify(serialise_domain_lens())


@app.route("/api/versions")
def api_get_all_versions():
    """API endpoint to get all available AOP-Wiki database versions with metadata.

    Returns:
        JSON response with list of versions sorted by date (newest first).
        Each version includes: version, graph_uri, and date.

    Example Response:
        {
            "versions": [
                {
                    "version": "2024-10-01",
                    "graph_uri": "http://aopwiki.org/graph/2024-10-01",
                    "date": "2024-10-01"
                },
                ...
            ],
            "count": 30
        }
    """
    try:
        versions = get_all_versions()
        return jsonify({
            "versions": versions,
            "count": len(versions)
        })
    except Exception as e:
        logger.error(f"Error getting all versions: {e}")
        return jsonify({"versions": [], "count": 0, "error": str(e)}), 500

@app.route("/api/properties/<entity_type>")
def api_get_properties(entity_type):
    """API endpoint to get properties for a specific entity type.

    Returns properties filtered by entity type from property_labels.csv,
    grouped by category (Essential, Content, Context, Assessment, Metadata).

    Args:
        entity_type (str): Entity type (AOP, KE, KER, or Stressor)

    Returns:
        JSON response with properties grouped by category.

    Example Response:
        {
            "entity_type": "KE",
            "properties": {
                "Essential": [
                    {"uri": "...", "label": "Title", "type": "Essential", "applies_to": "AOP|KE|KER|Stressor"},
                    ...
                ],
                "Content": [...],
                "Context": [...],
                "Assessment": [...],
                "Metadata": [...]
            }
        }
    """
    try:
        # Validate entity type
        valid_types = ['AOP', 'KE', 'KER', 'STRESSOR']
        if entity_type.upper() not in valid_types:
            return jsonify({
                "error": f"Invalid entity type. Must be one of: {', '.join(valid_types)}"
            }), 400

        properties = get_properties_for_entity(entity_type)
        return jsonify({
            "entity_type": entity_type.upper(),
            "properties": properties
        })
    except Exception as e:
        logger.error(f"Error getting properties for {entity_type}: {e}")
        return jsonify({
            "entity_type": entity_type,
            "properties": {},
            "error": str(e)
        }), 500

@app.route("/api/v2/manifest")
def api_v2_manifest():
    """Every plot the dashboard serves, from the plot registry (#178).

    One entry per plot: id, kind (snapshot/trend), the startup task that
    produces it, its methodology-note key, the page holding its container,
    whether it takes ?version=, its tuning args and bulk categories.
    """
    return jsonify({'plots': registry.manifest()})


@app.route("/api/plot-data/<plot_name>")
def api_plot_data(plot_name):
    """API endpoint to serve raw data for a plot as JSON.

    Returns the cached DataFrame for a given plot in JSON format,
    suitable for rendering as an HTML table by the frontend.

    Args:
        plot_name (str): Name of the plot whose data to retrieve

    Query Parameters:
        version (str, optional): Version string for latest_* plots

    Returns:
        dict: JSON with columns, rows, total_rows, truncated, plot_name, success
    """
    version = request.args.get('version', None)
    max_rows = 100

    # Per-version key for snapshot plots, no fallback to another version (#166);
    # recompute if this worker hasn't rendered it yet (#148).
    cache_key = plot_cache_key(plot_name, version)
    if cache_key not in _plot_data_cache and not rewarm_cache_key(_plot_data_cache, cache_key):
        return jsonify({'error': 'No data available', 'success': False}), 404

    try:
        df = _plot_data_cache[cache_key]
        if df is None or (hasattr(df, 'empty') and df.empty):
            return jsonify({'error': 'No data available', 'success': False}), 404

        total_rows = len(df)
        truncated = total_rows > max_rows
        display_df = df.head(max_rows).copy()
        display_df = display_df.fillna('')
        columns = list(display_df.columns)
        rows = display_df.to_dict('records')

        return jsonify({
            'columns': columns,
            'rows': rows,
            'total_rows': total_rows,
            'truncated': truncated,
            'plot_name': plot_name,
            'success': True
        })
    except Exception as e:
        logger.error(f"Error serving plot data for {plot_name}: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route("/api/plot/<plot_name>")
def get_plot(plot_name):
    """API endpoint to serve individual plots on demand for lazy loading.

    This endpoint allows individual plots to be loaded asynchronously,
    significantly reducing initial page load time by only rendering plots
    when they're actually viewed by the user.

    Supports version parameter for latest_* plots to enable historical version viewing.

    Args:
        plot_name (str): Name of the plot to render

    Query Parameters:
        version (str, optional): Version string for latest_* plots (e.g., "2024-10-01")

    Returns:
        dict: JSON response with plot HTML or error message
    """
    # Get version parameter from query string (for latest_* plots only)
    version = request.args.get('version', None)

    # Fast path: pre-rendered snapshot plots (no version requested and no
    # tuning args). Any tuning arg forces a re-render so the plot reflects the
    # user's choice instead of the cached default.
    _has_tuning = any(request.args.get(k) for k in registry.tuning_params())
    if not version and not _has_tuning and _serve_precomputed(plot_name):
        return jsonify({'html': _latest_precomputed_html[plot_name], 'success': True})

    if plot_name in LATEST_PLOT_FUNCTIONS:
        try:
            plot_function = LATEST_PLOT_FUNCTIONS[plot_name]
            # Pass through the tuning args this plot's spec declares (scope,
            # view, ...) — keeps the unified coverage toggles working without
            # overloading the route surface.
            extra_kwargs = {}
            for key in registry.BY_ID[plot_name].params:
                value = request.args.get(key)
                if value:
                    extra_kwargs[key] = value
            if version:
                html = plot_function(version, **extra_kwargs)
            else:
                html = plot_function(**extra_kwargs)
            return jsonify({'html': html, 'success': True})
        except Exception as e:
            logger.error(f"Error generating plot {plot_name} with version {version}: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    # Historical trend plots, served from their startup result
    elif plot_name in TREND_PLOT_OUTPUTS:
        plot_html = trend_recovery.html(plot_name, _startup_trend_html(plot_name))
        # If it's a callable (lambda function), execute it to generate the plot on-demand
        if callable(plot_html):
            try:
                plot_html = plot_html()
            except Exception as e:
                logger.error(f"Error generating on-demand plot {plot_name}: {e}")
                return jsonify({'error': str(e), 'success': False}), 500

        # Validate content: empty or fallback HTML should not count as success
        if not plot_html or (isinstance(plot_html, str) and len(plot_html.strip()) == 0):
            return jsonify({'error': f'Plot {plot_name} has no data available', 'success': False}), 500
        if isinstance(plot_html, str) and 'Data Unavailable' in plot_html:
            return jsonify({'error': f'Plot {plot_name} failed to load - data unavailable', 'success': False}), 500

        return jsonify({'html': plot_html, 'success': True})

    else:
        return jsonify({'error': f'Plot {plot_name} not found', 'success': False}), 404



# Set Plotly configuration for static images
@app.route("/old-dashboard")
def index():
    """Redirect legacy old-dashboard route to the modern snapshot page."""
    return redirect(url_for('database_snapshot'))


@app.route("/")
def landing():
    """Serve the navigation hub landing page with live data summary.

    Displays section cards for Database Snapshot, Historical Trends, and
    Network Analysis. Shows latest version number and headline entity counts
    from cached startup data. Includes expandable AOP-Wiki introduction.
    """
    # The startup value, deliberately: the headline counts below come from the
    # startup caches and have to describe the same version.
    latest_version = _latest_version or "Unknown"

    def _counts_from_cache():
        cached_df = _plot_data_cache.get(plot_cache_key('latest_entity_counts', _latest_version))
        if cached_df is not None and hasattr(cached_df, 'empty') and not cached_df.empty:
            try:
                return dict(zip(cached_df['Entity'], cached_df['Count']))
            except (KeyError, TypeError):
                return None
        return None

    entity_counts = _counts_from_cache()
    if entity_counts is None:
        # The counts come from the startup plot cache. If that computation failed
        # during warmup (e.g. a SPARQL timeout), the cache stays empty and the
        # landing badges would show nothing until the next restart. Recompute
        # once on demand so the page self-heals (#140). safe_plot_execution
        # swallows errors, so a still-down endpoint just leaves counts hidden
        # rather than erroring the page.
        try:
            safe_plot_execution(LATEST_PLOT_FUNCTIONS['latest_entity_counts'], _latest_version)
            entity_counts = _counts_from_cache()
        except Exception:
            logger.warning("Landing: on-demand entity_counts recompute failed", exc_info=True)

    return render_template("landing.html", version=latest_version, entity_counts=entity_counts, active_page='home')


@app.route("/snapshot")
def database_snapshot():
    """Serve the Database Snapshot page with version-selectable visualizations.

    Displays key metrics and visualizations from any version of the AOP-Wiki
    database, allowing users to explore current and historical snapshots.
    """
    return render_template("latest.html", methodology_notes=methodology_notes, sparql_endpoint=Config.SPARQL_PUBLIC_ENDPOINT, latest_graph_uri=_current_latest_graph_uri(), ontology_branches=available_ontology_branches(), domains=available_domains(), versioned_plots=sorted(LATEST_PLOT_FUNCTIONS), active_page='snapshot')


@app.route("/latest")
def latest_redirect():
    """Redirect /latest to /snapshot for backward compatibility."""
    return redirect(url_for('database_snapshot'))


@app.route("/trends")
def historical_trends():
    """Serve the Historical Trends page with time-series analysis.

    Displays evolution and growth patterns of the AOP-Wiki database
    over time using quarterly releases.
    """
    return render_template("trends_page.html", methodology_notes=methodology_notes, sparql_endpoint=Config.SPARQL_PUBLIC_ENDPOINT, latest_graph_uri=_current_latest_graph_uri(), active_page='trends')


@app.route("/dashboard")
def dashboard():
    """Redirect legacy dashboard route to the modern snapshot page."""
    return redirect(url_for('database_snapshot'))


@app.route("/network")
def network_page():
    """Serve the Network Analysis page."""
    return render_template("network.html", active_page='network')


@app.route("/about")
def about():
    """Serve the About page with project info, funding, and contact links."""
    return render_template("about.html", active_page='about')


@app.route("/api/network/graph")
def api_network_graph():
    """API endpoint: full graph data in Cytoscape.js JSON format."""
    try:
        graph_data = get_or_compute_network()
        return jsonify({
            'elements': graph_data['elements'],
            'version': graph_data['version'],
            'stats': graph_data['stats']
        })
    except Exception as e:
        logger.error(f"Network graph API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/network/metrics")
def api_network_metrics():
    """API endpoint: centrality metrics as JSON for sortable table."""
    try:
        graph_data = get_or_compute_network()
        return jsonify({
            'metrics': graph_data['metrics'],
            'version': graph_data['version']
        })
    except Exception as e:
        logger.error(f"Network metrics API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/network/communities")
def api_network_communities():
    """API endpoint: community detection results."""
    try:
        graph_data = get_or_compute_network()
        return jsonify({
            'communities': graph_data['communities'],
            'version': graph_data['version']
        })
    except Exception as e:
        logger.error(f"Network communities API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/download/network/metrics")
def download_network_metrics():
    """Download network centrality metrics as CSV."""
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'
    try:
        # Ensure network data has been computed so cache is populated
        get_or_compute_network()
        csv_data = get_csv_with_metadata('network_metrics', include_metadata=include_metadata)
        if not csv_data:
            return "No network metrics data available for download", 404
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={build_export_filename("network_metrics", "csv")}'}
        )
    except Exception as e:
        logger.error(f"Network metrics download error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/download/network/graph")
def download_network_graph():
    """Download network graph structure as JSON."""
    try:
        graph_data = get_or_compute_network()
        export_data = {
            'metadata': {
                'export_date': pd.Timestamp.now().isoformat(),
                'version': graph_data['version'],
                'stats': graph_data['stats']
            },
            'elements': graph_data['elements']
        }
        return Response(
            json.dumps(export_data, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename=network_graph_{graph_data["version"]}.json'}
        )
    except Exception as e:
        logger.error(f"Network graph download error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/usage")
def api_usage():
    """Aggregated, privacy-respecting usage statistics as JSON.

    Counts of plot views and downloads by plot, format, version and day. No
    personal data is collected or returned. Gated by ENABLE_USAGE_ANALYTICS.
    """
    return jsonify(get_summary())


@app.route("/stats")
def usage_stats_page():
    """Human-readable usage statistics page (reads /api/usage)."""
    return render_template("usage.html", active_page='stats')


# Run the Flask app
if __name__ == "__main__":
    logger.info(f"Starting AOP-Wiki RDF Dashboard on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    logger.info(f"SPARQL Endpoint: {Config.SPARQL_ENDPOINT}")
    logger.info(f"Configuration: {Config.get_config_dict()}")
    
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)
