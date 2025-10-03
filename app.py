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
    Generated with Claude Code (https://claude.ai/code)

"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import plotly.express as px
import plotly.io as pio
import os
from functools import reduce
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Validate configuration
if not Config.validate_config():
    logger.error("Invalid configuration detected, using defaults")

# Set default Plotly renderer
from flask import Flask, render_template, url_for, Response, redirect

app = Flask(
    __name__,
    static_folder="static",     # This is default, but good to be explicit
    template_folder="templates"
)

from plots import (
    plot_main_graph,
    plot_avg_per_aop,
    plot_network_density,
    plot_ke_components,
    plot_ke_components_percentage,
    plot_unique_ke_components,
    plot_bio_processes,
    plot_bio_objects,
    plot_author_counts,
    plot_aop_lifetime,
    plot_aop_property_presence,
    plot_aop_property_presence_unique_colors,
    plot_kes_by_kec_count,
    plot_latest_entity_counts,
    plot_latest_ke_components,
    plot_latest_network_density,
    plot_latest_avg_per_aop,
    plot_latest_ontology_usage,
    plot_latest_process_usage,
    plot_latest_object_usage,
    plot_latest_aop_completeness,
    plot_latest_aop_completeness_unique_colors,
    plot_latest_database_summary,
    plot_latest_ke_annotation_depth,
    check_sparql_endpoint_health,
    safe_plot_execution,
    get_latest_version,
    get_all_versions,
    _plot_data_cache,
    _plot_figure_cache,
    export_figure_as_image,
    get_csv_with_metadata,
    create_bulk_download
)

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
        - Individual plot timeout protection (Config.PLOT_TIMEOUT)
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
    
    # Check SPARQL endpoint health first
    if not check_sparql_endpoint_health():
        logger.error("SPARQL endpoint is not healthy, proceeding with degraded service")
    
    # Define plot functions and their expected results
    plot_tasks = [
        ('main_graph', lambda: safe_plot_execution(plot_main_graph)),
        ('avg_per_aop', lambda: safe_plot_execution(plot_avg_per_aop)),
        ('network_density', lambda: safe_plot_execution(plot_network_density)),
        ('ke_components', lambda: safe_plot_execution(plot_ke_components)),
        ('unique_ke_components', lambda: safe_plot_execution(plot_unique_ke_components)),
        ('ke_components_percentage', lambda: safe_plot_execution(plot_ke_components_percentage)),
        ('bio_processes', lambda: safe_plot_execution(plot_bio_processes)),
        ('bio_objects', lambda: safe_plot_execution(plot_bio_objects)),
        ('author_counts', lambda: safe_plot_execution(plot_author_counts)),
        ('aop_lifetime', lambda: safe_plot_execution(plot_aop_lifetime)),
        ('aop_property_presence', lambda: safe_plot_execution(plot_aop_property_presence)),
        ('aop_property_presence_unique', lambda: safe_plot_execution(plot_aop_property_presence_unique_colors)),
        ('kes_by_kec_count', lambda: safe_plot_execution(plot_kes_by_kec_count)),
        ('latest_entity_counts', lambda: safe_plot_execution(plot_latest_entity_counts)),
        ('latest_ke_components', lambda: safe_plot_execution(plot_latest_ke_components)),
        ('latest_network_density', lambda: safe_plot_execution(plot_latest_network_density)),
        ('latest_avg_per_aop', lambda: safe_plot_execution(plot_latest_avg_per_aop)),
        ('latest_ontology_usage', lambda: safe_plot_execution(plot_latest_ontology_usage)),
        ('latest_process_usage', lambda: safe_plot_execution(plot_latest_process_usage)),
        ('latest_object_usage', lambda: safe_plot_execution(plot_latest_object_usage)),
        ('latest_aop_completeness', lambda: safe_plot_execution(plot_latest_aop_completeness)),
        ('latest_aop_completeness_unique', lambda: safe_plot_execution(plot_latest_aop_completeness_unique_colors)),
        ('latest_database_summary', lambda: safe_plot_execution(plot_latest_database_summary)),
        ('latest_ke_annotation_depth', lambda: safe_plot_execution(plot_latest_ke_annotation_depth)),
    ]
    
    results = {}
    
    # Execute plots in parallel
    with ThreadPoolExecutor(max_workers=Config.PARALLEL_WORKERS) as executor:
        # Submit all tasks
        future_to_name = {executor.submit(task[1]): task[0] for task in plot_tasks}
        
        # Collect results as they complete
        for future in as_completed(future_to_name):
            plot_name = future_to_name[future]
            try:
                result = future.result(timeout=Config.PLOT_TIMEOUT)
                results[plot_name] = result
                logger.info(f"Plot {plot_name} completed successfully")
            except Exception as e:
                logger.error(f"Plot {plot_name} failed: {str(e)}")
                results[plot_name] = None
    
    total_time = time.time() - start_time
    successful_plots = sum(1 for v in results.values() if v is not None)
    logger.info(f"Plot computation completed in {total_time:.2f}s. {successful_plots}/{len(plot_tasks)} plots successful")
    
    return results

# --- Precompute plots at startup ---
plot_results = compute_plots_parallel()

# Extract results with fallbacks
try:
    graph_main_abs, graph_main_delta, df_all = plot_results.get('main_graph', (None, None, None))
    if graph_main_abs is None:
        graph_main_abs = graph_main_delta = ""
        df_all = pd.DataFrame()
except (TypeError, ValueError):
    graph_main_abs = graph_main_delta = ""
    df_all = pd.DataFrame()

try:
    graph_avg_abs, graph_avg_delta = plot_results.get('avg_per_aop', (None, None))
    if graph_avg_abs is None:
        graph_avg_abs = graph_avg_delta = ""
except (TypeError, ValueError):
    graph_avg_abs = graph_avg_delta = ""

# Extract other results with similar pattern
graph_density = plot_results.get('network_density') or ""
graph_components_abs, graph_components_delta = plot_results.get('ke_components', ("", ""))
graph_unique_abs, graph_unique_delta = plot_results.get('unique_ke_components', ("", ""))
graph_components_pct_abs, graph_components_pct_delta = plot_results.get('ke_components_percentage', ("", ""))
graph_bio_processes_abs, graph_bio_processes_delta = plot_results.get('bio_processes', ("", ""))
graph_bio_objects_abs, graph_bio_objects_delta = plot_results.get('bio_objects', ("", ""))
graph_authors_abs, graph_authors_delta = plot_results.get('author_counts', ("", ""))
graph_created, graph_modified, graph_scatter = plot_results.get('aop_lifetime', ("", "", ""))
graph_prop_abs, graph_prop_pct = plot_results.get('aop_property_presence', ("", ""))
graph_prop_unique_abs, graph_prop_unique_pct = plot_results.get('aop_property_presence_unique', ("", ""))
graph_kec_count_abs, graph_kec_count_delta = plot_results.get('kes_by_kec_count', ("", ""))

# Latest data plots
latest_entity_counts = plot_results.get('latest_entity_counts') or ""
latest_ke_components = plot_results.get('latest_ke_components') or ""
latest_network_density = plot_results.get('latest_network_density') or ""
latest_avg_per_aop = plot_results.get('latest_avg_per_aop') or ""
latest_ontology_usage = plot_results.get('latest_ontology_usage') or ""
latest_process_usage = plot_results.get('latest_process_usage') or ""
latest_object_usage = plot_results.get('latest_object_usage') or ""
latest_aop_completeness = plot_results.get('latest_aop_completeness') or ""
latest_aop_completeness_unique = plot_results.get('latest_aop_completeness_unique') or ""
latest_database_summary = plot_results.get('latest_database_summary') or ""
latest_ke_annotation_depth = plot_results.get('latest_ke_annotation_depth') or ""
# --- End of precomputed plots ---

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
                - status: "healthy", "degraded", or "error"
                - sparql_endpoint: "up" or "down"
                - plots_loaded: "X/Y" format showing successful plot ratio
                - timestamp: Unix timestamp of health check
            - int: HTTP status code (200 for healthy, 503 for degraded, 500 for error)
    
    HTTP Status Codes:
        200: Application is fully healthy (endpoint up, plots loaded)
        503: Application is degraded (endpoint down or no plots loaded)
        500: Health check itself failed (unexpected error)
    
    Example Response:
        >>> # Healthy response
        {
            "status": "healthy",
            "sparql_endpoint": "up", 
            "plots_loaded": "22/22",
            "timestamp": 1640995200.0
        }
        
        >>> # Degraded response  
        {
            "status": "degraded",
            "sparql_endpoint": "down",
            "plots_loaded": "0/22", 
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
        endpoint_healthy = check_sparql_endpoint_health()
        successful_plots = sum(1 for v in plot_results.values() if v is not None)
        total_plots = len(plot_results)
        
        health_status = {
            "status": "healthy" if endpoint_healthy and successful_plots > 0 else "degraded",
            "sparql_endpoint": "up" if endpoint_healthy else "down",
            "plots_loaded": f"{successful_plots}/{total_plots}",
            "timestamp": time.time()
        }
        
        return health_status, 200 if health_status["status"] == "healthy" else 503
        
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
    return render_template("status.html")

@app.route("/download/latest_entity_counts")
def download_latest_entity_counts():
    """Download CSV data for the Latest Entity Counts visualization.
    
    Provides CSV export functionality for the latest entity counts data,
    allowing users to download the raw data behind the bar chart visualization
    for further analysis or reporting purposes.
    
    The CSV includes:
        - Entity types (AOPs, KEs, KERs, Stressors, Authors)
        - Current counts for each entity type
        - Version information for data context
        - Metadata columns for reference
    
    Returns:
        Response: Flask Response object with CSV data as attachment
            - Content-Type: text/csv
            - Content-Disposition: attachment with filename
            - HTTP 200 on success, 404 if data unavailable, 500 on error
    
    CSV Format:
        Entity,Count,Version
        AOPs,245,2024-01-15
        KEs,1580,2024-01-15
        ...
        
    Error Handling:
        - 404: Data not available in cache (plot generation failed)
        - 500: Server error during CSV generation or file serving
    
    Example Usage:
        >>> import requests
        >>> response = requests.get("http://localhost:5000/download/latest_entity_counts")
        >>> if response.status_code == 200:
        ...     with open("entity_counts.csv", "wb") as f:
        ...         f.write(response.content)
        
        Or via browser: Direct download link in the dashboard interface
    
    Note:
        Data is cached from plot generation during application startup.
        If plots failed to generate, this endpoint returns 404.
    """
    plot_name = 'latest_entity_counts'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/latest_ke_components")
def download_latest_ke_components():
    """Download CSV data for Latest KE Components plot."""
    plot_name = 'latest_ke_components'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/latest_network_density")
def download_latest_network_density():
    """Download CSV data for Latest Network Density plot."""
    plot_name = 'latest_network_density'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/latest_avg_per_aop")
def download_latest_avg_per_aop():
    """Download CSV data for Latest Avg per AOP plot."""
    plot_name = 'latest_avg_per_aop'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/latest_process_usage")
def download_latest_process_usage():
    """Download CSV data for Latest Process Usage plot."""
    plot_name = 'latest_process_usage'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/latest_object_usage")
def download_latest_object_usage():
    """Download CSV data for Latest Object Usage plot."""
    plot_name = 'latest_object_usage'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/latest_aop_completeness")
def download_latest_aop_completeness():
    """Download CSV data for Latest AOP Completeness plot."""
    plot_name = 'latest_aop_completeness'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/latest_aop_completeness_unique")
def download_latest_aop_completeness_unique():
    """Download CSV data for Latest AOP Completeness (Unique Colors) plot."""
    plot_name = 'latest_aop_completeness_unique'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/aop_property_presence_unique_absolute")
def download_aop_property_presence_unique_absolute():
    """Download CSV data for AOP Property Presence Unique Colors (Absolute Count) plot."""
    plot_name = 'aop_property_presence_unique_absolute'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/aop_property_presence_unique_percentage")
def download_aop_property_presence_unique_percentage():
    """Download CSV data for AOP Property Presence Unique Colors (Percentage) plot."""
    plot_name = 'aop_property_presence_unique_percentage'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/latest_ke_annotation_depth")
def download_latest_ke_annotation_depth():
    """Download CSV data for Latest KE Annotation Depth plot."""
    plot_name = 'latest_ke_annotation_depth'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/main_graph_absolute")
def download_main_graph_absolute():
    """Download CSV data for Main Graph Absolute plot."""
    plot_name = 'main_graph_absolute'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

@app.route("/download/main_graph_delta")
def download_main_graph_delta():
    """Download CSV data for Main Graph Delta plot."""
    plot_name = 'main_graph_delta'
    export_format = request.args.get('format', 'csv').lower()
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'

    try:
        if export_format == 'csv':
            csv_data = get_csv_with_metadata(plot_name, include_metadata)
            if not csv_data:
                return "No data available for download", 404

            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={plot_name}.csv'}
            )

        elif export_format in ['png', 'svg']:
            image_bytes = export_figure_as_image(plot_name, export_format)
            if not image_bytes:
                return "No figure available for export", 404

            mimetype = f'image/{export_format}'
            return Response(
                image_bytes,
                mimetype=mimetype,
                headers={'Content-Disposition': f'attachment; filename={plot_name}.{export_format}'}
            )

        else:
            return f"Unsupported format: {export_format}. Use csv, png, or svg.", 400

    except Exception as e:
        logger.error(f"Download failed for {plot_name}: {str(e)}")
        return f"Download failed: {str(e)}", 500

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
        # Predefined plot categories
        categories = {
            'all': [
                'latest_entity_counts', 'latest_ke_components', 'latest_network_density',
                'latest_avg_per_aop', 'latest_process_usage', 'latest_object_usage',
                'latest_aop_completeness', 'latest_ke_annotation_depth'
            ],
            'database-state': ['latest_entity_counts'],
            'network-analysis': ['latest_network_density', 'latest_avg_per_aop'],
            'ke-analysis': ['latest_ke_components', 'latest_ke_annotation_depth'],
            'data-quality': ['latest_aop_completeness', 'latest_process_usage', 'latest_object_usage']
        }

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

        # Create ZIP file
        zip_bytes = create_bulk_download(plot_names, formats)
        if not zip_bytes:
            return "Failed to create bulk download", 500

        # Generate filename
        if category:
            filename = f"aopwiki_{category}_plots.zip"
        else:
            filename = f"aopwiki_{len(plot_names)}_plots.zip"

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

    # Map plot names to their corresponding variables or functions
    # For latest_* plots that support version parameter, we'll regenerate them
    # For historical trend plots, use pre-computed global variables
    plot_map = {
        'graph_main_abs': graph_main_abs,
        'graph_main_delta': graph_main_delta,
        'graph_avg_abs': graph_avg_abs,
        'graph_avg_delta': graph_avg_delta,
        'graph_density': graph_density,
        'graph_components_abs': graph_components_abs,
        'graph_components_delta': graph_components_delta,
        'graph_components_pct_abs': graph_components_pct_abs,
        'graph_components_pct_delta': graph_components_pct_delta,
        'graph_unique_abs': graph_unique_abs,
        'graph_unique_delta': graph_unique_delta,
        'graph_bio_processes_abs': graph_bio_processes_abs,
        'graph_bio_processes_delta': graph_bio_processes_delta,
        'graph_bio_objects_abs': graph_bio_objects_abs,
        'graph_bio_objects_delta': graph_bio_objects_delta,
        'graph_authors_abs': graph_authors_abs,
        'graph_authors_delta': graph_authors_delta,
        'graph_created': graph_created,
        'graph_modified': graph_modified,
        'graph_scatter': graph_scatter,
        'graph_prop_abs': graph_prop_abs,
        'graph_prop_pct': graph_prop_pct,
        'graph_prop_unique_abs': graph_prop_unique_abs,
        'graph_prop_unique_pct': graph_prop_unique_pct,
        'graph_kec_count_abs': graph_kec_count_abs,
        'graph_kec_count_delta': graph_kec_count_delta
    }

    # Handle latest_* plots dynamically with version support
    latest_plots_with_version = {
        'latest_entity_counts': plot_latest_entity_counts,
        'latest_ke_components': plot_latest_ke_components,
        'latest_network_density': plot_latest_network_density,
        'latest_avg_per_aop': plot_latest_avg_per_aop,
        'latest_process_usage': plot_latest_process_usage,
        'latest_object_usage': plot_latest_object_usage,
        'latest_aop_completeness': plot_latest_aop_completeness,
        'latest_aop_completeness_unique': plot_latest_aop_completeness_unique_colors,
        'latest_ontology_usage': plot_latest_ontology_usage,
        'latest_database_summary': plot_latest_database_summary,
        'latest_ke_annotation_depth': plot_latest_ke_annotation_depth,
    }

    # Handle latest_* plots without version support yet (use pre-computed)
    latest_plots_precomputed = {}

    # Check if it's a versioned latest plot
    if plot_name in latest_plots_with_version:
        try:
            plot_function = latest_plots_with_version[plot_name]
            html = plot_function(version) if version else plot_function()
            return jsonify({'html': html, 'success': True})
        except Exception as e:
            logger.error(f"Error generating plot {plot_name} with version {version}: {e}")
            return jsonify({'error': str(e), 'success': False}), 500

    # Check precomputed latest plots
    elif plot_name in latest_plots_precomputed:
        return jsonify({'html': latest_plots_precomputed[plot_name], 'success': True})

    # Check historical trend plots
    elif plot_name in plot_map:
        return jsonify({'html': plot_map[plot_name], 'success': True})

    else:
        return jsonify({'error': f'Plot {plot_name} not found', 'success': False}), 404



# Set Plotly configuration for static images
@app.route("/old-dashboard")
def index():
    """Serve the main dashboard page with all visualizations.
    
    Renders the primary dashboard interface containing all precomputed
    visualizations in a professional tabbed layout. This is the main
    entry point for users accessing the AOP-Wiki RDF Dashboard.
    
    The dashboard includes:
        - Latest Data tab: Current snapshot visualizations
        - Historical Trends tab: Time-series analysis 
        - Professional navigation with VHP4Safety branding
        - Interactive Plotly charts with download capabilities
        - Responsive design for desktop and mobile
        - CSV export buttons for all visualizations
    
    Visualization Categories:
        Latest Data:
            - Entity counts (bar chart)
            - KE component distribution (pie chart)
            - Network connectivity analysis
            - Ontology usage statistics
            - Data completeness metrics
            
        Historical Trends:
            - Main entity evolution over time
            - Average components per AOP trends
            - Network density changes
            - Author contribution patterns
            - Component annotation trends
    
    Returns:
        str: Rendered HTML template with all visualizations embedded
    
    Template Variables:
        Passes 20+ visualization HTML strings to index.html template:
        - graph_main_abs, graph_main_delta: Main entity trends
        - latest_entity_counts: Current entity statistics
        - latest_ke_components: Component distribution
        - And many more visualization variables
    
    Performance:
        - All plots are precomputed at startup for fast serving
        - Uses CDN-hosted Plotly.js for optimal loading
        - Responsive configuration for all chart types
        - Global caching prevents recomputation on each request
    
    Example:
        Access the dashboard: http://localhost:5000/
        
    Note:
        If plot generation failed during startup, some visualizations
        may display error messages or fallback content.
    """
    # Serve the original tabbed dashboard (for backward compatibility)
    return render_template("index.html", lazy_loading=True)


@app.route("/")
def landing():
    """Serve the clean landing page with navigation buttons.

    This is the new main entry point that provides a clean introduction
    to the service with prominent navigation buttons to Latest Data and
    Historical Trends pages.
    """
    return render_template("landing.html")


@app.route("/snapshot")
def database_snapshot():
    """Serve the Database Snapshot page with version-selectable visualizations.

    Displays key metrics and visualizations from any version of the AOP-Wiki
    database, allowing users to explore current and historical snapshots.
    """
    return render_template("latest.html")


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
    return render_template("trends_page.html")


@app.route("/dashboard")
def dashboard():
    """Serve the original tabbed dashboard (legacy route).

    Maintains backward compatibility for users who may have bookmarked
    the original tabbed interface.
    """
    return render_template("index.html", lazy_loading=True)





# Run the Flask app
if __name__ == "__main__":
    logger.info(f"Starting AOP-Wiki RDF Dashboard on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    logger.info(f"SPARQL Endpoint: {Config.SPARQL_ENDPOINT}")
    logger.info(f"Configuration: {Config.get_config_dict()}")
    
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)
