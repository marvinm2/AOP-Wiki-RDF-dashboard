from flask import Flask, render_template
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
from flask import Flask, render_template, url_for

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
    plot_database_size,
    plot_kes_by_kec_count,
    plot_latest_entity_counts,
    plot_latest_ke_components,
    check_sparql_endpoint_health,
    safe_plot_execution
)

def compute_plots_parallel() -> dict:
    """Compute all plots in parallel for better performance."""
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
        ('database_size', lambda: safe_plot_execution(plot_database_size)),
        ('kes_by_kec_count', lambda: safe_plot_execution(plot_kes_by_kec_count)),
        ('latest_entity_counts', lambda: safe_plot_execution(plot_latest_entity_counts)),
        ('latest_ke_components', lambda: safe_plot_execution(plot_latest_ke_components))
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
graph_db_size_abs, graph_db_size_delta = plot_results.get('database_size', ("", ""))
graph_kec_count_abs, graph_kec_count_delta = plot_results.get('kes_by_kec_count', ("", ""))

# Latest data plots
latest_entity_counts = plot_results.get('latest_entity_counts') or ""
latest_ke_components = plot_results.get('latest_ke_components') or ""
# --- End of precomputed plots ---

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring."""
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
    """Status page with real-time monitoring."""
    return render_template("status.html")

# Set Plotly configuration for static images
@app.route("/")
def index():
    return render_template(
        "index.html",
        graph_main_abs=graph_main_abs,
        graph_main_delta=graph_main_delta,
        graph_avg_abs=graph_avg_abs,
        graph_avg_delta=graph_avg_delta,
        graph_density=graph_density,
        graph_components_abs=graph_components_abs,
        graph_components_delta=graph_components_delta,
        graph_components_pct_abs=graph_components_pct_abs,
        graph_components_pct_delta=graph_components_pct_delta,
        graph_unique_abs=graph_unique_abs,        # NEW
        graph_unique_delta=graph_unique_delta,    # NEW
        graph_bio_processes_abs=graph_bio_processes_abs,
        graph_bio_processes_delta=graph_bio_processes_delta,
        graph_bio_objects_abs=graph_bio_objects_abs,
        graph_bio_objects_delta=graph_bio_objects_delta,
        graph_authors_abs=graph_authors_abs,
        graph_authors_delta=graph_authors_delta,
        graph_created=graph_created,
        graph_modified=graph_modified,
        graph_scatter=graph_scatter,
        graph_prop_abs=graph_prop_abs,
        graph_prop_pct=graph_prop_pct,
        graph_db_size_abs=graph_db_size_abs,
        graph_db_size_delta=graph_db_size_delta,
        graph_kec_count_abs=graph_kec_count_abs,
        graph_kec_count_delta=graph_kec_count_delta,
        latest_entity_counts=latest_entity_counts,
        latest_ke_components=latest_ke_components
    )





# Run the Flask app
if __name__ == "__main__":
    logger.info(f"Starting AOP-Wiki RDF Dashboard on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    logger.info(f"SPARQL Endpoint: {Config.SPARQL_ENDPOINT}")
    logger.info(f"Configuration: {Config.get_config_dict()}")
    
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)
