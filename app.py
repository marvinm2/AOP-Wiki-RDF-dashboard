from flask import Flask, render_template
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import plotly.express as px
import plotly.io as pio
import os
from functools import reduce

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
    plot_latest_ke_components
)

# --- Precompute plots at startup ---
graph_main_abs, graph_main_delta, df_all = plot_main_graph()
graph_avg_abs, graph_avg_delta = plot_avg_per_aop()
graph_density = plot_network_density()
graph_components_abs, graph_components_delta = plot_ke_components()
graph_unique_abs, graph_unique_delta = plot_unique_ke_components()
graph_components_pct_abs, graph_components_pct_delta = plot_ke_components_percentage()
graph_bio_processes_abs, graph_bio_processes_delta = plot_bio_processes()
graph_bio_objects_abs, graph_bio_objects_delta = plot_bio_objects()
graph_authors_abs, graph_authors_delta = plot_author_counts()
graph_created, graph_modified, graph_scatter = plot_aop_lifetime()
graph_prop_abs, graph_prop_pct = plot_aop_property_presence()
graph_db_size_abs, graph_db_size_delta = plot_database_size()
graph_kec_count_abs, graph_kec_count_delta = plot_kes_by_kec_count()

# --- Latest data plots ---
latest_entity_counts = plot_latest_entity_counts()
latest_ke_components = plot_latest_ke_components()
# --- End of precomputed plots ---

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
    app.run(host="0.0.0.0", port=5000)
