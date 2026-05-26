"""AOP-Wiki RDF Dashboard - Latest Data Snapshot Plot Functions.

This module contains all current snapshot visualization functions for the AOP-Wiki
RDF Dashboard. These functions analyze the most recent RDF graph version to provide
immediate insights into the current state of the AOP-Wiki database.

Key Features:
    - Current state analysis from latest RDF graph version
    - Immediate database composition overview
    - Real-time data quality assessment
    - Comprehensive CSV export functionality for all visualizations
    - Professional styling with VHP4Safety branding

Latest Data Plot Functions:
    Core Database State:
        - plot_latest_entity_counts(): Current entity distribution (AOPs, KEs, KERs, etc.)
        - plot_latest_database_summary(): Core entity summary overview
        - plot_latest_avg_per_aop(): Current average components per AOP

    Component Analysis:
        - plot_latest_ke_components(): KE component distribution
        - plot_latest_ke_annotation_depth(): Annotation depth analysis

    Connectivity Analysis:
        - plot_latest_network_density(): AOP connectivity assessment

    Ontology Usage:
        - plot_latest_ontology_usage(): General ontology term usage
        - plot_latest_process_usage(): Biological process ontology sources
        - plot_latest_object_usage(): Biological object ontology sources

    Data Quality:
        - plot_latest_aop_completeness(): Property completeness analysis
        - plot_latest_aop_completeness_unique_colors(): Enhanced completeness visualization

Visualization Characteristics:
    - Current state focus (latest RDF graph version only)
    - Interactive charts with hover details and responsive design
    - Consistent VHP4Safety color palette and professional styling
    - Automatic version detection and labeling
    - Global data caching for CSV export functionality

Data Processing:
    - Latest version detection via ORDER BY DESC(?graph) LIMIT 1
    - Robust SPARQL query execution with error handling
    - Data validation and quality checks
    - Version consistency across all visualizations

Performance Features:
    - Single-version queries for optimal speed
    - Efficient data processing and caching
    - Minimal SPARQL endpoint load
    - Fast dashboard loading and rendering

Author:
    Marvin Martens
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from .shared import (
    BRAND_COLORS, config, _plot_data_cache, _plot_figure_cache, run_sparql_query, safe_read_csv, create_fallback_plot,
    render_plot_html
)
from .organ_systems import (
    ORGAN_SYSTEM_BUCKETS,
    NO_ANNOTATION_BUCKET,
    SIGNAL_COLOURS,
    SIGNAL_ORDER,
    best_signal,
    classify_anatomy,
    classify_process,
    classify_text,
)

logger = logging.getLogger(__name__)


def _build_graph_filter(version: str = None) -> tuple[str, str]:
    """Helper function to build SPARQL graph filter clause based on version parameter.

    Args:
        version: Optional version string (e.g., "2024-10-01"). If None, returns latest.

    Returns:
        tuple: (where_filter, order_limit) - SPARQL clauses for WHERE and after GROUP BY
    """
    if version:
        # For specific version: add FILTER in WHERE, no ORDER/LIMIT needed
        where_filter = f'FILTER(?graph = <http://aopwiki.org/graph/{version}>)'
        order_limit = ''
    else:
        # For latest version: no extra WHERE filter, use ORDER BY LIMIT after GROUP BY
        where_filter = ''
        order_limit = 'ORDER BY DESC(?graph) LIMIT 1'

    return where_filter, order_limit


def plot_latest_entity_counts(version: str = None) -> str:
    """Create a bar chart visualization of current AOP entity counts from the latest RDF version.

    Generates an interactive bar chart showing the current state of all major AOP-Wiki
    entities in the most recent RDF graph version. This provides a snapshot view of
    the database size and composition, complementing the historical trend analysis.

    The function queries the latest RDF graph to count different entity types and
    creates a professionally styled bar chart with VHP4Safety branding. Data is
    automatically cached for CSV export functionality.

    Entity Types Analyzed:
        - AOPs (AdverseOutcomePathway): Complete adverse outcome pathways
        - KEs (KeyEvent): Individual key events within pathways
        - KERs (KeyEventRelationship): Relationships between key events
        - Stressors (nci:C54571): Chemical and physical stressors
        - Authors: Unique contributors to AOP content

    SPARQL Query Strategy:
        - Executes 5 separate queries for each entity type
        - Uses ORDER BY DESC(?graph) LIMIT 1 to get latest version
        - Ensures consistent version across all entity counts
        - Handles missing entities gracefully (returns 0 counts)

    Returns:
        str: Complete HTML string containing the interactive Plotly bar chart.
            Returns fallback HTML with error message if data unavailable.

    Data Processing:
        1. Executes entity-specific SPARQL queries for latest version
        2. Extracts counts and validates data integrity
        3. Creates structured DataFrame with entity types and counts
        4. Adds version and metadata columns for context
        5. Caches complete dataset in global _plot_data_cache

    Visualization Features:
        - Interactive bar chart with hover details
        - VHP4Safety color palette for brand consistency
        - Text labels on bars showing exact counts
        - Responsive design for mobile and desktop
        - Professional styling with proper margins
        - No legend (entity names on x-axis are self-explanatory)

    Global Caching:
        Stores complete dataset in _plot_data_cache['latest_entity_counts']:
        >>> {
        ...     'Entity': ['AOPs', 'KEs', 'KERs', 'Stressors', 'Authors'],
        ...     'Count': [142, 890, 780, 52, 28],
        ...     'Version': ['2024-01-15', '2024-01-15', ...]
        ... }

    Example Usage:
        >>> # Generate current entity snapshot
        >>> html_chart = plot_latest_entity_counts()
        >>> print(f"Generated chart HTML: {len(html_chart)} characters")

        >>> # Access cached data for analysis
        >>> from plots import _plot_data_cache
        >>> if 'latest_entity_counts' in _plot_data_cache:
        ...     df = _plot_data_cache['latest_entity_counts']
        ...     total_entities = df['Count'].sum()
        ...     print(f"Total entities in latest version: {total_entities}")

    Error Handling:
        - No query results: Returns "No data available" message
        - SPARQL failures: Handled by run_sparql_query_with_retry()
        - Version inconsistencies: Uses first available version
        - Visualization errors: Returns fallback HTML content

    Performance:
        - Single-pass data collection for efficiency
        - Minimal SPARQL query load (5 simple COUNT queries)
        - Cached results prevent recomputation
        - Optimized for dashboard loading speed

    Dashboard Integration:
        - Displayed in "Latest Data" tab of main dashboard
        - Provides immediate overview of database size
        - Links to CSV download functionality
        - Updates automatically with new RDF versions

    Version Handling:
        - Automatically detects most recent RDF graph version
        - Handles version string extraction from graph URIs
        - Ensures data consistency across all entity types
        - Adapts to new versions without code changes

    Note:
        This function provides the "current state" complement to historical
        trend analysis. It's designed to give users an immediate understanding
        of the AOP-Wiki's current scale and composition.
    """
    global _plot_data_cache

    # Build graph filter based on version parameter
    where_filter, order_limit = _build_graph_filter(version)

    sparql_queries = {
        "AOPs": f"""
            SELECT ?graph (COUNT(?aop) AS ?count)
            WHERE {{
                GRAPH ?graph {{ ?aop a aopo:AdverseOutcomePathway . }}
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                {where_filter}
            }}
            GROUP BY ?graph
            {order_limit}
        """,
        "KEs": f"""
            SELECT ?graph (COUNT(DISTINCT ?ke) AS ?count)
            WHERE {{
                GRAPH ?graph {{
                    ?ke a aopo:KeyEvent .
                }}
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                {where_filter}
            }}
            GROUP BY ?graph
            {order_limit}
        """,
        "KERs": f"""
            SELECT ?graph (COUNT(DISTINCT ?ker) AS ?count)
            WHERE {{
                GRAPH ?graph {{
                    ?ker a aopo:KeyEventRelationship  .
                }}
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                {where_filter}
            }}
            GROUP BY ?graph
            {order_limit}
        """,
        "Stressors": f"""
            SELECT ?graph (COUNT(?s) AS ?count)
            WHERE {{
                GRAPH ?graph {{
                    ?s a nci:C54571 .
                }}
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                {where_filter}
            }}
            GROUP BY ?graph
            {order_limit}
        """,
        "Authors": f"""
            SELECT ?graph (COUNT(DISTINCT ?c) AS ?count)
            WHERE {{
                GRAPH ?graph {{
                    ?aop a aopo:AdverseOutcomePathway ;
                         dc:creator ?c .
                }}
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                {where_filter}
            }}
            GROUP BY ?graph
            {order_limit}
        """
    }

    # Get latest counts for each entity type
    data = []
    latest_version = None

    for entity_type, query in sparql_queries.items():
        results = run_sparql_query(query)
        if results:
            count = int(results[0]["count"]["value"])
            version = results[0]["graph"]["value"].split("/")[-1]
            if latest_version is None:
                latest_version = version
            data.append({"Entity": entity_type, "Count": count})

    if not data:
        return "<p>No data available</p>"

    df = pd.DataFrame(data)
    df["Version"] = latest_version  # Add version column for context

    # Store in global cache for CSV download
    _plot_data_cache['latest_entity_counts'] = df

    fig = px.bar(
        df,
        x="Entity",
        y="Count",
        text="Count",
    )

    fig.update_traces(marker_color=BRAND_COLORS['blue'], textposition='outside')
    fig.update_layout(

        showlegend=False,

        margin=dict(l=50, r=20, t=50, b=50),
        yaxis=dict(title="Count"),
        xaxis=dict(title="Entity Type")
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_entity_counts'] = fig

    return render_plot_html(fig)


def plot_latest_ke_components(version: str = None) -> str:
    """Create a pie chart showing the current KE component distribution."""
    global _plot_data_cache

    # Determine target graph directly to avoid Virtuoso cross-graph timeout
    # with triple OPTIONALs on bioevent sub-predicates
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        # Get the latest graph
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:KeyEvent . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("KE Component Distribution", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    query_components = f"""
    SELECT (COUNT(?process) AS ?process_count)
           (COUNT(?object) AS ?object_count)
           (COUNT(?action) AS ?action_count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL {{ ?bioevent aopo:hasProcess ?process . }}
        OPTIONAL {{ ?bioevent aopo:hasObject ?object . }}
        OPTIONAL {{ ?bioevent aopo:hasAction ?action . }}
      }}
    }}
    """

    results = run_sparql_query(query_components)
    if not results:
        return create_fallback_plot("KE Component Distribution", "No data available")

    result = results[0]

    data = [
        {"Component": "Process", "Count": int(result["process_count"]["value"])},
        {"Component": "Object", "Count": int(result["object_count"]["value"])},
        {"Component": "Action", "Count": int(result["action_count"]["value"])}
    ]

    df = pd.DataFrame(data)
    df = df[df["Count"] > 0]  # Remove zero counts

    if df.empty:
        return "<p>No component data available</p>"

    # Add version for context and cache data
    df["Version"] = latest_version
    _plot_data_cache['latest_ke_components'] = df

    fig = px.pie(
        df,
        values="Count",
        names="Component",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=50)
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_ke_components'] = fig

    return render_plot_html(fig)


def plot_latest_network_density(version: str = None) -> str:
    """Analyze current AOP connectivity based on shared Key Events."""
    global _plot_data_cache

    # Build graph filter based on version parameter
    where_filter, order_limit = _build_graph_filter(version)

    # First, get total AOPs in latest version
    query_total = f"""
    SELECT ?graph (COUNT(DISTINCT ?aop) AS ?total_aops)
    WHERE {{
        GRAPH ?graph {{
            ?aop a aopo:AdverseOutcomePathway .
        }}
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        {where_filter}
    }}
    GROUP BY ?graph
    {order_limit}
    """

    # Second, get AOPs that share at least one KE with another AOP
    query_connected = f"""
    SELECT ?graph (COUNT(DISTINCT ?aop1) AS ?connected_aops)
    WHERE {{
        GRAPH ?graph {{
            ?aop1 a aopo:AdverseOutcomePathway ;
                  aopo:has_key_event ?ke .
            ?aop2 a aopo:AdverseOutcomePathway ;
                  aopo:has_key_event ?ke .
            FILTER(?aop1 != ?aop2)
        }}
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        {where_filter}
    }}
    GROUP BY ?graph
    {order_limit}
    """

    total_results = run_sparql_query(query_total)
    connected_results = run_sparql_query(query_connected)

    if not total_results:
        return create_fallback_plot("AOP Connectivity", "No AOP data available")

    total_aops = int(total_results[0]["total_aops"]["value"])
    connected_aops = int(connected_results[0]["connected_aops"]["value"]) if connected_results else 0
    isolated_aops = total_aops - connected_aops

    latest_version = total_results[0]["graph"]["value"].split("/")[-1]

    if total_aops == 0:
        return create_fallback_plot("AOP Connectivity", "No AOPs found in latest version")

    data = [
        {"Type": "Connected AOPs", "Count": connected_aops, "Description": "Share Key Events with other AOPs"},
        {"Type": "Isolated AOPs", "Count": isolated_aops, "Description": "Have unique Key Events"}
    ]
    df = pd.DataFrame(data)
    df["Version"] = latest_version  # Add version for context
    df["Total_AOPs"] = total_aops   # Add total for reference

    # Store in global cache for CSV download
    _plot_data_cache['latest_aop_connectivity'] = df

    fig = px.pie(
        df, values="Count", names="Type",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=50),
        annotations=[
            dict(text=f"Total AOPs: {total_aops}<br>Connected: {connected_aops}<br>Isolated: {isolated_aops}",
                 x=0.5, y=0.1, font_size=12, showarrow=False)
        ]
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_aop_connectivity'] = fig

    return render_plot_html(fig)


def plot_latest_avg_per_aop(version: str = None) -> str:
    """Create a bar chart showing current average KEs and KERs per AOP."""
    global _plot_data_cache

    where_filter, order_limit = _build_graph_filter(version)

    query_aops = f"""
        SELECT ?graph (COUNT(?aop) AS ?count)
        WHERE {{
            GRAPH ?graph {{ ?aop a aopo:AdverseOutcomePathway . }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            {where_filter}
        }}
        GROUP BY ?graph
        {order_limit}
    """
    query_kes = f"""
        SELECT ?graph (COUNT(?ke) AS ?count)
        WHERE {{
            GRAPH ?graph {{
                ?ke a aopo:KeyEvent .
            }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            {where_filter}
        }}
        GROUP BY ?graph
        {order_limit}
    """
    query_kers = f"""
        SELECT ?graph (COUNT(?ker) AS ?count)
        WHERE {{
            GRAPH ?graph {{
                ?ker a aopo:KeyEventRelationship .
            }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            {where_filter}
        }}
        GROUP BY ?graph
        {order_limit}
    """

    results_aops = run_sparql_query(query_aops)
    results_kes = run_sparql_query(query_kes)
    results_kers = run_sparql_query(query_kers)

    if not all([results_aops, results_kes, results_kers]):
        return create_fallback_plot("Average per AOP", "Insufficient data available")

    aop_count = int(results_aops[0]["count"]["value"])
    ke_count = int(results_kes[0]["count"]["value"])
    ker_count = int(results_kers[0]["count"]["value"])
    latest_version = results_aops[0]["graph"]["value"].split("/")[-1]

    if aop_count == 0:
        return create_fallback_plot("Average per AOP", "No AOPs found")

    avg_kes = ke_count / aop_count
    avg_kers = ker_count / aop_count

    data = [
        {"Metric": "Avg KEs per AOP", "Value": avg_kes},
        {"Metric": "Avg KERs per AOP", "Value": avg_kers}
    ]
    df = pd.DataFrame(data)
    df["Version"] = latest_version  # Add version for context
    df["AOP_Count"] = aop_count     # Add total AOPs for reference
    df["KE_Count"] = [ke_count, ke_count]    # Add raw counts for reference
    df["KER_Count"] = [ker_count, ker_count]  # Add raw counts for reference

    # Store in global cache for CSV download
    _plot_data_cache['latest_avg_per_aop'] = df

    fig = px.bar(
        df, x="Metric", y="Value",
        text="Value",
    )

    fig.update_traces(marker_color=BRAND_COLORS['blue'], texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(

        showlegend=False,

        margin=dict(l=50, r=20, t=50, b=50)
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_avg_per_aop'] = fig

    return render_plot_html(fig)


def plot_latest_ontology_usage(version: str = None) -> str:
    """Create a chart showing current ontology usage."""

    # Determine target graph directly for reliable query execution
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        # Get the latest graph
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:KeyEvent . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("Ontology Usage", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    query = f"""
    SELECT ?ontology (COUNT(DISTINCT ?term) AS ?count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        {{
          ?bioevent aopo:hasProcess ?term .
        }} UNION {{
          ?bioevent aopo:hasObject ?term .
        }} UNION {{
          ?bioevent aopo:hasAction ?term .
        }}

        BIND(
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/GO_"), "GO",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/MP_"), "MP",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/NBO_"), "NBO",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/MI_"), "MI",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/VT_"), "VT",
          IF(STRSTARTS(STR(?term), "http://purl.org/commons/record/mesh/"), "MESH",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/HP_"), "HP", "OTHER"))))))) AS ?ontology)
      }}
    }}
    GROUP BY ?ontology
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Ontology Usage", "No ontology data available")

    latest_data = results

    data = []
    for r in latest_data:
        if "ontology" in r:
            data.append({
                "Ontology": r["ontology"]["value"],
                "Terms": int(r["count"]["value"])
            })

    if not data:
        return create_fallback_plot("Ontology Usage", "No ontology terms found")

    df = pd.DataFrame(data)
    df["Version"] = latest_version
    _plot_data_cache['latest_ontology_usage'] = df

    fig = px.pie(
        df, values="Terms", names="Ontology",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=50)
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_ontology_usage'] = fig

    return render_plot_html(fig)


def plot_latest_process_usage(version: str = None) -> str:
    """Create a pie chart showing current ontology source distribution for biological processes."""
    global _plot_data_cache

    # First, determine which graph to use
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
    else:
        # Get the latest graph - using KeyEvent for faster query
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:KeyEvent . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("Process Ontology Sources", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]

    # Now query that specific graph
    query = f"""
    SELECT ?ontology (COUNT(DISTINCT ?process) AS ?count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        ?bioevent aopo:hasProcess ?process .

        BIND(
          IF(STRSTARTS(STR(?process), "http://purl.obolibrary.org/obo/GO_"), "GO",
          IF(STRSTARTS(STR(?process), "http://purl.obolibrary.org/obo/MP_"), "MP",
          IF(STRSTARTS(STR(?process), "http://purl.obolibrary.org/obo/NBO_"), "NBO",
          IF(STRSTARTS(STR(?process), "http://purl.obolibrary.org/obo/MI_"), "MI",
          IF(STRSTARTS(STR(?process), "http://purl.obolibrary.org/obo/VT_"), "VT",
          IF(STRSTARTS(STR(?process), "http://purl.org/commons/record/mesh/"), "MESH",
          IF(STRSTARTS(STR(?process), "http://purl.obolibrary.org/obo/HP_"), "HP", "OTHER"))))))) AS ?ontology)
      }}
    }}
    GROUP BY ?ontology
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Process Ontology Sources", "No process ontology data available")

    latest_version = target_graph.split("/")[-1]
    latest_data = results

    data = []
    for r in latest_data:
        if "ontology" in r and r["ontology"]["value"]:
            ontology = r["ontology"]["value"]
            count = int(r["count"]["value"])
            data.append({
                "Ontology": ontology,
                "Count": count
            })

    if not data:
        return create_fallback_plot("Process Ontology Sources", "No ontology source data found")

    df = pd.DataFrame(data).sort_values("Count", ascending=False)
    df["Version"] = latest_version  # Add version for context
    df["Component_Type"] = "Process"  # Add component type for clarity

    # Store in global cache for CSV download
    _plot_data_cache['latest_process_usage'] = df

    fig = px.pie(
        df, values="Count", names="Ontology",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=50)
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_process_usage'] = fig

    return render_plot_html(fig)


def plot_latest_object_usage(version: str = None) -> str:
    """Create a pie chart showing current ontology source distribution for biological objects."""
    global _plot_data_cache

    # First, determine which graph to use
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
    else:
        # Get the latest graph - using KeyEvent for faster query
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:KeyEvent . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("Object Ontology Sources", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]

    # Now query that specific graph
    query = f"""
    SELECT ?ontology (COUNT(DISTINCT ?object) AS ?count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        ?bioevent aopo:hasObject ?object .

        BIND(
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/GO_"), "GO",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/CHEBI_"), "CHEBI",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/PR_"), "PR",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/CL_"), "CL",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/UBERON_"), "UBERON",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/NCBITaxon_"), "NCBITaxon",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/DOID_"), "DOID",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/HP_"), "HP", "OTHER")))))))) AS ?ontology)
      }}
    }}
    GROUP BY ?ontology
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Object Ontology Sources", "No object ontology data available")

    latest_version = target_graph.split("/")[-1]
    latest_data = results

    data = []
    for r in latest_data:
        if "ontology" in r and r["ontology"]["value"]:
            ontology = r["ontology"]["value"]
            count = int(r["count"]["value"])
            data.append({
                "Ontology": ontology,
                "Count": count
            })

    if not data:
        return create_fallback_plot("Object Ontology Sources", "No ontology source data found")

    df = pd.DataFrame(data).sort_values("Count", ascending=False)
    df["Version"] = latest_version  # Add version for context
    df["Component_Type"] = "Object"  # Add component type for clarity

    # Store in global cache for CSV download
    _plot_data_cache['latest_object_usage'] = df

    fig = px.pie(
        df, values="Count", names="Ontology",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=50)
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_object_usage'] = fig

    return render_plot_html(fig)


def plot_latest_aop_completeness(version: str = None) -> str:
    """Create a chart showing current AOP data completeness for all properties."""
    global _plot_data_cache

    where_filter, order_limit = _build_graph_filter(version)

    # Default fallback properties for essential AOP completeness
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential"},
        {"uri": "http://purl.org/dc/terms/abstract", "label": "Abstract", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata"},
        {"uri": "http://aopkb.org/aop_ontology#has_key_event", "label": "Has Key Event", "type": "Content"},
        {"uri": "http://aopkb.org/aop_ontology#has_adverse_outcome", "label": "Has Adverse Outcome", "type": "Content"}
    ]
    properties_df = safe_read_csv("property_labels.csv", default_properties)
    properties = properties_df.to_dict(orient="records")

    # Query total number of AOPs
    total_query = f"""
    SELECT ?graph (COUNT(DISTINCT ?aop) AS ?total_aops)
    WHERE {{
        GRAPH ?graph {{ ?aop a aopo:AdverseOutcomePathway . }}
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        {where_filter}
    }}
    GROUP BY ?graph
    {order_limit}
    """
    total_results = run_sparql_query(total_query)
    if not total_results:
        return create_fallback_plot("AOP Completeness", "No AOP data available")

    total_aops = int(total_results[0]["total_aops"]["value"])
    latest_version = total_results[0]["graph"]["value"].split("/")[-1]

    # Use elegant single GROUP BY query to count all properties at once
    # This queries what properties AOPs actually have, much faster than checking each one

    # First, we need to ensure we query only the target graph
    # If version is specified, use it directly. Otherwise, query for latest graph first.
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
    else:
        target_graph = f"http://aopwiki.org/graph/{latest_version}"

    property_count_query = f"""
    SELECT DISTINCT ?p (COUNT(DISTINCT ?aop) AS ?n)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway ;
                 ?p ?o .
        }}
    }}
    GROUP BY ?p
    """

    results = run_sparql_query(property_count_query)

    # Build a map of property URI -> count
    property_counts = {}
    for result in results:
        prop_uri = result["p"]["value"]
        count = int(result["n"]["value"])
        property_counts[prop_uri] = count

    # Match with our property labels and calculate completeness
    # Only include properties that are actually used (count > 0)
    completeness_data = []
    for prop in properties:
        uri = prop["uri"]
        count = property_counts.get(uri, 0)

        # Skip properties with 0 count (not relevant for AOPs)
        if count == 0:
            continue

        completeness = (count / total_aops) * 100

        completeness_data.append({
            "Property": prop["label"],
            "Completeness": completeness,
            "Type": prop["type"],
            "URI": uri,
            "Count": count
        })

    df = pd.DataFrame(completeness_data)
    df["Version"] = latest_version  # Add version for context
    df["Total_AOPs"] = total_aops   # Add total AOPs for reference

    # Store in global cache for CSV download
    _plot_data_cache['latest_aop_completeness'] = df

    # Use centralized brand colors for consistency
    color_map = BRAND_COLORS['type_colors'].copy()
    # Add fallback for any missing types
    color_map.update({"Structure": BRAND_COLORS['accent']})

    fig = px.bar(
        df, x="Property", y="Completeness", color="Type",
        text="Completeness",
        color_discrete_map=color_map
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=100),
        yaxis=dict(title="Completeness (%)", range=[0, 105]),
        xaxis=dict(title="AOP Properties", tickangle=45),
        legend=dict(title="Property Type")
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_aop_completeness'] = fig

    return render_plot_html(fig)


def plot_latest_aop_completeness_unique_colors(version: str = None) -> str:
    """Create a chart showing current AOP data completeness with unique colors for each property."""
    global _plot_data_cache

    where_filter, order_limit = _build_graph_filter(version)

    # Default fallback properties for essential AOP completeness
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential"},
        {"uri": "http://purl.org/dc/terms/abstract", "label": "Abstract", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata"},
        {"uri": "http://aopkb.org/aop_ontology#has_key_event", "label": "Has Key Event", "type": "Content"},
        {"uri": "http://aopkb.org/aop_ontology#has_adverse_outcome", "label": "Has Adverse Outcome", "type": "Content"}
    ]
    properties_df = safe_read_csv("property_labels.csv", default_properties)
    properties = properties_df.to_dict(orient="records")

    # Query total number of AOPs
    total_query = f"""
    SELECT ?graph (COUNT(DISTINCT ?aop) AS ?total_aops)
    WHERE {{
        GRAPH ?graph {{ ?aop a aopo:AdverseOutcomePathway . }}
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        {where_filter}
    }}
    GROUP BY ?graph
    {order_limit}
    """
    total_results = run_sparql_query(total_query)
    if not total_results:
        return create_fallback_plot("AOP Completeness (Unique Colors)", "No AOP data available")

    total_aops = int(total_results[0]["total_aops"]["value"])
    latest_version = total_results[0]["graph"]["value"].split("/")[-1]

    # Use elegant single GROUP BY query to count all properties at once
    # This queries what properties AOPs actually have, much faster than checking each one

    # First, we need to ensure we query only the target graph
    # If version is specified, use it directly. Otherwise, query for latest graph first.
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
    else:
        target_graph = f"http://aopwiki.org/graph/{latest_version}"

    property_count_query = f"""
    SELECT DISTINCT ?p (COUNT(DISTINCT ?aop) AS ?n)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway ;
                 ?p ?o .
        }}
    }}
    GROUP BY ?p
    """

    results = run_sparql_query(property_count_query)

    # Build a map of property URI -> count
    property_counts = {}
    for result in results:
        prop_uri = result["p"]["value"]
        count = int(result["n"]["value"])
        property_counts[prop_uri] = count

    # Match with our property labels and calculate completeness
    completeness_data = []
    for prop in properties:
        uri = prop["uri"]
        count = property_counts.get(uri, 0)
        completeness = (count / total_aops) * 100

        completeness_data.append({
            "Property": prop["label"],
            "Completeness": completeness,
            "Type": prop["type"],
            "URI": uri,
            "Count": count
        })

    df = pd.DataFrame(completeness_data)
    df["Version"] = latest_version  # Add version for context
    df["Total_AOPs"] = total_aops   # Add total AOPs for reference

    # Store in global cache for CSV download with unique key
    _plot_data_cache['latest_aop_completeness_unique'] = df

    fig = px.bar(
        df, x="Property", y="Completeness",
        text="Completeness",
    )
    fig.update_traces(marker_color=BRAND_COLORS['blue'], texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(


        height=600,  # Larger height for better visibility
        margin=dict(l=50, r=20, t=70, b=120),  # More space for labels
        yaxis=dict(title="Completeness (%)", range=[0, 105]),
        xaxis=dict(title="AOP Properties", tickangle=45),
        showlegend=False  # Hide legend since colors are unique per property
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_aop_completeness_unique'] = fig

    return render_plot_html(fig)


def plot_latest_database_summary(version: str = None) -> str:
    """Create a simple summary chart of main entities in current version."""

    where_filter, order_limit = _build_graph_filter(version)

    # Ultra-simple separate queries for each entity type
    queries = {
        "AOPs": f"""
            SELECT ?graph (COUNT(?aop) AS ?count)
            WHERE {{
                GRAPH ?graph {{ ?aop a aopo:AdverseOutcomePathway . }}
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                {where_filter}
            }}
            GROUP BY ?graph
            {order_limit}
        """,
        "Key Events": f"""
            SELECT ?graph (COUNT(?ke) AS ?count)
            WHERE {{
                GRAPH ?graph {{ ?ke a aopo:KeyEvent . }}
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                {where_filter}
            }}
            GROUP BY ?graph
            {order_limit}
        """,
        "KE Relationships": f"""
            SELECT ?graph (COUNT(?ker) AS ?count)
            WHERE {{
                GRAPH ?graph {{ ?ker a aopo:KeyEventRelationship . }}
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                {where_filter}
            }}
            GROUP BY ?graph
            {order_limit}
        """
    }

    data = []
    latest_version = None

    for entity_type, query in queries.items():
        results = run_sparql_query(query)
        if results:
            count = int(results[0]["count"]["value"])
            version = results[0]["graph"]["value"].split("/")[-1]
            if latest_version is None:
                latest_version = version
            data.append({"Entity": entity_type, "Count": count})

    if not data:
        return create_fallback_plot("Core Entity Summary", "No database data available")

    df = pd.DataFrame(data)
    if latest_version is not None:
        df["Version"] = latest_version
    _plot_data_cache['latest_database_summary'] = df

    # Regular bar chart since values are in similar ranges
    fig = px.bar(
        df, x="Entity", y="Count",
        text="Count",
    )

    fig.update_traces(marker_color=BRAND_COLORS['blue'], textposition='outside')
    fig.update_layout(

        showlegend=False,

        margin=dict(l=50, r=20, t=50, b=50),
        yaxis=dict(title="Count")
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_database_summary'] = fig

    return render_plot_html(fig)


def plot_latest_ke_annotation_depth(version: str = None) -> str:
    """Show current distribution of Key Events by annotation depth (number of components)."""
    global _plot_data_cache

    # First, determine which graph to use
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
    else:
        # Get the latest graph - using KeyEvent for faster query
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:KeyEvent . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("KE Annotation Depth", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]

    # Now query that specific graph - get annotation depth for each KE
    query = f"""
    SELECT ?ke (COUNT(DISTINCT ?bioevent) AS ?annotation_depth)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?ke a aopo:KeyEvent .
        OPTIONAL {{ ?ke aopo:hasBiologicalEvent ?bioevent . }}
      }}
    }}
    GROUP BY ?ke
    """

    # First get all KE annotation depths
    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("KE Annotation Depth", "No annotation data available")

    # Get latest version and count distribution of annotation depths
    latest_version = target_graph.split("/")[-1]

    # Count how many KEs have each annotation depth
    from collections import Counter
    depth_counts = Counter()
    for r in results:
        depth = int(r["annotation_depth"]["value"])
        depth_counts[depth] += 1

    # Convert to list format
    latest_data = [{"annotation_depth": {"value": str(depth)}, "ke_count": {"value": str(count)}}
                   for depth, count in depth_counts.items()]

    data = []
    for r in latest_data:
        depth = int(r["annotation_depth"]["value"])
        count = int(r["ke_count"]["value"])
        depth_label = f"{depth} components" if depth > 0 else "No annotations"
        data.append({"Depth": depth_label, "KE Count": count, "Sort": depth})

    if not data:
        return create_fallback_plot("KE Annotation Depth", "No annotation depth data found")

    df = pd.DataFrame(data)
    df = df.sort_values("Sort")
    df["Version"] = latest_version  # Add version for context
    df["Numeric_Depth"] = df["Sort"]  # Add numeric depth for analysis

    # Store in global cache for CSV download
    _plot_data_cache['latest_ke_annotation_depth'] = df

    fig = px.pie(
        df, values="KE Count", names="Depth",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=50)
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_ke_annotation_depth'] = fig

    return render_plot_html(fig)


def plot_latest_aop_completeness_by_status(version: str = None) -> str:
    """Create a grouped bar chart showing AOP completeness scores grouped by OECD status.

    Analyzes completeness across property categories (Essential, Content, Context,
    Assessment, Metadata) for each OECD status category to test the hypothesis that
    endorsed/reviewed AOPs have better completeness than under development AOPs.

    Args:
        version: Optional version string to analyze historical data

    Returns:
        HTML string containing the Plotly visualization
    """
    global _plot_data_cache

    where_filter, order_limit = _build_graph_filter(version)

    # Default fallback properties for AOP completeness analysis
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential"},
        {"uri": "http://purl.org/dc/terms/abstract", "label": "Abstract", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata"},
        {"uri": "http://aopkb.org/aop_ontology#has_key_event", "label": "Has Key Event", "type": "Content"},
        {"uri": "http://aopkb.org/aop_ontology#has_adverse_outcome", "label": "Has Adverse Outcome", "type": "Content"}
    ]
    properties_df = safe_read_csv("property_labels.csv", default_properties)

    # Filter properties to only those that apply to AOPs
    aop_properties_df = properties_df[properties_df['applies_to'].str.contains('AOP', na=False)]
    properties = aop_properties_df.to_dict(orient="records")

    # Build property URI list for SPARQL filtering
    property_uris = [p["uri"] for p in properties]

    # Build FILTER clause for properties
    if property_uris:
        property_filter_values = ", ".join([f"<{uri}>" for uri in property_uris])
        property_filter = f"FILTER(?p IN ({property_filter_values}))"
    else:
        property_filter = ""

    # Get the target graph version
    version_query = f"""
    SELECT ?graph
    WHERE {{
        GRAPH ?graph {{ ?aop a aopo:AdverseOutcomePathway . }}
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        {where_filter}
    }}
    GROUP BY ?graph
    {order_limit}
    """
    version_results = run_sparql_query(version_query)
    if not version_results:
        return create_fallback_plot("AOP Completeness by OECD Status", "No AOP data available")

    latest_version = version_results[0]["graph"]["value"].split("/")[-1]
    target_graph = f"http://aopwiki.org/graph/{latest_version}"

    # Optimized aggregated query: Get counts of AOPs with each property grouped by status
    # This query aggregates in SPARQL instead of transferring all data to Python
    aggregated_query = f"""
    SELECT ?status ?p (COUNT(DISTINCT ?aop) AS ?count_with_property)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway .
            OPTIONAL {{
                ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
                BIND(STR(?status_obj) AS ?status)
            }}
            ?aop ?p ?o .
            {property_filter}
        }}
    }}
    GROUP BY ?status ?p
    """

    aggregated_results = run_sparql_query(aggregated_query)

    # Also get total count of AOPs per status
    status_count_query = f"""
    SELECT ?status (COUNT(DISTINCT ?aop) AS ?total_aops)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway .
            OPTIONAL {{
                ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
                BIND(STR(?status_obj) AS ?status)
            }}
        }}
    }}
    GROUP BY ?status
    """

    status_count_results = run_sparql_query(status_count_query)

    if not aggregated_results or not status_count_results:
        return create_fallback_plot("AOP Completeness by OECD Status", "No completeness data available")

    # Build status to total AOPs map
    status_totals = {}
    for result in status_count_results:
        status = result.get("status", {}).get("value", "No Status")
        total = int(result["total_aops"]["value"])
        status_totals[status] = total

    # Build property URI to metadata map
    property_metadata = {p["uri"]: p for p in properties}

    # Process aggregated results to calculate completeness by property type
    status_type_data = {}

    for result in aggregated_results:
        status = result.get("status", {}).get("value", "No Status")
        prop_uri = result["p"]["value"]
        count_with_property = int(result["count_with_property"]["value"])

        # Get property metadata
        prop_meta = property_metadata.get(prop_uri)
        if not prop_meta:
            continue

        prop_type = prop_meta["type"]

        if status not in status_type_data:
            status_type_data[status] = {}
        if prop_type not in status_type_data[status]:
            status_type_data[status][prop_type] = {"count": 0, "total": 0}

        # Add this property's contribution
        status_type_data[status][prop_type]["count"] += count_with_property
        status_type_data[status][prop_type]["total"] += status_totals.get(status, 0)

    # Convert to DataFrame for plotting
    data = []
    for status, type_data in status_type_data.items():
        for prop_type, counts in type_data.items():
            completeness = (counts["count"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            data.append({
                "OECD Status": status,
                "Property Type": prop_type,
                "Completeness": completeness,
                "Count": counts["count"],
                "Total": counts["total"]
            })

    if not data:
        return create_fallback_plot("AOP Completeness by OECD Status", "No completeness data found")

    df = pd.DataFrame(data)
    df["Version"] = latest_version  # Add version for context

    # Store in global cache for CSV download
    _plot_data_cache['latest_aop_completeness_by_status'] = df

    # Use centralized brand colors for consistency
    color_map = BRAND_COLORS['type_colors'].copy()
    # Add fallback for any missing types
    color_map.update({"Structure": BRAND_COLORS['accent']})

    # Create grouped bar chart
    fig = px.bar(
        df,
        x="OECD Status",
        y="Completeness",
        color="Property Type",
        text="Completeness",
        color_discrete_map=color_map,
        barmode="group"
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=100),
        yaxis=dict(title="Completeness (%)", range=[0, 105]),
        xaxis=dict(title="OECD Status", tickangle=45),
        legend=dict(title="Property Type")
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_aop_completeness_by_status'] = fig

    return render_plot_html(fig)


def plot_latest_ke_by_bio_level(version: str = None) -> str:
    """Show distribution of Key Events across biological levels of organization.

    Args:
        version: Optional version string. If None, uses latest.

    Returns:
        str: Plotly HTML string for embedding.
    """
    global _plot_data_cache

    # Determine target graph directly for reliable query execution
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:KeyEvent . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("KE by Biological Level", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    query = f"""
    SELECT ?level_label (COUNT(DISTINCT ?ke) AS ?count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?ke a aopo:KeyEvent .
        OPTIONAL {{
          ?ke <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25664> ?level_obj .
        }}
        BIND(IF(BOUND(?level_obj), STR(?level_obj), "Not Annotated") AS ?level_label)
      }}
    }}
    GROUP BY ?level_label
    ORDER BY DESC(?count)
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("KE by Biological Level", "No data available")

    data = []
    total_ke = 0
    not_annotated_count = 0
    for r in results:
        label_raw = r["level_label"]["value"]
        count = int(r["count"]["value"])
        total_ke += count
        # Extract human-readable name from URI or keep as-is
        if label_raw.startswith("http"):
            label = label_raw.split("/")[-1].replace("_", " ")
        else:
            label = label_raw
        if label == "Not Annotated":
            not_annotated_count = count
        data.append({"Biological Level": label, "KE Count": count})

    if not data:
        return create_fallback_plot("KE by Biological Level", "No biological level data found")

    df = pd.DataFrame(data).sort_values("KE Count", ascending=True)
    version_key = version or "latest"
    df["Version"] = latest_version
    _plot_data_cache[f'latest_ke_by_bio_level_{version_key}'] = df

    # Build subtitle note if data is very sparse
    subtitle = ""
    if total_ke > 0 and not_annotated_count / total_ke > 0.8:
        subtitle = "<br><sub>Note: >80% of KEs lack biological level annotations</sub>"

    fig = px.bar(
        df,
        x="KE Count",
        y="Biological Level",
        orientation='h',
        text="KE Count",
    )

    fig.update_traces(marker_color=BRAND_COLORS['blue'], textposition='outside')
    fig.update_layout(





        showlegend=False,

        margin=dict(l=150, r=30, t=80, b=60),
        yaxis=dict(title=""),
        xaxis=dict(title="Number of Key Events"),
    )

    _plot_figure_cache[f'latest_ke_by_bio_level_{version_key}'] = fig
    return render_plot_html(fig)


def plot_latest_taxonomic_groups(version: str = None) -> str:
    """Show which taxonomic groups are most represented across AOPs.

    Args:
        version: Optional version string. If None, uses latest.

    Returns:
        str: Plotly HTML string for embedding.
    """
    global _plot_data_cache

    # Determine target graph
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:AdverseOutcomePathway . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("Taxonomic Groups", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    # Query taxonomic applicability on AOPs (and KEs as fallback context)
    query = f"""
    SELECT ?taxon_label (COUNT(DISTINCT ?aop) AS ?aop_count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway .
        ?aop <http://purl.bioontology.org/ontology/NCBITAXON/131567> ?taxon_obj .
        BIND(STR(?taxon_obj) AS ?taxon_label)
      }}
    }}
    GROUP BY ?taxon_label
    ORDER BY DESC(?aop_count)
    LIMIT 25
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Taxonomic Groups", "No taxonomic applicability data found. This annotation may not be present in the current RDF data.")

    data = []
    for r in results:
        label_raw = r["taxon_label"]["value"]
        count = int(r["aop_count"]["value"])
        # Extract readable name from URI
        if label_raw.startswith("http"):
            label = label_raw.split("/")[-1].replace("_", " ")
        else:
            label = label_raw
        data.append({"Taxonomic Group": label, "AOP Count": count})

    if not data:
        return create_fallback_plot("Taxonomic Groups", "No taxonomic group data found")

    df = pd.DataFrame(data).sort_values("AOP Count", ascending=True)
    version_key = version or "latest"
    df["Version"] = latest_version
    _plot_data_cache[f'latest_taxonomic_groups_{version_key}'] = df

    fig = px.bar(
        df,
        x="AOP Count",
        y="Taxonomic Group",
        orientation='h',
        text="AOP Count",
    )

    fig.update_traces(marker_color=BRAND_COLORS['blue'], textposition='outside')
    fig.update_layout(





        showlegend=False,

        margin=dict(l=180, r=30, t=60, b=60),
        yaxis=dict(title=""),
        xaxis=dict(title="Number of AOPs"),
    )

    _plot_figure_cache[f'latest_taxonomic_groups_{version_key}'] = fig
    return render_plot_html(fig)


def plot_latest_entity_by_oecd_status(version: str = None) -> str:
    """Show entity count breakdowns (AOPs, KEs, KERs) by OECD status.

    Args:
        version: Optional version string. If None, uses latest.

    Returns:
        str: Plotly HTML string for embedding.
    """
    global _plot_data_cache

    # Determine target graph
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:AdverseOutcomePathway . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("Entity by OECD Status", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    # Count AOPs by OECD status
    aop_query = f"""
    SELECT ?status_label (COUNT(DISTINCT ?aop) AS ?count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {{
          ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
        }}
        BIND(IF(BOUND(?status_obj), STR(?status_obj), "No Status") AS ?status_label)
      }}
    }}
    GROUP BY ?status_label
    """

    # Count KEs by OECD status (via parent AOP)
    ke_query = f"""
    SELECT ?status_label (COUNT(DISTINCT ?ke) AS ?count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             aopo:has_key_event ?ke .
        OPTIONAL {{
          ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
        }}
        BIND(IF(BOUND(?status_obj), STR(?status_obj), "No Status") AS ?status_label)
      }}
    }}
    GROUP BY ?status_label
    """

    # Count KERs by OECD status (via parent AOP)
    ker_query = f"""
    SELECT ?status_label (COUNT(DISTINCT ?ker) AS ?count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             aopo:has_key_event_relationship ?ker .
        OPTIONAL {{
          ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
        }}
        BIND(IF(BOUND(?status_obj), STR(?status_obj), "No Status") AS ?status_label)
      }}
    }}
    GROUP BY ?status_label
    """

    aop_results = run_sparql_query(aop_query)
    ke_results = run_sparql_query(ke_query)
    ker_results = run_sparql_query(ker_query)

    if not aop_results and not ke_results and not ker_results:
        return create_fallback_plot("Entity by OECD Status", "No OECD status data available")

    data = []
    for entity_type, results in [("AOPs", aop_results), ("KEs", ke_results), ("KERs", ker_results)]:
        if results:
            for r in results:
                status_raw = r["status_label"]["value"]
                count = int(r["count"]["value"])
                # Clean up status label from URI
                if status_raw.startswith("http"):
                    status = status_raw.split("/")[-1].replace("_", " ")
                else:
                    status = status_raw
                data.append({
                    "Entity Type": entity_type,
                    "OECD Status": status,
                    "Count": count
                })

    if not data:
        return create_fallback_plot("Entity by OECD Status", "No entity status data found")

    df = pd.DataFrame(data)
    version_key = version or "latest"
    df["Version"] = latest_version
    _plot_data_cache[f'latest_entity_by_oecd_status_{version_key}'] = df

    # Use shared OECD status color mapping for consistency across plots
    status_colors = BRAND_COLORS['oecd_status']

    fig = px.bar(
        df,
        x="Entity Type",
        y="Count",
        color="OECD Status",
        barmode="group",
        text="Count",
        color_discrete_map=status_colors,
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(






        margin=dict(l=60, r=30, t=60, b=60),
        yaxis=dict(title="Count"),
        xaxis=dict(title="Entity Type"),
        legend=dict(title="OECD Status"),
    )

    _plot_figure_cache[f'latest_entity_by_oecd_status_{version_key}'] = fig
    return render_plot_html(fig)


def plot_latest_ke_reuse(version: str = None) -> str:
    """Show the most reused Key Events across AOPs (top 30).

    Includes AOP-Wiki entity links via Plotly customdata for click-to-open.

    Args:
        version: Optional version string. If None, uses latest.

    Returns:
        str: Plotly HTML string for embedding.
    """
    global _plot_data_cache

    # Determine target graph
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:AdverseOutcomePathway . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("Most Reused KEs", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    query = f"""
    SELECT ?ke (SAMPLE(?title_val) AS ?title) (COUNT(DISTINCT ?aop) AS ?aop_count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             aopo:has_key_event ?ke .
        OPTIONAL {{ ?ke dc:title ?title_val . }}
      }}
    }}
    GROUP BY ?ke
    HAVING (COUNT(DISTINCT ?aop) > 1)
    ORDER BY DESC(?aop_count)
    LIMIT 30
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Most Reused KEs", "No reused Key Events found")

    data = []
    for r in results:
        ke_uri = r["ke"]["value"]
        ke_id = ke_uri.split("/")[-1]
        title = r.get("title", {}).get("value", f"KE {ke_id}")
        aop_count = int(r["aop_count"]["value"])
        # Truncate title for readability
        title_display = title[:50] + "..." if len(title) > 50 else title
        label = f"KE {ke_id}: {title_display}"
        wiki_url = f"https://aopwiki.org/events/{ke_id}"
        data.append({
            "KE": label,
            "KE ID": ke_id,
            "Title": title,
            "AOP Count": aop_count,
            "wiki_url": wiki_url,
        })

    if not data:
        return create_fallback_plot("Most Reused KEs", "No KE reuse data found")

    df = pd.DataFrame(data).sort_values("AOP Count", ascending=True)
    version_key = version or "latest"
    df["Version"] = latest_version
    _plot_data_cache[f'latest_ke_reuse_{version_key}'] = df

    fig = px.bar(
        df,
        x="AOP Count",
        y="KE",
        orientation='h',
        text="AOP Count",
        custom_data=['wiki_url'],
    )

    fig.update_traces(marker_color=BRAND_COLORS['blue'], textposition='outside')
    fig.update_layout(





        showlegend=False,

        height=max(400, len(data) * 25 + 100),
        margin=dict(l=300, r=30, t=60, b=60),
        yaxis=dict(title=""),
        xaxis=dict(title="Number of AOPs"),
    )

    _plot_figure_cache[f'latest_ke_reuse_{version_key}'] = fig
    return render_plot_html(fig)


def plot_latest_ke_reuse_distribution(version: str = None) -> str:
    """Show the distribution of how many AOPs each KE belongs to (reuse histogram).

    Args:
        version: Optional version string. If None, uses latest.

    Returns:
        str: Plotly HTML string for embedding.
    """
    global _plot_data_cache

    # Determine target graph
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:AdverseOutcomePathway . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("KE Reuse Distribution", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    query = f"""
    SELECT ?ke (COUNT(DISTINCT ?aop) AS ?aop_count)
    WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             aopo:has_key_event ?ke .
      }}
    }}
    GROUP BY ?ke
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("KE Reuse Distribution", "No KE data available")

    # Count distribution using discrete bins: 1, 2, 3, 4, 5, 6-10, 11+
    from collections import Counter
    bin_counts = Counter()
    for r in results:
        aop_count = int(r["aop_count"]["value"])
        if aop_count <= 5:
            bin_label = str(aop_count)
        elif aop_count <= 10:
            bin_label = "6-10"
        else:
            bin_label = "11+"
        bin_counts[bin_label] += 1

    # Order bins properly
    bin_order = ["1", "2", "3", "4", "5", "6-10", "11+"]
    data = []
    for b in bin_order:
        if b in bin_counts:
            data.append({"AOPs per KE": b, "Number of KEs": bin_counts[b]})

    if not data:
        return create_fallback_plot("KE Reuse Distribution", "No distribution data found")

    df = pd.DataFrame(data)
    version_key = version or "latest"
    df["Version"] = latest_version
    _plot_data_cache[f'latest_ke_reuse_distribution_{version_key}'] = df

    fig = px.bar(
        df,
        x="AOPs per KE",
        y="Number of KEs",
        text="Number of KEs",
    )

    fig.update_traces(marker_color=BRAND_COLORS['blue'], textposition='outside')
    fig.update_layout(





        showlegend=False,

        margin=dict(l=60, r=30, t=60, b=60),
        yaxis=dict(title="Number of Key Events"),
        xaxis=dict(title="Number of AOPs a KE Belongs To", type='category'),
    )

    _plot_figure_cache[f'latest_ke_reuse_distribution_{version_key}'] = fig
    return render_plot_html(fig)


def plot_latest_ke_completeness_by_status(version: str = None) -> str:
    """Create a grouped bar chart showing KE completeness scores grouped by OECD status.

    Analyzes completeness of Key Events based on the OECD status of AOPs they belong to.
    A KE can belong to multiple AOPs with different statuses and will be counted for each.

    Args:
        version: Optional version string to analyze historical data

    Returns:
        HTML string containing the Plotly visualization
    """
    global _plot_data_cache

    where_filter, order_limit = _build_graph_filter(version)

    # Get properties relevant for KEs
    default_properties = [
        {"uri": "http://www.w3.org/2000/01/rdf-schema#label", "label": "Label", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/description", "label": "Description", "type": "Essential"}
    ]
    properties_df = safe_read_csv("property_labels.csv", default_properties)

    # Filter properties to only those that apply to KEs
    ke_properties_df = properties_df[properties_df['applies_to'].str.contains('KE', na=False)]
    properties = ke_properties_df.to_dict(orient="records")

    # Build property URI list for SPARQL filtering
    property_uris = [p["uri"] for p in properties]

    # Build FILTER clause for properties
    if property_uris:
        property_filter_values = ", ".join([f"<{uri}>" for uri in property_uris])
        property_filter = f"FILTER(?p IN ({property_filter_values}))"
    else:
        property_filter = ""

    # Get the target graph version
    version_query = f"""
    SELECT ?graph
    WHERE {{
        GRAPH ?graph {{ ?aop a aopo:AdverseOutcomePathway . }}
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        {where_filter}
    }}
    GROUP BY ?graph
    {order_limit}
    """
    version_results = run_sparql_query(version_query)
    if not version_results:
        return create_fallback_plot("KE Completeness by OECD Status", "No AOP data available")

    latest_version = version_results[0]["graph"]["value"].split("/")[-1]
    target_graph = f"http://aopwiki.org/graph/{latest_version}"

    # Optimized aggregated query: Get counts of KEs with each property grouped by AOP status
    # This combines the KE-AOP relationship, AOP status, and property presence in one query
    aggregated_query = f"""
    SELECT ?status ?p (COUNT(DISTINCT ?ke) AS ?count_with_property)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway ;
                 <http://aopkb.org/aop_ontology#has_key_event> ?ke .
            OPTIONAL {{
                ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
                BIND(STR(?status_obj) AS ?status)
            }}
            ?ke a aopo:KeyEvent ;
                ?p ?o .
            {property_filter}
        }}
    }}
    GROUP BY ?status ?p
    """

    aggregated_results = run_sparql_query(aggregated_query)

    # Also get total count of KEs per status (a KE can belong to multiple statuses)
    status_ke_count_query = f"""
    SELECT ?status (COUNT(DISTINCT ?ke) AS ?total_kes)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway ;
                 <http://aopkb.org/aop_ontology#has_key_event> ?ke .
            OPTIONAL {{
                ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
                BIND(STR(?status_obj) AS ?status)
            }}
        }}
    }}
    GROUP BY ?status
    """

    status_count_results = run_sparql_query(status_ke_count_query)

    if not aggregated_results or not status_count_results:
        return create_fallback_plot("KE Completeness by OECD Status", "No completeness data available")

    # Build status to total KEs map
    status_totals = {}
    for result in status_count_results:
        status = result.get("status", {}).get("value", "No Status")
        total = int(result["total_kes"]["value"])
        status_totals[status] = total

    # Build property URI to metadata map
    property_metadata = {p["uri"]: p for p in properties}

    # Process aggregated results to calculate completeness by property type
    status_type_data = {}

    for result in aggregated_results:
        status = result.get("status", {}).get("value", "No Status")
        prop_uri = result["p"]["value"]
        count_with_property = int(result["count_with_property"]["value"])

        # Get property metadata
        prop_meta = property_metadata.get(prop_uri)
        if not prop_meta:
            continue

        prop_type = prop_meta["type"]

        if status not in status_type_data:
            status_type_data[status] = {}
        if prop_type not in status_type_data[status]:
            status_type_data[status][prop_type] = {"count": 0, "total": 0}

        # Add this property's contribution
        status_type_data[status][prop_type]["count"] += count_with_property
        status_type_data[status][prop_type]["total"] += status_totals.get(status, 0)

    # Convert to DataFrame for plotting
    data = []
    for status, type_data in status_type_data.items():
        for prop_type, counts in type_data.items():
            completeness = (counts["count"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            data.append({
                "OECD Status": status,
                "Property Type": prop_type,
                "Completeness": completeness,
                "Count": counts["count"],
                "Total": counts["total"]
            })

    if not data:
        return create_fallback_plot("KE Completeness by OECD Status", "No completeness data found")

    df = pd.DataFrame(data)
    df["Version"] = latest_version  # Add version for context

    # Store in global cache for CSV download
    _plot_data_cache['latest_ke_completeness_by_status'] = df

    # Use centralized brand colors for consistency
    color_map = BRAND_COLORS['type_colors'].copy()
    # Add fallback for any missing types
    color_map.update({"Structure": BRAND_COLORS['accent']})

    # Create grouped bar chart
    fig = px.bar(
        df,
        x="OECD Status",
        y="Completeness",
        color="Property Type",
        text="Completeness",
        color_discrete_map=color_map,
        barmode="group"
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=100),
        yaxis=dict(title="Completeness (%)", range=[0, 105]),
        xaxis=dict(title="OECD Status of Parent AOPs", tickangle=45),
        legend=dict(title="Property Type")
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_ke_completeness_by_status'] = fig

    return render_plot_html(fig)


def plot_latest_ker_completeness_by_status(version: str = None) -> str:
    """Create a grouped bar chart showing KER completeness scores grouped by OECD status.

    Analyzes completeness of Key Event Relationships based on the OECD status of AOPs they belong to.
    A KER can belong to multiple AOPs with different statuses and will be counted for each.

    Args:
        version: Optional version string to analyze historical data

    Returns:
        HTML string containing the Plotly visualization
    """
    global _plot_data_cache

    where_filter, order_limit = _build_graph_filter(version)

    # Get properties relevant for KERs
    default_properties = [
        {"uri": "http://www.w3.org/2000/01/rdf-schema#label", "label": "Label", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/description", "label": "Description", "type": "Essential"}
    ]
    properties_df = safe_read_csv("property_labels.csv", default_properties)

    # Filter properties to only those that apply to KERs
    ker_properties_df = properties_df[properties_df['applies_to'].str.contains('KER', na=False)]
    properties = ker_properties_df.to_dict(orient="records")

    # Build property URI list for SPARQL filtering
    property_uris = [p["uri"] for p in properties]

    # Build FILTER clause for properties
    if property_uris:
        property_filter_values = ", ".join([f"<{uri}>" for uri in property_uris])
        property_filter = f"FILTER(?p IN ({property_filter_values}))"
    else:
        property_filter = ""

    # Get the target graph version
    version_query = f"""
    SELECT ?graph
    WHERE {{
        GRAPH ?graph {{ ?aop a aopo:AdverseOutcomePathway . }}
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        {where_filter}
    }}
    GROUP BY ?graph
    {order_limit}
    """
    version_results = run_sparql_query(version_query)
    if not version_results:
        return create_fallback_plot("KER Completeness by OECD Status", "No AOP data available")

    latest_version = version_results[0]["graph"]["value"].split("/")[-1]
    target_graph = f"http://aopwiki.org/graph/{latest_version}"

    # Optimized aggregated query: Get counts of KERs with each property grouped by AOP status
    # This combines the KER-AOP relationship, AOP status, and property presence in one query
    aggregated_query = f"""
    SELECT ?status ?p (COUNT(DISTINCT ?ker) AS ?count_with_property)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway ;
                 <http://aopkb.org/aop_ontology#has_key_event_relationship> ?ker .
            OPTIONAL {{
                ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
                BIND(STR(?status_obj) AS ?status)
            }}
            ?ker a aopo:KeyEventRelationship ;
                ?p ?o .
            {property_filter}
        }}
    }}
    GROUP BY ?status ?p
    """

    aggregated_results = run_sparql_query(aggregated_query)

    # Also get total count of KERs per status (a KER can belong to multiple statuses)
    status_ker_count_query = f"""
    SELECT ?status (COUNT(DISTINCT ?ker) AS ?total_kers)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway ;
                 <http://aopkb.org/aop_ontology#has_key_event_relationship> ?ker .
            OPTIONAL {{
                ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
                BIND(STR(?status_obj) AS ?status)
            }}
        }}
    }}
    GROUP BY ?status
    """

    status_count_results = run_sparql_query(status_ker_count_query)

    if not aggregated_results or not status_count_results:
        return create_fallback_plot("KER Completeness by OECD Status", "No completeness data available")

    # Build status to total KERs map
    status_totals = {}
    for result in status_count_results:
        status = result.get("status", {}).get("value", "No Status")
        total = int(result["total_kers"]["value"])
        status_totals[status] = total

    # Build property URI to metadata map
    property_metadata = {p["uri"]: p for p in properties}

    # Process aggregated results to calculate completeness by property type
    status_type_data = {}

    for result in aggregated_results:
        status = result.get("status", {}).get("value", "No Status")
        prop_uri = result["p"]["value"]
        count_with_property = int(result["count_with_property"]["value"])

        # Get property metadata
        prop_meta = property_metadata.get(prop_uri)
        if not prop_meta:
            continue

        prop_type = prop_meta["type"]

        if status not in status_type_data:
            status_type_data[status] = {}
        if prop_type not in status_type_data[status]:
            status_type_data[status][prop_type] = {"count": 0, "total": 0}

        # Add this property's contribution
        status_type_data[status][prop_type]["count"] += count_with_property
        status_type_data[status][prop_type]["total"] += status_totals.get(status, 0)

    # Convert to DataFrame for plotting
    data = []
    for status, type_data in status_type_data.items():
        for prop_type, counts in type_data.items():
            completeness = (counts["count"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            data.append({
                "OECD Status": status,
                "Property Type": prop_type,
                "Completeness": completeness,
                "Count": counts["count"],
                "Total": counts["total"]
            })

    if not data:
        return create_fallback_plot("KER Completeness by OECD Status", "No completeness data found")

    df = pd.DataFrame(data)
    df["Version"] = latest_version  # Add version for context

    # Store in global cache for CSV download
    _plot_data_cache['latest_ker_completeness_by_status'] = df

    # Use centralized brand colors for consistency
    color_map = BRAND_COLORS['type_colors'].copy()
    # Add fallback for any missing types
    color_map.update({"Structure": BRAND_COLORS['accent']})

    # Create grouped bar chart
    fig = px.bar(
        df,
        x="OECD Status",
        y="Completeness",
        color="Property Type",
        text="Completeness",
        color_discrete_map=color_map,
        barmode="group"
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(


        margin=dict(l=50, r=20, t=50, b=100),
        yaxis=dict(title="Completeness (%)", range=[0, 105]),
        xaxis=dict(title="OECD Status of Parent AOPs", tickangle=45),
        legend=dict(title="Property Type")
    )

    # Cache the figure object for image export (PNG/SVG/PDF)
    _plot_figure_cache['latest_ker_completeness_by_status'] = fig

    return render_plot_html(fig)


# plot_latest_annotation_heatmap removed — produced non-sensical output per UAT feedback


def plot_latest_ontology_diversity(version: str = None) -> str:
    """Create a bar chart showing unique ontology term counts per ontology source.

    Queries all distinct ontology-linked IRI values from biological process and object
    annotations, parses the URIs in Python to extract ontology prefixes, and counts
    unique terms per source (GO, CHEBI, UBERON, etc.).

    Args:
        version: Optional version string for historical snapshots.

    Returns:
        str: HTML string containing the interactive Plotly bar chart.
    """
    global _plot_data_cache

    # Determine target graph
    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?s a aopo:KeyEvent . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("Ontology Term Diversity", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    # Check cache
    version_key = version or "latest"
    cache_key = f"latest_ontology_diversity_{version_key}"
    if cache_key in _plot_figure_cache:
        cached_html = _plot_figure_cache.get(cache_key)
        if cached_html and isinstance(cached_html, str):
            return cached_html

    # Query all distinct ontology-linked IRI terms
    query = f"""
    SELECT DISTINCT ?term
    WHERE {{
        GRAPH <{target_graph}> {{
            {{
                ?ke a aopo:KeyEvent ;
                    aopo:hasBiologicalEvent ?be .
                ?be aopo:hasProcess ?term .
            }}
            UNION
            {{
                ?ke a aopo:KeyEvent ;
                    aopo:hasBiologicalEvent ?be .
                ?be aopo:hasObject ?term .
            }}
        }}
        FILTER(isIRI(?term))
    }}
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Ontology Term Diversity", "No ontology term data available")

    # Parse URIs to extract ontology prefix and count unique terms
    prefix_map = {
        "GO_": "GO",
        "CHEBI_": "CHEBI",
        "UBERON_": "UBERON",
        "NCBITaxon_": "NCBITaxon",
        "PR_": "PR",
        "CL_": "CL",
        "DOID_": "DOID",
        "HP_": "HP",
        "MP_": "MP",
        "NBO_": "NBO",
        "MI_": "MI",
        "VT_": "VT",
    }

    ontology_counts = {}
    for r in results:
        term_uri = r["term"]["value"]
        matched = False
        for fragment, prefix_name in prefix_map.items():
            if fragment in term_uri:
                ontology_counts[prefix_name] = ontology_counts.get(prefix_name, 0) + 1
                matched = True
                break
        if not matched:
            if "mesh/" in term_uri.lower():
                ontology_counts["MESH"] = ontology_counts.get("MESH", 0) + 1
            else:
                ontology_counts["Other"] = ontology_counts.get("Other", 0) + 1

    if not ontology_counts:
        return create_fallback_plot("Ontology Term Diversity", "No ontology terms found")

    # Build DataFrame
    data = [{"Ontology": k, "Unique Terms": v} for k, v in ontology_counts.items()]
    df = pd.DataFrame(data).sort_values("Unique Terms", ascending=False)
    df["Version"] = latest_version

    _plot_data_cache[cache_key] = df
    _plot_data_cache['latest_ontology_diversity'] = df

    fig = px.bar(
        df,
        x="Ontology",
        y="Unique Terms",
        text="Unique Terms",
    )

    fig.update_traces(marker_color=BRAND_COLORS['blue'], textposition='outside')
    fig.update_layout(

        showlegend=False,

        margin=dict(l=50, r=20, t=60, b=50),
        yaxis=dict(title="Number of Unique Terms"),
        xaxis=dict(title="Ontology Source")
    )

    _plot_figure_cache[cache_key] = fig
    _plot_figure_cache['latest_ontology_diversity'] = fig
    return render_plot_html(fig)


# ===========================================================================
# Organ-system coverage (AOP-level hybrid A + A' + B + C signal stack)
# ===========================================================================


def _resolve_target_graph(version: str = None) -> tuple[str, str] | None:
    """Return ``(graph_iri, version_label)`` for the given version, or None on failure."""
    if version:
        graph_iri = f"http://aopwiki.org/graph/{version}"
        return graph_iri, version

    version_query = """
    SELECT ?graph
    WHERE {
        GRAPH ?graph { ?s a aopo:KeyEvent . }
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY DESC(?graph)
    LIMIT 1
    """
    results = run_sparql_query(version_query)
    if not results:
        return None
    graph_iri = results[0]["graph"]["value"]
    return graph_iri, graph_iri.rsplit("/", 1)[-1]


def _query_aop_signal_rows(target_graph: str) -> list[dict]:
    """Pull the AOP/AO/KE/anatomy/process/title rows used by all four signals.

    Also fetches the biological-organisation level of each KE (NCI Thesaurus
    C25664) so the apical / AO-only views can filter by KE level.
    """
    query = f"""
    SELECT ?aop ?aop_title ?ao ?ke ?level ?organ ?cell ?obj ?proc
    WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             <http://purl.org/dc/elements/1.1/title> ?aop_title ;
             aopo:has_key_event ?ke .
        OPTIONAL {{ ?aop aopo:has_adverse_outcome ?ao . }}
        OPTIONAL {{ ?ke <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25664> ?level . }}
        OPTIONAL {{ ?ke aopo:OrganContext ?organ . }}
        OPTIONAL {{ ?ke aopo:CellTypeContext ?cell . }}
        OPTIONAL {{
          ?ke aopo:hasBiologicalEvent ?be .
          OPTIONAL {{ ?be aopo:hasObject ?obj . }}
          OPTIONAL {{ ?be aopo:hasProcess ?proc . }}
        }}
      }}
    }}
    """
    return run_sparql_query(query) or []


# KE biological-organisation levels considered "apical" — Tissue and above.
APICAL_LEVELS: frozenset = frozenset({"Tissue", "Organ", "Individual", "Population"})

# Sentinel level for Signal C rows (the source is the AOP title, not a KE).
_AOP_TITLE_LEVEL = "<aop-title>"


def _compute_coverage_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a granular per-(AOP, KE, Signal, Bucket) coverage dataframe.

    Each row records one classification attribution: which signal fired on
    which KE (or the AOP title, for Signal C) and what bucket it produced.
    Downstream :func:`_aggregate_per_aop` reduces this to the per-AOP view
    each plot needs, with optional KE-level scope filtering for the apical
    and AO-only variants.

    Columns:
        AOP, AOP Title, AO URI, KE, KE Level, Is AO, Organ System, Signal, Source
    """
    aop_titles: dict[str, str] = {}
    aop_ao: dict[str, str] = {}
    ke_level: dict[tuple[str, str], str] = {}  # (aop, ke) -> level

    granular_records: list[dict] = []

    for r in rows:
        aop = r.get("aop", {}).get("value")
        ke = r.get("ke", {}).get("value")
        if not aop or not ke:
            continue
        title = r.get("aop_title", {}).get("value", "")
        aop_titles.setdefault(aop, title)
        ao = r.get("ao", {}).get("value")
        if ao and aop not in aop_ao:
            aop_ao[aop] = ao

        level_raw = r.get("level", {}).get("value", "")
        if level_raw:
            level_clean = level_raw.split("/")[-1].replace("_", " ") if level_raw.startswith("http") else level_raw
        else:
            level_clean = ""
        if level_clean:
            ke_level[(aop, ke)] = level_clean

        is_ao = bool(ao) and ao == ke
        ke_level_now = ke_level.get((aop, ke), level_clean)

        def add(bucket: str, signal: str, source: str) -> None:
            granular_records.append({
                "AOP": aop,
                "AOP Title": title,
                "AO URI": ao or "",
                "KE": ke,
                "KE Level": ke_level_now,
                "Is AO": is_ao,
                "Organ System": bucket,
                "Signal": signal,
                "Source": source,
            })

        for predicate_var, source_label in (("organ", "OrganContext"), ("cell", "CellTypeContext")):
            iri = r.get(predicate_var, {}).get("value")
            if iri:
                for bucket in classify_anatomy(iri):
                    add(bucket, "A", source_label)

        obj_iri = r.get("obj", {}).get("value")
        if obj_iri:
            # Anatomy → Signal A'
            for bucket in classify_anatomy(obj_iri):
                add(bucket, "A'", "hasObject (UBERON/CL)")
            # Phenotype-as-object → Signal B (rare but present)
            for bucket in classify_process(obj_iri):
                add(bucket, "B", "hasObject (HP/MP via UPHENO)")

        proc_iri = r.get("proc", {}).get("value")
        if proc_iri:
            for bucket in classify_process(proc_iri):
                add(bucket, "B", "hasProcess (GO RO:0002296 / HP/MP UPHENO)")

    # Signal C — keywords on the AOP title (one row per matched bucket, no KE
    # attachment). The sentinel level marks these rows so the scope filters
    # can include or exclude them.
    for aop, title in aop_titles.items():
        for bucket in classify_text(title):
            granular_records.append({
                "AOP": aop,
                "AOP Title": title,
                "AO URI": aop_ao.get(aop, ""),
                "KE": "",
                "KE Level": _AOP_TITLE_LEVEL,
                "Is AO": False,
                "Organ System": bucket,
                "Signal": "C",
                "Source": "AOP title regex",
            })

    return pd.DataFrame(granular_records)


def _aggregate_per_aop(
    granular: pd.DataFrame,
    aop_universe: dict[str, str],
    scope: str,
) -> pd.DataFrame:
    """Reduce the granular df to one row per (AOP, bucket, best-signal) under a scope.

    Scope determines which granular rows count toward classification:

    - ``"all"``: every signal row.
    - ``"apical"``: rows where the KE is at Tissue level or higher, PLUS all
      Signal C rows (Signal C is on the AOP title, not a KE — kept regardless).
    - ``"ao"``: rows where the KE is the Adverse Outcome, PLUS all Signal C rows.

    AOPs not classified under the scope get a "No annotation" sentinel row,
    using ``aop_universe`` so AOPs with zero signal rows anywhere are still
    accounted for. ``aop_universe`` is ``{aop_iri: title}`` for every AOP in
    the snapshot.
    """
    if granular.empty:
        filtered = granular
    elif scope == "apical":
        filtered = granular[
            (granular["Signal"] == "C")
            | granular["KE Level"].isin(APICAL_LEVELS)
        ]
    elif scope == "ao":
        filtered = granular[(granular["Signal"] == "C") | granular["Is AO"]]
    elif scope == "all":
        filtered = granular
    else:
        raise ValueError(f"unknown scope: {scope!r}")

    if filtered.empty:
        per_pair = pd.DataFrame(
            columns=["AOP", "Organ System", "Signals", "Best Signal", "AOP Title", "AO URI"]
        )
    else:
        per_pair = (
            filtered.groupby(["AOP", "Organ System"], dropna=False)
            .agg(
                Signals=("Signal", lambda s: ",".join(sorted(set(s)))),
                **{"AOP Title": ("AOP Title", "first"), "AO URI": ("AO URI", "first")},
            )
            .reset_index()
        )
        per_pair["Best Signal"] = per_pair["Signals"].apply(
            lambda s: best_signal(s.split(",")) if s else None
        )

    classified_aops = set(per_pair["AOP"]) if not per_pair.empty else set()
    unclassified = set(aop_universe) - classified_aops
    sentinel_rows = [
        {
            "AOP": aop,
            "AOP Title": aop_universe.get(aop, ""),
            "AO URI": "",
            "Organ System": NO_ANNOTATION_BUCKET,
            "Signals": "",
            "Best Signal": None,
        }
        for aop in unclassified
    ]
    if sentinel_rows:
        per_pair = pd.concat([per_pair, pd.DataFrame(sentinel_rows)], ignore_index=True)

    return per_pair


def _query_aop_universe(target_graph: str) -> dict[str, str]:
    """Return {aop_iri: title} for every AOP in the snapshot, irrespective of
    whether any classifier fires on it. Used to seed the No-annotation sentinel
    rows so the per-AOP scope counts stay honest."""
    query = f"""
    SELECT ?aop ?aop_title WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             <http://purl.org/dc/elements/1.1/title> ?aop_title .
      }}
    }}
    """
    rows = run_sparql_query(query) or []
    return {
        r["aop"]["value"]: r.get("aop_title", {}).get("value", "")
        for r in rows
        if "aop" in r
    }


# Module-level cache for the granular per-(AOP, KE, signal) coverage dataframe.
# Keyed by version label so historical snapshot views don't collide. Separate
# from ``_plot_data_cache`` because the granular df is internal — the
# /api/plot-data toggles surface the aggregated per-bucket views instead.
_GRANULAR_COVERAGE_CACHE: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}


def _get_coverage_dataframe(
    version: str = None,
) -> tuple[pd.DataFrame, dict[str, str], str] | None:
    """Return ``(granular_df, aop_universe, version_label)`` for the snapshot.

    ``granular_df`` carries one row per (AOP, KE, signal, bucket) classification
    attribution — see :func:`_compute_coverage_rows`. ``aop_universe`` is
    ``{aop_iri: title}`` for every AOP in the snapshot (used by aggregators to
    seed No-annotation sentinels).
    """
    cache_key = version or "latest"
    cached = _GRANULAR_COVERAGE_CACHE.get(cache_key)
    if cached is not None:
        granular, aop_universe = cached
        version_label = (
            str(granular["Version"].iloc[0])
            if not granular.empty and "Version" in granular.columns
            else (version or "")
        )
        return granular, aop_universe, version_label

    target = _resolve_target_graph(version)
    if target is None:
        return None
    target_graph, version_label = target

    aop_universe = _query_aop_universe(target_graph)
    rows = _query_aop_signal_rows(target_graph)
    granular = _compute_coverage_rows(rows)
    if not granular.empty:
        granular["Version"] = version_label

    _GRANULAR_COVERAGE_CACHE[cache_key] = (granular, aop_universe)
    return granular, aop_universe, version_label


def _build_examples_per_pair(
    per_pair: pd.DataFrame, granular: pd.DataFrame
) -> dict[tuple[str, str], list[str]]:
    """Per (bucket, best_signal): the AOP titles that classified there."""
    examples: dict[tuple[str, str], list[str]] = {}
    classified = per_pair[per_pair["Best Signal"].notna()]
    for (bucket, signal), sub in classified.groupby(["Organ System", "Best Signal"]):
        titles = (
            sub.drop_duplicates(subset=["AOP"])["AOP Title"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        titles = sorted([t for t in titles if t], key=str.lower)
        examples[(bucket, signal)] = titles
    return examples


def _render_coverage_bar(
    per_pair: pd.DataFrame,
    *,
    granular: pd.DataFrame,
    aop_universe: dict[str, str],
    version_label: str,
    scope_label: str,
    percentage: bool,
    title: str,
):
    """Build the horizontal stacked Plotly bar shared by the four coverage views."""
    classified = per_pair[per_pair["Best Signal"].notna()]
    total_aops = len(aop_universe)
    no_anno = int((per_pair["Organ System"] == NO_ANNOTATION_BUCKET).sum())
    annotated = total_aops - no_anno

    grouped = (
        classified.groupby(["Organ System", "Best Signal"])["AOP"]
        .nunique()
        .reset_index(name="AOP count")
    )
    if percentage and total_aops > 0:
        grouped["Percentage"] = grouped["AOP count"] * 100.0 / total_aops
        x_col = "Percentage"
    else:
        x_col = "AOP count"

    bucket_order = [b for b in ORGAN_SYSTEM_BUCKETS if b in set(grouped["Organ System"])]
    bucket_totals = grouped.groupby("Organ System")[x_col].sum()
    bucket_order = sorted(bucket_order, key=lambda b: bucket_totals.get(b, 0))

    examples_per_pair = _build_examples_per_pair(per_pair, granular)

    fig = go.Figure()
    for signal in SIGNAL_ORDER:
        seg = grouped[grouped["Best Signal"] == signal]
        if seg.empty:
            continue
        seg = (
            seg.set_index("Organ System").reindex(bucket_order).fillna(0).reset_index()
        )

        customdata_examples = []
        for bucket in seg["Organ System"]:
            titles = examples_per_pair.get((bucket, signal), [])
            if not titles:
                preview = "—"
            elif len(titles) <= 3:
                preview = "<br>&nbsp;&nbsp;• " + "<br>&nbsp;&nbsp;• ".join(titles)
            else:
                head = titles[:3]
                preview = (
                    "<br>&nbsp;&nbsp;• "
                    + "<br>&nbsp;&nbsp;• ".join(head)
                    + f"<br>&nbsp;&nbsp;<i>… and {len(titles) - 3} more</i>"
                )
            customdata_examples.append(preview)

        if percentage:
            counts = seg["AOP count"].astype(int).tolist() if "AOP count" in seg.columns else [0] * len(seg)
            customdata = list(zip(customdata_examples, counts))
            hover = (
                "<b>%{y}</b><br>"
                f"Signal {signal}: " + "%{x:.1f}% of AOPs (%{customdata[1]} AOPs)"
                "<br><i>Example AOPs:</i>%{customdata[0]}<extra></extra>"
            )
        else:
            customdata = customdata_examples
            hover = (
                "<b>%{y}</b><br>"
                f"Signal {signal}: " + "%{x} AOPs"
                "<br><i>Example AOPs:</i>%{customdata}<extra></extra>"
            )

        fig.add_trace(
            go.Bar(
                x=seg[x_col],
                y=seg["Organ System"],
                name=f"Signal {signal}",
                orientation="h",
                marker_color=SIGNAL_COLOURS.get(signal),
                customdata=customdata,
                hovertemplate=hover,
            )
        )

    if percentage:
        subtitle = (
            f"{scope_label} · share of the {total_aops} AOPs in the snapshot"
        )
        x_axis_title = "% of AOPs in snapshot"
    else:
        subtitle = (
            f"{scope_label} · {annotated}/{total_aops} AOPs classified at least once · "
            f"{no_anno} unclassified at this scope"
        )
        x_axis_title = "Number of AOPs"

    fig.update_layout(
        barmode="stack",
        title={"text": f"{title}<br><sub>{subtitle}</sub>"},
        xaxis_title=x_axis_title,
        yaxis_title="",
        legend_title="Signal source",
        margin=dict(l=80, r=30, t=80, b=50),
    )
    return fig


_SCOPE_LABELS: dict[str, str] = {
    "all": "All KEs",
    "apical": "Apical KEs (Tissue / Organ / Individual / Population)",
    "ao": "Adverse Outcome KE only",
}


def _coverage_plot_for_scope(
    version: str | None,
    *,
    scope: str,
    cache_stub: str,
    title: str,
    percentage: bool = False,
) -> str:
    """Shared implementation for every snapshot organ-coverage plot variant."""
    global _plot_data_cache, _plot_figure_cache

    result = _get_coverage_dataframe(version)
    if result is None:
        return create_fallback_plot(title, "No AOP data in this snapshot")
    granular, aop_universe, version_label = result

    per_pair = _aggregate_per_aop(granular, aop_universe, scope=scope)
    per_pair = per_pair.copy()
    per_pair["Version"] = version_label
    per_pair["Scope"] = scope

    cache_key = f"{cache_stub}_{version or 'latest'}"
    _plot_data_cache[cache_key] = per_pair
    _plot_data_cache[cache_stub] = per_pair

    fig = _render_coverage_bar(
        per_pair,
        granular=granular,
        aop_universe=aop_universe,
        version_label=version_label,
        scope_label=_SCOPE_LABELS.get(scope, scope),
        percentage=percentage,
        title=title,
    )
    _plot_figure_cache[cache_key] = fig
    return render_plot_html(fig)


def plot_latest_organ_coverage(version: str = None) -> str:
    """AOP coverage of organ systems — all member KEs contribute (Signals A/A'/B/C)."""
    return _coverage_plot_for_scope(
        version,
        scope="all",
        cache_stub="latest_organ_coverage",
        title="AOP Coverage of Organ Systems",
    )


def plot_latest_organ_coverage_apical(version: str = None) -> str:
    """Apical-only view: only Tissue/Organ/Individual/Population KEs contribute
    to Signals A/A'/B. Signal C (AOP title) is kept. Surfaces what the AOP is
    *about* at the organ-and-up level rather than its molecular footprint.
    """
    return _coverage_plot_for_scope(
        version,
        scope="apical",
        cache_stub="latest_organ_coverage_apical",
        title="AOP Coverage of Organ Systems — Apical KEs only",
    )


def plot_latest_organ_coverage_ao_only(version: str = None) -> str:
    """AO-only view: only the Adverse Outcome KE contributes to Signals A/A'/B.
    Signal C (AOP title) is kept since titles routinely describe the AO.
    """
    return _coverage_plot_for_scope(
        version,
        scope="ao",
        cache_stub="latest_organ_coverage_ao_only",
        title="AOP Coverage of Organ Systems — Adverse Outcome only",
    )


def plot_latest_life_stage(version: str = None) -> str:
    """Distribution of AOPs by life-stage applicability (issue #22, narrowed).

    AOP-Wiki RDF exposes ``aopo:LifeStageContext`` as one or more free-text
    literals per AOP (e.g. "Adult", "Juvenile", "All life stages"). No
    structured sex-applicability predicate exists in the RDF — sex
    applicability is described inside free-text ``aopo:AopContext`` literals
    and is not machine-readable. This plot therefore reports life stage only;
    the methodology note records the gap.

    Args:
        version: Optional version string. If None, uses the latest snapshot.

    Returns:
        str: Plotly HTML horizontal bar chart.
    """
    global _plot_data_cache, _plot_figure_cache

    target = _resolve_target_graph(version)
    if target is None:
        return create_fallback_plot("AOP Life-Stage Applicability", "No graphs available")
    target_graph, version_label = target

    # COUNT(*) on AdverseOutcomePathway gives the denominator so "Not specified"
    # can be derived from the difference rather than guessed.
    query = f"""
    SELECT ?aop ?life_stage
    WHERE {{
      GRAPH <{target_graph}> {{
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {{ ?aop aopo:LifeStageContext ?life_stage . }}
      }}
    }}
    """
    results = run_sparql_query(query) or []
    if not results:
        return create_fallback_plot("AOP Life-Stage Applicability", "No AOP rows")

    per_aop: dict[str, set] = {}
    for r in results:
        aop = r.get("aop", {}).get("value")
        if not aop:
            continue
        bucket = per_aop.setdefault(aop, set())
        ls = r.get("life_stage", {}).get("value")
        if ls:
            bucket.add(ls.strip())

    total_aops = len(per_aop)
    UNSPECIFIED = "Not specified"

    label_counts: dict[str, int] = {}
    for aop, labels in per_aop.items():
        if not labels:
            label_counts[UNSPECIFIED] = label_counts.get(UNSPECIFIED, 0) + 1
        else:
            for lbl in labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

    df = pd.DataFrame(
        [{"Life stage": k, "AOP count": v} for k, v in label_counts.items()]
    )
    if df.empty:
        return create_fallback_plot("AOP Life-Stage Applicability", "No life-stage data")

    df = df.sort_values("AOP count", ascending=True)
    df["Version"] = version_label
    cache_key = f"latest_life_stage_{version or 'latest'}"
    _plot_data_cache[cache_key] = df
    _plot_data_cache["latest_life_stage"] = df

    fig = px.bar(
        df, x="AOP count", y="Life stage", orientation="h", text="AOP count"
    )
    fig.update_traces(marker_color=BRAND_COLORS["blue"], textposition="outside")
    subtitle = (
        f"AOPs annotated with at least one life-stage: "
        f"{total_aops - label_counts.get(UNSPECIFIED, 0)}/{total_aops}. "
        f"Sex applicability is not structured in the RDF — see methodology note."
    )
    fig.update_layout(
        title={"text": f"AOP Life-Stage Applicability<br><sub>{subtitle}</sub>"},
        margin=dict(l=200, r=30, t=80, b=50),
        xaxis_title="Number of AOPs",
        yaxis_title="",
    )

    _plot_figure_cache[cache_key] = fig
    return render_plot_html(fig)


def plot_latest_organ_coverage_percentage(version: str = None) -> str:
    """Percentage view of the all-KEs coverage plot."""
    return _coverage_plot_for_scope(
        version,
        scope="all",
        cache_stub="latest_organ_coverage_percentage",
        title="AOP Coverage of Organ Systems — % view",
        percentage=True,
    )


_SCOPE_TITLE_SUFFIX: dict[str, str] = {
    "all": "All KEs",
    "apical": "Apical KEs only",
    "ao": "Adverse Outcome only",
}


# Stable per-bucket colour palette for the pie chart. Plotly's Bold sequence
# is colour-blind tolerant and has 11 hues; we pad with greys for the two
# residual buckets so every snapshot keeps the same colour per bucket.
_BUCKET_PALETTE: dict[str, str] = {
    bucket: colour
    for bucket, colour in zip(
        ORGAN_SYSTEM_BUCKETS,
        [
            "#7F3C8D", "#11A579", "#3969AC", "#F2B701", "#E73F74", "#80BA5A",
            "#E68310", "#008695", "#CF1C90", "#f97b72", "#4b4b8f", "#A5AA99",
            "#888888",
        ],
    )
}
_BUCKET_PALETTE[NO_ANNOTATION_BUCKET] = "#cccccc"


def plot_latest_organ_coverage_unified(
    version: str = None,
    scope: str = "all",
    view: str = "absolute",
) -> str:
    """Unified coverage bar — accepts ``scope`` (all/apical/ao) and ``view``
    (absolute/percentage) so the front-end can drive both toggles from one
    plot name. The four legacy ``plot_latest_organ_coverage*`` functions
    remain as compatibility shims for direct CSV/PNG downloads.
    """
    if scope not in _SCOPE_LABELS:
        scope = "all"
    percentage = (view == "percentage")
    suffix = _SCOPE_TITLE_SUFFIX.get(scope, "All KEs")
    title = f"AOP Coverage of Organ Systems — {suffix}"
    # Cache under the bare stub so the generic /download/latest/<name> route
    # can find the most recently rendered scope/view combination for CSV/PNG.
    return _coverage_plot_for_scope(
        version,
        scope=scope,
        cache_stub="latest_organ_coverage_unified",
        title=title,
        percentage=percentage,
    )


def plot_latest_organ_coverage_pie(
    version: str = None,
    scope: str = "all",
) -> str:
    """Pie of AOP membership per organ-system bucket under a given scope.

    An AOP can be in multiple buckets; the pie counts each (AOP, bucket) pair
    once, so slices sum to bucket-memberships, not to total AOPs. An
    "Unclassified" slice surfaces AOPs that the scope leaves with no signal —
    deliberately visible because false-negative diagnosis is part of the
    coverage analysis.
    """
    global _plot_data_cache, _plot_figure_cache

    if scope not in _SCOPE_LABELS:
        scope = "all"

    result = _get_coverage_dataframe(version)
    if result is None:
        return create_fallback_plot(
            "Organ-system bucket distribution", "No AOP data in this snapshot"
        )
    granular, aop_universe, version_label = result

    per_pair = _aggregate_per_aop(granular, aop_universe, scope=scope)
    total_aops = len(aop_universe)

    classified = per_pair[
        (per_pair["Organ System"] != NO_ANNOTATION_BUCKET)
        & per_pair["Best Signal"].notna()
    ]
    bucket_counts = (
        classified.groupby("Organ System")["AOP"].nunique().to_dict()
    )

    unclassified = int(
        (per_pair["Organ System"] == NO_ANNOTATION_BUCKET).sum()
    )

    rows = []
    for bucket in ORGAN_SYSTEM_BUCKETS:
        count = int(bucket_counts.get(bucket, 0))
        if count > 0:
            rows.append({"Organ System": bucket, "AOPs": count})
    if unclassified > 0:
        rows.append({"Organ System": "Unclassified", "AOPs": unclassified})

    if not rows:
        return create_fallback_plot(
            "Organ-system bucket distribution",
            "No classifications at this scope",
        )

    df = pd.DataFrame(rows)
    df["Version"] = version_label
    df["Scope"] = scope

    cache_key = f"latest_organ_coverage_pie_{scope}_{version or 'latest'}"
    _plot_data_cache[cache_key] = df
    _plot_data_cache["latest_organ_coverage_pie"] = df

    colour_map = dict(_BUCKET_PALETTE)
    colour_map["Unclassified"] = "#cccccc"

    classified_aops = total_aops - unclassified
    memberships = int(df.loc[df["Organ System"] != "Unclassified", "AOPs"].sum())
    avg_buckets = (memberships / classified_aops) if classified_aops else 0.0
    subtitle = (
        f"{_SCOPE_LABELS.get(scope, scope)} · "
        f"{classified_aops}/{total_aops} AOPs classified · "
        f"{memberships} bucket-memberships "
        f"({avg_buckets:.2f} buckets/AOP on average)"
    )

    fig = px.pie(
        df,
        names="Organ System",
        values="AOPs",
        color="Organ System",
        color_discrete_map=colour_map,
        category_orders={
            "Organ System": list(ORGAN_SYSTEM_BUCKETS) + ["Unclassified"]
        },
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "AOPs: %{value}<br>"
            "Share of memberships: %{percent}<extra></extra>"
        ),
        sort=False,
    )
    fig.update_layout(
        title={"text": f"Organ-system bucket distribution<br><sub>{subtitle}</sub>"},
        margin=dict(l=30, r=30, t=80, b=30),
        legend_title="Bucket",
    )

    _plot_figure_cache[cache_key] = fig
    # Also cache under the bare stub so the generic /download/latest/<name>
    # PNG/SVG export route resolves without needing the scope suffix.
    _plot_figure_cache["latest_organ_coverage_pie"] = fig
    return render_plot_html(fig)


def plot_latest_multi_organ_aops(version: str = None) -> str:
    """Histogram of the number of distinct organ systems each AOP classifies into."""
    global _plot_data_cache, _plot_figure_cache

    result = _get_coverage_dataframe(version)
    if result is None:
        return create_fallback_plot("Multi-organ AOPs", "No AOP data in this snapshot")
    granular, aop_universe, version_label = result

    per_pair = _aggregate_per_aop(granular, aop_universe, scope="all")
    real = per_pair[
        (per_pair["Organ System"] != NO_ANNOTATION_BUCKET)
        & per_pair["Best Signal"].notna()
    ]
    counts = real.groupby("AOP")["Organ System"].nunique()
    total_aops = len(aop_universe)
    unclassified = total_aops - counts.shape[0]

    capped: dict[str, int] = {}
    for n, count in counts.value_counts().sort_index().items():
        key = "5+" if int(n) >= 5 else str(int(n))
        capped[key] = capped.get(key, 0) + int(count)
    if unclassified > 0:
        capped = {"0 (no signal)": unclassified, **capped}

    hist_df = pd.DataFrame(
        [{"Organ systems touched": k, "AOPs": v} for k, v in capped.items()]
    )

    def _order(key: str) -> int:
        if key == "0 (no signal)":
            return -1
        if key == "5+":
            return 99
        try:
            return int(key)
        except ValueError:
            return 0

    hist_df["__sort__"] = hist_df["Organ systems touched"].map(_order)
    hist_df = hist_df.sort_values("__sort__").drop(columns="__sort__").reset_index(drop=True)

    multi = counts[counts >= 2].sort_values(ascending=False)
    title_lookup = aop_universe
    buckets_lookup = (
        real.groupby("AOP")["Organ System"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .to_dict()
    )
    detail_rows = [
        {
            "AOP": aop,
            "AOP Title": title_lookup.get(aop, ""),
            "Organ systems touched": int(n),
            "Buckets": buckets_lookup.get(aop, ""),
            "Version": version_label,
        }
        for aop, n in multi.items()
    ]
    detail_df = pd.DataFrame(detail_rows)

    cache_key = f"latest_multi_organ_aops_{version or 'latest'}"
    _plot_data_cache[cache_key] = detail_df if not detail_df.empty else hist_df.copy()
    _plot_data_cache["latest_multi_organ_aops"] = detail_df if not detail_df.empty else hist_df.copy()

    multi_count = int(multi.shape[0])
    multi_pct = 100.0 * multi_count / total_aops if total_aops else 0.0
    subtitle = (
        f"{multi_count}/{total_aops} AOPs ({multi_pct:.0f}%) classify into 2 or more organ systems"
    )

    fig = px.bar(
        hist_df,
        x="Organ systems touched",
        y="AOPs",
        text="AOPs",
        color="Organ systems touched",
        color_discrete_sequence=BRAND_COLORS["palette"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title={"text": f"Multi-organ AOPs<br><sub>{subtitle}</sub>"},
        xaxis_title="Number of distinct organ systems an AOP classifies into",
        yaxis_title="Number of AOPs",
        showlegend=False,
        margin=dict(l=60, r=30, t=80, b=60),
    )

    _plot_figure_cache[cache_key] = fig
    return render_plot_html(fig)

def plot_latest_ke_mmo_coverage(version: str = None) -> str:
    """KE measurement-method (obo:MMO_0000000) coverage for the current snapshot.

    Per PLOT-IDEAS.md §4.1 (#68): the percentage of KEs that lack
    `obo:MMO_0000000` (free-text measurement-method / assay descriptions)
    is the single biggest completeness signal for NAM-ready AOPs and is
    currently invisible on the dashboard. Latest snapshot: ~36% of KEs
    carry an MMO description, ~64% don't.

    Renders a donut split: With MMO (magenta) vs Without MMO (primary).
    """
    global _plot_data_cache, _plot_figure_cache

    where_filter, order_limit = _build_graph_filter(version)
    query = f"""
        SELECT ?graph
               (SUM(IF(BOUND(?mmo), 1, 0)) AS ?with_mmo)
               (COUNT(?ke) AS ?total)
        WHERE {{
            GRAPH ?graph {{
                ?ke a aopo:KeyEvent .
                OPTIONAL {{ ?ke <http://purl.obolibrary.org/obo/MMO_0000000> ?mmo }}
            }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            {where_filter}
        }}
        GROUP BY ?graph
        {order_limit}
    """
    try:
        results = run_sparql_query(query)
        if not results:
            return create_fallback_plot("KE MMO Coverage", "No data available")

        row = results[0]
        with_mmo = int(row.get('with_mmo', {}).get('value', 0))
        total = int(row.get('total', {}).get('value', 0))
        without = max(0, total - with_mmo)
        version_str = row.get('graph', {}).get('value', '').rsplit('/', 1)[-1]

        df = pd.DataFrame({
            'Status': ['With MMO assay', 'Without MMO assay'],
            'KEs': [with_mmo, without],
            'Version': [version_str, version_str],
        })

        fig = px.pie(
            df, names='Status', values='KEs', hole=0.55,
            color='Status',
            color_discrete_map={
                'With MMO assay': BRAND_COLORS['magenta'],
                'Without MMO assay': BRAND_COLORS['primary'],
            },
        )
        pct_covered = (with_mmo / total * 100) if total else 0
        fig.update_layout(
            margin=dict(l=30, r=30, t=80, b=30),
            annotations=[dict(
                text=f"<b>{pct_covered:.0f}%</b><br>covered",
                x=0.5, y=0.5, font=dict(size=18, color=BRAND_COLORS['primary']),
                showarrow=False,
            )],
            title={"text": f"KE measurement-method coverage<br><sub>{with_mmo}/{total} KEs carry an obo:MMO_0000000 description (v{version_str})</sub>"},
        )

        cache_key = f"latest_ke_mmo_coverage_{version or 'latest'}"
        _plot_data_cache[cache_key] = df
        _plot_data_cache['latest_ke_mmo_coverage'] = df
        _plot_figure_cache[cache_key] = fig
        _plot_figure_cache['latest_ke_mmo_coverage'] = fig

        return render_plot_html(fig)

    except Exception as e:
        logger.error(f"Failed to generate KE MMO coverage: {e}")
        return create_fallback_plot("KE MMO Coverage", str(e))


def plot_latest_aop_aop_overlap(version: str = None, min_shared_kes: int = 5, max_pairs: int = 250) -> str:
    """AOP-AOP overlap network: nodes=AOPs, edges=#shared KEs (#67).

    Renders a force-directed graph showing pairs of AOPs that share a
    KE backbone. Nodes coloured by OECD status; node size = #KEs in
    the AOP; edge width = #shared KEs. Hover for AOP titles + edge
    detail.

    Args:
        version: Optional snapshot date string. Defaults to latest.
        min_shared_kes: Threshold for displaying an edge. Default 5 to
            keep the network legible (the v1 issue acceptance criteria
            says default 2 but with 584 AOPs that yields >2400 edges).
            Tune down via the methodology query box to inspect specific
            sub-clusters.
        max_pairs: Hard cap on edges rendered to protect Plotly perf.
    """
    import math
    try:
        import networkx as nx
    except ImportError:
        return create_fallback_plot("AOP-AOP Overlap Network", "networkx not installed")

    where_filter, order_limit = _build_graph_filter(version)

    # Resolve target graph explicitly (we need it for follow-on title/status queries).
    if version:
        graph_uri = f"http://aopwiki.org/graph/{version}"
    else:
        latest_q = """
        SELECT ?g WHERE {
            GRAPH ?g { ?s a aopo:AdverseOutcomePathway . }
            FILTER(STRSTARTS(STR(?g), "http://aopwiki.org/graph/"))
        } GROUP BY ?g ORDER BY DESC(?g) LIMIT 1
        """
        res = run_sparql_query(latest_q)
        if not res:
            return create_fallback_plot("AOP-AOP Overlap Network", "No data")
        graph_uri = res[0]['g']['value']

    version_str = graph_uri.rsplit('/', 1)[-1]

    # Edge query.
    edge_q = f"""
    SELECT ?aop1 ?aop2 (COUNT(DISTINCT ?ke) AS ?sharedKEs)
    WHERE {{
        GRAPH <{graph_uri}> {{
            ?aop1 a aopo:AdverseOutcomePathway ; aopo:has_key_event ?ke .
            ?aop2 a aopo:AdverseOutcomePathway ; aopo:has_key_event ?ke .
            FILTER (STR(?aop1) < STR(?aop2))
        }}
    }}
    GROUP BY ?aop1 ?aop2
    HAVING (COUNT(DISTINCT ?ke) >= {min_shared_kes})
    ORDER BY DESC(?sharedKEs)
    LIMIT {max_pairs}
    """
    edges = run_sparql_query(edge_q)
    if not edges:
        return create_fallback_plot(
            "AOP-AOP Overlap Network",
            f"No AOP pairs share {min_shared_kes}+ Key Events in this snapshot."
        )

    # Per-AOP metadata for nodes that appear in any edge.
    aop_uris = sorted({e['aop1']['value'] for e in edges} | {e['aop2']['value'] for e in edges})
    values_block = ' '.join(f'<{u}>' for u in aop_uris)
    meta_q = f"""
    SELECT ?aop ?title (COUNT(DISTINCT ?ke) AS ?ke_count) ?status
    WHERE {{
        GRAPH <{graph_uri}> {{
            VALUES ?aop {{ {values_block} }}
            ?aop a aopo:AdverseOutcomePathway ;
                 aopo:has_key_event ?ke .
            OPTIONAL {{ ?aop <http://purl.org/dc/elements/1.1/title> ?title }}
            OPTIONAL {{ ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?s . BIND(STR(?s) AS ?status) }}
        }}
    }}
    GROUP BY ?aop ?title ?status
    """
    meta = run_sparql_query(meta_q)
    meta_by_uri = {}
    for r in meta:
        uri = r.get('aop', {}).get('value', '')
        meta_by_uri[uri] = {
            'title': r.get('title', {}).get('value', uri.rsplit('/', 1)[-1]),
            'ke_count': int(r.get('ke_count', {}).get('value', 0)),
            'status': r.get('status', {}).get('value', 'No Status'),
        }

    # Build graph.
    G = nx.Graph()
    for u in aop_uris:
        info = meta_by_uri.get(u, {'title': u.rsplit('/', 1)[-1], 'ke_count': 0, 'status': 'No Status'})
        G.add_node(u, **info)
    for e in edges:
        G.add_edge(e['aop1']['value'], e['aop2']['value'],
                   shared_kes=int(e['sharedKEs']['value']))

    # Layout — spring is stable enough for sub-300-node graphs.
    pos = nx.spring_layout(G, k=1.5 / math.sqrt(len(G.nodes())), iterations=80, seed=42)

    oecd_colors = BRAND_COLORS.get('oecd_status', {})
    default_color = BRAND_COLORS['palette'][0]

    # Edge traces — one trace per width bucket so legend works cleanly.
    edge_x, edge_y, edge_widths, edge_hover = [], [], [], []
    for a, b, attrs in G.edges(data=True):
        x0, y0 = pos[a]; x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_widths.append(attrs['shared_kes'])
        edge_hover.append(f"{meta_by_uri.get(a, {}).get('title', a)} ↔ {meta_by_uri.get(b, {}).get('title', b)}: {attrs['shared_kes']} shared KEs")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(50,50,80,0.25)', width=1),
        hoverinfo='none',
        showlegend=False,
    )

    # Node traces grouped by OECD status.
    statuses = sorted({d['status'] for _, d in G.nodes(data=True)})
    node_traces = []
    for status in statuses:
        sub_nodes = [n for n, d in G.nodes(data=True) if d['status'] == status]
        xs = [pos[n][0] for n in sub_nodes]
        ys = [pos[n][1] for n in sub_nodes]
        texts = []
        sizes = []
        for n in sub_nodes:
            d = G.nodes[n]
            short = n.rsplit('/', 1)[-1]
            texts.append(f"<b>AOP {short}</b><br>{d['title']}<br>KEs: {d['ke_count']}<br>OECD: {d['status']}")
            sizes.append(8 + math.sqrt(max(1, d['ke_count'])) * 2.5)
        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(size=sizes, color=oecd_colors.get(status, default_color),
                        line=dict(color='white', width=1)),
            name=status,
            text=texts,
            hoverinfo='text',
        ))

    fig = go.Figure(data=[edge_trace, *node_traces])
    fig.update_layout(
        title={"text": f"AOP-AOP overlap network<br><sub>{G.number_of_nodes()} AOPs sharing {min_shared_kes}+ KEs ({G.number_of_edges()} edges, v{version_str})</sub>"},
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=30, r=30, t=80, b=30),
        height=700,
        hovermode='closest',
        legend=dict(orientation='v', x=1.02, y=1.0),
    )

    # Cache (per-edge data is the most useful CSV).
    edge_df = pd.DataFrame([
        {
            'aop1': a,
            'aop2': b,
            'aop1_title': meta_by_uri.get(a, {}).get('title', ''),
            'aop2_title': meta_by_uri.get(b, {}).get('title', ''),
            'shared_kes': attrs['shared_kes'],
            'Version': version_str,
        }
        for a, b, attrs in G.edges(data=True)
    ])
    cache_key = f"latest_aop_aop_overlap_{version or 'latest'}"
    _plot_data_cache[cache_key] = edge_df
    _plot_data_cache['latest_aop_aop_overlap'] = edge_df
    _plot_figure_cache[cache_key] = fig
    _plot_figure_cache['latest_aop_aop_overlap'] = fig

    return render_plot_html(fig)


def _compute_aop_maturity(graph_uri: str) -> pd.DataFrame:
    """Per-AOP maturity scores on 4 dimensions for the given snapshot graph.

    Computes the underlying aggregation that backs both
    `plot_latest_aop_maturity_index` (composite #64) and
    `plot_latest_qaop_readiness` (per-dimension #65). Pulled into a
    separate helper because both views need exactly the same numbers.

    Dimensions:
        1. mmo_pct       — % of the AOP's KEs that have `obo:MMO_0000000`
        2. evidence_pct  — % of the AOP's KERs that have `edam:data_2042`
        3. desc_pct      — % of the AOP's KEs that have a `dc:description`
        4. context_pct   — % of the AOP's KEs that have at least one of
                           OrganContext / CellTypeContext / LifeStageContext

    Note: the original issue prediction (#64 / #65) calls out
    `edam:operation_3799` (quantitative understanding) and `nciT:C80263`
    (essentiality of KE) as the 3rd and 4th dimensions. Both predicates
    return 0 hits against the live AOP-Wiki RDF — they are aspirational
    rather than populated. Substituted with description and biological
    context, which are populated and capture similar documentation depth.

    Returns:
        pd.DataFrame: columns aop_uri, aop_id, title, n_kes, n_kers,
        mmo_pct, evidence_pct, desc_pct, context_pct, maturity (composite).
    """
    queries = {
        # Per-AOP counts of KEs / KERs and sub-counts meeting each predicate.
        'base': """
            SELECT ?aop (COUNT(DISTINCT ?ke) AS ?n_kes)
                   (COUNT(DISTINCT ?ke_mmo) AS ?n_mmo)
                   (COUNT(DISTINCT ?ke_desc) AS ?n_desc)
                   (COUNT(DISTINCT ?ke_ctx) AS ?n_ctx)
            WHERE {
              GRAPH <__G__> {
                ?aop a aopo:AdverseOutcomePathway ;
                     aopo:has_key_event ?ke .
                OPTIONAL { ?ke <http://purl.obolibrary.org/obo/MMO_0000000> ?_m .
                           BIND(?ke AS ?ke_mmo) }
                OPTIONAL { ?ke <http://purl.org/dc/elements/1.1/description> ?_d .
                           BIND(?ke AS ?ke_desc) }
                OPTIONAL { ?ke ?_ctx_pred ?_ctx_val .
                           FILTER(?_ctx_pred IN (
                               <http://aopkb.org/aop_ontology#OrganContext>,
                               <http://aopkb.org/aop_ontology#CellTypeContext>,
                               <http://aopkb.org/aop_ontology#LifeStageContext>
                           ))
                           BIND(?ke AS ?ke_ctx) }
              }
            }
            GROUP BY ?aop
        """,
        # Per-AOP KER counts and how many KERs have empirical evidence.
        'kers': """
            SELECT ?aop (COUNT(DISTINCT ?ker) AS ?n_kers)
                   (COUNT(DISTINCT ?ker_ev) AS ?n_evidence)
            WHERE {
              GRAPH <__G__> {
                ?aop a aopo:AdverseOutcomePathway ;
                     aopo:has_key_event_relationship ?ker .
                OPTIONAL { ?ker <http://edamontology.org/data_2042> ?_ev .
                           BIND(?ker AS ?ker_ev) }
              }
            }
            GROUP BY ?aop
        """,
        # AOP titles for the chart axis.
        'titles': """
            SELECT ?aop ?title WHERE {
              GRAPH <__G__> {
                ?aop a aopo:AdverseOutcomePathway .
                OPTIONAL { ?aop <http://purl.org/dc/elements/1.1/title> ?title }
              }
            }
        """,
    }

    base_rows = run_sparql_query(queries['base'].replace('__G__', graph_uri))
    ker_rows = run_sparql_query(queries['kers'].replace('__G__', graph_uri))
    title_rows = run_sparql_query(queries['titles'].replace('__G__', graph_uri))

    base = pd.DataFrame([{
        'aop_uri': r['aop']['value'],
        'n_kes': int(r.get('n_kes', {}).get('value', 0)),
        'n_mmo': int(r.get('n_mmo', {}).get('value', 0)),
        'n_desc': int(r.get('n_desc', {}).get('value', 0)),
        'n_ctx': int(r.get('n_ctx', {}).get('value', 0)),
    } for r in base_rows])
    kers = pd.DataFrame([{
        'aop_uri': r['aop']['value'],
        'n_kers': int(r.get('n_kers', {}).get('value', 0)),
        'n_evidence': int(r.get('n_evidence', {}).get('value', 0)),
    } for r in ker_rows]) if ker_rows else pd.DataFrame(columns=['aop_uri', 'n_kers', 'n_evidence'])
    titles = pd.DataFrame([{
        'aop_uri': r['aop']['value'],
        'title': r.get('title', {}).get('value', r['aop']['value'].rsplit('/', 1)[-1]),
    } for r in title_rows])

    df = base.merge(kers, on='aop_uri', how='left').merge(titles, on='aop_uri', how='left')
    df[['n_kers', 'n_evidence']] = df[['n_kers', 'n_evidence']].fillna(0).astype(int)
    df['title'] = df['title'].fillna(df['aop_uri'].str.rsplit('/', n=1).str[-1])
    df['aop_id'] = df['aop_uri'].str.rsplit('/', n=1).str[-1]

    df['mmo_pct'] = (df['n_mmo'] / df['n_kes'].where(df['n_kes'] > 0) * 100).fillna(0)
    df['evidence_pct'] = (df['n_evidence'] / df['n_kers'].where(df['n_kers'] > 0) * 100).fillna(0)
    df['desc_pct'] = (df['n_desc'] / df['n_kes'].where(df['n_kes'] > 0) * 100).fillna(0)
    df['context_pct'] = (df['n_ctx'] / df['n_kes'].where(df['n_kes'] > 0) * 100).fillna(0)

    dims = ['mmo_pct', 'evidence_pct', 'desc_pct', 'context_pct']
    df['maturity'] = df[dims].mean(axis=1)
    for c in [*dims, 'maturity']:
        df[c] = df[c].round(1)

    return df


def _resolve_latest_graph_for_aop_maturity(version: str = None) -> str:
    if version:
        return f"http://aopwiki.org/graph/{version}"
    q = """
    SELECT ?graph WHERE {
        GRAPH ?graph { ?s a aopo:AdverseOutcomePathway . }
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    } GROUP BY ?graph ORDER BY DESC(?graph) LIMIT 1
    """
    res = run_sparql_query(q)
    return res[0]['graph']['value'] if res else ""


def plot_latest_aop_maturity_index(version: str = None) -> str:
    """Composite AOP maturity index (#64).

    Per-AOP weighted mean of four documentation-completeness sub-scores
    (% KEs with MMO, % KERs with edam:data_2042 evidence, % KEs with
    `dc:description`, % KEs with biological context). Default weights
    are equal (¼ each). Renders the top-30 AOPs by composite as a
    horizontal bar chart, with the four sub-scores stacked underneath
    via hover for transparency.
    """
    graph_uri = _resolve_latest_graph_for_aop_maturity(version)
    if not graph_uri:
        return create_fallback_plot("AOP Maturity Index", "No data")
    version_str = graph_uri.rsplit('/', 1)[-1]
    try:
        df = _compute_aop_maturity(graph_uri)
        if df.empty:
            return create_fallback_plot("AOP Maturity Index", "No AOPs in snapshot")

        top = df.sort_values('maturity', ascending=True).tail(30).copy()
        top['label'] = top['aop_id'] + " — " + top['title'].str.slice(0, 60)
        top['hover'] = top.apply(
            lambda r: (
                f"<b>{r['aop_id']}</b><br>{r['title']}<br>"
                f"Maturity: <b>{r['maturity']:.1f}</b><br>"
                f"MMO: {r['mmo_pct']:.0f}% • Evidence: {r['evidence_pct']:.0f}%<br>"
                f"Description: {r['desc_pct']:.0f}% • Context: {r['context_pct']:.0f}%<br>"
                f"KEs: {r['n_kes']} • KERs: {r['n_kers']}"
            ),
            axis=1,
        )

        fig = px.bar(
            top, x='maturity', y='label', orientation='h',
            color='maturity', color_continuous_scale=[
                BRAND_COLORS['primary'], BRAND_COLORS['blue'], BRAND_COLORS['magenta']
            ],
            labels={'maturity': 'Composite maturity (mean of 4 sub-scores, 0-100)', 'label': 'AOP'},
        )
        fig.update_traces(hovertemplate=top['hover'] + '<extra></extra>')
        fig.update_layout(
            title={"text": f"AOP Maturity Index — top 30<br><sub>Mean of (MMO%, Evidence%, Description%, Context%) per AOP (v{version_str})</sub>"},
            height=900,
            margin=dict(l=200, r=30, t=80, b=60),
            coloraxis_showscale=False,
        )
        fig.update_xaxes(range=[0, 100])

        cache_key = f"latest_aop_maturity_index_{version or 'latest'}"
        export_df = df[['aop_id', 'aop_uri', 'title', 'n_kes', 'n_kers',
                        'mmo_pct', 'evidence_pct', 'desc_pct', 'context_pct', 'maturity']].copy()
        export_df.insert(0, 'Version', version_str)
        _plot_data_cache[cache_key] = export_df
        _plot_data_cache['latest_aop_maturity_index'] = export_df
        _plot_figure_cache[cache_key] = fig
        _plot_figure_cache['latest_aop_maturity_index'] = fig

        return render_plot_html(fig)
    except Exception as e:
        logger.error(f"Failed to compute AOP maturity index: {e}")
        return create_fallback_plot("AOP Maturity Index", str(e))


def plot_latest_qaop_readiness(version: str = None) -> str:
    """Per-AOP qAOP-readiness scorecard (#65) — four sub-scores, top 20.

    Same underlying aggregation as the maturity index (#64), but
    presented as a grouped horizontal bar chart with one bar per
    sub-score per AOP. Sorted by composite descending. Lets users see
    where each AOP is strong vs weak across the four documentation
    dimensions.
    """
    graph_uri = _resolve_latest_graph_for_aop_maturity(version)
    if not graph_uri:
        return create_fallback_plot("qAOP Readiness Scorecard", "No data")
    version_str = graph_uri.rsplit('/', 1)[-1]
    try:
        df = _compute_aop_maturity(graph_uri)
        if df.empty:
            return create_fallback_plot("qAOP Readiness Scorecard", "No AOPs in snapshot")

        top = df.sort_values('maturity', ascending=False).head(20).copy()
        top['label'] = top['aop_id'] + " — " + top['title'].str.slice(0, 50)

        long = top.melt(
            id_vars=['label', 'aop_id', 'title', 'maturity'],
            value_vars=['mmo_pct', 'evidence_pct', 'desc_pct', 'context_pct'],
            var_name='Dimension', value_name='Score',
        )
        dim_labels = {
            'mmo_pct': 'Measurement (MMO)',
            'evidence_pct': 'Empirical evidence (edam:data_2042)',
            'desc_pct': 'KE description',
            'context_pct': 'Biological context',
        }
        long['Dimension'] = long['Dimension'].map(dim_labels)
        dim_order = list(dim_labels.values())

        fig = px.bar(
            long, x='Score', y='label', color='Dimension', orientation='h',
            barmode='group',
            category_orders={'Dimension': dim_order, 'label': top['label'].tolist()[::-1]},
            color_discrete_sequence=[BRAND_COLORS['magenta'], BRAND_COLORS['blue'],
                                     BRAND_COLORS['orange'], BRAND_COLORS['teal']],
            labels={'Score': '% covered (0-100)', 'label': 'AOP'},
        )
        fig.update_layout(
            title={"text": f"qAOP-readiness scorecard — top 20 AOPs by composite maturity<br><sub>Four sub-scores per AOP, sortable & downloadable (v{version_str})</sub>"},
            height=900,
            margin=dict(l=200, r=30, t=80, b=60),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        )
        fig.update_xaxes(range=[0, 105])

        cache_key = f"latest_qaop_readiness_{version or 'latest'}"
        export_df = df.sort_values('maturity', ascending=False)[[
            'aop_id', 'aop_uri', 'title', 'maturity',
            'mmo_pct', 'evidence_pct', 'desc_pct', 'context_pct',
            'n_kes', 'n_kers',
        ]].copy()
        export_df.insert(0, 'Version', version_str)
        _plot_data_cache[cache_key] = export_df
        _plot_data_cache['latest_qaop_readiness'] = export_df
        _plot_figure_cache[cache_key] = fig
        _plot_figure_cache['latest_qaop_readiness'] = fig

        return render_plot_html(fig)
    except Exception as e:
        logger.error(f"Failed to compute qAOP readiness: {e}")
        return create_fallback_plot("qAOP Readiness Scorecard", str(e))


# ---------------------------------------------------------------------------
# Curator backlog views (#70) — four ranked tables for editorial priorities
# ---------------------------------------------------------------------------


def _render_curator_table(df: pd.DataFrame, columns: list[tuple[str, str]],
                          title: str, subtitle: str = "") -> str:
    """Render a DataFrame as a Plotly go.Table with VHP4Safety styling."""
    if df.empty:
        return create_fallback_plot(title, f"No records match this criterion (0 rows). {subtitle}")
    header_values = [c[1] for c in columns]
    cell_values = [df[c[0]].astype(str).tolist() for c in columns]
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in header_values],
            fill_color=BRAND_COLORS['primary'],
            font=dict(color='white', size=12),
            align='left',
            height=32,
        ),
        cells=dict(
            values=cell_values,
            fill_color=[['#f9f9f9', 'white'] * (len(df) // 2 + 1)],
            font=dict(color=BRAND_COLORS['primary'], size=11),
            align='left',
            height=28,
        ),
    )])
    fig.update_layout(
        title={"text": f"{title}<br><sub>{subtitle}</sub>" if subtitle else title},
        margin=dict(l=20, r=20, t=80, b=20),
        height=min(180 + 30 * len(df), 700),
    )
    return fig


def plot_latest_curator_orphan_kes(version: str = None) -> str:
    """KEs that are not currently included in any AOP (#70 first table)."""
    graph_uri = _resolve_latest_graph_for_aop_maturity(version)
    if not graph_uri:
        return create_fallback_plot("Orphan KEs", "No data")
    version_str = graph_uri.rsplit('/', 1)[-1]
    q = f"""
    SELECT ?ke ?title ?modified WHERE {{
      GRAPH <{graph_uri}> {{
        ?ke a aopo:KeyEvent .
        OPTIONAL {{ ?ke <http://purl.org/dc/elements/1.1/title> ?title }}
        OPTIONAL {{ ?ke <http://purl.org/dc/terms/modified> ?modified }}
        FILTER NOT EXISTS {{ ?aop aopo:has_key_event ?ke }}
      }}
    }}
    ORDER BY ?ke
    LIMIT 200
    """
    try:
        rows = run_sparql_query(q)
        df = pd.DataFrame([{
            'KE ID': r['ke']['value'].rsplit('/', 1)[-1],
            'Title': r.get('title', {}).get('value', '—'),
            'Last modified': r.get('modified', {}).get('value', '—')[:10],
            'URI': r['ke']['value'],
            'Version': version_str,
        } for r in rows])
        fig = _render_curator_table(
            df,
            columns=[('KE ID', 'KE'), ('Title', 'Title'), ('Last modified', 'Modified')],
            title="Orphan Key Events",
            subtitle=f"KEs not currently included in any AOP — {len(df)} records (v{version_str}). Promote to an AOP or deprecate.",
        )
        if isinstance(fig, str):
            return fig
        cache_key = f"latest_curator_orphan_kes_{version or 'latest'}"
        _plot_data_cache[cache_key] = df
        _plot_data_cache['latest_curator_orphan_kes'] = df
        _plot_figure_cache[cache_key] = fig
        _plot_figure_cache['latest_curator_orphan_kes'] = fig
        return render_plot_html(fig)
    except Exception as e:
        logger.error(f"Failed to compute orphan-KEs view: {e}")
        return create_fallback_plot("Orphan KEs", str(e))


def plot_latest_curator_stale_aops(version: str = None) -> str:
    """AOPs not modified in > 2 years (#70 second table)."""
    from datetime import datetime, timedelta
    graph_uri = _resolve_latest_graph_for_aop_maturity(version)
    if not graph_uri:
        return create_fallback_plot("Stale AOPs", "No data")
    version_str = graph_uri.rsplit('/', 1)[-1]
    cutoff = (datetime.utcnow() - timedelta(days=730)).strftime('%Y-%m-%d')
    q = f"""
    SELECT ?aop ?title ?modified ?status WHERE {{
      GRAPH <{graph_uri}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             <http://purl.org/dc/terms/modified> ?modified .
        OPTIONAL {{ ?aop <http://purl.org/dc/elements/1.1/title> ?title }}
        OPTIONAL {{ ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?s . BIND(STR(?s) AS ?status) }}
        FILTER(STR(?modified) < "{cutoff}")
      }}
    }}
    ORDER BY ?modified
    LIMIT 200
    """
    try:
        rows = run_sparql_query(q)
        df = pd.DataFrame([{
            'AOP ID': r['aop']['value'].rsplit('/', 1)[-1],
            'Title': r.get('title', {}).get('value', '—'),
            'Last modified': r.get('modified', {}).get('value', '—')[:10],
            'OECD status': r.get('status', {}).get('value', 'No Status'),
            'URI': r['aop']['value'],
            'Version': version_str,
        } for r in rows])
        fig = _render_curator_table(
            df,
            columns=[('AOP ID', 'AOP'), ('Title', 'Title'),
                     ('Last modified', 'Modified'), ('OECD status', 'OECD')],
            title="Stale AOPs",
            subtitle=f"AOPs not modified since {cutoff} (>2 years ago) — {len(df)} records (v{version_str}). Showing oldest first.",
        )
        if isinstance(fig, str):
            return fig
        cache_key = f"latest_curator_stale_aops_{version or 'latest'}"
        _plot_data_cache[cache_key] = df
        _plot_data_cache['latest_curator_stale_aops'] = df
        _plot_figure_cache[cache_key] = fig
        _plot_figure_cache['latest_curator_stale_aops'] = fig
        return render_plot_html(fig)
    except Exception as e:
        logger.error(f"Failed to compute stale-AOPs view: {e}")
        return create_fallback_plot("Stale AOPs", str(e))


def plot_latest_curator_evidence_free_kers(version: str = None) -> str:
    """KERs reused across AOPs but missing edam:data_2042 (#70 third table).

    Highest-leverage curation target per the issue — a single evidence
    note on one KER lifts the maturity of every AOP referencing it.
    """
    graph_uri = _resolve_latest_graph_for_aop_maturity(version)
    if not graph_uri:
        return create_fallback_plot("Reused Evidence-Free KERs", "No data")
    version_str = graph_uri.rsplit('/', 1)[-1]
    q = f"""
    SELECT ?ker (COUNT(DISTINCT ?aop) AS ?n_aops)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?ker a aopo:KeyEventRelationship .
        ?aop aopo:has_key_event_relationship ?ker .
        FILTER NOT EXISTS {{ ?ker <http://edamontology.org/data_2042> ?_ }}
      }}
    }}
    GROUP BY ?ker
    HAVING (COUNT(DISTINCT ?aop) >= 2)
    ORDER BY DESC(?n_aops)
    LIMIT 100
    """
    try:
        rows = run_sparql_query(q)
        df = pd.DataFrame([{
            'KER ID': r['ker']['value'].rsplit('/', 1)[-1],
            'Reused across': int(r['n_aops']['value']),
            'URI': r['ker']['value'],
            'Version': version_str,
        } for r in rows])
        df['Reused across'] = df['Reused across'].astype(str) + ' AOPs'
        fig = _render_curator_table(
            df,
            columns=[('KER ID', 'KER'), ('Reused across', 'Reuse')],
            title="Reused KERs missing empirical evidence",
            subtitle=f"KERs in ≥2 AOPs that lack an edam:data_2042 evidence note — {len(df)} records (v{version_str}). Highest-leverage curation target: one fix lifts every referencing AOP.",
        )
        if isinstance(fig, str):
            return fig
        cache_key = f"latest_curator_evidence_free_kers_{version or 'latest'}"
        _plot_data_cache[cache_key] = df
        _plot_data_cache['latest_curator_evidence_free_kers'] = df
        _plot_figure_cache[cache_key] = fig
        _plot_figure_cache['latest_curator_evidence_free_kers'] = fig
        return render_plot_html(fig)
    except Exception as e:
        logger.error(f"Failed to compute evidence-free-KERs view: {e}")
        return create_fallback_plot("Reused Evidence-Free KERs", str(e))


def plot_latest_curator_abandoned_aops(version: str = None) -> str:
    """AOPs touched once at creation then never revisited (#70 fourth table).

    Heuristic: created == modified AND modified > 2 years old.
    """
    from datetime import datetime, timedelta
    graph_uri = _resolve_latest_graph_for_aop_maturity(version)
    if not graph_uri:
        return create_fallback_plot("Abandoned AOPs", "No data")
    version_str = graph_uri.rsplit('/', 1)[-1]
    cutoff = (datetime.utcnow() - timedelta(days=730)).strftime('%Y-%m-%d')
    q = f"""
    SELECT ?aop ?title ?created ?modified WHERE {{
      GRAPH <{graph_uri}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             <http://purl.org/dc/terms/created> ?created ;
             <http://purl.org/dc/terms/modified> ?modified .
        OPTIONAL {{ ?aop <http://purl.org/dc/elements/1.1/title> ?title }}
        FILTER(STR(?created) = STR(?modified))
        FILTER(STR(?modified) < "{cutoff}")
      }}
    }}
    ORDER BY ?modified
    LIMIT 200
    """
    try:
        rows = run_sparql_query(q)
        df = pd.DataFrame([{
            'AOP ID': r['aop']['value'].rsplit('/', 1)[-1],
            'Title': r.get('title', {}).get('value', '—'),
            'Created/last touch': r.get('modified', {}).get('value', '—')[:10],
            'URI': r['aop']['value'],
            'Version': version_str,
        } for r in rows])
        fig = _render_curator_table(
            df,
            columns=[('AOP ID', 'AOP'), ('Title', 'Title'), ('Created/last touch', 'Date')],
            title="First-pass-then-abandoned AOPs",
            subtitle=f"AOPs whose created date equals their modified date AND the touch is >2 years old — {len(df)} records (v{version_str}). Likely first-pass drafts left unfinished.",
        )
        if isinstance(fig, str):
            return fig
        cache_key = f"latest_curator_abandoned_aops_{version or 'latest'}"
        _plot_data_cache[cache_key] = df
        _plot_data_cache['latest_curator_abandoned_aops'] = df
        _plot_figure_cache[cache_key] = fig
        _plot_figure_cache['latest_curator_abandoned_aops'] = fig
        return render_plot_html(fig)
    except Exception as e:
        logger.error(f"Failed to compute abandoned-AOPs view: {e}")
        return create_fallback_plot("Abandoned AOPs", str(e))


def plot_latest_ke_wikipathways_overlay(version: str = None, min_mapped_genes: int = 3, top_n: int = 30) -> str:
    """KE ↔ WikiPathways overlay (#69) — pathway support per KE.

    For each KE with ≥`min_mapped_genes` HGNC mappings, queries WikiPathways
    directly (no SPARQL federation, since the federation surface is unreliable
    and the WP RDF predicate for gene symbols has moved). Returns the top-N
    KEs ranked by the number of WikiPathways pathways that touch at least
    one of the KE's mapped genes.

    Bridges Marvin's two pillars (D1 AOP-Wiki RDF + D2 WikiPathways). The
    visualisation is a horizontal bar chart of `#pathways supporting`; CSV
    download has the full (KE, pathway, supporting genes) triples for
    follow-on biology work.

    Note: WikiPathways queries are made from the dashboard's Python layer
    against `https://sparql.wikipathways.org/sparql` directly. Throttling /
    transient failures degrade gracefully (empty cells in the chart).
    """
    import requests as _requests

    graph_uri = _resolve_latest_graph_for_aop_maturity(version)
    if not graph_uri:
        return create_fallback_plot("KE ↔ WikiPathways Overlay", "No data")
    version_str = graph_uri.rsplit('/', 1)[-1]

    # 1. From AOPWikiRDF: KE → list of HGNC symbols (filtered to KEs with ≥N genes).
    ke_q = f"""
    SELECT ?ke ?title ?sym WHERE {{
      GRAPH <{graph_uri}> {{
        ?ke a aopo:KeyEvent ;
            <http://edamontology.org/data_1025> ?gene .
        ?gene <http://www.w3.org/2000/01/rdf-schema#label> ?sym .
        OPTIONAL {{ ?ke <http://purl.org/dc/elements/1.1/title> ?title }}
        {{ SELECT ?ke WHERE {{
            GRAPH <{graph_uri}> {{
              ?ke a aopo:KeyEvent ;
                  <http://edamontology.org/data_1025> ?g .
            }}
          }} GROUP BY ?ke HAVING (COUNT(DISTINCT ?g) >= {min_mapped_genes})
        }}
      }}
    }}
    """
    try:
        ke_rows = run_sparql_query(ke_q)
    except Exception as e:
        logger.error(f"KE→symbol query failed: {e}")
        return create_fallback_plot("KE ↔ WikiPathways Overlay", str(e))

    ke_to_syms: dict[str, set[str]] = {}
    ke_titles: dict[str, str] = {}
    for r in ke_rows:
        ke = r['ke']['value']
        sym = r.get('sym', {}).get('value', '').strip()
        if not sym:
            continue
        ke_to_syms.setdefault(ke, set()).add(sym)
        if 'title' in r and ke not in ke_titles:
            ke_titles[ke] = r.get('title', {}).get('value', '')

    if not ke_to_syms:
        return create_fallback_plot(
            "KE ↔ WikiPathways Overlay",
            f"No KEs with ≥{min_mapped_genes} HGNC mappings in this snapshot."
        )

    all_symbols = sorted({s for syms in ke_to_syms.values() for s in syms})

    # 2. From WikiPathways: symbol → pathways. Chunk to keep request size reasonable.
    sym_to_pathways: dict[str, set[tuple[str, str]]] = {}
    chunk = 80
    wp_endpoint = "https://sparql.wikipathways.org/sparql"
    for i in range(0, len(all_symbols), chunk):
        sub = all_symbols[i:i + chunk]
        values_block = ' '.join(f'"{s}"' for s in sub)
        wp_q = f"""
        PREFIX wp: <http://vocabularies.wikipathways.org/wp#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?sym ?pathway ?pwTitle WHERE {{
          ?g a wp:GeneProduct ;
             rdfs:label ?sym ;
             dcterms:isPartOf ?pathway .
          ?pathway a wp:Pathway ;
                   dc:title ?pwTitle .
          VALUES ?sym {{ {values_block} }}
        }}
        """
        try:
            resp = _requests.post(
                wp_endpoint,
                data={'query': wp_q},
                headers={'Accept': 'application/sparql-results+json'},
                timeout=45,
            )
            if resp.status_code != 200:
                logger.warning(f"WP chunk {i//chunk}: HTTP {resp.status_code}")
                continue
            for b in resp.json().get('results', {}).get('bindings', []):
                sym = b['sym']['value']
                pw = b['pathway']['value']
                title = b.get('pwTitle', {}).get('value', '')
                sym_to_pathways.setdefault(sym, set()).add((pw, title))
        except Exception as e:
            logger.warning(f"WP chunk {i//chunk} failed: {e}")

    # 3. Aggregate per KE.
    rows = []
    for ke, syms in ke_to_syms.items():
        pathway_genes: dict[tuple[str, str], set[str]] = {}
        for s in syms:
            for pw_tuple in sym_to_pathways.get(s, set()):
                pathway_genes.setdefault(pw_tuple, set()).add(s)
        n_pw = len(pathway_genes)
        rows.append({
            'ke_id': ke.rsplit('/', 1)[-1],
            'ke_uri': ke,
            'title': ke_titles.get(ke, '—'),
            'n_genes': len(syms),
            'n_pathways': n_pw,
            'top_pathways': '; '.join(f"{t} ({len(g)} genes)" for (pw, t), g in
                                       sorted(pathway_genes.items(),
                                              key=lambda kv: -len(kv[1]))[:5]),
        })

    df = pd.DataFrame(rows)
    df = df[df['n_pathways'] > 0].sort_values('n_pathways', ascending=False)
    if df.empty:
        return create_fallback_plot(
            "KE ↔ WikiPathways Overlay",
            "No WikiPathways pathways found for KE-mapped genes (WP endpoint may be throttling)."
        )

    top = df.head(top_n).copy()
    top['label'] = top['ke_id'] + " — " + top['title'].str.slice(0, 55)

    fig = px.bar(
        top, x='n_pathways', y='label', orientation='h',
        color='n_genes',
        color_continuous_scale=[BRAND_COLORS['sky_blue'], BRAND_COLORS['deep_magenta']],
        labels={'n_pathways': '#WikiPathways pathways supporting this KE',
                'label': 'Key Event', 'n_genes': '#HGNC genes mapped to this KE'},
        hover_data={'n_genes': True, 'top_pathways': True, 'label': False},
    )
    fig.update_layout(
        title={"text": f"KE ↔ WikiPathways overlay — top {len(top)} KEs by pathway support<br><sub>KEs with ≥{min_mapped_genes} HGNC mappings (v{version_str}, WP live)</sub>"},
        height=850,
        margin=dict(l=200, r=30, t=80, b=60),
    )

    cache_key = f"latest_ke_wikipathways_overlay_{version or 'latest'}"
    export = df.copy()
    export.insert(0, 'Version', version_str)
    _plot_data_cache[cache_key] = export
    _plot_data_cache['latest_ke_wikipathways_overlay'] = export
    _plot_figure_cache[cache_key] = fig
    _plot_figure_cache['latest_ke_wikipathways_overlay'] = fig

    return render_plot_html(fig)
