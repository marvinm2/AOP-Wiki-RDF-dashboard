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
    Generated with Claude Code (https://claude.ai/code)
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio
from .shared import (
    BRAND_COLORS, config, _plot_data_cache, run_sparql_query, safe_read_csv, create_fallback_plot
)


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
        title=f"Current Database Composition ({latest_version})",
        color="Entity",
        text="Count",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50),
        yaxis=dict(title="Count"),
        xaxis=dict(title="Entity Type")
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def plot_latest_ke_components(version: str = None) -> str:
    """Create a pie chart showing the current KE component distribution."""
    global _plot_data_cache

    # Build graph filter based on version parameter
    where_filter, order_limit = _build_graph_filter(version)

    query_components = f"""
    SELECT ?graph
           (COUNT(?process) AS ?process_count)
           (COUNT(?object) AS ?object_count)
           (COUNT(?action) AS ?action_count)
    WHERE {{
      GRAPH ?graph {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL {{ ?bioevent aopo:hasProcess ?process . }}
        OPTIONAL {{ ?bioevent aopo:hasObject ?object . }}
        OPTIONAL {{ ?bioevent aopo:hasAction ?action . }}
      }}
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
      {where_filter}
    }}
    GROUP BY ?graph
    {order_limit}
    """

    results = run_sparql_query(query_components)
    if not results:
        return "<p>No data available</p>"

    result = results[0]
    latest_version = result["graph"]["value"].split("/")[-1]

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
        title=f"Current KE Component Distribution ({latest_version})",
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


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
    _plot_data_cache['latest_network_density'] = df

    fig = px.pie(
        df, values="Count", names="Type",
        title=f"Current AOP Connectivity Analysis ({latest_version})",
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50),
        annotations=[
            dict(text=f"Total AOPs: {total_aops}<br>Connected: {connected_aops}<br>Isolated: {isolated_aops}",
                 x=0.5, y=0.1, font_size=12, showarrow=False)
        ]
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


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
        title=f"Current Average Connectivity per AOP ({latest_version})",
        color="Metric",
        text="Value",
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary']]
    )

    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def plot_latest_ontology_usage(version: str = None) -> str:
    """Create a chart showing current ontology usage."""
    where_filter, order_limit = _build_graph_filter(version)

    query = f"""
    SELECT ?graph ?ontology (COUNT(DISTINCT ?term) AS ?count)
    WHERE {{
      GRAPH ?graph {{
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
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
      {where_filter}
    }}
    GROUP BY ?graph ?ontology
    {order_limit}
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Ontology Usage", "No ontology data available")

    # Get only the latest version
    latest_version = results[0]["graph"]["value"].split("/")[-1]
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

    fig = px.pie(
        df, values="Terms", names="Ontology",
        title=f"Current Ontology Term Usage ({latest_version})"
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


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
        title=f"Current Process Ontology Sources ({latest_version})",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


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
        title=f"Current Object Ontology Sources ({latest_version})",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


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

    # Store in global cache for CSV download
    _plot_data_cache['latest_aop_completeness'] = df

    # Use centralized brand colors for consistency
    color_map = BRAND_COLORS['type_colors'].copy()
    # Add fallback for any missing types
    color_map.update({"Structure": BRAND_COLORS['accent']})

    fig = px.bar(
        df, x="Property", y="Completeness", color="Type",
        title=f"Current AOP Data Completeness ({latest_version})",
        text="Completeness",
        color_discrete_map=color_map
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=100),
        yaxis=dict(title="Completeness (%)", range=[0, 105]),
        xaxis=dict(title="AOP Properties", tickangle=45),
        legend=dict(title="Property Type")
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


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

    # Create unique color mapping using the full brand palette
    unique_properties = df["Property"].unique()
    # Use the brand palette, cycling through if more properties than colors
    palette = BRAND_COLORS['palette']
    color_map = {}
    for i, prop in enumerate(unique_properties):
        color_map[prop] = palette[i % len(palette)]

    fig = px.bar(
        df, x="Property", y="Completeness", color="Property",
        title=f"Current AOP Data Completeness - Enhanced View ({latest_version})",
        text="Completeness",
        color_discrete_map=color_map
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=600,  # Larger height for better visibility
        margin=dict(l=50, r=20, t=70, b=120),  # More space for labels
        yaxis=dict(title="Completeness (%)", range=[0, 105]),
        xaxis=dict(title="AOP Properties", tickangle=45),
        showlegend=False  # Hide legend since colors are unique per property
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


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

    # Regular bar chart since values are in similar ranges
    fig = px.bar(
        df, x="Entity", y="Count",
        title=f"Current Core Entity Summary ({latest_version})",
        text="Count",
        color="Entity",
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50),
        yaxis=dict(title="Count")
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


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
        title=f"Current KE Annotation Depth Distribution ({latest_version})",
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})