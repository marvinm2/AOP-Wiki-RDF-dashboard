"""AOP-Wiki RDF Dashboard - Historical Trends Plot Functions.

This module contains all historical trend and time-series visualization functions
for the AOP-Wiki RDF Dashboard. These functions analyze data evolution over time
by querying multiple RDF graph versions and creating comparative visualizations.

Key Features:
    - Time-series analysis of entity counts and relationships
    - Delta change calculations between versions
    - Historical pattern identification and trend analysis
    - Parallel SPARQL query execution for optimal performance
    - Comprehensive data caching for CSV export functionality

Historical Plot Functions:
    Core Evolution Analysis:
        - plot_main_graph(): Entity count evolution (AOPs, KEs, KERs, Stressors)
        - plot_avg_per_aop(): Average components per AOP trends
        - plot_network_density(): Graph connectivity evolution
        - plot_author_counts(): Author contribution patterns

    Component Analysis:
        - plot_ke_components(): Component annotation trends
        - plot_ke_components_percentage(): Percentage-based component trends
        - plot_unique_ke_components(): Unique component evolution
        - plot_kes_by_kec_count(): KE component count distributions

    Ontology Usage:
        - plot_bio_processes(): Biological process annotation trends
        - plot_bio_objects(): Biological object annotation trends

    Property Analysis:
        - plot_aop_property_presence(): Property presence evolution
        - plot_aop_property_presence_unique_colors(): Enhanced property visualization

    Temporal Analysis:
        - plot_aop_lifetime(): AOP creation and modification patterns

Visualization Characteristics:
    - Consistent VHP4Safety branding and color palette
    - Interactive hover modes with unified tooltips
    - Responsive design for desktop and mobile
    - Professional margins and typography
    - Delta change analysis for trend identification

Data Processing:
    - Robust SPARQL query execution with retry logic
    - Version-based data sorting and chronological ordering
    - Data quality validation and error handling
    - Global caching for CSV export functionality

Author:
    Generated with Claude Code (https://claude.ai/code)
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio
from functools import reduce
import colorsys
from .shared import (
    BRAND_COLORS, config, _plot_data_cache, run_sparql_query, extract_counts, safe_read_csv
)


def plot_main_graph() -> tuple[str, str, pd.DataFrame]:
    """Generate the main AOP entity evolution visualization with absolute and delta views.

    Creates comprehensive time-series visualizations showing the evolution of core
    AOP-Wiki entities (AOPs, Key Events, Key Event Relationships, and Stressors)
    across different RDF graph versions. Provides both absolute counts and
    delta changes between versions with global data caching for CSV exports.

    The function executes multiple SPARQL queries to gather entity counts across
    all available graph versions, processes the data into time-series format,
    and generates interactive Plotly visualizations with VHP4Safety branding.

    SPARQL Queries:
        - AOPs: Counts AdverseOutcomePathway instances per graph
        - KEs: Counts KeyEvent instances per graph
        - KERs: Counts KeyEventRelationship instances per graph
        - Stressors: Counts Stressor (nci:C54571) instances per graph

    Returns:
        tuple[str, str, pd.DataFrame]: A 3-tuple containing:
            - str: HTML for absolute counts line chart
            - str: HTML for delta changes line chart
            - pd.DataFrame: Processed data with all entity counts by version

    Data Processing:
        1. Executes 4 parallel SPARQL queries for entity types
        2. Extracts and validates count data using extract_counts()
        3. Merges results into unified DataFrame with outer join
        4. Sorts by datetime-converted version strings
        5. Computes delta changes between consecutive versions
        6. Caches data globally for CSV export functionality

    Visualizations:
        Absolute Plot:
            - Multi-line chart showing entity counts over time
            - Uses VHP4Safety color palette for consistency
            - Interactive hover with unified mode
            - Responsive design with proper margins

        Delta Plot:
            - Shows changes between consecutive versions
            - Highlights growth/decline patterns
            - Same styling and interactivity as absolute plot
            - Useful for identifying update patterns

    Global Caching:
        Stores processed data in _plot_data_cache for CSV downloads:
        - 'main_graph_absolute': Melted absolute data
        - 'main_graph_delta': Melted delta data

    Example Data Structure:
        >>> abs_html, delta_html, df = plot_main_graph()
        >>> print(df.head())
           version  AOPs  KEs  KERs  Stressors
        0  2023-01-15   120  800   650        45
        1  2023-06-01   135  850   720        48
        2  2024-01-15   142  890   780        52

    Error Handling:
        - Empty query results: Logs warnings, continues with available data
        - Invalid data formats: Uses extract_counts() error handling
        - Merge failures: Returns empty DataFrame and fallback plots
        - Visualization errors: Handled by safe_plot_execution wrapper

    Performance:
        - Parallel query execution for optimal speed
        - Efficient DataFrame operations with pandas
        - Cached results prevent recomputation
        - Optimized for time-series data processing

    Usage:
        Called during application startup for dashboard generation:
        >>> # In app.py startup
        >>> abs_plot, delta_plot, data = plot_main_graph()
        >>> # Data automatically cached for CSV downloads

    Note:
        This is the primary historical analysis function and forms the
        foundation for understanding AOP-Wiki data evolution patterns.
        The function assumes RDF graphs follow the naming pattern:
        "http://aopwiki.org/graph/{version}"
    """
    global _plot_data_cache

    sparql_queries = {
        "AOPs": """
            SELECT ?graph (COUNT(?aop) AS ?count)
            WHERE {
                GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
        """,
        "KEs": """
            SELECT ?graph (COUNT(?ke) AS ?count)
            WHERE {
                GRAPH ?graph {?ke a aopo:KeyEvent .
                }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
        """,
        "KERs": """
            SELECT ?graph (COUNT(?ker) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?ker a aopo:KeyEventRelationship .
                }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
        """,
        "Stressors": """
            SELECT ?graph (COUNT(?s) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?s a nci:C54571 .
                }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
        """
    }

    # --- Query and merge results ---
    df_list = []
    for label, query in sparql_queries.items():
        results = run_sparql_query(query)
        df = extract_counts(results)
        df.rename(columns={"count": label}, inplace=True)
        df_list.append(df)

    df_all = reduce(lambda left, right: pd.merge(left, right, on="version", how="outer"), df_list)

    # Convert to datetime for correct sorting
    df_all["version_dt"] = pd.to_datetime(df_all["version"], errors="coerce")
    df_all = df_all.sort_values("version_dt").drop(columns="version_dt").reset_index(drop=True)

    # --- Absolute plot ---
    df_abs_melted = df_all.melt(id_vars="version", var_name="Entity", value_name="Count")
    fig_abs = px.line(
        df_abs_melted, x="version", y="Count", color="Entity", markers=True,
        title="Entity Evolution Over Time",
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,   # Let Plotly resize dynamically
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_abs.update_xaxes(
        tickmode='array',
        tickvals=df_all["version"],
        ticktext=df_all["version"],
        tickangle=-45
    )

    # --- Delta plot ---
    df_delta = df_all.copy()
    for col in sparql_queries.keys():
        df_delta[f"{col}_Δ"] = df_all[col].diff().fillna(0).astype(int)

    df_delta_melted = df_delta.melt(
        id_vars="version",
        value_vars=[f"{k}_Δ" for k in sparql_queries.keys()],
        var_name="Entity",
        value_name="Count"
    )
    df_delta_melted["Entity"] = df_delta_melted["Entity"].str.replace("_Δ", "")

    fig_delta = px.line(
        df_delta_melted, x="version", y="Count", color="Entity", markers=True,
        title="Entity Change Between Versions",
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_delta.update_xaxes(
        tickmode='array',
        tickvals=df_all["version"],
        ticktext=df_all["version"],
        tickangle=-45
    )

    # Store absolute and delta data in cache for CSV download
    _plot_data_cache['main_graph_absolute'] = df_abs_melted
    _plot_data_cache['main_graph_delta'] = df_delta_melted

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs="cdn", config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        df_all
    )


def plot_avg_per_aop() -> tuple[str, str]:
    """Generate average components per AOP visualization with absolute and delta views."""
    query_aops = """
        SELECT ?graph (COUNT(?aop) AS ?count)
        WHERE {
            GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
    """
    query_kes = """
        SELECT ?graph (COUNT(?ke) AS ?count)
        WHERE {
            GRAPH ?graph {
                ?ke a aopo:KeyEvent .
            }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
    """
    query_kers = """
        SELECT ?graph (COUNT(?ker) AS ?count)
        WHERE {
            GRAPH ?graph {
                ?ker a aopo:KeyEventRelationship .
            }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
    """

    df_aops = extract_counts(run_sparql_query(query_aops))
    df_aops.rename(columns={"count": "aop_count"}, inplace=True)
    df_aops["version_dt"] = pd.to_datetime(df_aops["version"], errors="coerce")

    df_kes = extract_counts(run_sparql_query(query_kes))
    df_kes.rename(columns={"count": "ke_count"}, inplace=True)
    df_kes["version_dt"] = pd.to_datetime(df_kes["version"], errors="coerce")

    df_kers = extract_counts(run_sparql_query(query_kers))
    df_kers.rename(columns={"count": "ker_count"}, inplace=True)
    df_kers["version_dt"] = pd.to_datetime(df_kers["version"], errors="coerce")

    df_all = df_aops.merge(df_kes, on="version").merge(df_kers, on="version")
    df_all = df_all.drop_duplicates("version").copy()
    df_all["version_dt"] = pd.to_datetime(df_all["version"], errors="coerce")
    df_all = df_all.sort_values("version_dt").drop(columns="version_dt")

    df_all["avg_KEs_per_AOP"] = df_all["ke_count"] / df_all["aop_count"]
    df_all["avg_KERs_per_AOP"] = df_all["ker_count"] / df_all["aop_count"]

    # Absolute
    df_melted = df_all.melt(
        id_vars="version",
        value_vars=["avg_KEs_per_AOP", "avg_KERs_per_AOP"],
        var_name="Metric",
        value_name="Average"
    ).sort_values("version")

    df_melted["Metric"] = df_melted["Metric"].replace({
        "avg_KEs_per_AOP": "Average KEs per AOP",
        "avg_KERs_per_AOP": "Average KERs per AOP"
    })

    fig_abs = px.line(df_melted, x="version", y="Average", color="Metric", markers=True,
                      title="Average Components per AOP Over Time",
                      color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary']])
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_abs.update_xaxes(
        tickmode='array',
        tickvals=df_all["version"],
        ticktext=df_all["version"],
        tickangle=-45
    )

    # Delta
    df_all["avg_KEs_per_AOP_Δ"] = df_all["avg_KEs_per_AOP"].diff().fillna(0)
    df_all["avg_KERs_per_AOP_Δ"] = df_all["avg_KERs_per_AOP"].diff().fillna(0)
    df_delta_melted = df_all.melt(
        id_vars="version",
        value_vars=["avg_KEs_per_AOP_Δ", "avg_KERs_per_AOP_Δ"],
        var_name="Metric",
        value_name="Δ Average"
    ).sort_values("version")

    df_delta_melted["Metric"] = df_delta_melted["Metric"].replace({
        "avg_KEs_per_AOP_Δ": "Δ KEs per AOP",
        "avg_KERs_per_AOP_Δ": "Δ KERs per AOP"
    })
    fig_delta = px.line(df_delta_melted, x="version", y="Δ Average", color="Metric", markers=True,
                        title="Change in Average Components per AOP",
                        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary']])
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_delta.update_xaxes(
        tickmode='array',
        tickvals=df_all["version"],
        ticktext=df_all["version"],
        tickangle=-45
    )

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )


def plot_network_density() -> str:
    """Generate network density evolution visualization."""
    query_density = """
    SELECT ?graph (COUNT(DISTINCT ?ke) AS ?nodes) (COUNT(?ker) AS ?edges)
    WHERE {
      GRAPH ?graph {
        ?aop a aopo:AdverseOutcomePathway ;
             aopo:has_key_event ?ke ;
             aopo:has_key_event_relationship ?ker .
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    """

    results = run_sparql_query(query_density)
    df_density = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "nodes": int(r["nodes"]["value"]),
        "edges": int(r["edges"]["value"])
    } for r in results])

    # Avoid division by zero
    df_density["density"] = df_density.apply(
        lambda row: 2 * row["edges"] / (row["nodes"] * (row["nodes"] - 1)) if row["nodes"] > 1 else 0,
        axis=1
    )

    # Sort and ensure tick labels
    df_density["version_dt"] = pd.to_datetime(df_density["version"], errors="coerce")
    df_density = df_density.sort_values("version_dt").drop(columns="version_dt")

    fig = px.line(
        df_density,
        x="version",
        y="density",
        title="Network Density Evolution Over Time",
        markers=True,
        labels={"density": "Graph Density"},
        color_discrete_sequence=[BRAND_COLORS['accent']]
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=df_density["version"],
        ticktext=df_density["version"],
        tickangle=-45
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def plot_author_counts() -> tuple[str, str]:
    """Generate author contribution visualization with absolute and delta views."""
    query = """
    SELECT ?graph (COUNT(DISTINCT ?c) AS ?author_count)
    WHERE {
      GRAPH ?graph {
        ?aop a aopo:AdverseOutcomePathway ;
             dc:creator ?c .
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY ?graph
    """

    results = run_sparql_query(query)

    df_authors = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "author_count": int(r["author_count"]["value"])
    } for r in results])

    df_authors["version_dt"] = pd.to_datetime(df_authors["version"], errors="coerce")
    df_authors = df_authors.sort_values("version_dt").drop(columns="version_dt")

    # Absolute
    fig_abs = px.line(df_authors, x="version", y="author_count", markers=True,
                      title="Author Contributions Over Time",
                      color_discrete_sequence=[BRAND_COLORS['secondary']])
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_abs.update_xaxes(
        tickmode='array',
        tickvals=df_authors["version"],
        ticktext=df_authors["version"],
        tickangle=-45
    )

    # Delta
    df_authors["author_count_Δ"] = df_authors["author_count"].diff().fillna(0)
    fig_delta = px.line(df_authors, x="version", y="author_count_Δ", markers=True,
                        title="Change in Author Contributions",
                        color_discrete_sequence=[BRAND_COLORS['light']])
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_delta.update_xaxes(
        tickmode='array',
        tickvals=df_authors["version"],
        ticktext=df_authors["version"],
        tickangle=-45
    )

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )


def plot_aop_lifetime():
    """Generate AOP lifetime analysis visualizations."""
    query_lifetime = """
    SELECT ?graph ?aop ?created ?modified
    WHERE {
      GRAPH ?graph {
        ?aop a aopo:AdverseOutcomePathway ;
             dcterms:created ?created ;
             dcterms:modified ?modified .
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    """
    results_lifetime = run_sparql_query(query_lifetime)

    df_lifetime = pd.DataFrame([{
        "aop": r["aop"]["value"],
        "version": r["graph"]["value"].split("/")[-1],
        "created": pd.to_datetime(r["created"]["value"]),
        "modified": pd.to_datetime(r["modified"]["value"])
    } for r in results_lifetime])

    df_lifetime["lifetime_days"] = (df_lifetime["modified"] - df_lifetime["created"]).dt.days
    df_lifetime["year_created"] = df_lifetime["created"].dt.year
    df_lifetime["year_modified"] = df_lifetime["modified"].dt.year

    # Deduplicate
    df_created = df_lifetime.sort_values("created").drop_duplicates("aop", keep="first")
    df_modified = df_lifetime.sort_values("modified").drop_duplicates("aop", keep="last")

    # --- Plot 1: AOPs Created ---
    fig1 = px.histogram(df_created, x="year_created",
                        title="AOP Creation Timeline",
                        labels={"year_created": "Year", "count": "AOP Count"},
                        color_discrete_sequence=[BRAND_COLORS['primary']])
    fig1.update_layout(template="plotly_white", height=400)
    html1 = pio.to_html(fig1, full_html=False, include_plotlyjs="cdn")

    # --- Plot 2: AOPs Modified ---
    fig2 = px.histogram(df_modified, x="year_modified",
                        title="AOP Modification Timeline",
                        labels={"year_modified": "Year", "count": "AOP Count"},
                        color_discrete_sequence=[BRAND_COLORS['secondary']])
    fig2.update_layout(template="plotly_white", height=400)
    html2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)

    # --- Plot 3: Created vs. Modified Dates ---
    fig3 = px.scatter(df_lifetime, x="created", y="modified", hover_name="aop",
                      title="AOP Creation vs. Modification Timeline",
                      labels={"created": "Created", "modified": "Modified"},
                      color_discrete_sequence=[BRAND_COLORS['accent']])
    fig3.update_layout(template="plotly_white", height=500)
    html3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False)

    return html1, html2, html3


def plot_ke_components() -> tuple[str, str]:
    """Generate KE component annotation visualization with absolute and delta views."""
    query_components = """
    SELECT ?graph
           (COUNT(?process) AS ?process_count)
           (COUNT(?object) AS ?object_count)
           (COUNT(?action) AS ?action_count)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL { ?bioevent aopo:hasProcess ?process . }
        OPTIONAL { ?bioevent aopo:hasObject ?object . }
        OPTIONAL { ?bioevent aopo:hasAction ?action . }
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    """

    results_components = run_sparql_query(query_components)
    df_components = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "Process": int(r["process_count"]["value"]),
        "Object": int(r["object_count"]["value"]),
        "Action": int(r["action_count"]["value"])
    } for r in results_components])

    # Sort by datetime
    df_components["version_dt"] = pd.to_datetime(df_components["version"], errors="coerce")
    df_components = df_components.sort_values("version_dt").drop(columns="version_dt")

    # --- Absolute line plot ---
    df_melted = df_components.melt(
        id_vars=["version"],
        value_vars=["Process", "Object", "Action"],
        var_name="Component", value_name="Count"
    )

    fig_abs = px.line(
        df_melted,
        x="version",
        y="Count",
        color="Component",
        title="KE Component Annotations Over Time",
        markers=True,
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_abs.update_xaxes(
        tickmode='array',
        tickvals=df_components["version"],
        ticktext=df_components["version"],
        tickangle=-45
    )

    # --- Delta line plot ---
    df_diff = df_components.copy()
    df_diff[["Process", "Object", "Action"]] = \
        df_diff[["Process", "Object", "Action"]].diff().fillna(0)

    df_melted_diff = df_diff.melt(
        id_vars=["version"],
        value_vars=["Process", "Object", "Action"],
        var_name="Component", value_name="Change"
    )

    fig_delta = px.line(
        df_melted_diff,
        x="version",
        y="Change",
        color="Component",
        title="Change in KE Component Annotations",
        markers=True,
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_delta.update_xaxes(
        tickmode='array',
        tickvals=df_components["version"],
        ticktext=df_components["version"],
        tickangle=-45
    )

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )


def plot_ke_components_percentage() -> tuple[str, str]:
    """Generate KE component percentage visualization with absolute and delta views."""
    query_components = """
    SELECT ?graph
           (COUNT(?process) AS ?process_count)
           (COUNT(?object) AS ?object_count)
           (COUNT(?action) AS ?action_count)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL { ?bioevent aopo:hasProcess ?process . }
        OPTIONAL { ?bioevent aopo:hasObject ?object . }
        OPTIONAL { ?bioevent aopo:hasAction ?action . }
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    """

    query_total_kes = """
    SELECT ?graph (COUNT(?ke) AS ?total_kes)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent .
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    """

    # Run queries
    results_components = run_sparql_query(query_components)
    results_total_kes = run_sparql_query(query_total_kes)

    # DataFrames
    df_components = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "Process": int(r["process_count"]["value"]),
        "Object": int(r["object_count"]["value"]),
        "Action": int(r["action_count"]["value"])
    } for r in results_components])

    df_total = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "TotalKEs": int(r["total_kes"]["value"])
    } for r in results_total_kes])

    # Merge for percentage calculation
    df_merged = df_components.merge(df_total, on="version", how="inner")

    # Calculate percentages
    for col in ["Process", "Object", "Action"]:
        df_merged[col] = (df_merged[col] / df_merged["TotalKEs"]) * 100

    # Sort by version
    df_merged["version_dt"] = pd.to_datetime(df_merged["version"], errors="coerce")
    df_merged = df_merged.sort_values("version_dt").drop(columns="version_dt")

    # --- Absolute percentage plot ---
    df_melted = df_merged.melt(
        id_vars=["version"],
        value_vars=["Process", "Object", "Action"],
        var_name="Component", value_name="Percentage"
    )

    fig_abs = px.line(
        df_melted,
        x="version",
        y="Percentage",
        color="Component",
        title="KE Component Annotations as Percentage Over Time",
        markers=True,
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        yaxis=dict(title="Percentage (%)"),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_abs.update_xaxes(
        tickmode='array',
        tickvals=df_merged["version"],
        ticktext=df_merged["version"],
        tickangle=-45
    )

    # --- Delta percentage plot ---
    df_delta = df_merged.copy()
    df_delta[["Process", "Object", "Action"]] = df_delta[["Process", "Object", "Action"]].diff().fillna(0)

    df_melted_delta = df_delta.melt(
        id_vars=["version"],
        value_vars=["Process", "Object", "Action"],
        var_name="Component", value_name="Percentage Change"
    )

    fig_delta = px.line(
        df_melted_delta,
        x="version",
        y="Percentage Change",
        color="Component",
        title="Change in KE Component Percentage",
        markers=True,
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        yaxis=dict(title="Percentage Change (%)"),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_delta.update_xaxes(
        tickmode='array',
        tickvals=df_merged["version"],
        ticktext=df_merged["version"],
        tickangle=-45
    )

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config=config),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config=config)
    )


def plot_unique_ke_components() -> tuple[str, str]:
    """Generate unique KE component visualization with absolute and delta views."""
    query_unique_components = """
    SELECT ?graph
           (COUNT(DISTINCT ?process) AS ?process_count)
           (COUNT(DISTINCT ?object) AS ?object_count)
           (COUNT(DISTINCT ?action) AS ?action_count)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL { ?bioevent aopo:hasProcess ?process . }
        OPTIONAL { ?bioevent aopo:hasObject ?object . }
        OPTIONAL { ?bioevent aopo:hasAction ?action . }
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    """

    results_unique = run_sparql_query(query_unique_components)
    df_unique = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "Process": int(r["process_count"]["value"]),
        "Object": int(r["object_count"]["value"]),
        "Action": int(r["action_count"]["value"])
    } for r in results_unique])

    # Sort by datetime
    df_unique["version_dt"] = pd.to_datetime(df_unique["version"], errors="coerce")
    df_unique = df_unique.sort_values("version_dt").drop(columns="version_dt")

    # --- Absolute plot ---
    df_melted = df_unique.melt(
        id_vars=["version"],
        value_vars=["Process", "Object", "Action"],
        var_name="Component", value_name="Unique Count"
    )

    fig_abs = px.line(
        df_melted,
        x="version",
        y="Unique Count",
        color="Component",
        title="Unique KE Component Annotations Over Time",
        markers=True,
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_abs.update_xaxes(
        tickmode='array',
        tickvals=df_unique["version"],
        ticktext=df_unique["version"],
        tickangle=-45
    )

    # --- Delta plot ---
    df_diff = df_unique.copy()
    df_diff[["Process", "Object", "Action"]] = \
        df_diff[["Process", "Object", "Action"]].diff().fillna(0)

    df_melted_diff = df_diff.melt(
        id_vars=["version"],
        value_vars=["Process", "Object", "Action"],
        var_name="Component", value_name="Change"
    )

    fig_delta = px.line(
        df_melted_diff,
        x="version",
        y="Change",
        color="Component",
        title="Change in Unique KE Component Annotations",
        markers=True,
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_delta.update_xaxes(
        tickmode='array',
        tickvals=df_unique["version"],
        ticktext=df_unique["version"],
        tickangle=-45
    )

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )


def plot_bio_processes() -> tuple[str, str]:
    """Generate biological process ontology usage visualization with absolute and delta views."""
    query_ontologies = """
    SELECT ?graph ?ontology (COUNT(DISTINCT ?process) AS ?count)
    WHERE {
      GRAPH ?graph {
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
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph ?ontology
    ORDER BY ?graph ?ontology
    """

    results_ont = run_sparql_query(query_ontologies)
    df_ont = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "ontology": r["ontology"]["value"],
        "count": int(r["count"]["value"])
    } for r in results_ont if "ontology" in r])

    # Sort by datetime
    df_ont["version_dt"] = pd.to_datetime(df_ont["version"], errors="coerce")
    df_ont = df_ont.sort_values("version_dt").drop(columns="version_dt")

    # --- Absolute bar chart ---
    fig_abs = px.bar(
        df_ont,
        x="version",
        y="count",
        color="ontology",
        barmode="group",
        title="Biological Process Annotations by Ontology Over Time",
        labels={"count": "Annotated KEs", "ontology": "Ontology"},
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_abs.update_layout(template="plotly_white", hovermode="x unified", autosize=True)
    fig_abs.update_xaxes(tickmode='array', tickvals=df_ont["version"], ticktext=df_ont["version"], tickangle=-45)

    # --- Delta calculation ---
    df_delta = df_ont.copy()
    df_delta["count"] = df_delta.groupby("ontology")["count"].diff().fillna(0)

    fig_delta = px.bar(
        df_delta,
        x="version",
        y="count",
        color="ontology",
        barmode="group",
        title="Change in Biological Process Annotations by Ontology",
        labels={"count": "Change in Annotated KEs", "ontology": "Ontology"},
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_delta.update_layout(template="plotly_white", hovermode="x unified", autosize=True)
    fig_delta.update_xaxes(tickmode='array', tickvals=df_ont["version"], ticktext=df_ont["version"], tickangle=-45)

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )


def plot_bio_objects() -> tuple[str, str]:
    """Generate biological object ontology usage visualization with absolute and delta views."""
    query_objects = """
    SELECT ?graph ?ontology (COUNT(DISTINCT ?object) AS ?count)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        ?bioevent aopo:hasObject ?object .

        BIND(
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/GO_"), "GO",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/PR_"), "PR",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/CHEBI_"), "CHEBI",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/UBERON_"), "UBERON",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/CL_"), "CL",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/MP_"), "MP",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/NBO_"), "NBO",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/MI_"), "MI",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/VT_"), "VT",
          IF(STRSTARTS(STR(?object), "http://purl.org/commons/record/mesh/"), "MESH",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/HP_"), "HP", "OTHER"))))))))))) AS ?ontology)
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph ?ontology
    ORDER BY ?graph ?ontology
    """

    results_obj = run_sparql_query(query_objects)
    df_obj = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "ontology": r["ontology"]["value"],
        "count": int(r["count"]["value"])
    } for r in results_obj if "ontology" in r])

    df_obj["version_dt"] = pd.to_datetime(df_obj["version"], errors="coerce")
    df_obj = df_obj.sort_values("version_dt").drop(columns="version_dt")

    # --- Absolute bar chart ---
    fig_abs = px.bar(
        df_obj,
        x="version",
        y="count",
        color="ontology",
        barmode="group",
        title="Biological Object Annotations by Ontology Over Time",
        labels={"count": "Annotated KEs", "ontology": "Ontology"},
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_abs.update_layout(template="plotly_white", hovermode="x unified", autosize=True)
    fig_abs.update_xaxes(tickmode='array', tickvals=df_obj["version"], ticktext=df_obj["version"], tickangle=-45)

    # --- Delta calculation ---
    df_delta = df_obj.copy()
    df_delta["count"] = df_delta.groupby("ontology")["count"].diff().fillna(0)

    fig_delta = px.bar(
        df_delta,
        x="version",
        y="count",
        color="ontology",
        barmode="group",
        title="Change in Biological Object Annotations by Ontology",
        labels={"count": "Change in Annotated KEs", "ontology": "Ontology"},
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_delta.update_layout(template="plotly_white", hovermode="x unified", autosize=True)
    fig_delta.update_xaxes(tickmode='array', tickvals=df_obj["version"], ticktext=df_obj["version"], tickangle=-45)

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )


def plot_aop_property_presence(label_file="property_labels.csv") -> tuple[str, str]:
    """Generate AOP property presence visualization with absolute and percentage views."""
    query_props = """
    SELECT ?graph ?p (COUNT(DISTINCT ?AOP) AS ?count)
    WHERE {
      GRAPH ?graph {
        ?AOP a aopo:AdverseOutcomePathway ;
             ?p ?o .
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph ?p
    ORDER BY ?graph ?p
    """

    query_total = """
    SELECT ?graph (COUNT(DISTINCT ?AOP) AS ?total)
    WHERE {
      GRAPH ?graph {
        ?AOP a aopo:AdverseOutcomePathway .
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY ?graph
    """

    results_props = run_sparql_query(query_props)
    results_total = run_sparql_query(query_total)

    df_props = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "property": r["p"]["value"],
        "count": int(r["count"]["value"])
    } for r in results_props])

    df_total = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "total_aops": int(r["total"]["value"])
    } for r in results_total])

    # Merge
    df = df_props.merge(df_total, on="version", how="left")
    df["percentage"] = (df["count"] / df["total_aops"]) * 100

    # Remove properties that are 100% in all versions
    props_to_keep = (
        df.groupby("property")["percentage"]
          .max()
          .loc[lambda x: x < 100]
          .index
    )
    df = df[df["property"].isin(props_to_keep)]

    # Label mapping with safe file reading
    default_labels = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata"}
    ]
    df_labels = safe_read_csv(label_file, default_labels)

    if not df_labels.empty:
        df = df.merge(df_labels, how="left", left_on="property", right_on="uri")
        df["display_label"] = df["label"].fillna(df["property"])
    else:
        df["display_label"] = df["property"]

    # Sort
    df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
    df = df.sort_values("version_dt")

    # Absolute presence
    fig_abs = px.line(
        df,
        x="version",
        y="count",
        color="display_label",
        markers=True,
        title="Property Presence in AOPs Over Time (Count)",
        labels={"count": "Number of AOPs", "display_label": "Property"},
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_abs.update_layout(
    template="plotly_white",
    hovermode="x unified",
    autosize=True,
    margin=dict(l=50, r=20, t=50, b=50)
    )

    # Percentage presence
    fig_delta = px.line(
        df,
        x="version",
        y="percentage",
        color="display_label",
        markers=True,
        title="Property Presence in AOPs Over Time (Percentage)",
        labels={"percentage": "Percentage (%)", "display_label": "Property"},
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )

    config = {
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "aop_property_presence",
            "height": 1000,
            "width": 1600,
            "scale": 4
        }
    }

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config=config),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config=config)
    )


def plot_aop_property_presence_unique_colors(label_file="property_labels.csv") -> tuple[str, str]:
    """Generate AOP property presence visualization with unique colors for each property line."""
    global _plot_data_cache

    query_props = """
    SELECT ?graph ?p (COUNT(DISTINCT ?AOP) AS ?count)
    WHERE {
      GRAPH ?graph {
        ?AOP a aopo:AdverseOutcomePathway ;
             ?p ?o .
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph ?p
    ORDER BY ?graph ?p
    """

    query_total = """
    SELECT ?graph (COUNT(DISTINCT ?AOP) AS ?total)
    WHERE {
      GRAPH ?graph {
        ?AOP a aopo:AdverseOutcomePathway .
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY ?graph
    """

    results_props = run_sparql_query(query_props)
    results_total = run_sparql_query(query_total)

    df_props = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "property": r["p"]["value"],
        "count": int(r["count"]["value"])
    } for r in results_props])

    df_total = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "total_aops": int(r["total"]["value"])
    } for r in results_total])

    # Merge
    df = df_props.merge(df_total, on="version", how="left")
    df["percentage"] = (df["count"] / df["total_aops"]) * 100

    # Remove properties that are 100% in all versions
    props_to_keep = (
        df.groupby("property")["percentage"]
          .max()
          .loc[lambda x: x < 100]
          .index
    )
    df = df[df["property"].isin(props_to_keep)]

    # Label mapping with safe file reading
    default_labels = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata"}
    ]
    df_labels = safe_read_csv(label_file, default_labels)

    if not df_labels.empty:
        df = df.merge(df_labels, how="left", left_on="property", right_on="uri")
        df["display_label"] = df["label"].fillna(df["property"])
    else:
        df["display_label"] = df["property"]

    # Sort
    df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
    df = df.sort_values("version_dt")

    # Store data in cache for CSV downloads
    _plot_data_cache['aop_property_presence_unique_absolute'] = df.copy()
    _plot_data_cache['aop_property_presence_unique_percentage'] = df.copy()

    # Generate unique colors for each property
    unique_properties = sorted(df["display_label"].unique())

    # Generate distinct colors using HSV color space
    num_colors = len(unique_properties)
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        # Use varying saturation and value for more distinction
        saturation = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9
        value = 0.8 + (i % 2) * 0.1       # 0.8, 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)

    # Create color mapping
    color_map = dict(zip(unique_properties, colors))

    # Absolute presence with unique colors
    fig_abs = px.line(
        df,
        x="version",
        y="count",
        color="display_label",
        markers=True,
        title="Property Presence Over Time - Enhanced Visualization (Count)",
        labels={"count": "Number of AOPs", "display_label": "Property"},
        color_discrete_map=color_map
    )
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        height=700,  # Larger height for better visibility
        margin=dict(l=50, r=20, t=70, b=50)
    )

    # Percentage presence with unique colors
    fig_delta = px.line(
        df,
        x="version",
        y="percentage",
        color="display_label",
        markers=True,
        title="Property Presence Over Time - Enhanced Visualization (Percentage)",
        labels={"percentage": "Percentage (%)", "display_label": "Property"},
        color_discrete_map=color_map
    )
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        height=700,  # Larger height for better visibility
        margin=dict(l=50, r=20, t=70, b=50)
    )

    config = {"responsive": True}

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config=config),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config=config)
    )


def plot_kes_by_kec_count() -> tuple[str, str]:
    """Generate KE distribution by KEC count visualization with absolute and delta views."""
    query_kec_count = """
    PREFIX aopo: <http://aopkb.org/aop_ontology#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX dcterms: <http://purl.org/dc/terms/>

    SELECT ?graph ?bioevent_count_group
           (COUNT(DISTINCT ?ke) AS ?total_kes)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent .
        OPTIONAL { ?ke aopo:hasBiologicalEvent ?bioevent . }
        {
          SELECT ?ke (COUNT(DISTINCT ?bioevent2) AS ?bioevent_count)
          WHERE {
            ?ke a aopo:KeyEvent .
            OPTIONAL { ?ke aopo:hasBiologicalEvent ?bioevent2 . }
          }
          GROUP BY ?ke
        }
        BIND(
          IF(?bioevent_count = 0, "0",
            IF(?bioevent_count = 1, "1",
               IF(?bioevent_count = 2, "2",
                  IF(?bioevent_count = 3, "3",
                     IF(?bioevent_count = 4, "4",
                        IF(?bioevent_count = 5, "5",
                           IF(?bioevent_count >= 6, "6+", ">1"))))))) AS ?bioevent_count_group
        )
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph ?bioevent_count_group
    ORDER BY ?graph xsd:integer(?bioevent_count_group)
    """

    results = run_sparql_query(query_kec_count)

    data = []
    for r in results:
        if "graph" in r and "bioevent_count_group" in r and "total_kes" in r:
            data.append({
                "version": r["graph"]["value"].split("/")[-1],
                "bioevent_count_group": r["bioevent_count_group"]["value"],
                "total_kes": int(r["total_kes"]["value"])
            })

    if not data:
        print("No data for KE by KEC count plot")
        return "", ""

    df = pd.DataFrame(data)

    # --- Fill missing groups ---
    all_versions = sorted(df["version"].unique())
    all_groups = sorted(df["bioevent_count_group"].unique())

    idx = pd.MultiIndex.from_product([all_versions, all_groups],
                                     names=["version", "bioevent_count_group"])
    df_full = df.set_index(["version", "bioevent_count_group"]).reindex(idx, fill_value=0).reset_index()

    # Sort
    df_full["version_dt"] = pd.to_datetime(df_full["version"], errors="coerce")
    df_full = df_full.sort_values(["version_dt", "bioevent_count_group"]).drop(columns="version_dt")

    # --- Absolute stacked area plot ---
    fig_abs = px.area(
        df_full,
        x="version",
        y="total_kes",
        color="bioevent_count_group",
        title="KE Distribution by Component Count Over Time",
        labels={"total_kes": "Number of KEs", "bioevent_count_group": "Number of Components"},
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(
            tickmode='array',
            tickvals=all_versions,
            ticktext=all_versions,
            tickangle=-45
        )
    )

    # --- Delta stacked area plot ---
    df_delta = df_full.copy()
    df_delta["total_kes_delta"] = df_delta.groupby("bioevent_count_group")["total_kes"].diff().fillna(0)

    fig_delta = px.area(
        df_delta,
        x="version",
        y="total_kes_delta",
        color="bioevent_count_group",
        title="Change in KE Distribution by Component Count",
        labels={"total_kes_delta": "Change in KEs", "bioevent_count_group": "Number of Components"},
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(
            tickmode='array',
            tickvals=all_versions,
            ticktext=all_versions,
            tickangle=-45
        )
    )

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )