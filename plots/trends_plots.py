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
        - plot_aop_property_presence(): Property presence evolution with marker shapes

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
import logging
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed
from .shared import (
    BRAND_COLORS, config, _plot_data_cache, _plot_figure_cache,
    run_sparql_query, run_sparql_query_with_retry, extract_counts,
    safe_read_csv, create_fallback_plot, get_properties_for_entity,
    get_all_versions
)

logger = logging.getLogger(__name__)


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
    global _plot_data_cache, _plot_figure_cache

    try:
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

        if df_all.empty:
            logger.warning("Main graph query returned no results")
            return (
                create_fallback_plot("Entity Evolution Over Time", "No data available"),
                create_fallback_plot("Entity Change Between Versions", "No data available"),
                pd.DataFrame()
            )

        # Convert to datetime for correct sorting
        df_all["version_dt"] = pd.to_datetime(df_all["version"], errors="coerce")
        df_all = df_all.sort_values("version_dt").drop(columns="version_dt").reset_index(drop=True)

        # --- Absolute plot ---
        df_abs_melted = df_all.melt(id_vars="version", var_name="Entity", value_name="Count")
        # Clean data: fill NaN with 0 and ensure numeric type
        df_abs_melted["Count"] = pd.to_numeric(df_abs_melted["Count"], errors="coerce").fillna(0).astype(int)
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

        # Cache figures for image export
        _plot_figure_cache['aop_entity_counts_absolute'] = fig_abs
        _plot_figure_cache['aop_entity_counts_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs="cdn", config={"responsive": True}),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True}),
            df_all
        )

    except Exception as e:
        logger.error(f"Failed to generate main graph plots: {str(e)}")
        return (
            create_fallback_plot("Entity Evolution Over Time", str(e)),
            create_fallback_plot("Entity Change Between Versions", str(e)),
            pd.DataFrame()
        )


def plot_avg_per_aop() -> tuple[str, str]:
    """Generate average components per AOP visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
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

        if df_all.empty:
            logger.warning("Average per AOP query returned no results")
            return (
                create_fallback_plot("Average Components per AOP Over Time", "No data available"),
                create_fallback_plot("Change in Average Components per AOP", "No data available")
            )

        # Guard against division by zero
        df_all["avg_KEs_per_AOP"] = df_all.apply(
            lambda row: row["ke_count"] / row["aop_count"] if row["aop_count"] > 0 else 0, axis=1
        )
        df_all["avg_KERs_per_AOP"] = df_all.apply(
            lambda row: row["ker_count"] / row["aop_count"] if row["aop_count"] > 0 else 0, axis=1
        )

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

        # Cache data for CSV export
        _plot_data_cache['average_components_per_aop_absolute'] = df_melted.copy()
        _plot_data_cache['average_components_per_aop_delta'] = df_delta_melted.copy()

        # Cache figures for image export
        _plot_figure_cache['average_components_per_aop_absolute'] = fig_abs
        _plot_figure_cache['average_components_per_aop_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
        )

    except Exception as e:
        logger.error(f"Failed to generate average per AOP plots: {str(e)}")
        return (
            create_fallback_plot("Average Components per AOP Over Time", str(e)),
            create_fallback_plot("Change in Average Components per AOP", str(e))
        )


def plot_network_density() -> str:
    """Generate network density evolution visualization."""
    global _plot_data_cache, _plot_figure_cache

    try:
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

        if not results:
            logger.warning("Network density query returned no results")
            return create_fallback_plot("Network Density Evolution Over Time", "No data available")

        df_density = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "nodes": int(r["nodes"]["value"]),
            "edges": int(r["edges"]["value"])
        } for r in results])

        if df_density.empty:
            logger.warning("Network density DataFrame is empty")
            return create_fallback_plot("Network Density Evolution Over Time", "No data available")

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

        # Cache data for CSV export
        _plot_data_cache['aop_network_density'] = df_density.copy()

        # Cache figure for image export
        _plot_figure_cache['aop_network_density'] = fig

        return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})

    except Exception as e:
        logger.error(f"Failed to generate network density plot: {str(e)}")
        return create_fallback_plot("Network Density Evolution Over Time", str(e))


def plot_author_counts() -> tuple[str, str]:
    """Generate author contribution visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
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

        if not results:
            logger.warning("Author counts query returned no results")
            return (
                create_fallback_plot("Author Contributions Over Time", "No data available"),
                create_fallback_plot("Change in Author Contributions", "No data available")
            )

        df_authors = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "author_count": int(r["author_count"]["value"])
        } for r in results])

        if df_authors.empty:
            logger.warning("Author counts DataFrame is empty")
            return (
                create_fallback_plot("Author Contributions Over Time", "No data available"),
                create_fallback_plot("Change in Author Contributions", "No data available")
            )

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

        # Cache data for CSV export
        _plot_data_cache['aop_authors_absolute'] = df_authors.copy()
        _plot_data_cache['aop_authors_delta'] = df_authors[['version', 'author_count_Δ']].copy()

        # Cache figures for image export
        _plot_figure_cache['aop_authors_absolute'] = fig_abs
        _plot_figure_cache['aop_authors_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
        )

    except Exception as e:
        logger.error(f"Failed to generate author count plots: {str(e)}")
        return (
            create_fallback_plot("Author Contributions Over Time", str(e)),
            create_fallback_plot("Change in Author Contributions", str(e))
        )


def plot_aop_lifetime() -> tuple[str, str, str]:
    """Generate AOP lifetime analysis visualizations."""
    global _plot_data_cache, _plot_figure_cache

    fallback_created = create_fallback_plot("AOP Creation Timeline", "No data available")
    fallback_modified = create_fallback_plot("AOP Modification Timeline", "No data available")
    fallback_scatter = create_fallback_plot("AOP Creation vs. Modification Timeline", "No data available")

    try:
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

        if not results_lifetime:
            logger.warning("AOP lifetime query returned no results")
            return fallback_created, fallback_modified, fallback_scatter

        df_lifetime = pd.DataFrame([{
            "aop": r["aop"]["value"],
            "version": r["graph"]["value"].split("/")[-1],
            "created": pd.to_datetime(r["created"]["value"], errors="coerce"),
            "modified": pd.to_datetime(r["modified"]["value"], errors="coerce")
        } for r in results_lifetime])

        if df_lifetime.empty:
            logger.warning("AOP lifetime DataFrame is empty after parsing")
            return fallback_created, fallback_modified, fallback_scatter

        # Drop rows where date parsing failed
        df_lifetime = df_lifetime.dropna(subset=["created", "modified"])
        if df_lifetime.empty:
            logger.warning("AOP lifetime DataFrame is empty after dropping invalid dates")
            return fallback_created, fallback_modified, fallback_scatter

        df_lifetime["lifetime_days"] = (df_lifetime["modified"] - df_lifetime["created"]).dt.days
        df_lifetime["year_created"] = df_lifetime["created"].dt.year
        df_lifetime["year_modified"] = df_lifetime["modified"].dt.year

        # Deduplicate
        df_created = df_lifetime.sort_values("created").drop_duplicates("aop", keep="first")
        df_modified = df_lifetime.sort_values("modified").drop_duplicates("aop", keep="last")

        # Cache data for CSV export
        _plot_data_cache['aops_created_over_time'] = df_created.copy()
        _plot_data_cache['aops_modified_over_time'] = df_modified.copy()
        _plot_data_cache['aop_creation_vs_modification_timeline'] = df_lifetime.copy()

        # --- Plot 1: AOPs Created ---
        html1 = fallback_created
        try:
            fig1 = px.histogram(df_created, x="year_created",
                                title="AOP Creation Timeline",
                                labels={"year_created": "Year", "count": "AOP Count"},
                                color_discrete_sequence=[BRAND_COLORS['primary']])
            fig1.update_layout(template="plotly_white", height=400)
            html1 = pio.to_html(fig1, full_html=False, include_plotlyjs="cdn")
            _plot_figure_cache['aops_created_over_time'] = fig1
        except Exception as e:
            logger.error(f"Failed to generate AOP creation timeline plot: {str(e)}")

        # --- Plot 2: AOPs Modified ---
        html2 = fallback_modified
        try:
            fig2 = px.histogram(df_modified, x="year_modified",
                                title="AOP Modification Timeline",
                                labels={"year_modified": "Year", "count": "AOP Count"},
                                color_discrete_sequence=[BRAND_COLORS['secondary']])
            fig2.update_layout(template="plotly_white", height=400)
            html2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)
            _plot_figure_cache['aops_modified_over_time'] = fig2
        except Exception as e:
            logger.error(f"Failed to generate AOP modification timeline plot: {str(e)}")

        # --- Plot 3: Created vs. Modified Dates ---
        html3 = fallback_scatter
        try:
            fig3 = px.scatter(df_lifetime, x="created", y="modified", hover_name="aop",
                              title="AOP Creation vs. Modification Timeline",
                              labels={"created": "Created", "modified": "Modified"},
                              color_discrete_sequence=[BRAND_COLORS['accent']],
                              render_mode='svg')
            fig3.update_layout(template="plotly_white", height=500)
            html3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False)
            _plot_figure_cache['aop_creation_vs_modification_timeline'] = fig3
        except Exception as e:
            logger.error(f"Failed to generate AOP creation vs modification scatter plot: {str(e)}")

        return html1, html2, html3

    except Exception as e:
        logger.error(f"Failed to generate AOP lifetime plots: {str(e)}")
        return fallback_created, fallback_modified, fallback_scatter


def _query_ke_components_version(version_info, use_distinct=False):
    """Query KE component counts for a single version graph.

    Args:
        version_info: Dict with 'version' and 'graph_uri' keys
        use_distinct: If True, count DISTINCT values (for unique component queries)

    Returns:
        Dict with version, Process, Object, Action counts, or None on error
    """
    graph_uri = version_info["graph_uri"]
    version_str = version_info["version"]
    distinct = "DISTINCT " if use_distinct else ""

    query = f"""
    SELECT (COUNT({distinct}?process) AS ?process_count)
           (COUNT({distinct}?object) AS ?object_count)
           (COUNT({distinct}?action) AS ?action_count)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL {{ ?bioevent aopo:hasProcess ?process . }}
        OPTIONAL {{ ?bioevent aopo:hasObject ?object . }}
        OPTIONAL {{ ?bioevent aopo:hasAction ?action . }}
      }}
    }}
    """

    try:
        results = run_sparql_query_with_retry(query, max_retries=2)
        if results:
            r = results[0]
            return {
                "version": version_str,
                "Process": int(r["process_count"]["value"]),
                "Object": int(r["object_count"]["value"]),
                "Action": int(r["action_count"]["value"])
            }
    except Exception as e:
        logger.warning(f"KE components query failed for version {version_str}: {e}")
    return None


def plot_ke_components() -> tuple[str, str]:
    """Generate KE component annotation visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = get_all_versions()
        if not versions:
            logger.warning("No versions available for KE components query")
            return (
                create_fallback_plot("KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Annotations", "No data available")
            )

        logger.info(f"KE components: querying {len(versions)} versions in parallel")

        # Query each version in parallel to avoid Virtuoso cross-graph timeout
        data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_query_ke_components_version, v): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        data.append(result)
                except Exception as e:
                    logger.warning(f"KE components query timed out for version {version_str}: {e}")

        if not data:
            logger.warning("KE components query returned no results")
            return (
                create_fallback_plot("KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Annotations", "No data available")
            )

        df_components = pd.DataFrame(data)

        if df_components.empty:
            logger.warning("KE components DataFrame is empty")
            return (
                create_fallback_plot("KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Annotations", "No data available")
            )

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

        # Cache data for CSV export
        _plot_data_cache['ke_component_annotations_absolute'] = df_melted.copy()
        _plot_data_cache['ke_component_annotations_delta'] = df_melted_diff.copy()

        # Cache figures for image export
        _plot_figure_cache['ke_component_annotations_absolute'] = fig_abs
        _plot_figure_cache['ke_component_annotations_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
        )

    except Exception as e:
        logger.error(f"Failed to generate KE component annotations plots: {str(e)}")
        return (
            create_fallback_plot("KE Component Annotations Over Time", str(e)),
            create_fallback_plot("Change in KE Component Annotations", str(e))
        )


def _query_ke_components_percentage_version(version_info):
    """Query KE component counts and total KEs for a single version graph.

    Returns:
        Dict with version, Process, Object, Action, TotalKEs counts, or None on error
    """
    graph_uri = version_info["graph_uri"]
    version_str = version_info["version"]

    # Query components
    query_comp = f"""
    SELECT (COUNT(?process) AS ?process_count)
           (COUNT(?object) AS ?object_count)
           (COUNT(?action) AS ?action_count)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL {{ ?bioevent aopo:hasProcess ?process . }}
        OPTIONAL {{ ?bioevent aopo:hasObject ?object . }}
        OPTIONAL {{ ?bioevent aopo:hasAction ?action . }}
      }}
    }}
    """

    # Query total KEs
    query_total = f"""
    SELECT (COUNT(?ke) AS ?total_kes)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?ke a aopo:KeyEvent .
      }}
    }}
    """

    try:
        results_comp = run_sparql_query_with_retry(query_comp, max_retries=2)
        results_total = run_sparql_query_with_retry(query_total, max_retries=2)
        if results_comp and results_total:
            rc = results_comp[0]
            rt = results_total[0]
            return {
                "version": version_str,
                "Process": int(rc["process_count"]["value"]),
                "Object": int(rc["object_count"]["value"]),
                "Action": int(rc["action_count"]["value"]),
                "TotalKEs": int(rt["total_kes"]["value"])
            }
    except Exception as e:
        logger.warning(f"KE components percentage query failed for version {version_str}: {e}")
    return None


def plot_ke_components_percentage() -> tuple[str, str]:
    """Generate KE component percentage visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = get_all_versions()
        if not versions:
            logger.warning("No versions available for KE components percentage query")
            return (
                create_fallback_plot("KE Component Annotations as Percentage Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Percentage", "No data available")
            )

        logger.info(f"KE components percentage: querying {len(versions)} versions in parallel")

        # Query each version in parallel
        data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_query_ke_components_percentage_version, v): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        data.append(result)
                except Exception as e:
                    logger.warning(f"KE components percentage query timed out for version {version_str}: {e}")

        if not data:
            logger.warning("KE components percentage query returned no results")
            return (
                create_fallback_plot("KE Component Annotations as Percentage Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Percentage", "No data available")
            )

        df_merged = pd.DataFrame(data)

        # Filter out versions with zero total KEs to avoid division by zero
        df_merged = df_merged[df_merged["TotalKEs"] > 0]

        if df_merged.empty:
            logger.warning("KE components percentage: no versions with non-zero KE counts")
            return (
                create_fallback_plot("KE Component Annotations as Percentage Over Time", "No versions with KE data"),
                create_fallback_plot("Change in KE Component Percentage", "No versions with KE data")
            )

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

        # Cache data for CSV export
        _plot_data_cache['ke_components_percentage_absolute'] = df_melted.copy()
        _plot_data_cache['ke_components_percentage_delta'] = df_melted_delta.copy()

        # Cache figures for image export
        _plot_figure_cache['ke_components_percentage_absolute'] = fig_abs
        _plot_figure_cache['ke_components_percentage_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config=config),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config=config)
        )

    except Exception as e:
        logger.error(f"Failed to generate KE components percentage plots: {str(e)}")
        return (
            create_fallback_plot("KE Component Annotations as Percentage Over Time", str(e)),
            create_fallback_plot("Change in KE Component Percentage", str(e))
        )


def plot_unique_ke_components() -> tuple[str, str]:
    """Generate unique KE component visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = get_all_versions()
        if not versions:
            logger.warning("No versions available for unique KE components query")
            return (
                create_fallback_plot("Unique KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in Unique KE Component Annotations", "No data available")
            )

        logger.info(f"Unique KE components: querying {len(versions)} versions in parallel")

        # Query each version in parallel using shared helper with use_distinct=True
        data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_query_ke_components_version, v, True): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        data.append(result)
                except Exception as e:
                    logger.warning(f"Unique KE components query timed out for version {version_str}: {e}")

        if not data:
            logger.warning("Unique KE components query returned no results")
            return (
                create_fallback_plot("Unique KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in Unique KE Component Annotations", "No data available")
            )

        df_unique = pd.DataFrame(data)

        if df_unique.empty:
            logger.warning("Unique KE components DataFrame is empty")
            return (
                create_fallback_plot("Unique KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in Unique KE Component Annotations", "No data available")
            )

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

        # Cache data for CSV export
        _plot_data_cache['unique_ke_components_absolute'] = df_melted.copy()
        _plot_data_cache['unique_ke_components_delta'] = df_melted_diff.copy()

        # Cache figures for image export
        _plot_figure_cache['unique_ke_components_absolute'] = fig_abs
        _plot_figure_cache['unique_ke_components_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
        )

    except Exception as e:
        logger.error(f"Failed to generate unique KE component plots: {str(e)}")
        return (
            create_fallback_plot("Unique KE Component Annotations Over Time", str(e)),
            create_fallback_plot("Change in Unique KE Component Annotations", str(e))
        )


def plot_bio_processes() -> tuple[str, str]:
    """Generate biological process ontology usage visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
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

        if not results_ont:
            logger.warning("Biological process annotations query returned no results")
            return (
                create_fallback_plot("Biological Process Annotations by Ontology Over Time", "No data available"),
                create_fallback_plot("Change in Biological Process Annotations by Ontology", "No data available")
            )

        df_ont = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "ontology": r["ontology"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_ont if "ontology" in r])

        if df_ont.empty:
            logger.warning("Biological process annotations DataFrame is empty")
            return (
                create_fallback_plot("Biological Process Annotations by Ontology Over Time", "No data available"),
                create_fallback_plot("Change in Biological Process Annotations by Ontology", "No data available")
            )

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

        # Cache data for CSV export
        _plot_data_cache['biological_process_annotations_absolute'] = df_ont.copy()
        _plot_data_cache['biological_process_annotations_delta'] = df_delta.copy()

        # Cache figures for image export
        _plot_figure_cache['biological_process_annotations_absolute'] = fig_abs
        _plot_figure_cache['biological_process_annotations_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
        )

    except Exception as e:
        logger.error(f"Failed to generate biological process annotation plots: {str(e)}")
        return (
            create_fallback_plot("Biological Process Annotations by Ontology Over Time", str(e)),
            create_fallback_plot("Change in Biological Process Annotations by Ontology", str(e))
        )


def plot_bio_objects() -> tuple[str, str]:
    """Generate biological object ontology usage visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
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

        if not results_obj:
            logger.warning("Biological object annotations query returned no results")
            return (
                create_fallback_plot("Biological Object Annotations by Ontology Over Time", "No data available"),
                create_fallback_plot("Change in Biological Object Annotations by Ontology", "No data available")
            )

        df_obj = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "ontology": r["ontology"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_obj if "ontology" in r])

        if df_obj.empty:
            logger.warning("Biological object annotations DataFrame is empty")
            return (
                create_fallback_plot("Biological Object Annotations by Ontology Over Time", "No data available"),
                create_fallback_plot("Change in Biological Object Annotations by Ontology", "No data available")
            )

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

        # Cache data for CSV export
        _plot_data_cache['biological_object_annotations_absolute'] = df_obj.copy()
        _plot_data_cache['biological_object_annotations_delta'] = df_delta.copy()

        # Cache figures for image export
        _plot_figure_cache['biological_object_annotations_absolute'] = fig_abs
        _plot_figure_cache['biological_object_annotations_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
        )

    except Exception as e:
        logger.error(f"Failed to generate biological object annotation plots: {str(e)}")
        return (
            create_fallback_plot("Biological Object Annotations by Ontology Over Time", str(e)),
            create_fallback_plot("Change in Biological Object Annotations by Ontology", str(e))
        )


def plot_aop_property_presence(label_file="property_labels.csv") -> tuple[str, str]:
    """Generate AOP property presence visualization with absolute and percentage views.

    Uses marker shapes to differentiate properties when colors repeat, ensuring
    visual distinction across all properties.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
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

        if not results_props or not results_total:
            logger.warning("AOP property presence query returned no results")
            return (
                create_fallback_plot("Property Presence in AOPs Over Time (Count)", "No data available"),
                create_fallback_plot("Property Presence in AOPs Over Time (Percentage)", "No data available")
            )

        df_props = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "property": r["p"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_props])

        df_total = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "total_aops": int(r["total"]["value"])
        } for r in results_total])

        if df_props.empty or df_total.empty:
            logger.warning("AOP property presence DataFrames are empty")
            return (
                create_fallback_plot("Property Presence in AOPs Over Time (Count)", "No data available"),
                create_fallback_plot("Property Presence in AOPs Over Time (Percentage)", "No data available")
            )

        # Merge
        df = df_props.merge(df_total, on="version", how="left")
        # Guard against division by zero in percentage calculation
        df["percentage"] = df.apply(
            lambda row: (row["count"] / row["total_aops"]) * 100 if row["total_aops"] > 0 else 0, axis=1
        )

        # Remove properties that are 100% in all versions
        props_to_keep = (
            df.groupby("property")["percentage"]
              .max()
              .loc[lambda x: x < 100]
              .index
        )
        df = df[df["property"].isin(props_to_keep)]

        # Ensure complete data: fill missing property-version combinations with 0
        if not df.empty:
            all_versions = sorted(df['version'].unique())
            all_props = sorted(df['property'].unique())
            complete_index = pd.MultiIndex.from_product(
                [all_versions, all_props],
                names=['version', 'property']
            )
            df_complete = df.set_index(['version', 'property']).reindex(complete_index, fill_value=0).reset_index()

            # Merge with totals to get proper totals for each version
            df_complete = df_complete.merge(df_total, on="version", how="left")

            # Find the total column name (total_aops, total_kes, total_kers, or total_stressors)
            total_col = [col for col in df_complete.columns if col.startswith('total_')][0]

            # Recalculate percentage with the correct total column, guarding division by zero
            df_complete["percentage"] = df_complete.apply(
                lambda row: (row["count"] / row[total_col]) * 100 if row[total_col] > 0 else 0, axis=1
            )

            # Preserve display_label if it exists, otherwise will be added later
            if 'display_label' in df.columns:
                label_map = df[['property', 'display_label']].drop_duplicates().set_index('property')['display_label'].to_dict()
                df_complete['display_label'] = df_complete['property'].map(label_map)

            df = df_complete

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

        # Cache data for CSV export
        _plot_data_cache['aop_property_presence_absolute'] = df.copy()
        _plot_data_cache['aop_property_presence_percentage'] = df.copy()

        # Define marker shapes for visual distinction
        marker_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x', 'star',
                          'pentagon', 'hexagon', 'octagon', 'triangle-down', 'triangle-left', 'triangle-right']

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

        # Apply marker shapes to traces
        for i, trace in enumerate(fig_abs.data):
            trace.update(marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8))

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

        # Apply marker shapes to traces
        for i, trace in enumerate(fig_delta.data):
            trace.update(marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8))

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

        # Cache figures for image export
        _plot_figure_cache['aop_property_presence_absolute'] = fig_abs
        _plot_figure_cache['aop_property_presence_percentage'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config=config),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config=config)
        )

    except Exception as e:
        logger.error(f"Failed to generate AOP property presence plots: {str(e)}")
        return (
            create_fallback_plot("Property Presence in AOPs Over Time (Count)", str(e)),
            create_fallback_plot("Property Presence in AOPs Over Time (Percentage)", str(e))
        )


def plot_ke_property_presence(label_file="property_labels.csv") -> tuple[str, str]:
    """Generate Key Event property presence visualization with absolute and percentage views.

    Uses marker shapes to differentiate properties when colors repeat, ensuring
    visual distinction across all properties.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        query_props = """
        SELECT ?graph ?p (COUNT(DISTINCT ?ke) AS ?count)
        WHERE {
          GRAPH ?graph {
            ?ke a aopo:KeyEvent ;
                 ?p ?o .
          }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ?p
        ORDER BY ?graph ?p
        """

        query_total = """
        SELECT ?graph (COUNT(DISTINCT ?ke) AS ?total)
        WHERE {
          GRAPH ?graph {
            ?ke a aopo:KeyEvent .
          }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY ?graph
        """

        results_props = run_sparql_query(query_props)
        results_total = run_sparql_query(query_total)

        if not results_props or not results_total:
            logger.warning("KE property presence query returned no results")
            return (
                create_fallback_plot("Property Presence in Key Events Over Time (Count)", "No data available"),
                create_fallback_plot("Property Presence in Key Events Over Time (Percentage)", "No data available")
            )

        df_props = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "property": r["p"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_props])

        df_total = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "total_kes": int(r["total"]["value"])
        } for r in results_total])

        if df_props.empty or df_total.empty:
            return (
                create_fallback_plot("Property Presence in Key Events Over Time (Count)", "No data available"),
                create_fallback_plot("Property Presence in Key Events Over Time (Percentage)", "No data available")
            )

        # Merge
        df = df_props.merge(df_total, on="version", how="left")
        df["percentage"] = df.apply(
            lambda row: (row["count"] / row["total_kes"]) * 100 if row["total_kes"] > 0 else 0, axis=1
        )

        # Remove properties that are 100% in all versions
        props_to_keep = (
            df.groupby("property")["percentage"]
              .max()
              .loc[lambda x: x < 100]
              .index
        )
        df = df[df["property"].isin(props_to_keep)]

        # Ensure complete data: fill missing property-version combinations with 0
        if not df.empty:
            all_versions = sorted(df['version'].unique())
            all_props = sorted(df['property'].unique())
            complete_index = pd.MultiIndex.from_product(
                [all_versions, all_props],
                names=['version', 'property']
            )
            df_complete = df.set_index(['version', 'property']).reindex(complete_index, fill_value=0).reset_index()
            df_complete = df_complete.merge(df_total, on="version", how="left")
            total_col = [col for col in df_complete.columns if col.startswith('total_')][0]
            df_complete["percentage"] = df_complete.apply(
                lambda row: (row["count"] / row[total_col]) * 100 if row[total_col] > 0 else 0, axis=1
            )
            if 'display_label' in df.columns:
                label_map = df[['property', 'display_label']].drop_duplicates().set_index('property')['display_label'].to_dict()
                df_complete['display_label'] = df_complete['property'].map(label_map)
            df = df_complete

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

        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")

        _plot_data_cache['ke_property_presence_absolute'] = df.copy()
        _plot_data_cache['ke_property_presence_percentage'] = df.copy()

        marker_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x', 'star',
                          'pentagon', 'hexagon', 'octagon', 'triangle-down', 'triangle-left', 'triangle-right']

        fig_abs = px.line(df, x="version", y="count", color="display_label", markers=True,
                          title="Property Presence in Key Events Over Time (Count)",
                          labels={"count": "Number of KEs", "display_label": "Property"},
                          color_discrete_sequence=BRAND_COLORS['palette'])
        for i, trace in enumerate(fig_abs.data):
            trace.update(marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8))
        fig_abs.update_layout(template="plotly_white", hovermode="x unified", autosize=True,
                              margin=dict(l=50, r=20, t=50, b=50))

        fig_delta = px.line(df, x="version", y="percentage", color="display_label", markers=True,
                            title="Property Presence in Key Events Over Time (Percentage)",
                            labels={"percentage": "Percentage (%)", "display_label": "Property"},
                            color_discrete_sequence=BRAND_COLORS['palette'])
        for i, trace in enumerate(fig_delta.data):
            trace.update(marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8))
        fig_delta.update_layout(template="plotly_white", hovermode="x unified", autosize=True,
                                margin=dict(l=50, r=20, t=50, b=50))

        config = {"responsive": True}
        _plot_figure_cache['ke_property_presence_absolute'] = fig_abs
        _plot_figure_cache['ke_property_presence_percentage'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config=config),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config=config)
        )

    except Exception as e:
        logger.error(f"Failed to generate KE property presence plots: {str(e)}")
        return (
            create_fallback_plot("Property Presence in Key Events Over Time (Count)", str(e)),
            create_fallback_plot("Property Presence in Key Events Over Time (Percentage)", str(e))
        )


def plot_ker_property_presence(label_file="property_labels.csv") -> tuple[str, str]:
    """Generate Key Event Relationship property presence visualization with absolute and percentage views.

    Uses marker shapes to differentiate properties when colors repeat, ensuring
    visual distinction across all properties.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        results_props = run_sparql_query("""
        SELECT ?graph ?p (COUNT(DISTINCT ?ker) AS ?count)
        WHERE {
          GRAPH ?graph { ?ker a aopo:KeyEventRelationship ; ?p ?o . }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ?p ORDER BY ?graph ?p
        """)
        results_total = run_sparql_query("""
        SELECT ?graph (COUNT(DISTINCT ?ker) AS ?total)
        WHERE {
          GRAPH ?graph { ?ker a aopo:KeyEventRelationship . }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ORDER BY ?graph
        """)

        if not results_props or not results_total:
            logger.warning("KER property presence query returned no results")
            return (
                create_fallback_plot("Property Presence in Key Event Relationships Over Time (Count)", "No data available"),
                create_fallback_plot("Property Presence in Key Event Relationships Over Time (Percentage)", "No data available")
            )

        df_props = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "property": r["p"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_props])
        df_total = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "total_kers": int(r["total"]["value"])
        } for r in results_total])

        if df_props.empty or df_total.empty:
            return (
                create_fallback_plot("Property Presence in Key Event Relationships Over Time (Count)", "No data available"),
                create_fallback_plot("Property Presence in Key Event Relationships Over Time (Percentage)", "No data available")
            )

        df = df_props.merge(df_total, on="version", how="left")
        df["percentage"] = df.apply(
            lambda row: (row["count"] / row["total_kers"]) * 100 if row["total_kers"] > 0 else 0, axis=1
        )

        props_to_keep = df.groupby("property")["percentage"].max().loc[lambda x: x < 100].index
        df = df[df["property"].isin(props_to_keep)]

        if not df.empty:
            all_versions = sorted(df['version'].unique())
            all_props = sorted(df['property'].unique())
            complete_index = pd.MultiIndex.from_product([all_versions, all_props], names=['version', 'property'])
            df_complete = df.set_index(['version', 'property']).reindex(complete_index, fill_value=0).reset_index()
            df_complete = df_complete.merge(df_total, on="version", how="left")
            total_col = [col for col in df_complete.columns if col.startswith('total_')][0]
            df_complete["percentage"] = df_complete.apply(
                lambda row: (row["count"] / row[total_col]) * 100 if row[total_col] > 0 else 0, axis=1
            )
            if 'display_label' in df.columns:
                label_map = df[['property', 'display_label']].drop_duplicates().set_index('property')['display_label'].to_dict()
                df_complete['display_label'] = df_complete['property'].map(label_map)
            df = df_complete

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

        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")

        _plot_data_cache['ker_property_presence_absolute'] = df.copy()
        _plot_data_cache['ker_property_presence_percentage'] = df.copy()

        marker_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x', 'star',
                          'pentagon', 'hexagon', 'octagon', 'triangle-down', 'triangle-left', 'triangle-right']

        fig_abs = px.line(df, x="version", y="count", color="display_label", markers=True,
                          title="Property Presence in Key Event Relationships Over Time (Count)",
                          labels={"count": "Number of KERs", "display_label": "Property"},
                          color_discrete_sequence=BRAND_COLORS['palette'])
        for i, trace in enumerate(fig_abs.data):
            trace.update(marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8))
        fig_abs.update_layout(template="plotly_white", hovermode="x unified", autosize=True,
                              margin=dict(l=50, r=20, t=50, b=50))

        fig_delta = px.line(df, x="version", y="percentage", color="display_label", markers=True,
                            title="Property Presence in Key Event Relationships Over Time (Percentage)",
                            labels={"percentage": "Percentage (%)", "display_label": "Property"},
                            color_discrete_sequence=BRAND_COLORS['palette'])
        for i, trace in enumerate(fig_delta.data):
            trace.update(marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8))
        fig_delta.update_layout(template="plotly_white", hovermode="x unified", autosize=True,
                                margin=dict(l=50, r=20, t=50, b=50))

        config = {"responsive": True}
        _plot_figure_cache['ker_property_presence_absolute'] = fig_abs
        _plot_figure_cache['ker_property_presence_percentage'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config=config),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config=config)
        )

    except Exception as e:
        logger.error(f"Failed to generate KER property presence plots: {str(e)}")
        return (
            create_fallback_plot("Property Presence in Key Event Relationships Over Time (Count)", str(e)),
            create_fallback_plot("Property Presence in Key Event Relationships Over Time (Percentage)", str(e))
        )


def plot_stressor_property_presence(label_file="property_labels.csv") -> tuple[str, str]:
    """Generate Stressor property presence visualization with absolute and percentage views.

    Uses marker shapes to differentiate properties when colors repeat, ensuring
    visual distinction across all properties.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        results_props = run_sparql_query("""
        SELECT ?graph ?p (COUNT(DISTINCT ?s) AS ?count)
        WHERE {
          GRAPH ?graph { ?s a nci:C54571 ; ?p ?o . }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ?p ORDER BY ?graph ?p
        """)
        results_total = run_sparql_query("""
        SELECT ?graph (COUNT(DISTINCT ?s) AS ?total)
        WHERE {
          GRAPH ?graph { ?s a nci:C54571 . }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ORDER BY ?graph
        """)

        if not results_props or not results_total:
            logger.warning("Stressor property presence query returned no results")
            return (
                create_fallback_plot("Property Presence in Stressors Over Time (Count)", "No data available"),
                create_fallback_plot("Property Presence in Stressors Over Time (Percentage)", "No data available")
            )

        df_props = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "property": r["p"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_props])
        df_total = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "total_stressors": int(r["total"]["value"])
        } for r in results_total])

        if df_props.empty or df_total.empty:
            return (
                create_fallback_plot("Property Presence in Stressors Over Time (Count)", "No data available"),
                create_fallback_plot("Property Presence in Stressors Over Time (Percentage)", "No data available")
            )

        df = df_props.merge(df_total, on="version", how="left")
        df["percentage"] = df.apply(
            lambda row: (row["count"] / row["total_stressors"]) * 100 if row["total_stressors"] > 0 else 0, axis=1
        )

        props_to_keep = df.groupby("property")["percentage"].max().loc[lambda x: x < 100].index
        df = df[df["property"].isin(props_to_keep)]

        if not df.empty:
            all_versions = sorted(df['version'].unique())
            all_props = sorted(df['property'].unique())
            complete_index = pd.MultiIndex.from_product([all_versions, all_props], names=['version', 'property'])
            df_complete = df.set_index(['version', 'property']).reindex(complete_index, fill_value=0).reset_index()
            df_complete = df_complete.merge(df_total, on="version", how="left")
            total_col = [col for col in df_complete.columns if col.startswith('total_')][0]
            df_complete["percentage"] = df_complete.apply(
                lambda row: (row["count"] / row[total_col]) * 100 if row[total_col] > 0 else 0, axis=1
            )
            if 'display_label' in df.columns:
                label_map = df[['property', 'display_label']].drop_duplicates().set_index('property')['display_label'].to_dict()
                df_complete['display_label'] = df_complete['property'].map(label_map)
            df = df_complete

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

        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")

        _plot_data_cache['stressor_property_presence_absolute'] = df.copy()
        _plot_data_cache['stressor_property_presence_percentage'] = df.copy()

        marker_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x', 'star',
                          'pentagon', 'hexagon', 'octagon', 'triangle-down', 'triangle-left', 'triangle-right']

        fig_abs = px.line(df, x="version", y="count", color="display_label", markers=True,
                          title="Property Presence in Stressors Over Time (Count)",
                          labels={"count": "Number of Stressors", "display_label": "Property"},
                          color_discrete_sequence=BRAND_COLORS['palette'])
        for i, trace in enumerate(fig_abs.data):
            trace.update(marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8))
        fig_abs.update_layout(template="plotly_white", hovermode="x unified", autosize=True,
                              margin=dict(l=50, r=20, t=50, b=50))

        fig_delta = px.line(df, x="version", y="percentage", color="display_label", markers=True,
                            title="Property Presence in Stressors Over Time (Percentage)",
                            labels={"percentage": "Percentage (%)", "display_label": "Property"},
                            color_discrete_sequence=BRAND_COLORS['palette'])
        for i, trace in enumerate(fig_delta.data):
            trace.update(marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=8))
        fig_delta.update_layout(template="plotly_white", hovermode="x unified", autosize=True,
                                margin=dict(l=50, r=20, t=50, b=50))

        config = {"responsive": True}
        _plot_figure_cache['stressor_property_presence_absolute'] = fig_abs
        _plot_figure_cache['stressor_property_presence_percentage'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config=config),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config=config)
        )

    except Exception as e:
        logger.error(f"Failed to generate stressor property presence plots: {str(e)}")
        return (
            create_fallback_plot("Property Presence in Stressors Over Time (Count)", str(e)),
            create_fallback_plot("Property Presence in Stressors Over Time (Percentage)", str(e))
        )


def plot_kes_by_kec_count() -> tuple[str, str]:
    """Generate KE distribution by KEC count visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
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
            logger.warning("No data for KE by KEC count plot")
            return (
                create_fallback_plot("KE Distribution by Component Count Over Time", "No data available"),
                create_fallback_plot("Change in KE Distribution by Component Count", "No data available")
            )

        df = pd.DataFrame(data)

        all_versions = sorted(df["version"].unique())
        all_groups = sorted(df["bioevent_count_group"].unique())

        idx = pd.MultiIndex.from_product([all_versions, all_groups],
                                         names=["version", "bioevent_count_group"])
        df_full = df.set_index(["version", "bioevent_count_group"]).reindex(idx, fill_value=0).reset_index()

        df_full["version_dt"] = pd.to_datetime(df_full["version"], errors="coerce")
        df_full = df_full.sort_values(["version_dt", "bioevent_count_group"]).drop(columns="version_dt")

        fig_abs = px.area(
            df_full, x="version", y="total_kes", color="bioevent_count_group",
            title="KE Distribution by Component Count Over Time",
            labels={"total_kes": "Number of KEs", "bioevent_count_group": "Number of Components"},
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_abs.update_layout(template="plotly_white", hovermode="x unified", autosize=True,
                              margin=dict(l=50, r=20, t=50, b=50),
                              xaxis=dict(tickmode='array', tickvals=all_versions, ticktext=all_versions, tickangle=-45))

        df_delta = df_full.copy()
        df_delta["total_kes_delta"] = df_delta.groupby("bioevent_count_group")["total_kes"].diff().fillna(0)

        fig_delta = px.area(
            df_delta, x="version", y="total_kes_delta", color="bioevent_count_group",
            title="Change in KE Distribution by Component Count",
            labels={"total_kes_delta": "Change in KEs", "bioevent_count_group": "Number of Components"},
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_delta.update_layout(template="plotly_white", hovermode="x unified", autosize=True,
                                margin=dict(l=50, r=20, t=50, b=50),
                                xaxis=dict(tickmode='array', tickvals=all_versions, ticktext=all_versions, tickangle=-45))

        _plot_data_cache['kes_by_kec_count_absolute'] = df_full.copy()
        _plot_data_cache['kes_by_kec_count_delta'] = df_delta.copy()
        _plot_figure_cache['kes_by_kec_count_absolute'] = fig_abs
        _plot_figure_cache['kes_by_kec_count_delta'] = fig_delta

        return (
            pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
            pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
        )

    except Exception as e:
        logger.error(f"Failed to generate KE by KEC count plots: {str(e)}")
        return (
            create_fallback_plot("KE Distribution by Component Count Over Time", str(e)),
            create_fallback_plot("Change in KE Distribution by Component Count", str(e))
        )


def plot_entity_completeness_trends(label_file="property_labels.csv") -> str:
    """Generate entity completeness trend visualization showing average completeness percentage over time.

    Calculates average completeness for AOPs, KEs, KERs, and Stressors across all versions.
    Excludes properties that are 100% present in any version (e.g., rdf:type, mandatory fields)
    to focus on optional/variable properties that reflect true data richness.

    Completeness is calculated per entity as (properties present / non-ubiquitous properties) × 100,
    then averaged across all entities of that type in each version.

    Args:
        label_file: Path to CSV file containing property labels and types

    Returns:
        HTML string containing the Plotly visualization
    """
    global _plot_data_cache, _plot_figure_cache

    # Load property labels
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential", "applies_to": "AOP|KE|KER|Stressor"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata", "applies_to": "AOP|KE|KER"}
    ]
    df_labels = safe_read_csv(label_file, default_properties)

    if df_labels.empty:
        return create_fallback_plot("Entity Completeness Trends", "No property labels available")

    # Entity type configurations
    entity_configs = [
        {
            "name": "AOP",
            "rdf_type": "aopo:AdverseOutcomePathway",
            "variable": "?aop",
            "label": "AOPs"
        },
        {
            "name": "KE",
            "rdf_type": "aopo:KeyEvent",
            "variable": "?ke",
            "label": "Key Events"
        },
        {
            "name": "KER",
            "rdf_type": "aopo:KeyEventRelationship",
            "variable": "?ker",
            "label": "Key Event Relationships"
        },
        {
            "name": "Stressor",
            "rdf_type": "aopo:Stressor",
            "variable": "?stressor",
            "label": "Stressors"
        }
    ]

    all_completeness_data = []

    for config in entity_configs:
        entity_type = config["name"]
        rdf_type = config["rdf_type"]
        variable = config["variable"]
        display_label = config["label"]

        # Filter properties applicable to this entity type
        applicable_props = df_labels[
            df_labels["applies_to"].fillna("").str.contains(entity_type, case=False, na=False)
        ]

        if applicable_props.empty:
            continue

        applicable_uris = set(applicable_props["uri"].tolist())

        if not applicable_uris:
            continue

        # Query to identify 100% present properties (to exclude them)
        # First get total entity counts per version
        uri_filter_presence = ">, <".join(applicable_uris)
        query_total = f"""
        SELECT ?graph (COUNT(DISTINCT {variable}) AS ?total)
        WHERE {{
          GRAPH ?graph {{
            {variable} a {rdf_type} .
          }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph
        ORDER BY ?graph
        """

        query_presence = f"""
        SELECT ?graph ?p (COUNT(DISTINCT {variable}) AS ?count)
        WHERE {{
          GRAPH ?graph {{
            {variable} a {rdf_type} ;
                 ?p ?o .
            FILTER(?p IN (<{uri_filter_presence}>))
          }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph ?p
        ORDER BY ?graph ?p
        """

        try:
            results_total = run_sparql_query(query_total)
            results_presence = run_sparql_query(query_presence)

            # Build totals map
            totals_map = {}
            for r in results_total:
                version = r["graph"]["value"].split("/")[-1]
                totals_map[version] = int(r["total"]["value"])

            # Calculate property presence percentages
            prop_presence = {}
            for r in results_presence:
                version = r["graph"]["value"].split("/")[-1]
                prop_uri = r["p"]["value"]
                count = int(r["count"]["value"])
                total = totals_map.get(version, 1)
                percentage = (count / total) * 100

                if prop_uri not in prop_presence:
                    prop_presence[prop_uri] = []
                prop_presence[prop_uri].append(percentage)

            # Filter out properties that reach 100% in any version
            props_to_exclude = {
                prop_uri for prop_uri, percentages in prop_presence.items()
                if max(percentages) >= 100
            }

            # Update applicable URIs to exclude 100% properties
            applicable_uris = applicable_uris - props_to_exclude
            total_properties = len(applicable_uris)

            if total_properties == 0:
                logger.info(f"All properties for {entity_type} are 100% present, skipping completeness calculation")
                continue

        except Exception as e:
            logger.warning(f"Error filtering 100% properties for {entity_type}: {str(e)}, proceeding with all properties")
            total_properties = len(applicable_uris)

        # Query to get property counts per entity per version (using filtered properties)
        uri_filter = ">, <".join(applicable_uris)
        query = f"""
        SELECT ?graph {variable} (COUNT(DISTINCT ?p) AS ?prop_count)
        WHERE {{
          GRAPH ?graph {{
            {variable} a {rdf_type} ;
                 ?p ?o .
            FILTER(?p IN (<{uri_filter}>))
          }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph {variable}
        ORDER BY ?graph {variable}
        """

        try:
            results = run_sparql_query(query)

            if not results:
                continue

            # Process results to calculate average completeness per version
            version_data = {}
            for result in results:
                version = result["graph"]["value"].split("/")[-1]
                prop_count = int(result["prop_count"]["value"])
                completeness = (prop_count / total_properties) * 100

                if version not in version_data:
                    version_data[version] = []
                version_data[version].append(completeness)

            # Calculate average completeness per version
            for version, completeness_list in version_data.items():
                avg_completeness = sum(completeness_list) / len(completeness_list)
                all_completeness_data.append({
                    "version": version,
                    "entity_type": display_label,
                    "avg_completeness": avg_completeness,
                    "entity_count": len(completeness_list)
                })

        except Exception as e:
            logger.error(f"Error querying completeness for {entity_type}: {str(e)}")
            continue

    if not all_completeness_data:
        return create_fallback_plot("Entity Completeness Trends", "No completeness data available")

    # Create DataFrame
    df = pd.DataFrame(all_completeness_data)

    # Sort by version
    df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
    df = df.sort_values("version_dt")

    # Cache data for CSV export
    _plot_data_cache['entity_completeness_trends'] = df.copy()

    # Create line plot
    fig = px.line(
        df,
        x="version",
        y="avg_completeness",
        color="entity_type",
        markers=True,
        title="Entity Completeness Over Time (Excluding 100% Present Properties)",
        labels={
            "avg_completeness": "Average Completeness (%)",
            "entity_type": "Entity Type",
            "version": "Version"
        },
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    # Apply marker shapes for visual distinction
    marker_symbols = ['circle', 'square', 'diamond', 'triangle-up']
    for i, trace in enumerate(fig.data):
        trace.update(
            marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=10),
            line=dict(width=3)
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=100, b=50),
        yaxis=dict(title="Average Completeness (%)", range=[0, 105]),
        xaxis=dict(title="Version", tickangle=-45),
        legend=dict(title="Entity Type", orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5)
    )

    config = {"responsive": True}

    # Cache figure for image export
    _plot_figure_cache['entity_completeness_trends'] = fig

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)


def _query_boxplot_version(version_info, aop_props, ke_props, ker_props, all_props):
    """Query boxplot data for a single version.

    Runs three scoped queries per version:
    1. Entity totals (AOP, KE, KER counts)
    2. Property presence per entity type
    3. AOP network structure (KEs/KERs per AOP)

    Returns:
        Dict with 'version', 'totals', 'presence', 'network' data, or None on error
    """
    graph_uri = version_info["graph_uri"]
    version_str = version_info["version"]

    all_props_filter = ">, <".join(all_props)

    # Query 1: Entity totals for this version
    query_totals = f"""
    SELECT ?entity_type (COUNT(DISTINCT ?entity) AS ?total)
    WHERE {{
      GRAPH <{graph_uri}> {{
        {{
          ?entity a aopo:AdverseOutcomePathway .
          BIND("AOP" AS ?entity_type)
        }}
        UNION
        {{
          ?entity a aopo:KeyEvent .
          BIND("KE" AS ?entity_type)
        }}
        UNION
        {{
          ?entity a aopo:KeyEventRelationship .
          BIND("KER" AS ?entity_type)
        }}
      }}
    }}
    GROUP BY ?entity_type
    """

    # Query 2: Property presence per entity type
    query_presence = f"""
    SELECT ?entity_type ?p (COUNT(DISTINCT ?entity) AS ?count)
    WHERE {{
      GRAPH <{graph_uri}> {{
        {{
          ?entity a aopo:AdverseOutcomePathway ;
                  ?p ?o .
          BIND("AOP" AS ?entity_type)
        }}
        UNION
        {{
          ?entity a aopo:KeyEvent ;
                  ?p ?o .
          BIND("KE" AS ?entity_type)
        }}
        UNION
        {{
          ?entity a aopo:KeyEventRelationship ;
                  ?p ?o .
          BIND("KER" AS ?entity_type)
        }}
        FILTER(?p IN (<{all_props_filter}>))
      }}
    }}
    GROUP BY ?entity_type ?p
    """

    # Query 3: AOP network structure
    query_network = f"""
    SELECT ?aop
           (GROUP_CONCAT(DISTINCT ?ke; separator="|") AS ?kes)
           (GROUP_CONCAT(DISTINCT ?ker; separator="|") AS ?kers)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {{ ?aop aopo:has_key_event ?ke }}
        OPTIONAL {{ ?aop aopo:has_key_event_relationship ?ker }}
      }}
    }}
    GROUP BY ?aop
    """

    try:
        results_totals = run_sparql_query_with_retry(query_totals, max_retries=2)
        results_presence = run_sparql_query_with_retry(query_presence, max_retries=2)
        results_network = run_sparql_query_with_retry(query_network, max_retries=2)

        return {
            "version": version_str,
            "totals": results_totals or [],
            "presence": results_presence or [],
            "network": results_network or []
        }
    except Exception as e:
        logger.warning(f"Boxplot query failed for version {version_str}: {e}")
        return None


def _query_boxplot_entity_props(version_info, aop_props, ke_props, ker_props,
                                aop_prop_count, ke_prop_count, ker_prop_count):
    """Query per-entity property counts for a single version for composite score calculation.

    Returns:
        Dict with 'version', 'entity_props' (entity completeness data), 'network', or None on error
    """
    graph_uri = version_info["graph_uri"]
    version_str = version_info["version"]

    # Build UNION parts for entity property counts
    union_parts = []

    if aop_prop_count > 0:
        aop_uri_filter = ">, <".join(aop_props)
        union_parts.append(f"""
          {{
            ?entity a aopo:AdverseOutcomePathway ;
                 ?p ?o .
            FILTER(?p IN (<{aop_uri_filter}>))
            BIND("AOP" AS ?entity_type)
          }}
        """)

    if ke_prop_count > 0:
        ke_uri_filter = ">, <".join(ke_props)
        union_parts.append(f"""
          {{
            ?entity a aopo:KeyEvent ;
                ?p ?o .
            FILTER(?p IN (<{ke_uri_filter}>))
            BIND("KE" AS ?entity_type)
          }}
        """)

    if ker_prop_count > 0:
        ker_uri_filter = ">, <".join(ker_props)
        union_parts.append(f"""
          {{
            ?entity a aopo:KeyEventRelationship ;
                 ?p ?o .
            FILTER(?p IN (<{ker_uri_filter}>))
            BIND("KER" AS ?entity_type)
          }}
        """)

    if not union_parts:
        return {"version": version_str, "entity_props": [], "network": []}

    union_clause = "\nUNION\n".join(union_parts)

    # Query per-entity property counts
    query_props = f"""
    SELECT ?entity (COUNT(DISTINCT ?p) AS ?prop_count) ?entity_type
    WHERE {{
      GRAPH <{graph_uri}> {{
        {union_clause}
      }}
    }}
    GROUP BY ?entity ?entity_type
    """

    # Query AOP network structure
    query_network = f"""
    SELECT ?aop
           (GROUP_CONCAT(DISTINCT ?ke; separator="|") AS ?kes)
           (GROUP_CONCAT(DISTINCT ?ker; separator="|") AS ?kers)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {{ ?aop aopo:has_key_event ?ke }}
        OPTIONAL {{ ?aop aopo:has_key_event_relationship ?ker }}
      }}
    }}
    GROUP BY ?aop
    """

    try:
        results_props = run_sparql_query_with_retry(query_props, max_retries=2)
        results_network = run_sparql_query_with_retry(query_network, max_retries=2)

        return {
            "version": version_str,
            "entity_props": results_props or [],
            "network": results_network or []
        }
    except Exception as e:
        logger.warning(f"Boxplot entity props query failed for version {version_str}: {e}")
        return None


def plot_aop_completeness_boxplot(label_file="property_labels.csv") -> str:
    """Generate composite AOP completeness distribution boxplot showing completeness variance across versions.

    Creates a boxplot showing the distribution of **composite completeness scores** for AOPs.
    The composite score includes:
    - The AOP entity's own property completeness
    - Average completeness of all KEs (Key Events) linked to the AOP
    - Average completeness of all KERs (Key Event Relationships) linked to the AOP

    All completeness scores are averaged with equal weight (flat average of all entities in the AOP network).
    Excludes properties that are 100% present in any version to focus on optional properties.

    Uses per-version parallel queries to avoid Virtuoso MaxResultRows truncation.

    Args:
        label_file: Path to CSV file containing property labels and types

    Returns:
        HTML string containing the Plotly boxplot visualization
    """
    global _plot_data_cache, _plot_figure_cache

    # Load property labels from CSV
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential", "applies_to": "AOP|KE|KER|Stressor"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata", "applies_to": "AOP|KE|KER"}
    ]
    df_labels = safe_read_csv(label_file, default_properties)

    if df_labels.empty:
        return create_fallback_plot("Composite AOP Completeness Distribution", "No property labels available")

    # Load properties for each entity type from CSV
    aop_properties_df = df_labels[df_labels['applies_to'].str.contains('AOP', na=False)]
    ke_properties_df = df_labels[df_labels['applies_to'].str.contains('KE', na=False)]
    ker_properties_df = df_labels[df_labels['applies_to'].str.contains('KER', na=False)]

    # Get initial property sets
    aop_props_all = set(aop_properties_df['uri'].tolist())
    ke_props_all = set(ke_properties_df['uri'].tolist())
    ker_props_all = set(ker_properties_df['uri'].tolist())

    # Combine all properties for querying
    all_props = aop_props_all | ke_props_all | ker_props_all

    if not all_props:
        return create_fallback_plot("Composite AOP Completeness Distribution", "No properties found")

    # Get all versions for per-version parallel queries
    versions = get_all_versions()
    if not versions:
        return create_fallback_plot("Composite AOP Completeness Distribution", "No versions available")

    logger.info(f"Boxplot: querying {len(versions)} versions in parallel for property filtering")

    # Step 1: Query presence and totals per version to filter 100% properties
    try:
        version_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_query_boxplot_version, v, aop_props_all, ke_props_all, ker_props_all, all_props): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        version_data.append(result)
                except Exception as e:
                    logger.warning(f"Boxplot presence query timed out for version {version_str}: {e}")

        # Build totals map from per-version results: {version: {entity_type: total}}
        totals_map = {}
        for vd in version_data:
            version = vd["version"]
            totals_map[version] = {}
            for r in vd["totals"]:
                entity_type = r["entity_type"]["value"]
                total = int(r["total"]["value"])
                totals_map[version][entity_type] = total

        # Calculate property presence percentages by entity type
        # {entity_type: {prop_uri: [percentages]}}
        prop_presence = {"AOP": {}, "KE": {}, "KER": {}}

        for vd in version_data:
            version = vd["version"]
            for r in vd["presence"]:
                entity_type = r["entity_type"]["value"]
                prop_uri = r["p"]["value"]
                count = int(r["count"]["value"])
                total = totals_map.get(version, {}).get(entity_type, 1)
                percentage = (count / total) * 100

                if prop_uri not in prop_presence[entity_type]:
                    prop_presence[entity_type][prop_uri] = []
                prop_presence[entity_type][prop_uri].append(percentage)

        # Filter out properties that reach 100% in any version for each entity type
        aop_props_to_exclude = {
            prop_uri for prop_uri, percentages in prop_presence["AOP"].items()
            if max(percentages) >= 100
        }
        ke_props_to_exclude = {
            prop_uri for prop_uri, percentages in prop_presence["KE"].items()
            if max(percentages) >= 100
        }
        ker_props_to_exclude = {
            prop_uri for prop_uri, percentages in prop_presence["KER"].items()
            if max(percentages) >= 100
        }

        # Get filtered property sets
        aop_props = aop_props_all - aop_props_to_exclude
        ke_props = ke_props_all - ke_props_to_exclude
        ker_props = ker_props_all - ker_props_to_exclude

        aop_prop_count = len(aop_props)
        ke_prop_count = len(ke_props)
        ker_prop_count = len(ker_props)

    except Exception as e:
        logger.warning(f"Error filtering 100% properties: {str(e)}")
        # Fall back to using all properties
        aop_props = aop_props_all
        ke_props = ke_props_all
        ker_props = ker_props_all
        aop_prop_count = len(aop_props)
        ke_prop_count = len(ke_props)
        ker_prop_count = len(ker_props)

    if aop_prop_count == 0 and ke_prop_count == 0 and ker_prop_count == 0:
        return create_fallback_plot("Composite AOP Completeness Distribution", "All properties are 100% present")

    logger.info(f"Boxplot: querying {len(versions)} versions for per-entity property counts (AOP:{aop_prop_count}, KE:{ke_prop_count}, KER:{ker_prop_count} props)")

    # Step 2: Query per-entity property counts and network structure per version
    try:
        version_entity_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    _query_boxplot_entity_props, v,
                    aop_props, ke_props, ker_props,
                    aop_prop_count, ke_prop_count, ker_prop_count
                ): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        version_entity_data.append(result)
                except Exception as e:
                    logger.warning(f"Boxplot entity query timed out for version {version_str}: {e}")

        if not version_entity_data:
            return create_fallback_plot("Composite AOP Completeness Distribution", "No entity data available")

        # Build network map and completeness maps from per-version results
        network_map = {}
        aop_completeness = {}
        ke_completeness = {}
        ker_completeness = {}

        for vd in version_entity_data:
            version = vd["version"]

            # Build network map for this version
            network_map[version] = {}
            for r in vd["network"]:
                aop_uri = r["aop"]["value"]
                network_map[version][aop_uri] = {"kes": set(), "kers": set()}

                if "kes" in r and r["kes"] and r["kes"]["value"]:
                    network_map[version][aop_uri]["kes"] = set(r["kes"]["value"].split("|"))
                if "kers" in r and r["kers"] and r["kers"]["value"]:
                    network_map[version][aop_uri]["kers"] = set(r["kers"]["value"].split("|"))

            # Build completeness maps for this version
            for r in vd["entity_props"]:
                entity_uri = r["entity"]["value"]
                entity_type = r["entity_type"]["value"]
                prop_count = int(r["prop_count"]["value"])

                if entity_type == "AOP":
                    completeness = (prop_count / aop_prop_count) * 100 if aop_prop_count > 0 else 0
                    if version not in aop_completeness:
                        aop_completeness[version] = {}
                    aop_completeness[version][entity_uri] = completeness

                elif entity_type == "KE":
                    completeness = (prop_count / ke_prop_count) * 100 if ke_prop_count > 0 else 0
                    if version not in ke_completeness:
                        ke_completeness[version] = {}
                    ke_completeness[version][entity_uri] = completeness

                elif entity_type == "KER":
                    completeness = (prop_count / ker_prop_count) * 100 if ker_prop_count > 0 else 0
                    if version not in ker_completeness:
                        ker_completeness[version] = {}
                    ker_completeness[version][entity_uri] = completeness

        # Step 3: Calculate composite AOP completeness scores
        completeness_data = []

        for version, aops in network_map.items():
            for aop_uri, network in aops.items():
                entity_scores = []

                # Add AOP entity completeness
                if version in aop_completeness and aop_uri in aop_completeness[version]:
                    entity_scores.append(aop_completeness[version][aop_uri])
                elif aop_prop_count > 0:
                    entity_scores.append(0)  # AOP exists but has no properties

                # Add KE completeness scores
                for ke_uri in network["kes"]:
                    if version in ke_completeness and ke_uri in ke_completeness[version]:
                        entity_scores.append(ke_completeness[version][ke_uri])
                    elif ke_prop_count > 0:
                        entity_scores.append(0)  # KE exists but has no properties

                # Add KER completeness scores
                for ker_uri in network["kers"]:
                    if version in ker_completeness and ker_uri in ker_completeness[version]:
                        entity_scores.append(ker_completeness[version][ker_uri])
                    elif ker_prop_count > 0:
                        entity_scores.append(0)  # KER exists but has no properties

                # Calculate composite score as flat average
                if entity_scores:
                    composite_score = sum(entity_scores) / len(entity_scores)
                    completeness_data.append({
                        "version": version,
                        "aop_uri": aop_uri,
                        "completeness": composite_score,
                        "entity_count": len(entity_scores)
                    })

        if not completeness_data:
            return create_fallback_plot("Composite AOP Completeness Distribution", "No completeness data available")

        # Create DataFrame
        df = pd.DataFrame(completeness_data)

        # Sort by version
        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")

        # Cache data for CSV export
        _plot_data_cache['aop_completeness_boxplot'] = df.copy()

        # Create boxplot
        fig = px.box(
            df,
            x="version",
            y="completeness",
            title="Composite AOP Completeness Distribution Over Time",
            labels={
                "completeness": "Composite Completeness (%)",
                "version": "Version"
            }
        )

        fig.update_traces(
            marker=dict(opacity=0.6, color=BRAND_COLORS["primary"]),
            line=dict(width=2)
        )

        fig.update_layout(
            template="plotly_white",
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=100),
            yaxis=dict(title="Composite Completeness (%)", range=[0, 105]),
            xaxis=dict(title="Version", tickangle=-45),
            showlegend=False
        )

        config = {"responsive": True}

        # Cache figure for image export
        _plot_figure_cache['aop_completeness_boxplot'] = fig

        return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)

    except Exception as e:
        logger.error(f"Error creating composite AOP completeness boxplot: {str(e)}")
        return create_fallback_plot("Composite AOP Completeness Distribution", f"Error: {str(e)}")


def plot_oecd_completeness_trend(label_file="property_labels.csv") -> str:
    """Generate OECD completeness trend visualization showing mean completeness per OECD status over time.

    Replaces the removed plot_aop_completeness_boxplot_by_status() which hit Virtuoso limits.
    This version queries aggregated means per status per version (~5 rows per version),
    producing ~75 total data points instead of 240K raw rows.

    Strategy:
        - For each version, query per-AOP property counts and OECD status
        - Compute mean completeness per OECD status in Python
        - Plot as a line chart with one line per status

    Args:
        label_file: Path to CSV file containing property labels and types

    Returns:
        HTML string containing the Plotly line chart visualization
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        # Load property labels
        default_properties = [
            {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential", "applies_to": "AOP|KE|KER|Stressor"},
            {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata", "applies_to": "AOP|KE|KER"}
        ]
        df_labels = safe_read_csv(label_file, default_properties)

        if df_labels.empty:
            return create_fallback_plot("OECD Completeness Trend", "No property labels available")

        # Filter properties applicable to AOPs
        aop_props_df = df_labels[
            df_labels["applies_to"].fillna("").str.contains("AOP", case=False, na=False)
        ]
        if aop_props_df.empty:
            return create_fallback_plot("OECD Completeness Trend", "No AOP properties configured")

        aop_property_uris = set(aop_props_df["uri"].tolist())
        total_properties = len(aop_property_uris)

        if total_properties == 0:
            return create_fallback_plot("OECD Completeness Trend", "No AOP properties configured")

        # Build property filter for SPARQL
        prop_filter_values = ", ".join([f"<{uri}>" for uri in aop_property_uris])

        # Get all versions
        versions = get_all_versions()
        if not versions:
            return create_fallback_plot("OECD Completeness Trend", "No versions available")

        logger.info(f"OECD completeness trend: querying {len(versions)} versions with {total_properties} AOP properties")

        # Query function for a single version
        def query_version(version_info):
            """Query per-AOP property count and OECD status for a single version."""
            graph_uri = version_info["graph_uri"]
            version_str = version_info["version"]

            query = f"""
            SELECT ?aop (STR(?status_obj) AS ?status) (COUNT(DISTINCT ?p) AS ?prop_count)
            WHERE {{
                GRAPH <{graph_uri}> {{
                    ?aop a aopo:AdverseOutcomePathway .
                    OPTIONAL {{
                        ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
                    }}
                    ?aop ?p ?o .
                    FILTER(?p IN ({prop_filter_values}))
                }}
            }}
            GROUP BY ?aop ?status_obj
            """

            try:
                results = run_sparql_query_with_retry(query, max_retries=2)
                if not results:
                    return []

                # Collect per-AOP completeness with status
                version_data = []
                for r in results:
                    status = r.get("status", {}).get("value", "No Status") if "status" in r else "No Status"
                    prop_count = int(r["prop_count"]["value"])
                    completeness = (prop_count / total_properties) * 100

                    version_data.append({
                        "version": version_str,
                        "status": status,
                        "completeness": completeness
                    })

                return version_data
            except Exception as e:
                logger.warning(f"OECD completeness query failed for version {version_str}: {e}")
                return []

        # Query versions in parallel (4 workers to avoid overloading endpoint)
        all_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_version = {
                executor.submit(query_version, v): v["version"]
                for v in versions
            }
            for future in as_completed(future_to_version):
                version_str = future_to_version[future]
                try:
                    result = future.result(timeout=60)
                    all_data.extend(result)
                except Exception as e:
                    logger.warning(f"OECD completeness query timed out for version {version_str}: {e}")

        if not all_data:
            return create_fallback_plot("OECD Completeness Trend", "No completeness data available")

        # Create DataFrame and compute mean completeness per status per version
        df_raw = pd.DataFrame(all_data)

        # Aggregate: mean completeness per (version, status)
        df = df_raw.groupby(["version", "status"], as_index=False).agg(
            mean_completeness=("completeness", "mean"),
            aop_count=("completeness", "count")
        )

        # Sort by version date
        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")

        # Cache data for CSV export
        _plot_data_cache['oecd_completeness_trend'] = df.copy()

        logger.info(f"OECD completeness trend: {len(df)} data points across {df['version'].nunique()} versions, {df['status'].nunique()} statuses")

        # Use Plotly qualitative colors for OECD statuses
        unique_statuses = sorted(df['status'].unique())
        plotly_colors = px.colors.qualitative.Plotly
        status_color_map = {
            status: plotly_colors[i % len(plotly_colors)]
            for i, status in enumerate(unique_statuses)
        }

        # Create line chart
        fig = px.line(
            df,
            x="version",
            y="mean_completeness",
            color="status",
            markers=True,
            title="Mean AOP Completeness by OECD Status Over Time",
            labels={
                "mean_completeness": "Mean Completeness (%)",
                "status": "OECD Status",
                "version": "Version"
            },
            color_discrete_map=status_color_map
        )

        # Apply marker shapes for visual distinction
        marker_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross',
                          'star', 'hexagon', 'pentagon', 'triangle-down', 'x']
        for i, trace in enumerate(fig.data):
            trace.update(
                marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=9),
                line=dict(width=2.5)
            )

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            autosize=True,
            margin=dict(l=50, r=150, t=50, b=50),
            yaxis=dict(title="Mean Completeness (%)", range=[0, 105]),
            xaxis=dict(title="Version", tickangle=-45),
            legend=dict(
                title="OECD Status",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        plotly_config = {"responsive": True}

        # Cache figure for image export
        _plot_figure_cache['oecd_completeness_trend'] = fig

        return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=plotly_config)

    except Exception as e:
        logger.error(f"Error creating OECD completeness trend: {str(e)}")
        return create_fallback_plot("OECD Completeness Trend", f"Error: {str(e)}")


def plot_aop_completeness_boxplot_by_status():
    """
    Plot composite AOP completeness distribution grouped by OECD status across all versions.
    
    Similar to plot_aop_completeness_boxplot() but adds color-grouped visualization
    by OECD approval status (Approved, Under Review, Under Development, No Status).
    
    Uses fallback strategy: queries status from latest version and applies across
    all historical versions to avoid timeout issues.
    
    Returns:
        str: HTML string of the Plotly boxplot visualization
    """
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio

    # Load properties from CSV for each entity type
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential"},
        {"uri": "http://purl.org/dc/terms/abstract", "label": "Abstract", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata"},
    ]
    properties_df = safe_read_csv("property_labels.csv", default_properties)

    # Filter properties for each entity type
    aop_properties_df = properties_df[properties_df['applies_to'].str.contains('AOP', na=False)]
    ke_properties_df = properties_df[properties_df['applies_to'].str.contains('KE', na=False)]
    ker_properties_df = properties_df[properties_df['applies_to'].str.contains('KER', na=False)]

    aop_props = set([p["uri"] for _, p in aop_properties_df.iterrows()])
    ke_props = set([p["uri"] for _, p in ke_properties_df.iterrows()])
    ker_props = set([p["uri"] for _, p in ker_properties_df.iterrows()])

    aop_prop_count = len(aop_props)
    ke_prop_count = len(ke_props)
    ker_prop_count = len(ker_props)

    if aop_prop_count == 0 and ke_prop_count == 0 and ker_prop_count == 0:
        return create_fallback_plot("Composite AOP Completeness by OECD Status", "No properties configured")
    
    # Phase 1: Optimized network structure query - combine network + status in single query
    # Get individual AOP-KE and AOP-KER relationships with status (but no entity IN filters later)
    query_network = """
    SELECT ?graph ?aop (STR(?status_obj) AS ?status) ?ke ?ker
    WHERE {
      GRAPH ?graph {
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {
            ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj
        }
        OPTIONAL { ?aop aopo:has_key_event ?ke }
        OPTIONAL { ?aop aopo:has_key_event_relationship ?ker }
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    ORDER BY ?graph ?aop
    """

    try:
        results_network = run_sparql_query(query_network)

        if not results_network:
            return create_fallback_plot("Composite AOP Completeness by OECD Status", "No AOP network data available")

        # Build network map from individual rows
        network_map = {}  # {version: {aop_uri: {"kes": set(), "kers": set(), "status": str}}}
        all_aops = set()
        all_kes = set()
        all_kers = set()

        for r in results_network:
            version = r["graph"]["value"].split("/")[-1]
            aop_uri = r["aop"]["value"]
            status = r.get("status", {}).get("value", "No Status") if "status" in r else "No Status"

            # Track all unique entities
            all_aops.add(aop_uri)

            if version not in network_map:
                network_map[version] = {}
            if aop_uri not in network_map[version]:
                network_map[version][aop_uri] = {"kes": set(), "kers": set(), "status": status}

            # Add KE if present
            if "ke" in r and r["ke"]:
                ke_uri = r["ke"]["value"]
                network_map[version][aop_uri]["kes"].add(ke_uri)
                all_kes.add(ke_uri)

            # Add KER if present
            if "ker" in r and r["ker"]:
                ker_uri = r["ker"]["value"]
                network_map[version][aop_uri]["kers"].add(ker_uri)
                all_kers.add(ker_uri)

        logger.info(f"Phase 1 complete: Found {len(all_aops)} unique AOPs, {len(all_kes)} unique KEs, {len(all_kers)} unique KERs across all versions")

        # Phase 2: Query property counts with optimized filtering
        # Use property URI filtering but avoid large entity IN filters

        # Build property filter strings
        if aop_prop_count > 0:
            aop_prop_filter_values = ", ".join([f"<{uri}>" for uri in aop_props])
            aop_prop_filter = f"FILTER(?p IN ({aop_prop_filter_values}))"
        else:
            aop_prop_filter = ""

        if ke_prop_count > 0:
            ke_prop_filter_values = ", ".join([f"<{uri}>" for uri in ke_props])
            ke_prop_filter = f"FILTER(?p IN ({ke_prop_filter_values}))"
        else:
            ke_prop_filter = ""

        if ker_prop_count > 0:
            ker_prop_filter_values = ", ".join([f"<{uri}>" for uri in ker_props])
            ker_prop_filter = f"FILTER(?p IN ({ker_prop_filter_values}))"
        else:
            ker_prop_filter = ""

        # Query AOP property counts
        if aop_prop_count > 0:
            query_aop = f"""
            SELECT ?graph ?aop (COUNT(DISTINCT ?p) AS ?prop_count)
            WHERE {{
              GRAPH ?graph {{
                ?aop a aopo:AdverseOutcomePathway ;
                     ?p ?o .
                {aop_prop_filter}
              }}
              FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }}
            GROUP BY ?graph ?aop
            ORDER BY ?graph ?aop
            """
            results_aop = run_sparql_query(query_aop)
        else:
            results_aop = []

        # Query KE property counts
        if ke_prop_count > 0:
            query_ke = f"""
            SELECT ?graph ?ke (COUNT(DISTINCT ?p) AS ?prop_count)
            WHERE {{
              GRAPH ?graph {{
                ?ke a aopo:KeyEvent ;
                    ?p ?o .
                {ke_prop_filter}
              }}
              FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }}
            GROUP BY ?graph ?ke
            ORDER BY ?graph ?ke
            """
            results_ke = run_sparql_query(query_ke)
        else:
            results_ke = []

        # Query KER property counts
        if ker_prop_count > 0:
            query_ker = f"""
            SELECT ?graph ?ker (COUNT(DISTINCT ?p) AS ?prop_count)
            WHERE {{
              GRAPH ?graph {{
                ?ker a aopo:KeyEventRelationship ;
                     ?p ?o .
                {ker_prop_filter}
              }}
              FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }}
            GROUP BY ?graph ?ker
            ORDER BY ?graph ?ker
            """
            results_ker = run_sparql_query(query_ker)
        else:
            results_ker = []
        
        # Step 3: Build completeness maps
        
        aop_completeness = {}
        for r in results_aop:
            version = r["graph"]["value"].split("/")[-1]
            entity_uri = r["aop"]["value"]
            prop_count = int(r["prop_count"]["value"])
            completeness = (prop_count / aop_prop_count) * 100 if aop_prop_count > 0 else 0
            
            if version not in aop_completeness:
                aop_completeness[version] = {}
            aop_completeness[version][entity_uri] = completeness
        
        ke_completeness = {}
        for r in results_ke:
            version = r["graph"]["value"].split("/")[-1]
            entity_uri = r["ke"]["value"]
            prop_count = int(r["prop_count"]["value"])
            completeness = (prop_count / ke_prop_count) * 100 if ke_prop_count > 0 else 0
            
            if version not in ke_completeness:
                ke_completeness[version] = {}
            ke_completeness[version][entity_uri] = completeness
        
        ker_completeness = {}
        for r in results_ker:
            version = r["graph"]["value"].split("/")[-1]
            entity_uri = r["ker"]["value"]
            prop_count = int(r["prop_count"]["value"])
            completeness = (prop_count / ker_prop_count) * 100 if ker_prop_count > 0 else 0
            
            if version not in ker_completeness:
                ker_completeness[version] = {}
            ker_completeness[version][entity_uri] = completeness
        
        # Step 4: Calculate composite scores WITH status
        completeness_data = []
        
        for version, aops in network_map.items():
            for aop_uri, network in aops.items():
                entity_scores = []
                
                # Add AOP entity completeness
                if version in aop_completeness and aop_uri in aop_completeness[version]:
                    entity_scores.append(aop_completeness[version][aop_uri])
                elif aop_prop_count > 0:
                    entity_scores.append(0)
                
                # Add KE completeness scores
                for ke_uri in network["kes"]:
                    if version in ke_completeness and ke_uri in ke_completeness[version]:
                        entity_scores.append(ke_completeness[version][ke_uri])
                    elif ke_prop_count > 0:
                        entity_scores.append(0)
                
                # Add KER completeness scores
                for ker_uri in network["kers"]:
                    if version in ker_completeness and ker_uri in ker_completeness[version]:
                        entity_scores.append(ker_completeness[version][ker_uri])
                    elif ker_prop_count > 0:
                        entity_scores.append(0)
                
                # Calculate composite score
                if entity_scores:
                    composite_score = sum(entity_scores) / len(entity_scores)
                    # Get status from network_map (already captured in Phase 1)
                    status = network["status"]

                    completeness_data.append({
                        "version": version,
                        "aop_uri": aop_uri,
                        "status": status,
                        "completeness": composite_score,
                        "entity_count": len(entity_scores)
                    })
        
        if not completeness_data:
            return create_fallback_plot("Composite AOP Completeness by OECD Status", "No completeness data available")
        
        # Create DataFrame
        df = pd.DataFrame(completeness_data)
        
        # Sort by version
        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")
        
        # Cache data for CSV export
        _plot_data_cache['aop_completeness_boxplot_by_status'] = df.copy()

        # Dynamically extract all unique statuses and assign colors
        unique_statuses = sorted(df['status'].unique())

        # Use Plotly's qualitative color palette for consistent, distinguishable colors
        # Plotly Plotly palette provides 10 distinct colors
        plotly_colors = px.colors.qualitative.Plotly

        # Build dynamic color map
        status_color_map = {}
        for i, status in enumerate(unique_statuses):
            status_color_map[status] = plotly_colors[i % len(plotly_colors)]

        logger.info(f"Found {len(unique_statuses)} unique OECD statuses: {unique_statuses}")

        # Create boxplot with color grouping by status
        fig = px.box(
            df,
            x="version",
            y="completeness",
            color="status",
            title="Composite AOP Completeness Distribution by OECD Status",
            labels={
                "completeness": "Composite Completeness (%)",
                "version": "Version",
                "status": "OECD Status"
            },
            color_discrete_map=status_color_map
        )
        
        fig.update_traces(
            marker=dict(opacity=0.6),
            line=dict(width=2)
        )
        
        fig.update_layout(
            template="plotly_white",
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=100),
            yaxis=dict(title="Composite Completeness (%)", range=[0, 105]),
            xaxis=dict(title="Version", tickangle=-45),
            showlegend=True,
            legend=dict(
                title="OECD Status",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        config = {"responsive": True}
        
        # Cache figure for image export
        _plot_figure_cache['aop_completeness_boxplot_by_status'] = fig
        
        return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)
    
    except Exception as e:
        logger.error(f"Error creating status-separated AOP completeness boxplot: {str(e)}")
        return create_fallback_plot("Composite AOP Completeness by OECD Status", f"Error: {str(e)}")
