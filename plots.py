import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
from functools import reduce
import time
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Validate configuration on import
if not Config.validate_config():
    logger.error("Invalid configuration detected, using defaults")

SPARQL_ENDPOINT = Config.SPARQL_ENDPOINT
MAX_RETRIES = Config.SPARQL_MAX_RETRIES
RETRY_DELAY = Config.SPARQL_RETRY_DELAY
TIMEOUT = Config.SPARQL_TIMEOUT

config = {
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",       # You can also allow 'svg', 'jpeg', 'webp'
        "filename": "plot_download",
        "height": 500,         # Increased resolution
        "width": 800,
        "scale": 4             # Multiplies resolution
    }
}


def check_sparql_endpoint_health() -> bool:
    """Check if SPARQL endpoint is accessible and responsive."""
    try:
        parsed_url = urlparse(SPARQL_ENDPOINT)
        if parsed_url.scheme not in ['http', 'https']:
            logger.error(f"Invalid SPARQL endpoint URL: {SPARQL_ENDPOINT}")
            return False
            
        # Simple health check query
        test_query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o } LIMIT 1"
        sparql = SPARQLWrapper(SPARQL_ENDPOINT)
        sparql.setTimeout(10)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(test_query)
        
        result = sparql.query().convert()
        logger.info(f"SPARQL endpoint {SPARQL_ENDPOINT} is healthy")
        return True
        
    except Exception as e:
        logger.error(f"SPARQL endpoint health check failed: {str(e)}")
        return False


def run_sparql_query_with_retry(query: str, max_retries: int = MAX_RETRIES) -> List[Dict[str, Any]]:
    """Execute SPARQL query with retry logic and error handling."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            sparql = SPARQLWrapper(SPARQL_ENDPOINT)
            sparql.setTimeout(TIMEOUT)
            sparql.setReturnFormat(JSON)
            sparql.setQuery(query)
            
            result = sparql.query().convert()
            execution_time = time.time() - start_time
            
            if execution_time > 10:
                logger.warning(f"Slow query execution: {execution_time:.2f}s")
            
            bindings = result.get("results", {}).get("bindings", [])
            logger.info(f"Query executed successfully, returned {len(bindings)} results")
            return bindings
            
        except SPARQLExceptions.QueryBadFormed as e:
            logger.error(f"Bad SPARQL query: {str(e)}")
            break  # Don't retry for syntax errors
            
        except SPARQLExceptions.EndPointNotFound as e:
            logger.error(f"SPARQL endpoint not found: {str(e)}")
            last_exception = e
            break  # Don't retry for endpoint errors
            
        except (SPARQLExceptions.SPARQLWrapperException, requests.exceptions.RequestException) as e:
            last_exception = e
            logger.warning(f"Query attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            
        except Exception as e:
            logger.error(f"Unexpected error in SPARQL query: {str(e)}")
            last_exception = e
            break
    
    logger.error(f"All {max_retries} query attempts failed. Last error: {str(last_exception)}")
    return []


def run_sparql_query(query: str) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    return run_sparql_query_with_retry(query)

def extract_counts(results: List[Dict[str, Any]], var_name: str = "count") -> pd.DataFrame:
    """Extract version and count data with error handling."""
    if not results:
        logger.warning(f"No results to extract for {var_name}")
        return pd.DataFrame(columns=["version", var_name])
    
    try:
        data = []
        for r in results:
            if "graph" not in r or var_name not in r:
                logger.warning(f"Missing required fields in result: {r}")
                continue
                
            try:
                version = r["graph"]["value"].split("/")[-1]
                count = int(r[var_name]["value"])
                data.append({"version": version, var_name: count})
            except (KeyError, ValueError, IndexError) as e:
                logger.warning(f"Error processing result {r}: {str(e)}")
                continue
        
        if not data:
            logger.warning(f"No valid data extracted for {var_name}")
            return pd.DataFrame(columns=["version", var_name])
            
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error in extract_counts: {str(e)}")
        return pd.DataFrame(columns=["version", var_name])


def create_fallback_plot(title: str, error_message: str) -> str:
    """Create a fallback plot when data is unavailable."""
    fig = px.scatter(x=[0], y=[0], title=f"{title} - Data Unavailable")
    fig.add_annotation(
        x=0, y=0,
        text=f"Unable to load data: {error_message}",
        showarrow=False,
        font=dict(size=16, color="red"),
        bgcolor="white",
        bordercolor="red",
        borderwidth=2
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def safe_plot_execution(plot_func, *args, **kwargs) -> Any:
    """Safely execute a plot function with error handling."""
    try:
        start_time = time.time()
        result = plot_func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(f"Plot function {plot_func.__name__} executed in {execution_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error in plot function {plot_func.__name__}: {str(e)}")
        
        # Return appropriate fallback based on expected return type
        if hasattr(plot_func, '__annotations__'):
            return_type = plot_func.__annotations__.get('return', str)
            if return_type == str:
                return create_fallback_plot(plot_func.__name__, str(e))
            elif 'tuple' in str(return_type):
                fallback = create_fallback_plot(plot_func.__name__, str(e))
                # Return tuple of fallbacks based on expected length
                if 'plot_main_graph' in plot_func.__name__:
                    return fallback, fallback, pd.DataFrame()
                else:
                    return fallback, fallback
        
        # Default fallback
        return create_fallback_plot(plot_func.__name__, str(e))

def plot_main_graph() -> tuple[str, str, pd.DataFrame]:
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
        title="AOP Entity Counts (Absolute)"
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
        title="AOP Entity Counts (Delta Between Versions)"
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

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs="cdn", config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        df_all
    )


def plot_avg_per_aop() -> tuple[str, str]:
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
                      title="Average KEs and KERs per AOP (Absolute)")
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
                        title="Average KEs and KERs per AOP (Delta)")
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
        title="Network Density of KE-KER Graph",
        markers=True,
        labels={"density": "Graph Density"}
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
                      title="Unique AOP Authors per Version")
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
                        title="Change in Unique AOP Authors per Version")
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
                        title="Unique AOPs Created per Year",
                        labels={"year_created": "Year", "count": "AOP Count"},
                        color_discrete_sequence=["#636EFA"])
    fig1.update_layout(template="plotly_white", height=400)
    html1 = pio.to_html(fig1, full_html=False, include_plotlyjs="cdn")

    # --- Plot 2: AOPs Modified ---
    fig2 = px.histogram(df_modified, x="year_modified",
                        title="Unique AOPs Last Modified per Year",
                        labels={"year_modified": "Year", "count": "AOP Count"},
                        color_discrete_sequence=["#EF553B"])
    fig2.update_layout(template="plotly_white", height=400)
    html2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)

    # --- Plot 3: Created vs. Modified Dates ---
    fig3 = px.scatter(df_lifetime, x="created", y="modified", hover_name="aop",
                      title="AOP Creation vs. Last Modification Dates",
                      labels={"created": "Created", "modified": "Modified"},
                      color_discrete_sequence=["#AB63FA"])
    fig3.update_layout(template="plotly_white", height=500)
    html3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False)

    return html1, html2, html3

def plot_ke_components() -> tuple[str, str]:
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
        title="Total Number of KE Component Annotations Over Time",
        markers=True
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
        title="Change in KE Component Annotations Between Versions",
        markers=True
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
        title="KE Component Annotations as % of Total KEs",
        markers=True
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
        title="Change in KE Component % Between Versions",
        markers=True
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
        title="Total Unique KE Component Annotations Over Time",
        markers=True
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
        title="Change in Unique KE Component Annotations Between Versions",
        markers=True
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
        title="KEs Annotated with Biological Processes by Ontology (Absolute)",
        labels={"count": "Annotated KEs", "ontology": "Ontology"}
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
        title="Change in Biological Process Annotations by Ontology (Delta)",
        labels={"count": "Change in Annotated KEs", "ontology": "Ontology"}
    )
    fig_delta.update_layout(template="plotly_white", hovermode="x unified", autosize=True)
    fig_delta.update_xaxes(tickmode='array', tickvals=df_ont["version"], ticktext=df_ont["version"], tickangle=-45)

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )

def plot_bio_objects() -> tuple[str, str]:
    query_objects = """
    SELECT ?graph ?ontology (COUNT(DISTINCT ?object) AS ?count)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        ?bioevent aopo:hasObject ?object .

        BIND(
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/GO_"), "GO",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/MP_"), "MP",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/NBO_"), "NBO",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/MI_"), "MI",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/VT_"), "VT",
          IF(STRSTARTS(STR(?object), "http://purl.org/commons/record/mesh/"), "MESH",
          IF(STRSTARTS(STR(?object), "http://purl.obolibrary.org/obo/HP_"), "HP", "OTHER"))))))) AS ?ontology)
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
        title="KEs Annotated with Biological Objects by Ontology (Absolute)",
        labels={"count": "Annotated KEs", "ontology": "Ontology"}
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
        title="Change in Biological Object Annotations by Ontology (Delta)",
        labels={"count": "Change in Annotated KEs", "ontology": "Ontology"}
    )
    fig_delta.update_layout(template="plotly_white", hovermode="x unified", autosize=True)
    fig_delta.update_xaxes(tickmode='array', tickvals=df_obj["version"], ticktext=df_obj["version"], tickangle=-45)

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )

def plot_aop_property_presence(label_file="property_labels.csv") -> tuple[str, str]:
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

    # Label mapping
    if os.path.exists(label_file):
        df_labels = pd.read_csv(label_file)
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
        title="Presence of Properties in AOPs Over Time (Distinct Count)",
        labels={"count": "Number of AOPs", "display_label": "Property"}
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
        title="Presence of Properties in AOPs Over Time (Percentage)",
        labels={"percentage": "Percentage (%)", "display_label": "Property"}
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


def plot_database_size() -> tuple[str, str]:
    query_triples = """
    SELECT ?graph (COUNT(*) AS ?triples)
    WHERE { 
      GRAPH ?graph { ?s ?p ?o } 
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY ?graph
    """

    results = run_sparql_query(query_triples)
    df_triples = pd.DataFrame([{
        "version": r["graph"]["value"].split("/")[-1],
        "triples": int(r["triples"]["value"])
    } for r in results])

    # Sort by date
    df_triples["version_dt"] = pd.to_datetime(df_triples["version"], errors="coerce")
    df_triples = df_triples.sort_values("version_dt").drop(columns="version_dt")

    # Absolute plot
    fig_abs = px.line(
        df_triples,
        x="version",
        y="triples",
        markers=True,
        title="Database Size Over Time",
        labels={"triples": "Total Triples", "version": "Version"}
    )
    fig_abs.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_abs.update_xaxes(
        tickmode='array',
        tickvals=df_triples["version"],
        ticktext=df_triples["version"],
        tickangle=-45
    )

    # Delta plot
    df_triples["triples_delta"] = df_triples["triples"].diff().fillna(0)
    fig_delta = px.line(
        df_triples,
        x="version",
        y="triples_delta",
        markers=True,
        title="Change in Database Size Between Versions",
        labels={"triples_delta": "Change in Triples", "version": "Version"}
    )
    fig_delta.update_layout(
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig_delta.update_xaxes(
        tickmode='array',
        tickvals=df_triples["version"],
        ticktext=df_triples["version"],
        tickangle=-45
    )

    return (
        pio.to_html(fig_abs, full_html=False, include_plotlyjs=False, config={"responsive": True}),
        pio.to_html(fig_delta, full_html=False, include_plotlyjs=False, config={"responsive": True})
    )


def plot_kes_by_kec_count() -> tuple[str, str]:
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
        title="Distribution of KEs by Number of KECs Over Time",
        labels={"total_kes": "Number of KEs", "bioevent_count_group": "Number of KECs"}
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
        title="Change in KEs by Number of KECs Over Time",
        labels={"total_kes_delta": "Change in KEs", "bioevent_count_group": "Number of KECs"}
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


def plot_latest_entity_counts() -> str:
    """Create a bar chart showing the latest version's entity counts."""
    sparql_queries = {
        "AOPs": """
            SELECT ?graph (COUNT(?aop) AS ?count)
            WHERE {
                GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
            ORDER BY DESC(?graph)
            LIMIT 1
        """,
        "KEs": """
            SELECT ?graph (COUNT(?ke) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?aop a aopo:AdverseOutcomePathway ;
                         aopo:has_key_event ?ke .
                }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
            ORDER BY DESC(?graph)
            LIMIT 1
        """,
        "KERs": """
            SELECT ?graph (COUNT(?ker) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?aop a aopo:AdverseOutcomePathway ;
                         aopo:has_key_event_relationship ?ker .
                }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
            ORDER BY DESC(?graph)
            LIMIT 1
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
            ORDER BY DESC(?graph)
            LIMIT 1
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
    
    fig = px.bar(
        df,
        x="Entity",
        y="Count",
        title=f"Latest AOP Entity Counts ({latest_version})",
        color="Entity",
        text="Count"
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


def plot_latest_ke_components() -> str:
    """Create a pie chart showing the latest version's KE component distribution."""
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
    ORDER BY DESC(?graph)
    LIMIT 1
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
    
    fig = px.pie(
        df,
        values="Count",
        names="Component",
        title=f"KE Component Distribution ({latest_version})"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})
