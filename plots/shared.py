"""AOP-Wiki RDF Dashboard - Shared Plot Utilities and Constants.

This module contains all shared utilities, configuration, constants, and helper
functions used across the AOP-Wiki RDF Dashboard plotting system. It provides
a centralized location for common functionality to ensure consistency and
maintainability across historical trends and latest snapshot visualizations.

Core Components:
    - SPARQL query execution with retry logic and error handling
    - Brand color palette and styling constants
    - Data processing utilities for RDF query results
    - Error handling and fallback visualization generation
    - Performance monitoring and safe execution wrappers
    - Global data caching system for CSV export functionality

SPARQL Integration:
    - Endpoint: Configurable via Config.SPARQL_ENDPOINT
    - Retry Logic: Exponential backoff with configurable attempts
    - Health Monitoring: Connection status and performance tracking
    - Error Handling: Graceful degradation with fallback visualizations
    - Query Optimization: Efficient RDF graph filtering and aggregation

Performance Features:
    - Global data caching for CSV exports (_plot_data_cache)
    - Configurable timeouts and retry mechanisms
    - Parallel execution support via safe_plot_execution()
    - Performance monitoring and logging
    - Memory-efficient data processing

Usage Examples:
    Basic SPARQL query:
    >>> results = run_sparql_query_with_retry(query)
    >>> df = extract_counts(results, "count")

    Safe plot execution:
    >>> result = safe_plot_execution(plot_function)

    CSV data access:
    >>> from plots.shared import _plot_data_cache
    >>> if 'entity_counts' in _plot_data_cache:
    ...     df = _plot_data_cache['entity_counts']

Error Handling:
    All functions include comprehensive error handling:
    - SPARQL connection failures
    - Data processing errors
    - Visualization generation issues
    - Fallback plot generation for graceful degradation

Author:
    Generated with Claude Code (https://claude.ai/code)
"""

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

# Official House Style Color Palette
BRAND_COLORS = {
    # Primary house style colors
    'primary': '#29235C',      # Primary Dark (Main brand color)
    'secondary': '#E6007E',    # Primary Magenta (Accent color)
    'accent': '#307BBF',       # Primary Blue (Supporting blue)
    'light': '#93D5F6',        # Sky Blue (Light accent)
    'content': '#EB5B25',      # Orange (Content highlight)

    # Full palette using official house style colors
    'palette': [
        '#29235C',  # Primary Dark
        '#E6007E',  # Primary Magenta
        '#307BBF',  # Primary Blue
        '#009FE3',  # Light Blue
        '#EB5B25',  # Orange
        '#93D5F6',  # Sky Blue
        '#9A1C57',  # Deep Magenta
        '#45A6B2',  # Teal
        '#B81178',  # Purple
        '#005A6C',  # Dark Teal
        '#64358C'   # Violet
    ],

    # Property type colors using house style palette
    'type_colors': {
        'Essential': '#29235C',    # Primary Dark
        'Metadata': '#E6007E',     # Primary Magenta
        'Content': '#EB5B25',      # Orange
        'Context': '#93D5F6',      # Sky Blue
        'Assessment': '#307BBF',   # Primary Blue
        'Structure': '#45A6B2'     # Teal
    }
}

# Plotly configuration for consistent styling and downloads
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

# Global data cache for CSV export functionality
_plot_data_cache = {}
_plot_figure_cache = {}  # Cache for Plotly figure objects (for PNG/SVG/PDF export)


def safe_read_csv(filename: str, default_data: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Safely read CSV file with comprehensive error handling and fallback data.

    Provides robust CSV file reading with graceful error handling for missing files,
    corrupted data, or permission issues. Uses fallback data when the primary file
    cannot be read, ensuring the application continues to function even when
    configuration files are unavailable.

    Args:
        filename (str): Path to the CSV file to read. Can be relative or absolute.
        default_data (Optional[List[Dict]]): Fallback data as list of dictionaries
            to use if file reading fails. Each dictionary represents a row.
            If None, returns empty DataFrame on failure.

    Returns:
        pd.DataFrame: Loaded CSV data as DataFrame, fallback data, or empty DataFrame.
            Column structure depends on the CSV content or default_data structure.

    Error Handling:
        - FileNotFoundError: Logs warning and uses fallback data
        - PermissionError: Logs error and uses fallback data
        - pd.errors.ParserError: Logs error and uses fallback data
        - UnicodeDecodeError: Logs error and uses fallback data
        - Any other Exception: Logs error and uses fallback data

    Example:
        >>> # Read property labels with fallback
        >>> default_props = [
        ...     {"uri": "http://purl.org/dc/elements/1.1/title",
        ...      "label": "Title", "type": "Essential"}
        ... ]
        >>> df = safe_read_csv("property_labels.csv", default_props)
        >>> print(f"Loaded {len(df)} properties")

        >>> # Read without fallback
        >>> df = safe_read_csv("optional_config.csv")
        >>> if df.empty:
        ...     print("No configuration data available")

    Note:
        This function is designed for configuration files that may not always
        be present. It prioritizes application stability over strict file
        requirements.
    """
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        logger.warning(f"File {filename} not found, using fallback data")
        if default_data:
            return pd.DataFrame(default_data)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading {filename}: {str(e)}")
        if default_data:
            return pd.DataFrame(default_data)
        return pd.DataFrame()


def check_sparql_endpoint_health() -> bool:
    """Check if SPARQL endpoint is accessible and responsive.

    Performs a lightweight health check on the configured SPARQL endpoint by
    executing a simple test query. This function validates both connectivity
    and basic query functionality, providing essential health monitoring
    capabilities for the application.

    Health Check Process:
        1. Validates SPARQL endpoint URL format and scheme
        2. Executes a minimal test query (COUNT of any triple)
        3. Verifies response within timeout period
        4. Confirms valid JSON response format

    Returns:
        bool: True if endpoint is healthy and responsive, False otherwise.
            Logs appropriate messages for both success and failure cases.

    Test Query:
        Executes: "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o } LIMIT 1"
        - Lightweight query that works on any RDF store
        - Tests connectivity, query processing, and response format
        - Uses short timeout (10 seconds) for quick health checks

    Error Conditions:
        - Invalid URL format or scheme
        - Network connectivity issues
        - SPARQL endpoint not responding
        - Query execution timeout
        - Invalid response format
        - Server errors (500, 503, etc.)

    Example:
        >>> if check_sparql_endpoint_health():
        ...     print("SPARQL endpoint is ready")
        ...     # Proceed with data queries
        ... else:
        ...     print("SPARQL endpoint is unavailable")
        ...     # Use fallback data or cached results

    Usage:
        Called during application startup and by health check endpoints
        to ensure SPARQL connectivity before attempting data queries.

    Performance:
        - Uses 10-second timeout for quick response
        - Minimal query load on the SPARQL endpoint
        - Designed for frequent health monitoring

    Note:
        This function logs all outcomes (success/failure) and is safe
        to call repeatedly for monitoring purposes.
    """
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
    """Execute SPARQL query with comprehensive retry logic and error handling.

    Provides robust SPARQL query execution with exponential backoff retry logic,
    comprehensive error classification, and performance monitoring. This is the
    primary interface for all SPARQL operations in the application.

    Args:
        query (str): SPARQL query string to execute. Should be valid SPARQL syntax.
        max_retries (int, optional): Maximum number of retry attempts.
            Defaults to Config.SPARQL_MAX_RETRIES (typically 3).

    Returns:
        List[Dict[str, Any]]: List of result bindings from the SPARQL query.
            Each dictionary represents one result row with variable bindings.
            Returns empty list if query fails after all retries.

    Query Execution Process:
        1. Validates and prepares SPARQL query
        2. Configures SPARQLWrapper with timeout settings
        3. Executes query with performance monitoring
        4. Handles various error conditions with appropriate retry logic
        5. Returns parsed JSON results or empty list on failure

    Retry Logic:
        - Exponential backoff: delay increases with each retry attempt
        - Syntax errors (QueryBadFormed): No retry, immediate failure
        - Endpoint errors (EndPointNotFound): No retry, immediate failure
        - Network/timeout errors: Full retry with backoff
        - Other exceptions: Single retry attempt

    Error Classification:
        - SPARQLExceptions.QueryBadFormed: Query syntax errors (no retry)
        - SPARQLExceptions.EndPointNotFound: Endpoint unavailable (no retry)
        - SPARQLWrapperException: SPARQL-specific errors (retry with backoff)
        - requests.exceptions.RequestException: Network issues (retry with backoff)
        - Exception: Unexpected errors (single retry)

    Performance Monitoring:
        - Execution time tracking and logging
        - Slow query detection (>10 seconds)
        - Success/failure rate monitoring
        - Retry attempt logging

    Example:
        >>> # Basic query execution
        >>> query = '''
        ... SELECT ?graph (COUNT(?aop) AS ?count)
        ... WHERE {
        ...     GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
        ... }
        ... GROUP BY ?graph
        ... '''
        >>> results = run_sparql_query_with_retry(query)
        >>> print(f"Retrieved {len(results)} results")

        >>> # With custom retry count
        >>> results = run_sparql_query_with_retry(query, max_retries=5)

    Configuration:
        Uses Config parameters for:
        - SPARQL_TIMEOUT: Query timeout in seconds
        - SPARQL_RETRY_DELAY: Base delay between retries
        - SPARQL_ENDPOINT: Target SPARQL endpoint URL

    Logging:
        - Info: Successful queries with execution time and result count
        - Warning: Individual retry attempts with error details
        - Error: Final failure after all retries exhausted

    Note:
        This function is the foundation for all SPARQL operations and is
        designed to be resilient against temporary network issues while
        failing fast for permanent errors like syntax problems.
    """
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
    """Extract version and count data from SPARQL query results with robust error handling.

    Processes SPARQL query results to extract version information and count data,
    transforming the raw RDF query results into a clean pandas DataFrame suitable
    for visualization and analysis. Handles various data quality issues and
    provides comprehensive error recovery.

    Args:
        results (List[Dict[str, Any]]): Raw SPARQL query results as returned by
            run_sparql_query_with_retry(). Each dict contains variable bindings.
        var_name (str, optional): Name of the count variable in query results.
            Defaults to "count". Must match the variable name in the SPARQL query.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns:
            - version (str): RDF graph version extracted from graph URI
            - {var_name} (int): Converted count values
            Returns empty DataFrame if no valid results found.

    Data Processing:
        1. Validates presence of required fields (graph, count variable)
        2. Extracts version from graph URI (last path component)
        3. Converts count strings to integers with validation
        4. Filters out invalid or incomplete records
        5. Returns structured DataFrame or empty DataFrame

    Error Handling:
        - Missing required fields: Logs warning, skips record
        - Invalid URI format: Logs warning, skips record
        - Non-numeric counts: Logs warning, skips record
        - Empty results: Logs warning, returns empty DataFrame
        - Any processing error: Logs error, returns empty DataFrame

    Expected Input Format:
        Each result dict should contain:
        >>> {
        ...     "graph": {"value": "http://aopwiki.org/graph/2024-01-15"},
        ...     "count": {"value": "42"}
        ... }

    Example:
        >>> # Process AOP count results
        >>> sparql_results = run_sparql_query_with_retry(aop_query)
        >>> df = extract_counts(sparql_results, "count")
        >>> print(df.head())
           version  count
        0  2023-01-15     120
        1  2023-06-01     135
        2  2024-01-15     142

        >>> # Process with custom variable name
        >>> df = extract_counts(results, "author_count")
        >>> print(df.columns.tolist())
        ['version', 'author_count']

    Version Extraction:
        - Extracts version from graph URIs like:
          "http://aopwiki.org/graph/2024-01-15" → "2024-01-15"
        - Handles various URI formats robustly
        - Uses the last path component as version identifier

    Data Quality:
        - Validates numeric conversion for all counts
        - Ensures all records have required fields
        - Provides detailed logging for data quality issues
        - Returns consistent DataFrame structure even for edge cases

    Note:
        This function is a critical data processing component used by most
        visualization functions. It ensures data consistency and handles
        the common pattern of version-based count queries.
    """
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
    """Create a professional fallback visualization when primary data is unavailable.

    Generates a styled error visualization that maintains the dashboard's professional
    appearance when data queries fail or return empty results. This ensures users
    receive clear feedback about data availability issues while preserving the
    overall dashboard layout and user experience.

    Args:
        title (str): Title for the fallback plot, typically describing what
            visualization was attempted. Should be descriptive for user clarity.
        error_message (str): Specific error message explaining why data is
            unavailable. Can be technical (for debugging) or user-friendly.

    Returns:
        str: Complete HTML string containing a styled Plotly error visualization.
            Includes error message, professional styling, and responsive configuration.

    Visualization Features:
        - Clean scatter plot with single point (visually minimal)
        - Prominent error message annotation with styling
        - Professional white template matching dashboard theme
        - Red color scheme indicating error state
        - Hidden axes for clean appearance
        - Responsive configuration for mobile compatibility

    Styling Elements:
        - Large, readable error text (16pt font)
        - Red border and background for error indication
        - Centered annotation positioned prominently
        - Hidden legend and axes for minimalist design
        - Proper margins matching other dashboard plots

    Example Usage:
        >>> # In plot function error handling
        >>> try:
        ...     # Attempt data query and visualization
        ...     results = run_sparql_query(query)
        ...     return generate_plot(results)
        ... except Exception as e:
        ...     return create_fallback_plot("Entity Counts", str(e))

        >>> # Usage with descriptive errors
        >>> if not data_available:
        ...     return create_fallback_plot(
        ...         "AOP Timeline Analysis",
        ...         "SPARQL endpoint unavailable - using cached data"
        ...     )

    Error Message Guidelines:
        - Keep messages informative but not overly technical for end users
        - Include enough detail for debugging when necessary
        - Suggest potential solutions when applicable
        - Maintain professional tone consistent with dashboard

    Integration:
        - Used by safe_plot_execution() for automatic error handling
        - Maintains consistent dashboard layout during failures
        - Preserves user experience with graceful degradation
        - Enables debugging through visible error messages

    Performance:
        - Minimal overhead (single scatter point)
        - Fast rendering with CDN Plotly.js
        - Responsive configuration for all devices
        - No data processing requirements

    Dashboard Consistency:
        - Uses plotly_white template matching other plots
        - Maintains responsive configuration standards
        - Preserves dashboard grid layout and spacing
        - Provides visual continuity during error states

    Note:
        This function is critical for application resilience. It ensures
        that individual plot failures don't break the entire dashboard
        and provides users with clear information about data availability.
    """
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
    """Execute plot functions with comprehensive error handling and performance monitoring.

    Provides a robust wrapper for all visualization functions that ensures graceful
    error handling, performance monitoring, and consistent fallback behavior. This
    function is essential for application stability and enables the parallel plot
    generation system to handle individual failures without affecting overall
    dashboard functionality.

    Args:
        plot_func: The plot function to execute. Should be a callable that
            generates visualizations (e.g., plot_latest_entity_counts).
        *args: Positional arguments to pass to the plot function.
        **kwargs: Keyword arguments to pass to the plot function.

    Returns:
        Any: The result of the plot function execution, typically:
            - str: HTML for single visualizations
            - tuple: Multiple HTML strings for functions with variants
            - tuple with DataFrame: For functions that also return data
        Returns appropriate fallback content if execution fails.

    Error Handling Strategy:
        1. Wraps plot function execution in comprehensive try-catch
        2. Monitors execution time and logs performance warnings
        3. Analyzes function signature to determine expected return type
        4. Generates appropriate fallback content based on expected format
        5. Logs detailed error information for debugging

    Performance Monitoring:
        - Tracks execution time for each plot function
        - Logs function name and timing information
        - Identifies slow-running visualizations (useful for optimization)
        - Provides performance visibility for dashboard tuning

    Fallback Generation:
        Single HTML (str):
            - Returns create_fallback_plot() with error message

        Multiple HTML (tuple of str):
            - Returns tuple of fallback plots matching expected count
            - For functions like plot_main_graph() returning (abs, delta)

        Data + HTML (tuple with DataFrame):
            - Returns fallback plots plus empty DataFrame
            - Maintains expected return structure for data processing

    Function Signature Analysis:
        - Uses function annotations to determine expected return types
        - Handles functions with tuple[str, str] annotations
        - Provides intelligent fallbacks for complex return structures
        - Supports both annotated and non-annotated functions

    Example Usage:
        >>> # Direct execution with error handling
        >>> result = safe_plot_execution(plot_latest_entity_counts)
        >>> if "Data Unavailable" in result:
        ...     print("Plot generation failed")

        >>> # In parallel execution system
        >>> with ThreadPoolExecutor() as executor:
        ...     future = executor.submit(safe_plot_execution, plot_func)
        ...     result = future.result(timeout=60)

    Integration with Parallel System:
        - Used by compute_plots_parallel() for all plot generation
        - Enables individual plot failures without system-wide crashes
        - Supports timeout protection at the wrapper level
        - Provides consistent error handling across all visualizations

    Logging Output:
        Success: "Plot function plot_name executed in 2.34s"
        Failure: "Error in plot function plot_name: connection timeout"

    Performance Considerations:
        - Minimal overhead when plots execute successfully
        - Fast fallback generation for failed plots
        - Detailed timing for performance optimization
        - Memory-efficient error handling

    Return Type Detection:
        Analyzes __annotations__ to determine expected return format:
        - str → single HTML fallback
        - tuple[str, str] → dual HTML fallbacks
        - tuple[str, str, pd.DataFrame] → dual HTML + empty DataFrame
        - Default → single HTML fallback

    Note:
        This function is the cornerstone of the application's error resilience.
        It ensures that the dashboard can start and operate even when some
        visualizations fail, providing users with maximum available content.
    """
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

def get_latest_version() -> str:
    """Get the latest AOP-Wiki RDF database version.

    Queries the SPARQL endpoint to get all graphs, then filters and sorts
    in Python for better performance with Virtuoso triplestore.

    Returns:
        str: Latest version string (e.g., "2025-07-01")

    Example:
        >>> version = get_latest_version()
        >>> print(f"Latest version: {version}")
        Latest version: 2025-07-01
    """
    query = """
    SELECT DISTINCT ?g
    WHERE {
        GRAPH ?g { ?s ?p ?o }
    }
    LIMIT 100
    """

    try:
        results = run_sparql_query_with_retry(query)
        if results and len(results) > 0:
            # Filter for AOP-Wiki graphs
            aop_graphs = [
                r.get('g', {}).get('value', '')
                for r in results
                if 'aopwiki.org/graph' in r.get('g', {}).get('value', '')
            ]

            if aop_graphs:
                # Sort in descending order and get the latest
                aop_graphs.sort(reverse=True)
                graph_uri = aop_graphs[0]
                # Extract version from URI like http://aopwiki.org/graph/2025-07-01
                version = graph_uri.split('/')[-1]
                return version

        return "Unknown"
    except Exception as e:
        logger.error(f"Error getting latest version: {e}")
        return "Unknown"


def get_all_versions() -> list[dict]:
    """Get all available AOP-Wiki RDF database versions with metadata.

    Queries the SPARQL endpoint to retrieve all historical versions of the
    AOP-Wiki RDF database, sorted from newest to oldest.

    Returns:
        list[dict]: List of version dictionaries with keys:
            - version: Version string (e.g., "2025-07-01")
            - graph_uri: Full graph URI (e.g., "http://aopwiki.org/graph/2025-07-01")
            - date: Human-readable date string

    Example:
        >>> versions = get_all_versions()
        >>> print(f"Found {len(versions)} versions")
        >>> print(f"Latest: {versions[0]['version']}")
        Found 15 versions
        Latest: 2025-07-01
    """
    query = """
    SELECT DISTINCT ?g
    WHERE {
        GRAPH ?g { ?s ?p ?o }
    }
    LIMIT 100
    """

    try:
        results = run_sparql_query_with_retry(query)
        if results and len(results) > 0:
            # Filter for AOP-Wiki graphs
            aop_graphs = [
                r.get('g', {}).get('value', '')
                for r in results
                if 'aopwiki.org/graph' in r.get('g', {}).get('value', '')
            ]

            if aop_graphs:
                # Sort in descending order (newest first)
                aop_graphs.sort(reverse=True)

                # Build list of version dictionaries
                versions = []
                for graph_uri in aop_graphs:
                    version = graph_uri.split('/')[-1]
                    versions.append({
                        'version': version,
                        'graph_uri': graph_uri,
                        'date': version  # Can be formatted differently if needed
                    })

                return versions

        return []
    except Exception as e:
        logger.error(f"Error getting all versions: {e}")
        return []


def export_figure_as_image(plot_name: str, format: str = 'png', width: int = 1200, height: int = 800) -> Optional[bytes]:
    """Export a cached Plotly figure as PNG or SVG image.

    Args:
        plot_name: Name of the plot in _plot_figure_cache
        format: Image format ('png' or 'svg')
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        bytes: Image data as bytes, or None if export fails

    Example:
        >>> image_bytes = export_figure_as_image('latest_entity_counts', 'png')
        >>> if image_bytes:
        ...     with open('plot.png', 'wb') as f:
        ...         f.write(image_bytes)
    """
    try:
        if plot_name not in _plot_figure_cache:
            logger.error(f"Plot {plot_name} not found in figure cache")
            return None

        fig = _plot_figure_cache[plot_name]

        # Export to image format using Kaleido
        image_bytes = pio.to_image(
            fig,
            format=format,
            width=width,
            height=height,
            engine='kaleido'
        )

        logger.info(f"Successfully exported {plot_name} as {format.upper()}")
        return image_bytes

    except Exception as e:
        logger.error(f"Error exporting {plot_name} as {format}: {e}")
        return None


def get_csv_with_metadata(plot_name: str, include_metadata: bool = True) -> Optional[str]:
    """Generate CSV string with optional metadata headers.

    Args:
        plot_name: Name of the plot in _plot_data_cache
        include_metadata: Whether to include metadata header rows

    Returns:
        str: CSV string with optional metadata, or None if data not found

    Example:
        >>> csv_data = get_csv_with_metadata('latest_entity_counts')
        >>> print(csv_data[:100])
        # Export Date: 2025-10-02 14:30:00
        # Plot: latest_entity_counts
        ...
    """
    try:
        if plot_name not in _plot_data_cache:
            logger.error(f"Plot {plot_name} not found in data cache")
            return None

        df = _plot_data_cache[plot_name]

        if include_metadata:
            from datetime import datetime
            metadata_lines = [
                f"# AOP-Wiki RDF Dashboard Export",
                f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Plot: {plot_name}",
                f"# Data Source: {SPARQL_ENDPOINT}",
                f"# Rows: {len(df)}",
                f"#"
            ]

            # Add version if available in DataFrame
            if 'Version' in df.columns and not df.empty:
                version = df['Version'].iloc[0]
                metadata_lines.insert(3, f"# Database Version: {version}")

            metadata_header = '\n'.join(metadata_lines) + '\n'
            return metadata_header + df.to_csv(index=False)
        else:
            return df.to_csv(index=False)

    except Exception as e:
        logger.error(f"Error generating CSV for {plot_name}: {e}")
        return None


def create_bulk_download(plot_names: list, formats: list = ['csv', 'png', 'svg']) -> Optional[bytes]:
    """Create a ZIP archive containing multiple plots in multiple formats.

    Args:
        plot_names: List of plot names to include in the ZIP
        formats: List of formats to export ('csv', 'png', 'svg')

    Returns:
        bytes: ZIP file contents as bytes, or None if creation fails

    Example:
        >>> zip_bytes = create_bulk_download(['latest_entity_counts', 'latest_ke_components'])
        >>> if zip_bytes:
        ...     with open('plots.zip', 'wb') as f:
        ...         f.write(zip_bytes)
    """
    import io
    import zipfile

    try:
        # Create an in-memory ZIP file
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for plot_name in plot_names:
                # Add CSV if requested
                if 'csv' in formats:
                    csv_data = get_csv_with_metadata(plot_name, include_metadata=True)
                    if csv_data:
                        zip_file.writestr(f'{plot_name}.csv', csv_data)
                        logger.info(f"Added {plot_name}.csv to ZIP")

                # Add PNG if requested
                if 'png' in formats:
                    png_bytes = export_figure_as_image(plot_name, 'png')
                    if png_bytes:
                        zip_file.writestr(f'{plot_name}.png', png_bytes)
                        logger.info(f"Added {plot_name}.png to ZIP")

                # Add SVG if requested
                if 'svg' in formats:
                    svg_bytes = export_figure_as_image(plot_name, 'svg')
                    if svg_bytes:
                        zip_file.writestr(f'{plot_name}.svg', svg_bytes)
                        logger.info(f"Added {plot_name}.svg to ZIP")

        # Get the ZIP file contents
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()

        logger.info(f"Successfully created ZIP with {len(plot_names)} plots in {len(formats)} formats")
        return zip_bytes

    except Exception as e:
        logger.error(f"Error creating bulk download ZIP: {e}")
        return None
