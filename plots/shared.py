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
    Marvin Martens
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
import time
import logging
import threading
from collections import OrderedDict
import requests
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from config import Config

logger = logging.getLogger(__name__)

class VersionedPlotCache:
    """Cache with TTL expiry, max-version cap, and pinned latest version.

    - Latest/current version is pinned and never evicted
    - Historical versions follow TTL + max cap rules
    - Evicts oldest historical entry when cap is reached
    - Thread-safe for concurrent access from Gunicorn gthread workers
    - Supports dict-like [] access for backward compatibility
    """

    def __init__(self, max_versions: int = 5, ttl_seconds: int = 1800):
        self._data = OrderedDict()          # key -> (value, timestamp)
        self._pinned_prefix = None           # e.g., "2025-07-01"
        self._max_versions = max_versions
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def pin_version(self, version: str):
        """Pin a version so its entries are never evicted."""
        self._pinned_prefix = version

    def get(self, key: str, default=None):
        with self._lock:
            if key not in self._data:
                return default
            value, ts = self._data[key]
            # Check TTL for non-pinned entries
            if not self._is_pinned(key) and time.time() - ts > self._ttl:
                del self._data[key]
                return default
            # Move to end (most recently used)
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value):
        with self._lock:
            self._data[key] = (value, time.time())
            self._data.move_to_end(key)
            self._evict_if_needed()

    def __contains__(self, key):
        with self._lock:
            if key not in self._data:
                return False
            value, ts = self._data[key]
            # Check TTL for non-pinned entries
            if not self._is_pinned(key) and time.time() - ts > self._ttl:
                del self._data[key]
                return False
            return True

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getitem__(self, key):
        result = self.get(key)
        if result is None and key not in self._data:
            raise KeyError(key)
        return result

    def keys(self):
        """Return keys, filtering out expired entries."""
        with self._lock:
            now = time.time()
            valid_keys = []
            expired_keys = []
            for k in self._data:
                value, ts = self._data[k]
                if not self._is_pinned(k) and now - ts > self._ttl:
                    expired_keys.append(k)
                else:
                    valid_keys.append(k)
            for k in expired_keys:
                del self._data[k]
            return valid_keys

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._data.clear()

    def _is_pinned(self, key: str) -> bool:
        return self._pinned_prefix is not None and self._pinned_prefix in key

    def _evict_if_needed(self):
        """Evict oldest non-pinned entries when over version cap."""
        # Count distinct versions (extract version from keys like 'latest_entity_counts_2024-01-15')
        versions = set()
        for k in self._data:
            parts = k.rsplit('_', 1)
            if len(parts) > 1 and '-' in parts[-1]:
                versions.add(parts[-1])

        while len(versions) > self._max_versions:
            # Find oldest non-pinned entry
            for k in list(self._data.keys()):
                if not self._is_pinned(k):
                    del self._data[k]
                    parts = k.rsplit('_', 1)
                    if len(parts) > 1:
                        versions.discard(parts[-1])
                    break
            else:
                break  # All entries are pinned


# Validate configuration on import
if not Config.validate_config():
    logger.error("Invalid configuration detected, using defaults")

SPARQL_ENDPOINT = Config.SPARQL_ENDPOINT
MAX_RETRIES = Config.SPARQL_MAX_RETRIES
RETRY_DELAY = Config.SPARQL_RETRY_DELAY
TIMEOUT = Config.SPARQL_TIMEOUT

# Official House Style Color Palette — single source of truth for all chart colors
BRAND_COLORS = {
    'primary': '#29235C',      # VHP4Safety deep purple
    'magenta': '#E6007E',      # VHP4Safety magenta (CTA)
    'blue': '#307BBF',         # VHP4Safety blue (navigation)
    'light_blue': '#009FE3',   # VHP4Safety light blue
    'orange': '#EB5B25',       # VHP4Safety orange
    'sky_blue': '#93D5F6',     # VHP4Safety sky blue
    'deep_magenta': '#9A1C57', # Deep magenta accent
    'teal': '#45A6B2',         # Teal accent
    'dark_teal': '#005A6C',    # Dark teal
    'violet': '#64358C',       # Violet accent
    'warm_pink': '#B81178',    # Warm pink accent
    'palette': [
        '#29235C', '#E6007E', '#307BBF', '#009FE3', '#EB5B25',
        '#93D5F6', '#9A1C57', '#45A6B2', '#B81178', '#005A6C', '#64358C',
    ],
    # OECD Status color mapping — consistent across all OECD-related plots
    'oecd_status': {
        'EAGMST Under Review': '#307BBF',      # blue
        'Under Development': '#009FE3',         # light blue
        'TFHA/WNT Endorsed': '#29235C',         # primary (deep purple)
        'WNT Endorsed': '#E6007E',              # magenta
        'Approved': '#EB5B25',                  # orange
        'No Status': '#999999',                 # grey
        'EAGMST Under Development': '#45A6B2',  # teal
        'Not OECD': '#93D5F6',                  # sky blue
    },
    # Legacy aliases for backward compatibility
    'secondary': '#E6007E',
    'accent': '#307BBF',
    'light': '#93D5F6',
    'content': '#EB5B25',
    # Property type colors using house style palette
    'type_colors': {
        'Essential': '#29235C',    # Primary Dark
        'Metadata': '#E6007E',     # Primary Magenta
        'Content': '#EB5B25',      # Orange
        'Context': '#93D5F6',      # Sky Blue
        'Assessment': '#307BBF',   # Primary Blue
        'Structure': '#45A6B2'     # Teal
    },
    # Network graph role colors — MIE/KE/AO node differentiation
    'network_roles': {
        'MIE': '#EB5B25',   # orange - molecular initiating event
        'KE': '#307BBF',    # blue - key event (intermediate)
        'AO': '#E6007E',    # magenta - adverse outcome
    },
}

# VHP4Safety Plotly custom template — registered as default so all figures get brand styling
_vhp4safety_template = go.layout.Template(
    layout=go.Layout(
        colorway=BRAND_COLORS['palette'],
        font=dict(family="Arial, sans-serif", size=13, color=BRAND_COLORS['primary']),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="#e0e0e0",
            linecolor="#cccccc",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="#e0e0e0",
            linecolor="#cccccc",
            zeroline=False,
            tickfont=dict(size=11),
        ),
    )
)

pio.templates["vhp4safety"] = _vhp4safety_template
pio.templates.default = "plotly_white+vhp4safety"

# Standardized Plotly HTML config for all to_html() calls.
# The Plotly camera (toImage) is intentionally removed: image downloads go
# through the HTML PNG/SVG/CSV export routes, which render each figure at its
# own dimensions instead of a fixed, hard-coded size.
PLOTLY_HTML_CONFIG = {
    "responsive": True,
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"],
}

# Legacy alias for backward compatibility (old name: config)
config = PLOTLY_HTML_CONFIG


def render_plot_html(fig, include_plotlyjs=False):
    """Render a Plotly figure to HTML with standardized config.

    Uses shared PLOTLY_HTML_CONFIG for consistent toolbar behavior
    (hidden until hover, no Plotly logo, responsive sizing).
    """
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config=PLOTLY_HTML_CONFIG,
    )

# Global data cache for CSV export functionality (TTL + LRU eviction, pinned latest version)
_plot_data_cache = VersionedPlotCache(max_versions=5, ttl_seconds=1800)
_plot_figure_cache = VersionedPlotCache(max_versions=5, ttl_seconds=1800)


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


def get_properties_for_entity(entity_type: str) -> Dict[str, List[Dict[str, str]]]:
    """Get properties for a specific entity type, grouped by property category.

    Reads property_labels.csv and filters properties based on the applies_to column,
    then groups them by type (Essential, Content, Context, Assessment, Metadata).

    Args:
        entity_type (str): Entity type to filter by (AOP, KE, KER, or Stressor).
            Case-insensitive.

    Returns:
        Dict[str, List[Dict[str, str]]]: Dictionary with property types as keys
            (Essential, Content, Context, Assessment, Metadata) and lists of
            property dictionaries as values. Each property dict contains:
            - uri: Property URI
            - label: Human-readable label
            - type: Property category
            - applies_to: Pipe-separated entity types

    Example:
        >>> props = get_properties_for_entity('KE')
        >>> essential_props = props.get('Essential', [])
        >>> print(f"KE has {len(essential_props)} essential properties")
        >>> for prop in essential_props:
        ...     print(f"  - {prop['label']}")

    Note:
        Properties with applies_to="NONE" are excluded from all entity types.
        Properties are included if entity_type appears in the pipe-separated
        applies_to column (e.g., "AOP|KE|KER" includes AOP, KE, and KER).
    """
    entity_type = entity_type.upper()

    # Read property labels CSV
    df = safe_read_csv('property_labels.csv')

    if df.empty:
        logger.warning("property_labels.csv is empty or could not be read")
        return {}

    # Filter properties that apply to this entity type
    # applies_to is pipe-separated, e.g., "AOP|KE|KER"
    filtered_df = df[df['applies_to'].str.contains(entity_type, case=False, na=False)]

    # Group by property type
    grouped = {}
    property_types = ['Essential', 'Content', 'Context', 'Assessment', 'Metadata']

    for prop_type in property_types:
        type_df = filtered_df[filtered_df['type'] == prop_type]
        if not type_df.empty:
            grouped[prop_type] = type_df.to_dict('records')
        else:
            grouped[prop_type] = []

    return grouped


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
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config=PLOTLY_HTML_CONFIG)


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
                # Count expected tuple elements from annotation
                type_str = str(return_type)
                str_count = type_str.count('str')
                if str_count >= 3 and 'DataFrame' not in type_str:
                    return tuple([fallback] * str_count)
                elif 'DataFrame' in type_str:
                    return fallback, fallback, pd.DataFrame()
                else:
                    return fallback, fallback

        # Default fallback
        return create_fallback_plot(plot_func.__name__, str(e))

def get_latest_version() -> str:
    """Get the latest AOP-Wiki RDF database version.

    Pushes the prefix filter and ordering into SPARQL so Virtuoso returns
    only the latest AOP-Wiki graph URI directly, without a Python-side
    filter pass. Removes the previous LIMIT 100 ceiling so this still
    works if the triplestore ever holds more than 100 named graphs (#52).

    Returns:
        str: Latest version string (e.g., "2026-04-01"), or "Unknown" on failure.
    """
    query = """
    SELECT ?g
    WHERE {
        GRAPH ?g { ?s ?p ?o }
        FILTER(STRSTARTS(STR(?g), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?g
    ORDER BY DESC(?g)
    LIMIT 1
    """

    try:
        results = run_sparql_query_with_retry(query)
        if results:
            graph_uri = results[0].get('g', {}).get('value', '')
            if graph_uri:
                # Extract version from URI like http://aopwiki.org/graph/2026-04-01
                return graph_uri.rsplit('/', 1)[-1]
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


# SPARQL class URIs for the four headline entity types tracked across versions
# (also used by the entity birth/death helper below).
ENTITY_TYPE_CLASSES: Dict[str, str] = {
    "AOPs": "aopo:AdverseOutcomePathway",
    "KEs": "aopo:KeyEvent",
    "KERs": "aopo:KeyEventRelationship",
    "Stressors": "nci:C54571",
}


def fetch_entity_uris_in_graph(graph_uri: str, entity_class: str) -> set:
    """Fetch the set of entity URIs of a given class in a single named graph.

    Args:
        graph_uri: Full graph URI (e.g. "http://aopwiki.org/graph/2025-07-01").
        entity_class: Prefixed SPARQL class name (e.g. "aopo:KeyEvent",
            "nci:C54571"). The endpoint must have the corresponding prefix
            bound — Virtuoso's AOP-Wiki deployment has aopo/nci preconfigured.

    Returns:
        set[str]: Set of distinct entity URIs in the graph.
    """
    query = f"""
        SELECT DISTINCT ?e
        WHERE {{
            GRAPH <{graph_uri}> {{ ?e a {entity_class} . }}
        }}
    """
    try:
        results = run_sparql_query_with_retry(query)
        return {r.get('e', {}).get('value', '') for r in results if r.get('e', {}).get('value')}
    except Exception as e:
        logger.error(f"Error fetching entities of class {entity_class} in {graph_uri}: {e}")
        return set()


def fetch_entity_uris_by_version(
    entity_type: str,
    versions: List[str],
    max_workers: int = 5,
) -> Dict[str, set]:
    """Fetch the entity URI set for one class across all given versions in parallel.

    Internal helper shared by `diff_entities_between_versions` (#71) and by
    the cumulative-removed view (#72). Centralised so both views run the
    same per-version fetch implementation.

    Args:
        entity_type: A key in `ENTITY_TYPE_CLASSES`.
        versions: List of YYYY-MM-DD version strings to fetch.
        max_workers: ThreadPool size for the parallel per-version queries.

    Returns:
        dict[version_str, set[str]]: Per-version sets of distinct entity URIs.
        Versions whose query fails resolve to an empty set (logged).
    """
    if entity_type not in ENTITY_TYPE_CLASSES:
        raise ValueError(
            f"Unknown entity_type {entity_type!r}; expected one of "
            f"{sorted(ENTITY_TYPE_CLASSES)}"
        )
    cache_key = (entity_type, tuple(sorted(set(versions))))
    cached = _entity_uris_cache.get(cache_key)
    if cached is not None:
        return {v: cached[v].copy() for v in versions if v in cached}

    entity_class = ENTITY_TYPE_CLASSES[entity_type]
    # Single bulk SPARQL — returns (?graph, ?e) for ALL versions at once and
    # group by graph in Python. Replaces the previous per-version fan-out
    # (33 small queries per entity type) with one round-trip; on the live
    # multi-endpoint that's ~50K rows for KERs, well under the 100K cap.
    bulk_query = f"""
        SELECT ?graph ?e WHERE {{
            GRAPH ?graph {{ ?e a {entity_class} . }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
    """
    uris_by_version: Dict[str, set] = {v: set() for v in versions}
    try:
        results = run_sparql_query_with_retry(bulk_query)
    except Exception as e:
        logger.error(f"Bulk fetch failed for {entity_type}: {e}")
        results = []

    for r in results:
        graph_uri = r.get('graph', {}).get('value', '')
        e_uri = r.get('e', {}).get('value', '')
        if not graph_uri or not e_uri:
            continue
        v = graph_uri.rsplit('/', 1)[-1]
        if v in uris_by_version:
            uris_by_version[v].add(e_uri)

    _entity_uris_cache[cache_key] = uris_by_version
    return {v: uris_by_version[v].copy() for v in versions if v in uris_by_version}


# Module-level cache shared by diff_entities_between_versions (#71) and the
# cumulative-removed plot (#72). With the new bulk-query path above, the
# cache means the second plot incurs zero SPARQL traffic.
_entity_uris_cache: Dict[tuple, Dict[str, set]] = {}


def diff_entities_between_versions(
    entity_type: str,
    versions: Optional[List[str]] = None,
    max_workers: int = 5,
) -> pd.DataFrame:
    """Compute the per-version gross added/removed entity flow ("birth/death" curve).

    For each adjacent snapshot pair (v_prev → v_curr), counts entities that
    are present in v_curr but not v_prev (added) and present in v_prev but
    not v_curr (removed). A net delta of zero can hide significant churn —
    this helper exposes it. Used by `plot_entity_birth_death` (#71) and
    reusable by other PRs that need version-to-version set differences
    (#66 KE migration map, #70 curator view, #72 cumulative removals).

    Args:
        entity_type: One of the keys in `ENTITY_TYPE_CLASSES`
            ("AOPs", "KEs", "KERs", "Stressors").
        versions: Optional pre-sorted list of YYYY-MM-DD version strings.
            If omitted, queried via `get_all_versions()`.
        max_workers: ThreadPool size for the per-version fetch queries.

    Returns:
        pd.DataFrame with one row per transition (indexed by v_curr) and
        columns:
            - version (str): v_curr (the newer of the pair).
            - prev_version (str): v_prev.
            - entity_type (str): the input entity_type.
            - added_count (int)
            - removed_count (int)
            - stable_count (int)
            - added_uris (str): semicolon-joined list of added URIs.
            - removed_uris (str): semicolon-joined list of removed URIs.
        The earliest version has no v_prev and is therefore omitted from
        the result.
    """
    if versions is None:
        versions = [v['version'] for v in get_all_versions()]
    # Chronological order (YYYY-MM-DD sorts lexicographically).
    versions = sorted(set(versions))

    if len(versions) < 2:
        return pd.DataFrame(columns=[
            'version', 'prev_version', 'entity_type',
            'added_count', 'removed_count', 'stable_count',
            'added_uris', 'removed_uris',
        ])

    uris_by_version = fetch_entity_uris_by_version(entity_type, versions, max_workers=max_workers)

    rows = []
    for prev, curr in zip(versions, versions[1:]):
        prev_set = uris_by_version.get(prev, set())
        curr_set = uris_by_version.get(curr, set())
        added = curr_set - prev_set
        removed = prev_set - curr_set
        stable = prev_set & curr_set
        rows.append({
            'version': curr,
            'prev_version': prev,
            'entity_type': entity_type,
            'added_count': len(added),
            'removed_count': len(removed),
            'stable_count': len(stable),
            'added_uris': ';'.join(sorted(added)),
            'removed_uris': ';'.join(sorted(removed)),
        })

    return pd.DataFrame(rows)


def build_export_filename(plot_name: str, format: str, version: str = None) -> str:
    """Build self-documenting export filename with date and optional version.

    For trend plots (multi-version): includes export date only.
    For latest plots (single version): includes both version and export date.

    Args:
        plot_name: Name of the plot (underscores will be converted to hyphens)
        format: File format extension (csv, png, svg)
        version: Optional version string for latest-data exports

    Returns:
        str: Filename like 'aop-entity-counts_2026-02-21_v2025-12-01.csv'

    Examples:
        >>> build_export_filename('latest_entity_counts', 'csv', '2025-12-01')
        'latest-entity-counts_2026-02-21_v2025-12-01.csv'
        >>> build_export_filename('ke_components_absolute', 'png')
        'ke-components-absolute_2026-02-21.png'
    """
    from datetime import datetime
    date_str = datetime.now().strftime('%Y-%m-%d')
    clean_name = plot_name.replace('_', '-')

    if version:
        return f"{clean_name}_{date_str}_v{version}.{format}"
    return f"{clean_name}_{date_str}.{format}"


def export_figure_as_image(
    plot_name: str,
    format: str = 'png',
    width: Optional[int] = None,
    height: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[bytes]:
    """Export a cached Plotly figure as PNG or SVG image.

    Args:
        plot_name: Name of the plot in _plot_figure_cache
        format: Image format ('png' or 'svg')
        width: Image width in pixels. None (default) renders at the figure's own
            layout width so the export matches the plot as designed rather than a
            hard-coded size.
        height: Image height in pixels. None (default) renders at the figure's own
            layout height — important for the tall, dynamic-height plots that a
            fixed height would otherwise squash.
        start: Optional YYYY-MM-DD lower bound for the snapshot range (#44).
            Applied as an x-axis clamp on snapshot-keyed trend plots.
        end: Optional YYYY-MM-DD upper bound for the snapshot range (#44).

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

        # If a date-range is requested, clamp the x-axis to the matching window.
        if start or end:
            try:
                fig = _clamp_figure_to_range(fig, start, end)
            except Exception as e:  # never let a clamp failure break the export
                logger.warning(f"Could not apply date-range clamp to {plot_name}: {e}")

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


def apply_snapshot_xaxis(fig, title: str = "Snapshot date"):
    """Format the quarterly-snapshot x-axis of a trend figure.

    Trend plots feed YYYY-MM-DD **strings** to Plotly, which auto-detects a
    date axis and thins the tick labels to fit — one per year or two at the
    ~500px card widths used on /trends.

    Two things break that and should not be reintroduced:

    * ``tickmode='array'`` with every snapshot as an explicit tickval. All 33
      dates then render at -45° inside the card and collapse into an illegible
      smear (#127). Let Plotly choose the ticks instead.
    * Converting the x values to datetimes. ``_figure_x_categories`` only
      collects ``str`` x-values, so the /trends date-range selector would
      silently stop clamping the figure (see ``_clamp_figure_to_range``).

    Args:
        fig: Plotly figure whose x-axis carries YYYY-MM-DD strings.
        title: Axis title. Defaults to the reader-facing "Snapshot date"
            rather than the raw ``version`` column name.
    """
    fig.update_xaxes(title_text=title, tickangle=0)
    return fig


def _clamp_figure_to_range(fig, start: Optional[str], end: Optional[str]):
    """Return a deep-copied figure with its x-axis clamped to [start, end].

    Trend plots render YYYY-MM-DD strings on the x-axis, which Plotly
    auto-detects as ``type='date'``. The clamp range must therefore be
    expressed in the SAME units — ISO date strings (±12h padded so boundary
    markers don't clip). Passing ordinal indices to a date axis makes Plotly
    reinterpret them as milliseconds-since-epoch and the window collapses to
    1970 (the bug fixed in #44; regression-guarded by #61). Only when the
    bounds are not YYYY-MM-DD do we fall back to ordinal index positions.

    The input figure is never mutated; a deep copy is returned.
    """
    import copy
    from datetime import datetime, timedelta

    fig = copy.deepcopy(fig)
    versions = _figure_x_categories(fig)
    if not versions:
        return fig

    s_str = start if start else versions[0]
    e_str = end if end else versions[-1]
    try:
        s_dt = datetime.strptime(s_str, '%Y-%m-%d') - timedelta(hours=12)
        e_dt = datetime.strptime(e_str, '%Y-%m-%d') + timedelta(hours=12)
        fig.update_xaxes(range=[s_dt.isoformat(), e_dt.isoformat()], autorange=False)
    except ValueError:
        # Not a YYYY-MM-DD axis — fall back to ordinal positions.
        s_idx = _index_at_or_after(versions, start) if start else 0
        e_idx = _index_at_or_before(versions, end) if end else len(versions) - 1
        if s_idx is not None and e_idx is not None and s_idx <= e_idx:
            fig.update_xaxes(range=[s_idx - 0.5, e_idx + 0.5], autorange=False)
    return fig


def _figure_x_categories(fig) -> list:
    """Collect the sorted union of x-values across all traces of a figure.

    Trend plots have a categorical x-axis whose values are YYYY-MM-DD strings.
    Different traces (e.g. one per Entity) may have different per-trace x
    lists; the union — sorted lexicographically (which equals chronological
    for YYYY-MM-DD) — gives a stable index space.
    """
    seen = set()
    for tr in getattr(fig, 'data', []) or []:
        xs = getattr(tr, 'x', None)
        if xs is None:
            continue
        for x in xs:
            if isinstance(x, str):
                seen.add(x)
    return sorted(seen)


def _index_at_or_after(versions: list, target: str) -> Optional[int]:
    for i, v in enumerate(versions):
        if v >= target:
            return i
    return None


def _index_at_or_before(versions: list, target: str) -> Optional[int]:
    last = None
    for i, v in enumerate(versions):
        if v <= target:
            last = i
    return last


def get_csv_with_metadata(
    plot_name: str,
    include_metadata: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[str]:
    """Generate CSV string with optional metadata headers.

    Args:
        plot_name: Name of the plot in _plot_data_cache
        include_metadata: Whether to include metadata header rows
        start: Optional YYYY-MM-DD lower bound (inclusive) on the snapshot
            version column (#44). Rows are filtered against a `version` or
            `Version` column; plots without one are returned unfiltered.
        end: Optional YYYY-MM-DD upper bound (inclusive).

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

        # Apply optional snapshot-range filter (#44). YYYY-MM-DD strings sort
        # lexicographically the same as chronologically, so a direct string
        # comparison is correct.
        version_col = 'version' if 'version' in df.columns else ('Version' if 'Version' in df.columns else None)
        if version_col and (start or end):
            mask = pd.Series(True, index=df.index)
            if start:
                mask &= df[version_col].astype(str) >= start
            if end:
                mask &= df[version_col].astype(str) <= end
            df = df[mask]

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
            if start or end:
                metadata_lines.insert(
                    4,
                    f"# Snapshot range: {start or '(unbounded)'} → {end or '(unbounded)'}"
                )

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


def create_bulk_download(
    plot_names: list,
    formats: list = ['csv', 'png', 'svg'],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[bytes]:
    """Create a ZIP archive containing multiple plots in multiple formats.

    Args:
        plot_names: List of plot names to include in the ZIP
        formats: List of formats to export ('csv', 'png', 'svg')
        start: Optional YYYY-MM-DD lower bound forwarded to each export (#44).
        end: Optional YYYY-MM-DD upper bound forwarded to each export.

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
                    csv_data = get_csv_with_metadata(plot_name, include_metadata=True, start=start, end=end)
                    if csv_data:
                        zip_file.writestr(f'{plot_name}.csv', csv_data)
                        logger.info(f"Added {plot_name}.csv to ZIP")

                # Add PNG if requested
                if 'png' in formats:
                    png_bytes = export_figure_as_image(plot_name, 'png', start=start, end=end)
                    if png_bytes:
                        zip_file.writestr(f'{plot_name}.png', png_bytes)
                        logger.info(f"Added {plot_name}.png to ZIP")

                # Add SVG if requested
                if 'svg' in formats:
                    svg_bytes = export_figure_as_image(plot_name, 'svg', start=start, end=end)
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
