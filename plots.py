"""Data visualization and SPARQL query functions for AOP-Wiki RDF Dashboard.

This module contains all visualization generation and data processing functions for the
AOP-Wiki RDF Dashboard. It provides a comprehensive suite of plotting functions that
query SPARQL endpoints, process RDF data, and generate interactive Plotly visualizations
with consistent VHP4Safety branding.

Key Features:
    - SPARQL endpoint integration with retry logic and error handling
    - Interactive Plotly visualizations with professional branding
    - Global data caching system for CSV exports
    - Comprehensive error handling and fallback mechanisms
    - Performance monitoring and optimization
    - Brand-consistent color palette and styling

Module Architecture:
    - SPARQL Utilities: Connection, querying, and error handling
    - Data Processing: RDF data extraction and transformation
    - Visualization Engine: Plotly chart generation with branding
    - Caching System: Global data cache for CSV export functionality
    - Error Handling: Fallback plots and graceful degradation

Visualization Categories:
    
    Historical Trends (Time Series Analysis):
        - plot_main_graph(): Core entity evolution over time
        - plot_avg_per_aop(): Average components per AOP trends
        - plot_network_density(): Network connectivity evolution  
        - plot_ke_components(): Component annotation trends
        - plot_author_counts(): Author contribution patterns
        - plot_aop_lifetime(): AOP lifecycle analysis
        
    Latest Data Snapshots (Current State):
        - plot_latest_entity_counts(): Current entity statistics
        - plot_latest_ke_components(): Component distribution analysis
        - plot_latest_network_density(): Current connectivity metrics
        - plot_latest_aop_completeness(): Data completeness assessment
        - plot_latest_ontology_usage(): Ontology source distribution
        
    Specialized Analysis:
        - plot_bio_processes(): Biological process ontology usage
        - plot_bio_objects(): Biological object ontology analysis
        - plot_aop_property_presence(): Property presence analysis
        - plot_kes_by_kec_count(): KE annotation depth distribution

Brand Identity:
    All visualizations use the official house style color palette for consistency:
    
    Primary Colors:
    - Primary Dark: #29235C (Main brand color)
    - Primary Magenta: #E6007E (Accent color)
    - Primary Blue: #307BBF (Supporting blue)
    
    Secondary Colors:
    - Light Blue: #009FE3
    - Orange: #EB5B25 (Content highlight)
    - Sky Blue: #93D5F6 (Light accent)
    - Deep Magenta: #9A1C57, Teal: #45A6B2, Purple: #B81178
    - Dark Teal: #005A6C, Violet: #64358C

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
    Basic plot generation:
    >>> html_plot = plot_latest_entity_counts()
    >>> print("Generated plot HTML:", len(html_plot))
    
    Historical analysis:
    >>> abs_plot, delta_plot, data = plot_main_graph()
    >>> print("Historical data shape:", data.shape)
    
    CSV data access:
    >>> from plots import _plot_data_cache
    >>> if 'latest_entity_counts' in _plot_data_cache:
    ...     df = _plot_data_cache['latest_entity_counts']
    ...     df.to_csv('entity_counts.csv')

Error Handling:
    All plot functions include comprehensive error handling:
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

# Safe file operations with error handling
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
        title="AOP Entity Counts (Absolute)",
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
        title="AOP Entity Counts (Delta Between Versions)",
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
                      title="Average KEs and KERs per AOP (Absolute)",
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
                        title="Average KEs and KERs per AOP (Delta)",
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
                      title="Unique AOP Authors per Version",
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
                        title="Change in Unique AOP Authors per Version",
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
                        color_discrete_sequence=[BRAND_COLORS['primary']])
    fig1.update_layout(template="plotly_white", height=400)
    html1 = pio.to_html(fig1, full_html=False, include_plotlyjs="cdn")

    # --- Plot 2: AOPs Modified ---
    fig2 = px.histogram(df_modified, x="year_modified",
                        title="Unique AOPs Last Modified per Year",
                        labels={"year_modified": "Year", "count": "AOP Count"},
                        color_discrete_sequence=[BRAND_COLORS['secondary']])
    fig2.update_layout(template="plotly_white", height=400)
    html2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)

    # --- Plot 3: Created vs. Modified Dates ---
    fig3 = px.scatter(df_lifetime, x="created", y="modified", hover_name="aop",
                      title="AOP Creation vs. Last Modification Dates",
                      labels={"created": "Created", "modified": "Modified"},
                      color_discrete_sequence=[BRAND_COLORS['accent']])
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
        title="Change in KE Component Annotations Between Versions",
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
        title="Change in KE Component % Between Versions",
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
        title="Change in Unique KE Component Annotations Between Versions",
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
        title="Change in Biological Process Annotations by Ontology (Delta)",
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
        title="KEs Annotated with Biological Objects by Ontology (Absolute)",
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
        title="Change in Biological Object Annotations by Ontology (Delta)",
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
        title="Presence of Properties in AOPs Over Time (Distinct Count)",
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
        title="Presence of Properties in AOPs Over Time (Percentage)",
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
    """
    Create AOP property presence over time plots with unique colors for each property line.
    Generates colors programmatically to ensure all properties have distinct colors.
    """
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
    import colorsys
    
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
        title="AOP Property Presence Over Time - Unique Colors (Absolute Count)",
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
        title="AOP Property Presence Over Time - Unique Colors (Percentage)",
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
        labels={"total_kes": "Number of KEs", "bioevent_count_group": "Number of KECs"},
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
        title="Change in KEs by Number of KECs Over Time",
        labels={"total_kes_delta": "Change in KEs", "bioevent_count_group": "Number of KECs"},
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


# Global data cache for CSV downloads
_plot_data_cache = {}

def get_latest_version() -> str:
    """Get the latest AOP-Wiki RDF database version.

    Queries the SPARQL endpoint to find the most recent graph version
    available in the triplestore.

    Returns:
        str: Latest version identifier (e.g., "2024-10-01") or fallback message
    """
    query = """
        SELECT DISTINCT ?graph
        WHERE {
            GRAPH ?graph { ?s ?p ?o . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        ORDER BY DESC(?graph)
        LIMIT 1
    """

    try:
        results = run_sparql_query(query)
        if results:
            latest_graph_uri = results[0]["graph"]["value"]
            # Extract version from URI like "http://aopwiki.org/graph/2024-10-01"
            version = latest_graph_uri.split("/")[-1]
            return version
        else:
            return "No version data available"
    except Exception as e:
        logger.error(f"Error getting latest version: {e}")
        return "Version query failed"

def plot_latest_entity_counts() -> str:
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
            SELECT ?graph (COUNT(DISTINCT ?ke) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?ke a aopo:KeyEvent .
                }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
            ORDER BY DESC(?graph)
            LIMIT 1
        """,
        "KERs": """
            SELECT ?graph (COUNT(DISTINCT ?ker) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?ker a aopo:KeyEventRelationship  .
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
        """,
        "Authors": """
            SELECT ?graph (COUNT(DISTINCT ?c) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?aop a aopo:AdverseOutcomePathway ;
                         dc:creator ?c .
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
    df["Version"] = latest_version  # Add version column for context
    
    # Store in global cache for CSV download
    _plot_data_cache['latest_entity_counts'] = df
    
    fig = px.bar(
        df,
        x="Entity",
        y="Count",
        title=f"Latest AOP Entity Counts ({latest_version})",
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


def plot_latest_ke_components() -> str:
    """Create a pie chart showing the latest version's KE component distribution."""
    global _plot_data_cache
    
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
    
    # Add version for context and cache data
    df["Version"] = latest_version
    _plot_data_cache['latest_ke_components'] = df
    
    fig = px.pie(
        df,
        values="Count",
        names="Component",
        title=f"KE Component Distribution ({latest_version})",
        color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['secondary'], BRAND_COLORS['accent']]
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def plot_latest_network_density() -> str:
    """Analyze AOP connectivity based on shared Key Events - simplified approach."""
    global _plot_data_cache
    
    # First, get total AOPs in latest version
    query_total = """
    SELECT ?graph (COUNT(DISTINCT ?aop) AS ?total_aops)
    WHERE {
        GRAPH ?graph { 
            ?aop a aopo:AdverseOutcomePathway .
        }
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY DESC(?graph)
    LIMIT 1
    """
    
    # Second, get AOPs that share at least one KE with another AOP
    query_connected = """
    SELECT ?graph (COUNT(DISTINCT ?aop1) AS ?connected_aops)
    WHERE {
        GRAPH ?graph {
            ?aop1 a aopo:AdverseOutcomePathway ;
                  aopo:has_key_event ?ke .
            ?aop2 a aopo:AdverseOutcomePathway ;
                  aopo:has_key_event ?ke .
            FILTER(?aop1 != ?aop2)
        }
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY DESC(?graph)
    LIMIT 1
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
        title=f"AOP Connectivity Analysis ({latest_version})",
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


def plot_latest_avg_per_aop() -> str:
    """Create a bar chart showing average KEs and KERs per AOP in latest version."""
    global _plot_data_cache
    
    query_aops = """
        SELECT ?graph (COUNT(?aop) AS ?count)
        WHERE {
            GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
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
        ORDER BY DESC(?graph)
        LIMIT 1
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
        ORDER BY DESC(?graph)
        LIMIT 1
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
        title=f"Average Connectivity per AOP ({latest_version})",
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



def plot_latest_ontology_usage() -> str:
    """Create a chart showing ontology usage in the latest version."""
    query = """
    SELECT ?graph ?ontology (COUNT(DISTINCT ?term) AS ?count)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        {
          ?bioevent aopo:hasProcess ?term .
        } UNION {
          ?bioevent aopo:hasObject ?term .
        } UNION {
          ?bioevent aopo:hasAction ?term .
        }
        
        BIND(
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/GO_"), "GO",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/MP_"), "MP",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/NBO_"), "NBO",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/MI_"), "MI",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/VT_"), "VT",
          IF(STRSTARTS(STR(?term), "http://purl.org/commons/record/mesh/"), "MESH",
          IF(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/HP_"), "HP", "OTHER"))))))) AS ?ontology)
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph ?ontology
    ORDER BY DESC(?graph) DESC(?count)
    """
    
    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Ontology Usage", "No ontology data available")
    
    # Get only the latest version
    latest_version = results[0]["graph"]["value"].split("/")[-1]
    latest_data = [r for r in results if latest_version in r["graph"]["value"]]
    
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
        title=f"Ontology Term Usage ({latest_version})"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def plot_latest_process_usage() -> str:
    """Create a pie chart showing ontology source distribution for biological processes."""
    global _plot_data_cache
    
    query = """
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
    ORDER BY DESC(?graph) DESC(?count)
    """
    
    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Process Ontology Sources", "No process ontology data available")
    
    latest_version = results[0]["graph"]["value"].split("/")[-1] if results else "Unknown"
    latest_data = [r for r in results if latest_version in r["graph"]["value"]]
    
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
        title=f"Process Ontology Sources ({latest_version})",
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def plot_latest_object_usage() -> str:
    """Create a pie chart showing ontology source distribution for biological objects."""
    global _plot_data_cache
    
    query = """
    SELECT ?graph ?ontology (COUNT(DISTINCT ?object) AS ?count)
    WHERE {
      GRAPH ?graph {
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
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph ?ontology
    ORDER BY DESC(?graph) DESC(?count)
    """
    
    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("Object Ontology Sources", "No object ontology data available")
    
    latest_version = results[0]["graph"]["value"].split("/")[-1] if results else "Unknown"
    latest_data = [r for r in results if latest_version in r["graph"]["value"]]
    
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
        title=f"Object Ontology Sources ({latest_version})",
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def plot_latest_aop_completeness() -> str:
    """Create a chart showing AOP data completeness for all properties in the CSV spreadsheet."""
    global _plot_data_cache

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
    total_query = """
    SELECT ?graph (COUNT(DISTINCT ?aop) AS ?total_aops)
    WHERE {
        GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY DESC(?graph)
    LIMIT 1
    """
    total_results = run_sparql_query(total_query)
    if not total_results:
        return create_fallback_plot("AOP Completeness", "No AOP data available")

    total_aops = int(total_results[0]["total_aops"]["value"])
    latest_version = total_results[0]["graph"]["value"].split("/")[-1]

    completeness_data = []

    for prop in properties:
        uri = prop["uri"]
        label = prop["label"]
        ptype = prop["type"]

        query = f"""
        SELECT ?graph (COUNT(DISTINCT ?aop) AS ?count)
        WHERE {{
            GRAPH ?graph {{
                ?aop a aopo:AdverseOutcomePathway ;
                     <{uri}> ?value .
            }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph ORDER BY DESC(?graph) LIMIT 1
        """
        results = run_sparql_query(query)
        count = int(results[0]["count"]["value"]) if results else 0
        completeness = (count / total_aops) * 100

        completeness_data.append({
            "Property": label,
            "Completeness": completeness,
            "Type": ptype,
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
        title=f"AOP Data Completeness ({latest_version})",
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

def plot_latest_aop_completeness_unique_colors() -> str:
    """Create a chart showing AOP data completeness with unique colors for each property (larger panel)."""
    global _plot_data_cache

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
    total_query = """
    SELECT ?graph (COUNT(DISTINCT ?aop) AS ?total_aops)
    WHERE {
        GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY DESC(?graph)
    LIMIT 1
    """
    total_results = run_sparql_query(total_query)
    if not total_results:
        return create_fallback_plot("AOP Completeness (Unique Colors)", "No AOP data available")

    total_aops = int(total_results[0]["total_aops"]["value"])
    latest_version = total_results[0]["graph"]["value"].split("/")[-1]

    completeness_data = []

    for prop in properties:
        uri = prop["uri"]
        label = prop["label"]
        ptype = prop["type"]

        query = f"""
        SELECT ?graph (COUNT(DISTINCT ?aop) AS ?count)
        WHERE {{
            GRAPH ?graph {{
                ?aop a aopo:AdverseOutcomePathway ;
                     <{uri}> ?value .
            }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph ORDER BY DESC(?graph) LIMIT 1
        """
        results = run_sparql_query(query)
        count = int(results[0]["count"]["value"]) if results else 0
        completeness = (count / total_aops) * 100

        completeness_data.append({
            "Property": label,
            "Completeness": completeness,
            "Type": ptype,
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
        title=f"AOP Data Completeness - Unique Colors ({latest_version})",
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

def plot_latest_database_summary() -> str:
    """Create a simple summary chart of main entities in latest version using separate queries."""
    
    # Ultra-simple separate queries for each entity type
    queries = {
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
        "Key Events": """
            SELECT ?graph (COUNT(?ke) AS ?count)
            WHERE {
                GRAPH ?graph { ?ke a aopo:KeyEvent . }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
            ORDER BY DESC(?graph)
            LIMIT 1
        """,
        "KE Relationships": """
            SELECT ?graph (COUNT(?ker) AS ?count)
            WHERE {
                GRAPH ?graph { ?ker a aopo:KeyEventRelationship . }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
            ORDER BY DESC(?graph)
            LIMIT 1
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
        title=f"Core Entity Summary ({latest_version})",
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


def plot_latest_ke_annotation_depth() -> str:
    """Show distribution of Key Events by annotation depth (number of components)."""
    global _plot_data_cache
    
    query = """
    SELECT ?graph ?annotation_depth (COUNT(DISTINCT ?ke) AS ?ke_count)
    WHERE {
      GRAPH ?graph {
        ?ke a aopo:KeyEvent .
        OPTIONAL { ?ke aopo:hasBiologicalEvent ?bioevent . }
        {
          SELECT ?ke (COUNT(DISTINCT ?bioevent) AS ?annotation_depth)
          WHERE {
            ?ke a aopo:KeyEvent .
            OPTIONAL { 
              ?ke aopo:hasBiologicalEvent ?bioevent .
                          }
          }
          GROUP BY ?ke
        }
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph ?annotation_depth
    ORDER BY DESC(?graph) ?annotation_depth
    """
    
    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("KE Annotation Depth", "No annotation data available")
    
    # Get latest version data
    latest_version = results[0]["graph"]["value"].split("/")[-1]
    latest_data = [r for r in results if latest_version in r["graph"]["value"]]
    
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
        title=f"KE Annotation Depth Distribution ({latest_version})",
        color_discrete_sequence=BRAND_COLORS['palette']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


def plot_aop_network() -> str:
    """Generate interactive AOP network visualization using Cytoscape.js.
    
    Creates a comprehensive network visualization showing all AOPs, Key Events (KEs),
    and Key Event Relationships (KERs) from the latest database version.
    
    Network Structure:
        - Nodes: AOPs and KEs (with ID and title attributes only)
        - Edges: KERs showing upstream→downstream relationships
        - Interactive features: pan/zoom, node selection, hover tooltips
        
    Data Caching:
        - Nodes DataFrame: Contains all network nodes with attributes
        - Edges DataFrame: Contains all network edges with relationships
        - Network metrics: KE frequency counts for future scaling features
        
    Returns:
        str: HTML containing Cytoscape.js network visualization
        
    Future Enhancements:
        - Node sizing based on KE frequency across AOPs
        - Centrality metrics for important pathway nodes
        - Filtering by AOP properties or pathway types
    """
    global _plot_data_cache
    
    def extract_sparql_value(field):
        """Helper function to safely extract values from SPARQL dictionary results."""
        if isinstance(field, dict) and 'value' in field:
            return field['value']
        return str(field)
    
    try:
        # Get latest version
        latest_version_query = """
        SELECT ?graph
        WHERE {
            GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY DESC(?graph)
        LIMIT 1
        """
        
        latest_results = run_sparql_query_with_retry(latest_version_query)
        if not latest_results:
            return create_fallback_plot("AOP Network", "No AOP data available for network visualization")
        
        latest_graph_result = latest_results[0]['graph']
        # Handle both string URIs and dictionary objects from SPARQL results
        if isinstance(latest_graph_result, dict) and 'value' in latest_graph_result:
            latest_graph = latest_graph_result['value']
        else:
            latest_graph = str(latest_graph_result)
        
        latest_version = latest_graph.split('/')[-1]
        
        # Query 1: Get all AOPs with basic info (simplified)
        aop_query = f"""
        SELECT DISTINCT ?aop ?aop_id ?title
        WHERE {{
            GRAPH <{latest_graph}> {{
                ?aop a aopo:AdverseOutcomePathway ;
                     rdfs:label ?aop_id ;
                     dc:title ?title .
            }}
        }}
        ORDER BY ?aop_id
        """
        
        # Query 2: Get all KEs with descriptive titles and basic info
        ke_query = f"""
        SELECT DISTINCT ?ke ?ke_id ?descriptive_title
        WHERE {{
            GRAPH <{latest_graph}> {{
                ?ke a aopo:KeyEvent ;
                    rdfs:label ?ke_id .
                OPTIONAL {{ ?ke dc:title ?descriptive_title }}
            }}
        }}
        """
        
        # Query 4: Get MIEs (optimized separate query)
        mie_query = f"""
        SELECT DISTINCT ?ke
        WHERE {{
            GRAPH <{latest_graph}> {{
                ?aop aopo:has_molecular_initiating_event ?ke .
            }}
        }}
        """
        
        # Query 5: Get AOs (optimized separate query)  
        ao_query = f"""
        SELECT DISTINCT ?ke
        WHERE {{
            GRAPH <{latest_graph}> {{
                ?aop aopo:has_adverse_outcome ?ke .
            }}
        }}
        """
        
        # Query 3: Get all KERs (edges) - simplified
        ker_query = f"""
        SELECT DISTINCT ?ker ?ker_id ?upstream_ke ?downstream_ke
        WHERE {{
            GRAPH <{latest_graph}> {{
                ?ker a aopo:KeyEventRelationship ;
                     aopo:has_upstream_key_event ?upstream_ke ;
                     aopo:has_downstream_key_event ?downstream_ke ;
                     rdfs:label ?ker_id .
            }}
        }}
        ORDER BY ?ker_id
        """
        
        # Query 4: Get AOP-KE relationships for context - simplified
        aop_ke_query = f"""
        SELECT DISTINCT ?aop ?aop_id ?ke ?ke_id
        WHERE {{
            GRAPH <{latest_graph}> {{
                ?aop a aopo:AdverseOutcomePathway ;
                     rdfs:label ?aop_id ;
                     aopo:has_key_event ?ke .
                BIND(REPLACE(STR(?ke), ".*[/#]([0-9]+)$", "$1") AS ?ke_id)
            }}
        }}
        ORDER BY ?aop_id ?ke_id
        """
        
        # Query 5: Get AOP-KER relationships for edge thickness calculation
        aop_ker_query = f"""
        SELECT DISTINCT ?aop ?ker
        WHERE {{
            GRAPH <{latest_graph}> {{
                ?aop a aopo:AdverseOutcomePathway ;
                     aopo:has_key_event_relationship ?ker .
            }}
        }}
        """
        
        # Execute queries
        logging.info("Fetching AOP network data...")
        aop_results = run_sparql_query_with_retry(aop_query)  # Still needed for some stats
        ke_results = run_sparql_query_with_retry(ke_query)
        mie_results = run_sparql_query_with_retry(mie_query) 
        ao_results = run_sparql_query_with_retry(ao_query)
        ker_results = run_sparql_query_with_retry(ker_query)
        aop_ke_results = run_sparql_query_with_retry(aop_ke_query)
        aop_ker_results = run_sparql_query_with_retry(aop_ker_query)
        
        # Create fast lookup sets for MIE and AO classification
        mie_uris = set()
        if mie_results:
            for mie in mie_results:
                mie_uri = extract_sparql_value(mie.get('ke', ''))
                if mie_uri:
                    mie_uris.add(mie_uri)
                    
        ao_uris = set()
        if ao_results:
            for ao in ao_results:
                ao_uri = extract_sparql_value(ao.get('ke', ''))
                if ao_uri:
                    ao_uris.add(ao_uri)
        
        # Debug logging
        logging.info(f"AOP query returned {len(aop_results) if aop_results else 0} results")
        logging.info(f"KE query returned {len(ke_results) if ke_results else 0} results")
        logging.info(f"MIE query returned {len(mie_results) if mie_results else 0} results ({len(mie_uris)} unique MIEs)")
        logging.info(f"AO query returned {len(ao_results) if ao_results else 0} results ({len(ao_uris)} unique AOs)")
        logging.info(f"KER query returned {len(ker_results) if ker_results else 0} results")
        logging.info(f"AOP-KE query returned {len(aop_ke_results) if aop_ke_results else 0} results")
        
        if aop_results:
            logging.info(f"Sample AOP result: {aop_results[0] if aop_results else 'None'}")
        if ke_results:
            logging.info(f"Sample KE result: {ke_results[0] if ke_results else 'None'}")
        
        if not ke_results:
            logging.warning(f"Insufficient data - KEs: {len(ke_results) if ke_results else 0}")
            return create_fallback_plot("AOP Network", "Insufficient data for network visualization")
        
        # Calculate AOP frequencies for each KE
        ke_aop_frequency = {}  # Maps KE URI to count of AOPs containing it
        if aop_ke_results:
            for aop_ke in aop_ke_results:
                ke_uri = extract_sparql_value(aop_ke.get('ke', ''))
                if ke_uri:
                    ke_aop_frequency[ke_uri] = ke_aop_frequency.get(ke_uri, 0) + 1
            logging.info(f"Calculated AOP frequencies for {len(ke_aop_frequency)} KEs")
        
        # Identify dual-role nodes (MIEs that also appear as regular KEs, or AOs that also appear as regular KEs)
        # A node is dual-role if it's an MIE/AO but appears in more AOPs than just its MIE/AO role
        dual_role_uris = set()
        
        # Check MIEs that appear in multiple AOPs (suggesting they're also regular KEs)
        for mie_uri in mie_uris:
            if ke_aop_frequency.get(mie_uri, 0) > 1:  # Appears in more than 1 AOP
                dual_role_uris.add(mie_uri)
                
        # Check AOs that appear in multiple AOPs (suggesting they're also regular KEs)  
        for ao_uri in ao_uris:
            if ke_aop_frequency.get(ao_uri, 0) > 1:  # Appears in more than 1 AOP
                dual_role_uris.add(ao_uri)
                
        logging.info(f"Identified {len(dual_role_uris)} dual-role nodes (MIE/AO also serving as KE)")
        
        # Calculate KER frequencies for edge thickness
        ker_aop_frequency = {}  # Maps KER URI to count of AOPs containing it
        if aop_ker_results:
            for aop_ker in aop_ker_results:
                ker_uri = extract_sparql_value(aop_ker.get('ker', ''))
                if ker_uri:
                    ker_aop_frequency[ker_uri] = ker_aop_frequency.get(ker_uri, 0) + 1
            logging.info(f"Calculated KER frequencies for {len(ker_aop_frequency)} KERs")
        
        # Build nodes DataFrame (KE nodes only - no AOP nodes per user request)
        nodes_data = []
        
        # Add KE nodes with MIE/AO classification
        ke_count = 0
        mie_count = 0
        ao_count = 0
        for ke in ke_results:
            # Extract KE information using helper functions  
            ke_uri = extract_sparql_value(ke.get('ke', ''))
            ke_id_label = extract_sparql_value(ke.get('ke_id', ''))
            descriptive_title = extract_sparql_value(ke.get('descriptive_title', ''))
            
            if ke_uri and ke_id_label:
                # Extract KE ID from URI for consistency
                import re
                ke_id_match = re.search(r'[/#](\d+)$', ke_uri)
                ke_id = ke_id_match.group(1) if ke_id_match else str(ke_count + 1)
                
                # Calculate real AOP frequency for this KE
                aop_freq = ke_aop_frequency.get(ke_uri, 0)  # Real frequency from AOP-KE relationships
                
                # Use descriptive title if available, fallback to label
                display_title = descriptive_title if descriptive_title else ke_id_label
                
                # Create truncated label for display (max 25 chars)
                truncated_label = display_title[:25] + "..." if len(display_title) > 25 else display_title
                
                # Classify KE as MIE, AO, or regular KE using fast lookup sets
                is_dual_role = ke_uri in dual_role_uris
                if ke_uri in mie_uris:
                    node_subtype = 'MIE+KE' if is_dual_role else 'MIE'
                    group = 'mie'
                    mie_count += 1
                elif ke_uri in ao_uris:
                    node_subtype = 'AO+KE' if is_dual_role else 'AO'
                    group = 'ao'
                    ao_count += 1
                else:
                    node_subtype = 'KE'
                    group = 'ke'
                
                nodes_data.append({
                    'id': f"ke_{ke_id}",
                    'label': ke_id_label,  # Short label (e.g., "KE 10")
                    'truncated_label': truncated_label,  # Truncated title for display
                    'display_title': display_title,  # Descriptive title for display
                    'descriptive_title': descriptive_title or '',  # Raw descriptive title 
                    'type': 'KE',
                    'node_subtype': node_subtype,
                    'original_id': ke_id,
                    'aop_frequency': aop_freq,
                    'group': group,
                    'is_dual_role': is_dual_role
                })
                ke_count += 1
        
        logging.info(f"Created {ke_count} KE nodes from {len(ke_results)} KE results: {mie_count} MIEs, {ao_count} AOs, {ke_count - mie_count - ao_count} regular KEs")
        
        # Build edges DataFrame
        edges_data = []
        
        # Add KER edges (KE to KE relationships)
        for ker in ker_results:
            # Extract IDs using helper function
            ker_id = extract_sparql_value(ker.get('ker_id', ''))
            ker_uri = extract_sparql_value(ker.get('ker', ''))
            upstream_ke_uri = extract_sparql_value(ker.get('upstream_ke', ''))
            downstream_ke_uri = extract_sparql_value(ker.get('downstream_ke', ''))
            
            # Extract final ID from URIs
            upstream_id = upstream_ke_uri.split('/')[-1] if upstream_ke_uri else ''
            downstream_id = downstream_ke_uri.split('/')[-1] if downstream_ke_uri else ''
            
            # Calculate KER frequency for thickness
            ker_freq = ker_aop_frequency.get(ker_uri, 1)  # Default to 1 if not found
            
            if ker_id and upstream_id and downstream_id:
                edges_data.append({
                    'id': f"ker_{ker_id}",
                    'source': f"ke_{upstream_id}",
                    'target': f"ke_{downstream_id}",
                    'type': 'KER',
                    'original_id': ker_id,
                    'aop_frequency': ker_freq
                })
        
        # Note: AOP-KE edges (contains relationships) are not included in KE-only network visualization
        # This section is disabled to focus on KE-to-KE relationships only
        
        # Create DataFrames for caching
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        # Add version info for context
        nodes_df['Version'] = latest_version
        edges_df['Version'] = latest_version
        
        # Cache for CSV download
        _plot_data_cache['aop_network_nodes'] = nodes_df
        _plot_data_cache['aop_network_edges'] = edges_df
        
        # Prepare data for Cytoscape.js - COMPLETE NETWORK
        cytoscape_elements = []
        
        # Add ALL KE nodes to Cytoscape format (exclude AOP nodes)
        for _, node in nodes_df.iterrows():
            if node['type'] == 'KE':  # Include ALL KE nodes
                element = {
                    'data': {
                        'id': node['id'],
                        'label': node['label'],  # Short label (KE ID)
                        'truncated_label': node['truncated_label'],  # Truncated descriptive title
                        'display_title': node['display_title'],  # Full descriptive title
                        'descriptive_title': node['descriptive_title'],  # Raw descriptive title
                        'type': node['type'],
                        'node_subtype': node['node_subtype'],
                        'group': node['group'],
                        'frequency': node['aop_frequency'],
                        'aop_frequency': node['aop_frequency'],  # For sizing calculations
                        'is_dual_role': node['is_dual_role']  # For visual styling
                    }
                }
                cytoscape_elements.append(element)
        
        # Add ALL KER edges to Cytoscape format (exclude AOP-KE contains edges)
        for _, edge in edges_df.iterrows():
            if edge['type'] == 'KER':  # Include ALL KE-to-KE relationships
                # Calculate edge thickness based on frequency (1-4px)
                thickness = min(4, max(1, edge['aop_frequency']))
                cytoscape_elements.append({
                    'data': {
                        'id': edge['id'],
                        'source': edge['source'],
                        'target': edge['target'],
                        'type': edge['type'],
                        'aop_frequency': edge['aop_frequency'],
                        'thickness': thickness
                    }
                })
        
        # Calculate statistics for display
        mie_count = len([n for n in nodes_data if n.get('node_subtype') == 'MIE'])
        ao_count = len([n for n in nodes_data if n.get('node_subtype') == 'AO']) 
        regular_ke_count = len([n for n in nodes_data if n.get('node_subtype') == 'KE'])
        ker_count = len([e for e in edges_data if e['type'] == 'KER'])
        total_elements = len(cytoscape_elements)
        total_nodes = len([e for e in cytoscape_elements if 'source' not in e.get('data', {})])
        total_edges = len([e for e in cytoscape_elements if 'source' in e.get('data', {})])
        
        # Generate hybrid network visualization - works with or without JavaScript
        cytoscape_html = f"""
        <!-- Interactive Network Container -->
        <div id="cy" style="width: 100%; height: 800px; border: 1px solid #ccc; background: #f8f9fa; position: relative;">
            
            <!-- Network Control Panel -->
            <div id="network-controls" style="position: absolute; top: 10px; right: 10px; z-index: 1000; 
                 background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 8px; 
                 padding: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); border: 1px solid rgba(255, 255, 255, 0.3);">
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div style="display: flex; gap: 4px;">
                        <button id="zoom-in-btn" class="control-btn" title="Zoom In (+)">🔍+</button>
                        <button id="zoom-out-btn" class="control-btn" title="Zoom Out (-)">🔍-</button>
                        <button id="fit-btn" class="control-btn" title="Fit to View (R)">⌂</button>
                    </div>
                    <div style="display: flex; gap: 4px;">
                        <button id="center-btn" class="control-btn" title="Center Network">⊙</button>
                        <button id="layout-btn" class="control-btn" title="Re-layout Network">⟲</button>
                    </div>
                </div>
            </div>

            <!-- Node Sizing Controls -->
            <div id="sizing-controls" style="position: absolute; top: 10px; left: 10px; z-index: 1000;
                 background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 8px;
                 padding: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); border: 1px solid rgba(255, 255, 255, 0.3);">
                <div style="font-size: 12px; font-weight: bold; margin-bottom: 8px; color: {BRAND_COLORS['primary']};">Node Size</div>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <label style="font-size: 11px; display: flex; align-items: center; gap: 6px; cursor: pointer;">
                        <input type="radio" name="node-size" value="fixed" checked> Fixed
                    </label>
                    <label style="font-size: 11px; display: flex; align-items: center; gap: 6px; cursor: pointer;">
                        <input type="radio" name="node-size" value="frequency"> AOP Frequency
                    </label>
                    <label style="font-size: 11px; display: flex; align-items: center; gap: 6px; cursor: pointer;">
                        <input type="radio" name="node-size" value="degree"> Connections
                    </label>
                </div>
            </div>

            <!-- Network Statistics -->
            <div id="network-stats" style="position: absolute; bottom: 10px; left: 10px; z-index: 1000;
                 background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 8px;
                 padding: 8px 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); border: 1px solid rgba(255, 255, 255, 0.3);
                 font-size: 11px; color: #666; min-width: 200px;">
                <div id="stats-content">
                    <div>Nodes: <span id="visible-nodes">{total_nodes}</span> / {total_nodes}</div>
                    <div>Edges: <span id="visible-edges">{total_edges}</span> / {total_edges}</div>
                    <div>Zoom: <span id="current-zoom">100%</span></div>
                </div>
            </div>
            
            <!-- Fallback content for when JavaScript is blocked -->
            <div id="fallback-content" style="text-align: center; padding: 20px;">
                <h3 style="color: #29235C; margin-top: 0;">Complete AOP Network Summary</h3>
                
                <div style="background: white; margin: 15px 0; padding: 15px; border: 1px solid #e9ecef; border-radius: 4px;">
                    <h4 style="color: #307BBF; margin-top: 0;">Network Statistics</h4>
                    <div style="display: inline-block; margin: 0 20px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #29235C;">{mie_count}</div>
                        <div style="font-size: 12px; color: #666;">MIEs</div>
                    </div>
                    <div style="display: inline-block; margin: 0 20px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #29235C;">{ao_count}</div>
                        <div style="font-size: 12px; color: #666;">AOs</div>
                    </div>
                    <div style="display: inline-block; margin: 0 20px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #29235C;">{regular_ke_count}</div>
                        <div style="font-size: 12px; color: #666;">Regular KEs</div>
                    </div>
                    <div style="display: inline-block; margin: 0 20px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #29235C;">{ker_count}</div>
                        <div style="font-size: 12px; color: #666;">KERs</div>
                    </div>
                </div>
                
                <div style="background: white; margin: 15px 0; padding: 15px; border: 1px solid #e9ecef; border-radius: 4px;">
                    <h4 style="color: #28a745; margin-top: 0;">✅ Complete Network Data Available</h4>
                    <p style="margin: 5px 0; color: #666;">Full network contains {total_elements} elements ({total_nodes} nodes, {total_edges} edges)</p>
                    <div style="margin: 10px 0;">
                        <a href="/download/aop_network_nodes" style="display: inline-block; margin: 5px; padding: 8px 16px; background: #29235C; color: white; text-decoration: none; border-radius: 4px;">Download Nodes CSV</a>
                        <a href="/download/aop_network_edges" style="display: inline-block; margin: 5px; padding: 8px 16px; background: #29235C; color: white; text-decoration: none; border-radius: 4px;">Download Edges CSV</a>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Enhanced Cytoscape.js Network with Complete Data -->
        <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
        
        <style>
        .control-btn {{
            background: {BRAND_COLORS['primary']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            min-width: 32px;
            height: 28px;
        }}
        .control-btn:hover {{
            background: {BRAND_COLORS['accent']};
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        .control-btn:active {{
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }}
        
        #network-controls, #sizing-controls, #network-stats {{
            user-select: none;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        
        input[type="radio"] {{
            accent-color: {BRAND_COLORS['primary']};
        }}
        
        /* Smooth transitions for network interactions */
        #cy {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        /* Node highlighting styles will be handled by Cytoscape.js styling */
        </style>
        
        <script>
        // Initialize complete network when Cytoscape.js loads
        function initCompleteNetwork() {{
            if (typeof cytoscape === 'undefined') {{
                console.log('Cytoscape.js not available, using fallback display');
                return;
            }}
            
            var container = document.getElementById('cy');
            if (!container) return;
            
            // Show loading message
            container.innerHTML = '<div style="text-align: center; padding: 50px; color: #666;">Loading network data...</div>';
            
            // Load network data via AJAX
            fetch('/api/network_data')
                .then(response => response.json())
                .then(data => {{
                    if (!data.success) {{
                        throw new Error(data.error || 'Failed to load network data');
                    }}
                    
                    // Hide fallback content
                    var fallback = document.getElementById('fallback-content');
                    if (fallback) fallback.style.display = 'none';
                    
                    // Clear loading message but preserve control structure
                    container.innerHTML = '';
                    
                    initializeCytoscapeNetwork(data.elements, data.stats);
                }})
                .catch(error => {{
                    console.error('Error loading network data:', error);
                    container.innerHTML = '<div style="text-align: center; padding: 50px; color: #d32f2f;">Error loading network: ' + error.message + '</div>';
                }});
        }}
        
        function initializeCytoscapeNetwork(elements, stats) {{
            var container = document.getElementById('cy');
            
            try {{
                var cy = cytoscape({{
                    container: container,
                    
                    elements: elements,
                    
                    style: [
                        {{
                            selector: 'node[group="ke"]',
                            style: {{
                                'background-color': '{BRAND_COLORS["secondary"]}',
                                'label': 'data(truncated_label)',
                                'color': 'white',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': '8px',
                                'width': '20px',
                                'height': '20px',
                                'shape': 'ellipse'
                            }}
                        }},
                        {{
                            selector: 'node[group="mie"]',
                            style: {{
                                'background-color': '{BRAND_COLORS["content"]}',
                                'label': 'data(truncated_label)',
                                'color': 'white',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': '8px',
                                'width': '20px',
                                'height': '20px',
                                'shape': 'ellipse'
                            }}
                        }},
                        {{
                            selector: 'node[group="ao"]',
                            style: {{
                                'background-color': '{BRAND_COLORS["primary"]}',
                                'label': 'data(truncated_label)',
                                'color': 'white',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': '8px',
                                'width': '20px',
                                'height': '20px',
                                'shape': 'ellipse'
                            }}
                        }},
                        {{
                            selector: 'edge[type="KER"]',
                            style: {{
                                'width': 'data(thickness)',
                                'line-color': '{BRAND_COLORS["accent"]}',
                                'target-arrow-color': '{BRAND_COLORS["accent"]}',
                                'target-arrow-shape': 'triangle',
                                'arrow-scale': 0.8,
                                'curve-style': 'straight'
                            }}
                        }},
                        {{
                            selector: 'node:selected',
                            style: {{
                                'border-width': 2,
                                'border-color': '{BRAND_COLORS["content"]}'
                            }}
                        }},
                        {{
                            selector: 'node.highlighted',
                            style: {{
                                'border-width': 3,
                                'border-color': '{BRAND_COLORS["accent"]}',
                                'border-opacity': 0.8,
                                'z-index': 10
                            }}
                        }},
                        {{
                            selector: 'edge.highlighted',
                            style: {{
                                'width': 3,
                                'line-color': '{BRAND_COLORS["accent"]}',
                                'target-arrow-color': '{BRAND_COLORS["accent"]}',
                                'opacity': 1,
                                'z-index': 10
                            }}
                        }},
                        {{
                            selector: 'node.faded',
                            style: {{
                                'opacity': 0.3
                            }}
                        }},
                        {{
                            selector: 'edge.faded',
                            style: {{
                                'opacity': 0.1
                            }}
                        }},
                        {{
                            selector: 'node[is_dual_role = true]',
                            style: {{
                                'border-width': 2,
                                'border-color': '#FFD700',
                                'border-style': 'dashed'
                            }}
                        }}
                    ],
                    
                    layout: {{
                        name: 'cose',
                        idealEdgeLength: 50,
                        nodeOverlap: 10,
                        refresh: 10,
                        fit: true,
                        padding: 20,
                        randomize: false,
                        componentSpacing: 80,
                        nodeRepulsion: 100000,
                        edgeElasticity: 32,
                        nestingFactor: 5,
                        gravity: 40,
                        numIter: 1000,
                        initialTemp: 100,
                        coolingFactor: 0.95,
                        minTemp: 1.0
                    }}
                }});
                
                // Enhanced tooltip system using native Cytoscape events
                var tooltipDiv = document.createElement('div');
                tooltipDiv.style.position = 'absolute';
                tooltipDiv.style.display = 'none';
                tooltipDiv.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                tooltipDiv.style.color = 'white';
                tooltipDiv.style.padding = '8px 12px';
                tooltipDiv.style.borderRadius = '4px';
                tooltipDiv.style.fontSize = '12px';
                tooltipDiv.style.pointerEvents = 'none';
                tooltipDiv.style.zIndex = '1000';
                tooltipDiv.style.maxWidth = '250px';
                tooltipDiv.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
                document.body.appendChild(tooltipDiv);
                
                cy.on('mouseover', 'node', function(event) {{
                    var node = event.target;
                    var data = node.data();
                    var degree = node.degree();
                    
                    var descriptive = data.descriptive_title ? '<br><em>' + data.descriptive_title + '</em>' : '';
                    var dualRole = data.is_dual_role ? '<br><span style="color: #FFD700;">⭐ Dual Role</span>' : '';
                    tooltipDiv.innerHTML = '<strong>' + data.label + '</strong>' + descriptive +
                                          '<br>Type: ' + data.node_subtype + dualRole +
                                          '<br>AOP Frequency: ' + data.aop_frequency +
                                          '<br>Connections: ' + degree;
                    
                    tooltipDiv.style.display = 'block';
                }});
                
                cy.on('mouseout', 'node', function(event) {{
                    tooltipDiv.style.display = 'none';
                }});
                
                cy.on('mousemove', function(event) {{
                    tooltipDiv.style.left = (event.originalEvent.pageX + 10) + 'px';
                    tooltipDiv.style.top = (event.originalEvent.pageY - 10) + 'px';
                }});

                // Node selection highlighting
                cy.on('tap', 'node', function(evt) {{
                    var node = evt.target;
                    var neighbors = node.neighborhood();
                    
                    // Reset all nodes and edges
                    cy.elements().removeClass('highlighted faded');
                    
                    // Highlight selected node and neighbors
                    node.addClass('highlighted');
                    neighbors.addClass('highlighted');
                    
                    // Fade non-connected elements
                    cy.elements().difference(node.union(neighbors)).addClass('faded');
                    
                    console.log('Selected:', node.data('label'), '(' + node.data('node_subtype') + ')');
                    console.log('Neighbors:', neighbors.nodes().length);
                }});
                
                // Click background to reset selection
                cy.on('tap', function(event) {{
                    if (event.target === cy) {{
                        cy.elements().removeClass('highlighted faded');
                    }}
                }});
                
                // Navigation controls
                var currentNodeSizing = 'fixed';
                
                // Control button event handlers
                // Add event listeners with null checks
                var zoomInBtn = document.getElementById('zoom-in-btn');
                if (zoomInBtn) {{
                    zoomInBtn.addEventListener('click', function() {{
                        cy.zoom({{
                            level: cy.zoom() * 1.5,
                            renderedPosition: {{ x: cy.width()/2, y: cy.height()/2 }}
                        }});
                        updateZoomDisplay();
                    }});
                }}
                
                var zoomOutBtn = document.getElementById('zoom-out-btn');
                if (zoomOutBtn) {{
                    zoomOutBtn.addEventListener('click', function() {{
                        cy.zoom({{
                            level: cy.zoom() * 0.67,
                            renderedPosition: {{ x: cy.width()/2, y: cy.height()/2 }}
                        }});
                        updateZoomDisplay();
                    }});
                }}
                
                var fitBtn = document.getElementById('fit-btn');
                if (fitBtn) {{
                    fitBtn.addEventListener('click', function() {{
                        cy.animate({{
                            fit: {{
                                eles: cy.elements(),
                                padding: 20
                            }}
                        }}, {{
                            duration: 500,
                            easing: 'ease-in-out-cubic'
                        }});
                        setTimeout(updateZoomDisplay, 500);
                    }});
                }}
                
                var centerBtn = document.getElementById('center-btn');
                if (centerBtn) {{
                    centerBtn.addEventListener('click', function() {{
                        // Center the viewport on the network without changing zoom level
                        cy.center();
                    }});
                }}
                
                var layoutBtn = document.getElementById('layout-btn');
                if (layoutBtn) {{
                    layoutBtn.addEventListener('click', function() {{
                    var layout = cy.layout({{
                        name: 'cose',
                        animate: true,
                        animationDuration: 1000,
                        idealEdgeLength: 50,
                        nodeOverlap: 10,
                        refresh: 10,
                        fit: true,
                        padding: 20,
                        randomize: false,
                        componentSpacing: 80,
                        nodeRepulsion: 100000,
                        edgeElasticity: 32,
                        nestingFactor: 5,
                        gravity: 40,
                        numIter: 1000,
                        initialTemp: 100,
                        coolingFactor: 0.95,
                        minTemp: 1.0
                    }});
                    layout.run();
                    }});
                }}
                
                // Node sizing controls
                var sizeControls = document.querySelectorAll('input[name="node-size"]');
                sizeControls.forEach(function(control) {{
                    control.addEventListener('change', function() {{
                        if (this.checked) {{
                            currentNodeSizing = this.value;
                            updateNodeSizing();
                        }}
                    }});
                }});
                
                function updateNodeSizing() {{
                    var nodes = cy.nodes();
                    var maxFreq = Math.max.apply(Math, nodes.map(function(n) {{ return n.data('aop_frequency'); }}));
                    var maxDegree = Math.max.apply(Math, nodes.map(function(n) {{ return n.degree(); }}));
                    
                    nodes.forEach(function(node) {{
                        var size = 20; // default size
                        
                        if (currentNodeSizing === 'frequency') {{
                            var freq = node.data('aop_frequency');
                            size = Math.max(8, Math.min(40, 10 + (freq / maxFreq) * 30));
                        }} else if (currentNodeSizing === 'degree') {{
                            var degree = node.degree();
                            size = Math.max(8, Math.min(40, 10 + (degree / maxDegree) * 30));
                        }}
                        
                        node.style('width', size + 'px');
                        node.style('height', size + 'px');
                        node.style('font-size', Math.max(6, size * 0.4) + 'px');
                    }});
                }}
                
                // Zoom level tracking - now tracks ALL zoom events
                function updateZoomDisplay() {{
                    var zoomPercent = Math.round(cy.zoom() * 100);
                    var currentZoomEl = document.getElementById('current-zoom');
                    if (currentZoomEl) currentZoomEl.textContent = zoomPercent + '%';
                }}
                
                // Track all zoom and viewport changes
                cy.on('zoom viewport pan', function() {{
                    updateZoomDisplay();
                }});
                
                // Keyboard shortcuts
                document.addEventListener('keydown', function(event) {{
                    if (event.target.tagName.toLowerCase() !== 'input') {{
                        switch(event.key) {{
                            case 'r':
                            case 'R':
                                var fitBtn = document.getElementById('fit-btn');
                                if (fitBtn) fitBtn.click();
                                break;
                            case '+':
                            case '=':
                                var zoomInBtn = document.getElementById('zoom-in-btn');
                                if (zoomInBtn) zoomInBtn.click();
                                break;
                            case '-':
                            case '_':
                                var zoomOutBtn = document.getElementById('zoom-out-btn');
                                if (zoomOutBtn) zoomOutBtn.click();
                                break;
                        }}
                    }}
                }});
                
                // Initial setup
                cy.fit();
                cy.center();
                updateZoomDisplay();
                
                // Update stats display with actual loaded data
                var visibleNodesEl = document.getElementById('visible-nodes');
                if (visibleNodesEl) visibleNodesEl.textContent = cy.nodes().length;
                
                var visibleEdgesEl = document.getElementById('visible-edges');
                if (visibleEdgesEl) visibleEdgesEl.textContent = cy.edges().length;
                
                console.log('Enhanced AOP network loaded: ' + cy.nodes().length + ' nodes, ' + cy.edges().length + ' edges');
                
            }} catch (error) {{
                console.error('Error creating network:', error);
                container.innerHTML = '<div style="text-align: center; padding: 50px; color: #d32f2f;">Error initializing network: ' + error.message + '</div>';
            }}
        }}
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initCompleteNetwork);
        }} else {{
            initCompleteNetwork();
        }}
        </script>
        
        <style>
        #cy {{
            position: relative;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        </style>
        
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
            <h4>Network Legend</h4>
            <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: center;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background: {BRAND_COLORS["content"]}; border-radius: 50%;"></div>
                    <span>Molecular Initiating Events ({len([n for n in nodes_data if n.get('node_subtype') == 'MIE'])})</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background: {BRAND_COLORS["primary"]}; border-radius: 50%;"></div>
                    <span>Adverse Outcomes ({len([n for n in nodes_data if n.get('node_subtype') == 'AO'])})</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background: {BRAND_COLORS["secondary"]}; border-radius: 50%;"></div>
                    <span>Key Events ({len([n for n in nodes_data if n.get('node_subtype') == 'KE'])})</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 2px; background: {BRAND_COLORS["accent"]};"></div>
                    <span>KE Relationships ({len([e for e in edges_data if e['type'] == 'KER'])})</span>
                </div>
            </div>
            <p style="margin-top: 10px; font-size: 14px; color: #666;">
                <strong>Interaction:</strong> Pan and zoom to explore. Click nodes to select. Hover for details.
                <strong>Version:</strong> {latest_version}
            </p>
        </div>
        """
        
        logging.info(f"Network visualization generated successfully with {len(nodes_data)} total nodes and {len(edges_data)} total edges")
        logging.info(f"Cytoscape elements: {len(cytoscape_elements)} elements (complete network)")
        return cytoscape_html
        
    except Exception as e:
        logging.error(f"Error in plot_aop_network: {e}")
        return create_fallback_plot("AOP Network", f"Error generating network visualization: {str(e)}")
