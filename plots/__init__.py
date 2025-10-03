"""AOP-Wiki RDF Dashboard - Plot Functions Module.

This module provides a comprehensive suite of visualization functions for the AOP-Wiki
RDF Dashboard. It combines historical trend analysis and current snapshot visualizations
to provide complete insights into AOP-Wiki data evolution and current state.

Module Structure:
    - shared: Common utilities, SPARQL helpers, and constants
    - trends_plots: Historical and time-series analysis functions
    - latest_plots: Current snapshot and state analysis functions

Available Plot Functions:

Historical Trends (trends_plots):
    Core Evolution:
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

Current Snapshots (latest_plots):
    Database State:
        - plot_latest_entity_counts(): Current entity distribution
        - plot_latest_database_summary(): Core entity summary
        - plot_latest_avg_per_aop(): Current average components per AOP

    Component Analysis:
        - plot_latest_ke_components(): Current KE component distribution
        - plot_latest_ke_annotation_depth(): Current annotation depth analysis

    Connectivity:
        - plot_latest_network_density(): Current AOP connectivity

    Ontology Usage:
        - plot_latest_ontology_usage(): Current ontology term usage
        - plot_latest_process_usage(): Current biological process sources
        - plot_latest_object_usage(): Current biological object sources

    Data Quality:
        - plot_latest_aop_completeness(): Current property completeness
        - plot_latest_aop_completeness_unique_colors(): Enhanced completeness view

Shared Utilities (shared):
    Core Functions:
        - run_sparql_query_with_retry(): Robust SPARQL execution
        - extract_counts(): Data extraction and processing
        - safe_plot_execution(): Error-safe plot generation
        - check_sparql_endpoint_health(): Endpoint monitoring
        - create_fallback_plot(): Error visualization generation

    Constants and Configuration:
        - BRAND_COLORS: Official VHP4Safety color palette
        - config: Plotly configuration settings
        - _plot_data_cache: Global data cache for CSV exports

Features:
    - Consistent VHP4Safety branding across all visualizations
    - Comprehensive error handling and graceful degradation
    - Global data caching system for CSV export functionality
    - Performance monitoring and optimization
    - Responsive design for desktop and mobile devices

Usage Examples:
    Basic plot generation:
    >>> from plots import plot_latest_entity_counts
    >>> html_plot = plot_latest_entity_counts()

    Historical analysis:
    >>> from plots import plot_main_graph
    >>> abs_plot, delta_plot, data = plot_main_graph()

    Safe execution:
    >>> from plots import safe_plot_execution, plot_latest_ke_components
    >>> result = safe_plot_execution(plot_latest_ke_components)

    Data access:
    >>> from plots import _plot_data_cache
    >>> if 'latest_entity_counts' in _plot_data_cache:
    ...     df = _plot_data_cache['latest_entity_counts']

Author:
    Generated with Claude Code (https://claude.ai/code)
"""

# Import shared utilities and constants
from .shared import (
    # Core SPARQL functions
    run_sparql_query,
    run_sparql_query_with_retry,
    extract_counts,
    check_sparql_endpoint_health,
    get_latest_version,
    get_all_versions,

    # Utility functions
    safe_read_csv,
    safe_plot_execution,
    create_fallback_plot,
    export_figure_as_image,
    get_csv_with_metadata,
    create_bulk_download,

    # Constants and configuration
    BRAND_COLORS,
    config,
    _plot_data_cache,
    _plot_figure_cache,
)

# Import all historical trend plot functions
from .trends_plots import (
    # Core evolution analysis
    plot_main_graph,
    plot_avg_per_aop,
    plot_network_density,
    plot_author_counts,

    # Component analysis
    plot_ke_components,
    plot_ke_components_percentage,
    plot_unique_ke_components,
    plot_kes_by_kec_count,

    # Ontology usage analysis
    plot_bio_processes,
    plot_bio_objects,

    # Property analysis
    plot_aop_property_presence,
    plot_aop_property_presence_unique_colors,

    # Temporal analysis
    plot_aop_lifetime,
)

# Import all current snapshot plot functions
from .latest_plots import (
    # Database state analysis
    plot_latest_entity_counts,
    plot_latest_database_summary,
    plot_latest_avg_per_aop,

    # Component analysis
    plot_latest_ke_components,
    plot_latest_ke_annotation_depth,

    # Connectivity analysis
    plot_latest_network_density,

    # Ontology usage analysis
    plot_latest_ontology_usage,
    plot_latest_process_usage,
    plot_latest_object_usage,

    # Data quality analysis
    plot_latest_aop_completeness,
    plot_latest_aop_completeness_unique_colors,
)

# Define module version and metadata
__version__ = "2.0.0"
__author__ = "Claude Code (https://claude.ai/code)"

# Define public API - all functions that should be importable
__all__ = [
    # Shared utilities
    'run_sparql_query',
    'run_sparql_query_with_retry',
    'extract_counts',
    'check_sparql_endpoint_health',
    'get_latest_version',
    'get_all_versions',
    'safe_read_csv',
    'safe_plot_execution',
    'create_fallback_plot',
    'export_figure_as_image',
    'get_csv_with_metadata',
    'create_bulk_download',
    'BRAND_COLORS',
    'config',
    '_plot_data_cache',
    '_plot_figure_cache',

    # Historical trends functions
    'plot_main_graph',
    'plot_avg_per_aop',
    'plot_network_density',
    'plot_author_counts',
    'plot_ke_components',
    'plot_ke_components_percentage',
    'plot_unique_ke_components',
    'plot_kes_by_kec_count',
    'plot_bio_processes',
    'plot_bio_objects',
    'plot_aop_property_presence',
    'plot_aop_property_presence_unique_colors',
    'plot_aop_lifetime',

    # Current snapshot functions
    'plot_latest_entity_counts',
    'plot_latest_database_summary',
    'plot_latest_avg_per_aop',
    'plot_latest_ke_components',
    'plot_latest_ke_annotation_depth',
    'plot_latest_network_density',
    'plot_latest_ontology_usage',
    'plot_latest_process_usage',
    'plot_latest_object_usage',
    'plot_latest_aop_completeness',
    'plot_latest_aop_completeness_unique_colors',
]

# Module-level convenience functions
def get_available_functions():
    """Return a dictionary of all available plot functions organized by category.

    Returns:
        dict: Dictionary with categories as keys and lists of function names as values.
    """
    return {
        'historical_trends': [
            'plot_main_graph',
            'plot_avg_per_aop',
            'plot_network_density',
            'plot_author_counts',
            'plot_ke_components',
            'plot_ke_components_percentage',
            'plot_unique_ke_components',
            'plot_kes_by_kec_count',
            'plot_bio_processes',
            'plot_bio_objects',
            'plot_aop_property_presence',
            'plot_aop_property_presence_unique_colors',
            'plot_aop_lifetime',
        ],
        'current_snapshots': [
            'plot_latest_entity_counts',
            'plot_latest_database_summary',
            'plot_latest_avg_per_aop',
            'plot_latest_ke_components',
            'plot_latest_ke_annotation_depth',
            'plot_latest_network_density',
            'plot_latest_ontology_usage',
            'plot_latest_process_usage',
            'plot_latest_object_usage',
            'plot_latest_aop_completeness',
            'plot_latest_aop_completeness_unique_colors',
        ],
        'utilities': [
            'run_sparql_query',
            'run_sparql_query_with_retry',
            'extract_counts',
            'check_sparql_endpoint_health',
            'safe_read_csv',
            'safe_plot_execution',
            'create_fallback_plot',
        ]
    }


def get_cached_data_keys():
    """Return a list of all available cached data keys for CSV export.

    Returns:
        list: List of keys available in _plot_data_cache.
    """
    return list(_plot_data_cache.keys())


def clear_plot_cache():
    """Clear all cached plot data.

    This function clears the global _plot_data_cache dictionary,
    freeing memory and forcing regeneration of cached data on next access.
    """
    _plot_data_cache.clear()