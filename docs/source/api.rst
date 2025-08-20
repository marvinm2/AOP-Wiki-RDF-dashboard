API Documentation
=================

This section provides detailed API documentation for the AOP-Wiki RDF Dashboard's web endpoints and internal functions.

Web API Endpoints
-----------------

Dashboard Routes
~~~~~~~~~~~~~~~

**GET /** 
   Main dashboard page with interactive visualizations
   
   * **Returns**: HTML page with embedded Plotly charts
   * **Content-Type**: text/html
   * **Features**: Tabbed interface, responsive design, CSV download links

Health and Monitoring
~~~~~~~~~~~~~~~~~~~~

**GET /health**
   Health check endpoint for monitoring systems
   
   * **Returns**: JSON health status
   * **Content-Type**: application/json
   * **Status Codes**:
     * 200: Healthy (SPARQL endpoint up, plots loaded)
     * 503: Degraded (endpoint down or plots failed)
     * 500: Error (health check failed)
   
   **Response Format**:
   
   .. code-block:: json
   
      {
        "status": "healthy",
        "sparql_endpoint": "up",
        "plots_loaded": "22/22",
        "timestamp": 1640995200.0
      }

**GET /status**
   Human-readable status monitoring page
   
   * **Returns**: HTML monitoring interface
   * **Content-Type**: text/html
   * **Features**: Real-time metrics, system status, debugging info

CSV Download Endpoints
~~~~~~~~~~~~~~~~~~~~~

All visualization data can be downloaded as CSV files through dedicated endpoints:

**Latest Data Downloads**:

* ``GET /download/latest_entity_counts`` - Current entity counts
* ``GET /download/latest_ke_components`` - KE component distribution  
* ``GET /download/latest_network_density`` - Network connectivity data
* ``GET /download/latest_avg_per_aop`` - Average components per AOP
* ``GET /download/latest_process_usage`` - Process ontology usage
* ``GET /download/latest_object_usage`` - Object ontology usage
* ``GET /download/latest_aop_completeness`` - Data completeness metrics
* ``GET /download/latest_ke_annotation_depth`` - Annotation depth analysis

**Historical Data Downloads**:

* ``GET /download/main_graph_absolute`` - Historical entity counts (absolute)
* ``GET /download/main_graph_delta`` - Historical entity changes (delta)

**Response Format**:
   * **Content-Type**: text/csv
   * **Content-Disposition**: attachment; filename="{plot_name}.csv"
   * **Status Codes**:
     * 200: Success, CSV data returned
     * 404: Data not available (plot generation failed)
     * 500: Server error during CSV generation

Internal API Functions
----------------------

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: config.Config.get_config_dict

.. autofunction:: config.Config.validate_config

SPARQL Query Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: plots.check_sparql_endpoint_health

.. autofunction:: plots.run_sparql_query_with_retry

.. autofunction:: plots.run_sparql_query

.. autofunction:: plots.extract_counts

Data Processing Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: plots.safe_read_csv

.. autofunction:: plots.create_fallback_plot

.. autofunction:: plots.safe_plot_execution

Historical Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: plots.plot_main_graph

.. autofunction:: plots.plot_avg_per_aop

.. autofunction:: plots.plot_network_density

.. autofunction:: plots.plot_ke_components

.. autofunction:: plots.plot_author_counts

.. autofunction:: plots.plot_aop_lifetime

Latest Data Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: plots.plot_latest_entity_counts

.. autofunction:: plots.plot_latest_ke_components

.. autofunction:: plots.plot_latest_network_density

.. autofunction:: plots.plot_latest_avg_per_aop

.. autofunction:: plots.plot_latest_aop_completeness

Application Functions
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: app.compute_plots_parallel

.. autofunction:: app.health_check

.. autofunction:: app.status_page

.. autofunction:: app.index

Global Variables
---------------

**plots._plot_data_cache**
   Global dictionary storing cached DataFrames for CSV downloads
   
   * **Type**: Dict[str, pd.DataFrame]
   * **Keys**: Plot names (e.g., 'latest_entity_counts', 'main_graph_absolute')
   * **Values**: Processed DataFrames with visualization data
   * **Usage**: Populated during plot generation, accessed by download endpoints

**plots.BRAND_COLORS**
   VHP4Safety color palette for consistent visualization styling
   
   * **Type**: Dict[str, Union[str, List[str], Dict[str, str]]]
   * **Structure**:
     * primary, secondary, accent, light, content: Individual colors
     * palette: List of colors for multi-series plots
     * type_colors: Colors mapped to property types

Error Handling
--------------

The API uses consistent error handling patterns:

**HTTP Status Codes**:
   * 200: Success
   * 404: Resource not found (typically missing cached data)
   * 500: Internal server error
   * 503: Service unavailable (health check degraded)

**Error Responses**:
   * JSON endpoints return structured error objects
   * HTML endpoints return user-friendly error pages
   * CSV endpoints return plain text error messages

**Fallback Mechanisms**:
   * Individual plot failures don't affect overall dashboard
   * Missing visualizations show error messages with context
   * Health checks provide detailed failure information

Performance Characteristics
--------------------------

**Startup Performance**:
   * Parallel plot generation using ThreadPoolExecutor
   * Typical startup time: 10-30 seconds (depends on data size)
   * Configurable parallelism via PARALLEL_WORKERS

**Runtime Performance**:
   * Dashboard page: <1 second (precomputed plots)
   * Health check: <2 seconds (includes endpoint test)
   * CSV downloads: <1 second (cached data)

**Memory Usage**:
   * Base application: ~50MB
   * Plot data cache: Variable (typically 10-100MB)
   * Per request: Minimal additional memory

**Scalability**:
   * Single-process Flask application
   * Suitable for small-to-medium user loads
   * Can be deployed with gunicorn for higher concurrency