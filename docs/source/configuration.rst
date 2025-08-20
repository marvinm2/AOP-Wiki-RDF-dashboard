Configuration Guide
==================

The AOP-Wiki RDF Dashboard uses environment variables for configuration, providing 
flexible deployment options while maintaining sensible defaults.

Environment Variables
---------------------

SPARQL Configuration
~~~~~~~~~~~~~~~~~~~

**SPARQL_ENDPOINT**
   * **Default**: ``http://localhost:8890/sparql``
   * **Description**: URL of the SPARQL endpoint containing AOP-Wiki RDF data
   * **Example**: ``export SPARQL_ENDPOINT=http://aopwiki.example.com:8890/sparql``

**SPARQL_TIMEOUT**
   * **Default**: ``30``
   * **Description**: Query timeout in seconds
   * **Range**: 1-300 seconds
   * **Example**: ``export SPARQL_TIMEOUT=60``

**SPARQL_MAX_RETRIES**
   * **Default**: ``3``
   * **Description**: Maximum retry attempts for failed queries
   * **Range**: 1-10 retries
   * **Example**: ``export SPARQL_MAX_RETRIES=5``

**SPARQL_RETRY_DELAY**
   * **Default**: ``2``
   * **Description**: Base delay between retries in seconds (uses exponential backoff)
   * **Range**: 1-10 seconds
   * **Example**: ``export SPARQL_RETRY_DELAY=3``

Flask Web Server
~~~~~~~~~~~~~~~

**FLASK_HOST**
   * **Default**: ``0.0.0.0``
   * **Description**: Host address for Flask server binding
   * **Options**: ``0.0.0.0`` (all interfaces), ``127.0.0.1`` (localhost only)
   * **Example**: ``export FLASK_HOST=127.0.0.1``

**FLASK_PORT**
   * **Default**: ``5000``
   * **Description**: Port number for Flask web server
   * **Range**: 1-65535
   * **Example**: ``export FLASK_PORT=8080``

**FLASK_DEBUG**
   * **Default**: ``False``
   * **Description**: Enable Flask debug mode (development only)
   * **Values**: ``true``, ``false``
   * **Example**: ``export FLASK_DEBUG=true``

Performance Tuning
~~~~~~~~~~~~~~~~~~

**PARALLEL_WORKERS**
   * **Default**: ``5``
   * **Description**: Number of parallel workers for plot generation
   * **Range**: 1-20 (adjust based on CPU cores and memory)
   * **Example**: ``export PARALLEL_WORKERS=8``

**PLOT_TIMEOUT**
   * **Default**: ``60``
   * **Description**: Timeout for individual plot generation in seconds
   * **Range**: 30-300 seconds
   * **Example**: ``export PLOT_TIMEOUT=120``

Logging and Monitoring
~~~~~~~~~~~~~~~~~~~~~

**LOG_LEVEL**
   * **Default**: ``INFO``
   * **Description**: Logging verbosity level
   * **Values**: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
   * **Example**: ``export LOG_LEVEL=DEBUG``

**ENABLE_HEALTH_CHECK**
   * **Default**: ``True``
   * **Description**: Enable health check endpoints (/health, /status)
   * **Values**: ``true``, ``false``
   * **Example**: ``export ENABLE_HEALTH_CHECK=false``

**ENABLE_PERFORMANCE_LOGGING**
   * **Default**: ``True``
   * **Description**: Enable detailed performance logging
   * **Values**: ``true``, ``false``
   * **Example**: ``export ENABLE_PERFORMANCE_LOGGING=false``

Configuration Validation
------------------------

The application automatically validates all configuration settings at startup:

* **URL Validation**: SPARQL endpoint URLs are checked for proper format
* **Range Validation**: Numeric values are validated against acceptable ranges
* **Type Validation**: Boolean values are properly converted from strings
* **Consistency Checks**: Related parameters are validated for logical consistency

If validation fails, the application logs warnings and uses default values.

Deployment Examples
------------------

Development Setup
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Minimal development setup
   export SPARQL_ENDPOINT=http://localhost:8890/sparql
   export FLASK_DEBUG=true
   export LOG_LEVEL=DEBUG
   python app.py

Production Setup
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Production environment
   export SPARQL_ENDPOINT=http://prod-endpoint:8890/sparql
   export FLASK_HOST=0.0.0.0
   export FLASK_PORT=5000
   export LOG_LEVEL=INFO
   export PARALLEL_WORKERS=10
   export SPARQL_TIMEOUT=60
   gunicorn --bind 0.0.0.0:5000 --workers 4 app:app

Docker Environment
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker run -p 5000:5000 \
     -e SPARQL_ENDPOINT=http://endpoint:8890/sparql \
     -e PARALLEL_WORKERS=8 \
     -e SPARQL_TIMEOUT=90 \
     -e LOG_LEVEL=INFO \
     aop-dashboard

Configuration Best Practices
----------------------------

**Security**
   * Use ``FLASK_HOST=127.0.0.1`` for development or private networks
   * Disable debug mode in production (``FLASK_DEBUG=false``)
   * Use reverse proxy (nginx) for production deployments

**Performance**
   * Set ``PARALLEL_WORKERS`` to match available CPU cores
   * Increase timeouts for large datasets or slow networks
   * Use ``LOG_LEVEL=WARNING`` in production for better performance

**Reliability**
   * Increase ``SPARQL_MAX_RETRIES`` for unstable network connections
   * Set appropriate ``SPARQL_TIMEOUT`` based on dataset size
   * Enable health checks for monitoring (default: enabled)

**Monitoring**
   * Keep ``ENABLE_PERFORMANCE_LOGGING=true`` for performance insights
   * Use ``LOG_LEVEL=INFO`` for balanced logging in production
   * Monitor health endpoints for automated monitoring systems

Troubleshooting Configuration
-----------------------------

**Application won't start:**
   * Check SPARQL_ENDPOINT URL format
   * Verify FLASK_PORT is not in use
   * Check file permissions and Python path

**Slow performance:**
   * Increase PARALLEL_WORKERS (if CPU allows)
   * Increase SPARQL_TIMEOUT for large queries
   * Check network latency to SPARQL endpoint

**Connection issues:**
   * Verify SPARQL endpoint accessibility
   * Increase SPARQL_MAX_RETRIES and SPARQL_RETRY_DELAY
   * Check firewall and network configuration