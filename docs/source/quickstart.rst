Quick Start Guide
=================

This guide will help you get the AOP-Wiki RDF Dashboard up and running quickly.

Prerequisites
-------------

* Python 3.8 or higher
* Access to a SPARQL endpoint with AOP-Wiki RDF data
* Web browser for accessing the dashboard

Installation
------------

1. **Clone or download the project**

2. **Install Python dependencies:**

   .. code-block:: bash

      pip install -r requirements.txt

   The main dependencies include:
   
   * Flask (web framework)
   * pandas (data processing)
   * plotly (visualizations)
   * SPARQLWrapper (RDF queries)

Configuration
-------------

The application uses environment variables for configuration. You can either:

**Option 1: Set environment variables**

.. code-block:: bash

   export SPARQL_ENDPOINT=http://localhost:8890/sparql
   export FLASK_HOST=0.0.0.0
   export FLASK_PORT=5000
   export LOG_LEVEL=INFO

**Option 2: Use defaults**

The application will use sensible defaults if no environment variables are set:

* SPARQL endpoint: http://localhost:8890/sparql
* Host: 0.0.0.0 
* Port: 5000
* Log level: INFO

Running the Application
-----------------------

**Development mode:**

.. code-block:: bash

   python app.py

**Production mode with gunicorn:**

.. code-block:: bash

   pip install gunicorn
   gunicorn --bind 0.0.0.0:5000 app:app

**Docker deployment:**

.. code-block:: bash

   docker build -t aop-dashboard .
   docker run -p 5000:5000 \
     -e SPARQL_ENDPOINT=http://your-endpoint:8890/sparql \
     aop-dashboard

Accessing the Dashboard
-----------------------

Once running, open your web browser and navigate to:

* Local development: http://localhost:5000
* Network access: http://your-server-ip:5000

You should see the AOP-Wiki RDF Dashboard with two main tabs:

1. **Latest Data**: Current snapshot of entity counts and analysis
2. **Historical Trends**: Time-series analysis of data evolution

Features Overview
-----------------

**Interactive Visualizations**
   All charts are interactive with hover details, zoom, and pan capabilities

**CSV Downloads**
   Every visualization has a download button for accessing the underlying data

**Health Monitoring**
   Check application status at http://localhost:5000/health

**Real-time Status**
   Monitor system health at http://localhost:5000/status

Troubleshooting
---------------

**No plots appearing:**
   
* Check that your SPARQL endpoint is accessible
* Verify the endpoint contains AOP-Wiki RDF data
* Check application logs for connection errors

**Slow loading:**
   
* Reduce PARALLEL_WORKERS if running on limited hardware
* Check network connectivity to SPARQL endpoint
* Consider increasing SPARQL_TIMEOUT for large datasets

**CSV downloads not working:**
   
* Ensure plots generated successfully during startup
* Check browser console for JavaScript errors
* Verify Flask Response objects are properly formatted

Performance Tuning
-------------------

For optimal performance, consider these environment variables:

.. code-block:: bash

   # Increase workers for faster startup (if you have CPU cores)
   export PARALLEL_WORKERS=8
   
   # Increase timeouts for large datasets
   export SPARQL_TIMEOUT=60
   export PLOT_TIMEOUT=120
   
   # Tune retry behavior
   export SPARQL_MAX_RETRIES=5
   export SPARQL_RETRY_DELAY=3