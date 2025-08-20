AOP-Wiki RDF Dashboard Documentation
=====================================

Welcome to the AOP-Wiki RDF Dashboard documentation. This Flask-based application provides 
interactive visualizations for analyzing Adverse Outcome Pathway (AOP) data evolution over time.

The dashboard features comprehensive SPARQL integration, professional visualizations with 
official house style branding, and extensive CSV export capabilities for all charts.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   quickstart
   configuration
   api

Key Features
------------

* **Interactive Dashboard**: Professional web interface with tabbed navigation
* **Historical Analysis**: Time-series visualizations showing data evolution
* **Latest Data Snapshots**: Current state analysis and entity counting
* **SPARQL Integration**: Robust RDF query system with retry logic
* **CSV Export System**: Download data behind every visualization
* **Performance Optimized**: Parallel plot generation and caching
* **Error Resilient**: Graceful degradation with fallback visualizations

Quick Start
-----------

1. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

2. Configure SPARQL endpoint:

   .. code-block:: bash

      export SPARQL_ENDPOINT=http://localhost:8890/sparql

3. Run the application:

   .. code-block:: bash

      python app.py

4. Open http://localhost:5000 in your browser

Architecture Overview
--------------------

The application consists of three main components:

**app.py**
    Flask application with parallel plot computation and CSV download endpoints

**plots.py** 
    Comprehensive visualization library with SPARQL integration and data processing

**config.py**
    Centralized configuration management with validation and environment variables

Visualization Categories
-----------------------

**Historical Trends**
    * Entity evolution over time (AOPs, KEs, KERs, Stressors)
    * Network density analysis
    * Author contribution patterns
    * Component annotation trends

**Latest Data Analysis**
    * Current entity counts and distribution
    * Data completeness assessment
    * Ontology usage analysis
    * Network connectivity metrics

**Specialized Analysis**
    * AOP lifecycle analysis
    * Property presence trends
    * Biological process/object ontology usage
    * Key Event annotation depth

Performance Features
-------------------

* **Parallel Processing**: Concurrent plot generation using ThreadPoolExecutor
* **Global Caching**: Data cached for CSV exports without recomputation
* **Error Handling**: Individual plot failures don't affect overall system
* **Health Monitoring**: Real-time endpoint status and performance tracking

Brand Integration
----------------

All visualizations use the official house style color palette:

**Primary Colors:**

* Primary Dark: #29235C (Main brand color)
* Primary Magenta: #E6007E (Accent color)  
* Primary Blue: #307BBF (Supporting blue)

**Secondary Colors:**

* Light Blue: #009FE3
* Orange: #EB5B25 (Content highlight)
* Sky Blue: #93D5F6 (Light accent)
* Deep Magenta: #9A1C57
* Teal: #45A6B2
* Purple: #B81178
* Dark Teal: #005A6C
* Violet: #64358C

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`