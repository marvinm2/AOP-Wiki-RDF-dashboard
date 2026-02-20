# Technology Stack

**Analysis Date:** 2026-02-20

## Languages

**Primary:**
- Python 3.11 - All backend application code and data visualization logic

**Secondary:**
- HTML5 - Web interface templates
- CSS3 - Styling and responsive design
- JavaScript - Frontend interactivity and AJAX requests

## Runtime

**Environment:**
- Python 3.11 (slim Docker image: `python:3.11-slim`)
- Flask WSGI application server

**Package Manager:**
- pip
- Lockfile: `requirements.txt` (pinned versions not specified; uses latest compatible)

## Frameworks

**Core Web:**
- Flask 3.x - Lightweight WSGI web framework for HTTP endpoints and template rendering
  - Location: `app.py` - Main application entry point

**Data Visualization:**
- Plotly 5.x - Interactive visualization library with client-side rendering
  - SVG/PNG/WebP export capabilities
  - Used across `plots/latest_plots.py`, `plots/trends_plots.py`

**Data Processing:**
- Pandas - Data manipulation, filtering, and transformation
  - CSV reading/writing with fallback error handling
  - DataFrame operations for aggregation and analysis

**Semantic Web:**
- SPARQLWrapper 2.x - SPARQL query client for RDF endpoint communication
  - JSON-RDF response handling
  - Exception handling for query failures

## Key Dependencies

**Critical:**
- `plotly` - Interactive visualization rendering (web and image export)
- `pandas` - Data transformation and analysis
- `flask` - HTTP server and request routing
- `SPARQLWrapper` - SPARQL protocol client for RDF data access
- `requests` - HTTP client for health checks and external communication

**Infrastructure:**
- `Werkzeug` (Flask dependency) - WSGI utilities
- `Jinja2` (Flask dependency) - Template rendering for HTML pages

## Configuration

**Environment Variables:**
All configuration loaded from environment variables via `config.py`:

- `SPARQL_ENDPOINT` - URL of SPARQL endpoint (default: `http://localhost:8890/sparql`)
- `SPARQL_TIMEOUT` - Query timeout in seconds (default: `30`)
- `SPARQL_MAX_RETRIES` - Retry attempts for failed queries (default: `3`)
- `SPARQL_RETRY_DELAY` - Delay between retries in seconds (default: `2`)
- `PARALLEL_WORKERS` - Concurrent plot generation threads (default: `5`)
- `PLOT_TIMEOUT` - Individual plot generation timeout in seconds (default: `60`)
- `LOG_LEVEL` - Logging verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: `INFO`)
- `FLASK_HOST` - Server bind address (default: `0.0.0.0`)
- `FLASK_PORT` - Server port number (default: `5000`)
- `FLASK_DEBUG` - Enable Flask debug mode (default: `False`)
- `ENABLE_HEALTH_CHECK` - Enable `/health` and `/status` endpoints (default: `True`)
- `ENABLE_PERFORMANCE_LOGGING` - Enable performance metrics logging (default: `True`)

**Build Configuration:**
- `Dockerfile` - Multi-stage container image for Python 3.11
- `docker-compose.yml` - Single service composition for local development
  - Maps port 5000:5000
  - Sets SPARQL_ENDPOINT to Docker host bridge: `http://172.17.0.1:8890/sparql`

**Configuration Files:**
- `config.py` - Centralized configuration class with validation
  - Loads from environment variables
  - Validates numeric ranges and URL formats
  - Provides `Config.get_config_dict()` for runtime inspection
  - Provides `Config.validate_config()` for startup validation

## Platform Requirements

**Development:**
- Python 3.11+
- Docker and Docker Compose (optional, for containerized development)
- SPARQL endpoint accessible at `http://localhost:8890/sparql` or via `SPARQL_ENDPOINT` env var
- Property labels CSV file: `property_labels.csv` (optional, has fallback)

**Production:**
- Docker container runtime (recommended)
- External SPARQL 1.1 endpoint compatible with Virtuoso or similar
- Minimum 512MB RAM for Flask + plotting workload
- Network connectivity to SPARQL endpoint

**Performance Tuning:**
- Parallel workers configurable via `PARALLEL_WORKERS` (default: 5)
- SPARQL timeout configurable via `SPARQL_TIMEOUT` (default: 30s)
- Plot generation timeout configurable via `PLOT_TIMEOUT` (default: 60s)

---

*Stack analysis: 2026-02-20*
