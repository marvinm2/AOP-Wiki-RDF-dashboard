# External Integrations

**Analysis Date:** 2026-02-20

## APIs & External Services

**SPARQL Endpoint (RDF Data Source):**
- Service: SPARQL 1.1 Protocol-compliant endpoint (typically Virtuoso or similar)
- What it's used for: Querying AOP-Wiki RDF data (AOPs, KEs, KERs, Stressors, properties, annotations)
- SDK/Client: `SPARQLWrapper` (Python package)
- Configuration: `Config.SPARQL_ENDPOINT` (env var or default `http://localhost:8890/sparql`)
- Retry Logic: Exponential backoff with configurable attempts
  - Max retries: `Config.SPARQL_MAX_RETRIES` (default: 3)
  - Retry delay: `Config.SPARQL_RETRY_DELAY` (default: 2 seconds)
  - Query timeout: `Config.SPARQL_TIMEOUT` (default: 30 seconds)
- Implementation: `plots/shared.py` - `run_sparql_query_with_retry()`

**SPARQL Health Monitoring:**
- Service: Same SPARQL endpoint
- What it's used for: Endpoint availability checks via `/health` and `/status` endpoints
- Implementation: `plots/shared.py` - `check_sparql_endpoint_health()`
- HTTP Client: `requests` library

## Data Storage

**Databases:**
- Type: RDF Graph Database (Virtuoso or compatible SPARQL 1.1 endpoint)
  - Connection: Via SPARQL HTTP protocol over network
  - Client: `SPARQLWrapper` with JSON response format
  - No ORM used (direct SPARQL queries)

**File Storage:**
- Local filesystem only for:
  - `property_labels.csv` - Property metadata with fallback support
  - Plot cache directory (optional, not currently persisted)
  - HTML templates in `templates/`
  - Static assets in `static/`

**Caching:**
- In-memory caching only (not persistent):
  - Global `_plot_data_cache` dictionary in `plots/shared.py` - Stores DataFrame results for CSV export
  - Global `_plot_figure_cache` dictionary in `plots/shared.py` - Stores Plotly figure objects for PNG/SVG/PDF export
  - Cache lifetime: Application session only (cleared on restart)
  - No Redis or external cache store

## Authentication & Identity

**Auth Provider:**
- Type: Custom/None - No authentication layer
- Implementation: Direct HTTP access to SPARQL endpoint with no credentials
- Security Model: Assumes endpoint is behind VPN or firewall, or is public read-only

**SPARQL Endpoint Credentials:**
- Currently not supported
- If needed: Would require extending `SPARQLWrapper` configuration in `plots/shared.py`

## Monitoring & Observability

**Error Tracking:**
- Service: None (built-in logging only)
- Implementation: Python `logging` module configured via `config.py`
- Log level: Configurable via `Config.LOG_LEVEL` (default: INFO)
- Log format: `%(asctime)s - %(levelname)s - %(message)s`

**Logs:**
- Approach: Console/STDOUT logging only
  - No log aggregation service integrated
  - Suitable for Docker container environments (logs captured via container runtime)
  - Performance logging: Optional via `Config.ENABLE_PERFORMANCE_LOGGING` (default: True)

**Status Monitoring:**
- Endpoint: `/health` - JSON health check response
- Endpoint: `/status` - HTML status page with real-time metrics
- Implementation: `app.py` - `health_check()` and `status_page()` routes
- Checks: SPARQL endpoint connectivity and plot cache status

## CI/CD & Deployment

**Hosting:**
- Target: Docker container runtime
- Image: `python:3.11-slim` base image
- Container: Defined in `Dockerfile`
- Port: 5000 (configurable via `Config.FLASK_PORT`)
- Health check support: HTTP endpoints on `/health`

**CI Pipeline:**
- Service: None detected
- No GitHub Actions, GitLab CI, or similar configured in repository

**Container Orchestration:**
- Docker Compose: `docker-compose.yml` for local development
  - Single service: `visualizer`
  - Port mapping: 5000:5000
  - Restart policy: `unless-stopped`

## Environment Configuration

**Required Environment Variables:**
- `SPARQL_ENDPOINT` - URL of SPARQL endpoint (must be network-accessible)

**Optional Environment Variables (with sensible defaults):**
- `SPARQL_TIMEOUT=30`
- `SPARQL_MAX_RETRIES=3`
- `SPARQL_RETRY_DELAY=2`
- `PARALLEL_WORKERS=5`
- `PLOT_TIMEOUT=60`
- `LOG_LEVEL=INFO`
- `FLASK_HOST=0.0.0.0`
- `FLASK_PORT=5000`
- `FLASK_DEBUG=False`
- `ENABLE_HEALTH_CHECK=True`
- `ENABLE_PERFORMANCE_LOGGING=True`

**Secrets Location:**
- Not applicable - No secrets currently used
- If credentials needed for SPARQL endpoint: Would be passed via env vars (not committed to git)

**Configuration Validation:**
- Startup validation: `Config.validate_config()` called in `app.py` and `plots/shared.py`
- Validates: URL format, numeric ranges (1-65535 for port, positive for timeouts)
- Logs errors but continues with defaults on validation failure

## Webhooks & Callbacks

**Incoming:**
- None - Application is pull-only (initiates all requests to SPARQL endpoint)

**Outgoing:**
- None - Application does not send webhooks or callbacks to external services
- Only output: HTTP responses and static file downloads to web browser clients

## Data Export Integration

**CSV Export System:**
- Approach: Server-side generation and download
- Implementation: Flask routes in `app.py` with `/download/<plot_name>` endpoints
- Data source: Global `_plot_data_cache` dictionary populated during plot generation
- Format: RFC 4180 CSV via Pandas `to_csv()`
- Metadata: Included in CSV with query timestamp and version information

**Image Export Integration:**
- Approach: Client-side Plotly download buttons + server-side generation
- Implementation: Plotly's built-in PNG/SVG buttons configured in `plots/shared.py`
- Formats supported: PNG, SVG, WebP (configurable in Plotly config)
- Server endpoint: `/download/<plot_name>/image` for programmatic PNG/SVG/PDF export

---

*Integration audit: 2026-02-20*
