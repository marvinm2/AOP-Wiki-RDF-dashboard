# Technology Stack

**Project:** AOP-Wiki RDF Dashboard - Performance, Network Analysis, and Platform Deployment
**Researched:** 2026-02-20
**Mode:** Ecosystem (Stack dimension for SPARQL-based data monitoring dashboards)

## Executive Summary

This is a subsequent milestone for an existing Flask/Plotly dashboard. The stack is established and working -- Flask 3.x, Plotly 5.x, Pandas, SPARQLWrapper against Virtuoso. The research question is not "what framework to use" but rather "what libraries to add for network analysis, query optimization, and production deployment" while respecting the constraint that no framework migration is planned.

The key additions needed are: (1) NetworkX for graph algorithms (centrality, PageRank, clustering), (2) Cytoscape.js extensions for interactive network rendering (already partially integrated), (3) a production WSGI server (Gunicorn) for VHP platform deployment, and (4) SPARQL query optimization patterns that work within Virtuoso's constraints.

## Current Stack (Verified from Codebase)

| Technology | Current State | Pinned? | Notes |
|------------|---------------|---------|-------|
| Python | 3.11 | Yes (Dockerfile) | `python:3.11-slim` base image |
| Flask | latest (unpinned) | No | `requirements.txt` has no version pins |
| Plotly | latest (unpinned) | No | Used for 30+ visualizations |
| Pandas | latest (unpinned) | No | DataFrame operations, CSV export |
| SPARQLWrapper | latest (unpinned) | No | SPARQL endpoint communication |
| Cytoscape.js | 3.26.0 | Yes (CDN) | Loaded from unpkg in legacy `plots.py` |
| Virtuoso | External | N/A | Controlled SPARQL endpoint |
| Docker | docker-compose 3.8 | Yes | Single-service deployment |

**Critical observation:** No version pinning in `requirements.txt`. This is a deployment risk -- any `pip install` could pull breaking changes. Pin all versions before adding new dependencies.

## Recommended Stack Additions

### Network Analysis (Python-side)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| NetworkX | >=3.2 | Graph algorithms: centrality, PageRank, clustering coefficient, community detection | The standard Python graph library. Mature (20+ years), well-documented, integrates natively with Pandas DataFrames. All required algorithms (degree/betweenness/closeness centrality, PageRank, clustering coefficient) are built-in. No viable alternative for Python graph analysis at this scale. | HIGH |
| python-louvain (community) | >=0.16 | Community detection for AOP network clustering | Implements the Louvain method for community detection on top of NetworkX. The standard approach for finding communities in biological/knowledge networks. Required for Issue #11's "clustering" requirement. | MEDIUM |

**Why NetworkX and not igraph or graph-tool:**
- **igraph** (python-igraph): Faster for very large graphs (100K+ nodes), but AOP-Wiki has hundreds of KEs/KERs, not millions. The C backend adds compilation complexity in Docker builds with `python:3.11-slim`. Not worth the added build complexity for this graph size.
- **graph-tool**: Even faster but requires Boost C++ libraries. Extremely painful to install in containers. Overkill for this use case.
- **NetworkX**: Pure Python, zero compilation dependencies, `pip install` just works in slim Docker images. For graphs under 10K nodes (AOP-Wiki is well under this), performance is not a bottleneck.

**How NetworkX fits the architecture:**
- Build NetworkX graphs from SPARQL query results (KE nodes + KER edges) in Python
- Compute centrality/PageRank/clustering server-side, cache results in `_plot_data_cache`
- Serialize metrics as node attributes for Cytoscape.js rendering (node size = centrality, color = community)
- This avoids complex SPARQL aggregations that timeout Virtuoso -- do the math in Python, not in SPARQL

### Interactive Network Visualization (Frontend)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Cytoscape.js | >=3.28 (upgrade from 3.26.0) | Interactive graph rendering with pan/zoom/select | Already partially integrated in legacy `plots.py`. Mature, performant for medium graphs, extensive layout algorithms. Biological network visualization is literally its origin (Cytoscape desktop for bioinformatics). Perfect domain fit for AOP networks. | HIGH |
| cytoscape-cola | >=2.5 | Force-directed layout with constraints | Cola.js layout plugin for Cytoscape.js. Better than default layouts for directed biological networks because it supports edge-length constraints and alignment, producing cleaner AOP pathway visualizations than random force-directed layouts. | MEDIUM |
| cytoscape-popper | >=2.0 | Rich tooltips on hover | Integrates Tippy.js/Popper.js with Cytoscape nodes for styled tooltip popups showing KE details, centrality scores, AOP membership. Current implementation uses basic native Cytoscape events for tooltips. | MEDIUM |

**Why Cytoscape.js and not D3.js, Sigma.js, or vis.js:**
- **D3.js**: Maximum flexibility but requires writing everything from scratch. No built-in graph layouts, node interaction, or biological-network conventions. Development effort is 5-10x higher for equivalent functionality.
- **Sigma.js**: Optimized for rendering very large graphs (100K+ nodes). AOP-Wiki has hundreds of nodes. Sigma's API is lower-level and less feature-rich for medium graphs. Wrong tool for this scale.
- **vis.js**: Decent alternative but less maintained than Cytoscape.js. Missing specialized layout algorithms. No biological network heritage.
- **Cytoscape.js**: Already in the codebase. Rich plugin ecosystem. Purpose-built for the exact type of network this project visualizes. Keep it and extend it.

**Why NOT Dash Cytoscape:**
The project uses Flask templates with server-rendered Plotly HTML, not Dash. Adding `dash-cytoscape` would require either migrating to Dash (out of scope, per PROJECT.md constraints) or running Dash alongside Flask (unnecessary complexity). Instead, continue using Cytoscape.js directly via CDN in Flask templates, which is the pattern already established in `plots.py`.

### SPARQL Query Optimization

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| rdflib | >=7.0 | Local SPARQL query validation, RDF parsing for cached results | Can validate SPARQL syntax before sending to Virtuoso (catch errors early). Can parse small RDF result sets locally. Useful for building a local cache layer if needed. | MEDIUM |

**Why rdflib is optional, not required:**
The project already uses SPARQLWrapper effectively. rdflib would primarily help with: (1) validating SPARQL queries during development/testing, (2) potentially caching RDF subgraphs locally to avoid repeated Virtuoso round-trips. However, the more impactful optimization path is restructuring queries (which is a code pattern, not a library) and computing aggregations in Python with NetworkX/Pandas rather than in SPARQL.

**SPARQL optimization is primarily a pattern concern, not a library concern:**
- Move complex aggregations from SPARQL to Python (already proven: 10 queries reduced to 4 with 95% data reduction)
- Use Virtuoso-specific hints: `OPTION (ORDER)` for join ordering, `DEFINE sql:select-option "order"`
- Implement query decomposition: fetch raw triples, aggregate in Pandas
- Cache intermediate results in `_plot_data_cache` across related plot computations
- Use `VALUES` clauses for batch lookups instead of `FILTER(IN(...))`

### Production Deployment

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Gunicorn | >=22.0 | Production WSGI server | Flask's built-in server (`flask run`) is explicitly documented as not suitable for production. The current Dockerfile uses `flask run`. For VHP platform deployment (Issue #3), Gunicorn is the standard production WSGI server for Flask apps. Simple configuration, proven reliability, works with existing code without changes. | HIGH |
| Flask-CORS | >=4.0 | Cross-origin request headers | Identified in CONCERNS.md as missing. If VHP platform hosts the dashboard on a different subdomain than other VHP tools, CORS headers are required. Simple drop-in middleware. | MEDIUM |
| Flask-Caching | >=2.1 | Response-level caching with TTL and eviction | CONCERNS.md identifies unbounded cache growth as a performance bottleneck. Flask-Caching provides `SimpleCache` (in-memory with TTL) and `FileSystemCache` as drop-in replacements for the current manual `_plot_data_cache` dict. Adds LRU eviction and automatic expiry. | MEDIUM |

**Why Gunicorn and not uWSGI or Waitress:**
- **uWSGI**: More features but significantly more complex configuration. Historically harder to debug. Gunicorn is simpler and Flask's official documentation recommends it.
- **Waitress**: Windows-compatible but this project is Linux/Docker only. Gunicorn has better performance characteristics and broader deployment documentation.
- **Gunicorn**: One-line Dockerfile change (`CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]`). Handles concurrent requests properly, unlike Flask's dev server.

**Why NOT Redis/Memcached for caching:**
The dashboard is a single-container deployment serving read-only data. Adding Redis would require a second container in docker-compose, increasing operational complexity for minimal benefit. Flask-Caching's `SimpleCache` (in-process LRU) is sufficient for a single-instance deployment. If VHP platform requires horizontal scaling later, upgrade to Redis-backed caching at that point.

### Testing

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytest | >=8.0 | Test framework | No tests exist (identified in CONCERNS.md). pytest is the standard Python test framework. Required before adding network analysis features to prevent regressions. | HIGH |
| pytest-flask | >=1.3 | Flask test client integration | Provides `client` fixture for testing Flask routes without starting a server. Essential for testing the `/api/plot/<name>` endpoints and CSV download routes. | HIGH |
| responses | >=0.25 | Mock HTTP responses for SPARQL queries | Allows mocking SPARQLWrapper HTTP calls in tests without a live Virtuoso endpoint. Alternative to `unittest.mock` with a cleaner API for HTTP mocking specifically. | MEDIUM |

### Development Quality

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| ruff | >=0.3 | Linting and formatting | Replaces flake8 + black + isort with a single fast tool. Catches the broad exception handling issues identified in CONCERNS.md. | MEDIUM |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Graph analysis | NetworkX | igraph | C compilation in slim Docker; overkill for graph size under 10K nodes |
| Graph analysis | NetworkX | graph-tool | Boost C++ dependency; Docker build nightmare; wrong scale for this project |
| Network viz | Cytoscape.js (CDN) | Dash Cytoscape | Would require Dash alongside Flask or full migration; adds framework complexity |
| Network viz | Cytoscape.js (CDN) | D3.js | 5-10x more development effort for equivalent graph features |
| Network viz | Cytoscape.js (CDN) | Sigma.js | Optimized for 100K+ node graphs; less feature-rich for medium biological networks |
| WSGI server | Gunicorn | uWSGI | More complex configuration; Gunicorn is Flask's recommended production server |
| WSGI server | Gunicorn | Waitress | Less performant on Linux; designed for Windows compatibility which is irrelevant here |
| Caching | Flask-Caching (SimpleCache) | Redis | Second container adds operational complexity; single-instance deployment doesn't need distributed cache |
| Community detection | python-louvain | cdlib | cdlib pulls in heavy ML dependencies; python-louvain is lightweight and sufficient |
| Testing | pytest + responses | unittest + unittest.mock | pytest is more ergonomic; responses is purpose-built for HTTP mocking |
| Linting | ruff | flake8 + black + isort | ruff replaces all three with 10-100x faster execution |

## Version Pinning Strategy

**Immediate action required:** Pin all dependencies in `requirements.txt`. The current unpinned state is a deployment risk.

```
# requirements.txt - Pin to known-working versions
flask>=3.0,<4.0
pandas>=2.1,<3.0
plotly>=5.18,<6.0
SPARQLWrapper>=2.0,<3.0
requests>=2.31,<3.0

# New additions for network analysis
networkx>=3.2,<4.0
python-louvain>=0.16,<1.0

# Production deployment
gunicorn>=22.0,<23.0

# Optional additions
flask-cors>=4.0,<5.0
flask-caching>=2.1,<3.0

# Testing (dev only)
pytest>=8.0,<9.0
pytest-flask>=1.3,<2.0
responses>=0.25,<1.0
ruff>=0.3
```

**Version range strategy:** Use `>=minimum,<next_major` to get patch/minor updates while preventing breaking changes from major version bumps.

## Installation

```bash
# Core (production)
pip install flask pandas plotly SPARQLWrapper requests networkx python-louvain gunicorn flask-cors flask-caching

# Dev dependencies
pip install pytest pytest-flask responses ruff
```

## Dockerfile Changes for Production

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Production WSGI server instead of Flask dev server
CMD ["gunicorn", "--workers", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]
```

**Key changes:**
- Replace `flask run` with `gunicorn`
- 4 workers (adjust based on VHP platform CPU allocation)
- 120-second timeout to accommodate the 75-second Composite AOP Completeness query during startup
- Remove `FLASK_APP` env var (Gunicorn uses `app:app` module:variable syntax)

## Cytoscape.js CDN Upgrade

```html
<!-- Upgrade from 3.26.0 to latest stable -->
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>

<!-- Add layout plugin for better AOP pathway visualization -->
<script src="https://unpkg.com/cytoscape-cola@2.5.1/cytoscape-cola.js"></script>
<script src="https://unpkg.com/webcola@3.4.0/WebCola/cola.min.js"></script>

<!-- Add tooltip plugin for rich node/edge information -->
<script src="https://unpkg.com/@popperjs/core@2.11.8/dist/umd/popper.min.js"></script>
<script src="https://unpkg.com/cytoscape-popper@2.0.0/cytoscape-popper.js"></script>
```

**Note:** Cytoscape.js version 3.30.4 is based on training data (May 2025 cutoff). Verify the actual latest stable version at https://github.com/cytoscape/cytoscape.js/releases before implementation. **Confidence: LOW on exact version number.**

## Architecture Integration Pattern

The key architectural insight is that network analysis should happen in Python (NetworkX), not in SPARQL. This avoids the Virtuoso timeout problem entirely.

```
SPARQL Query (simple)     Python Processing        Frontend Rendering
---------------------     ------------------       ------------------
Fetch KE nodes       -->  Build NetworkX graph -->  Cytoscape.js with
Fetch KER edges      -->  Compute centrality   -->  node size = centrality
Fetch AOP membership -->  Compute PageRank     -->  node color = community
                     -->  Detect communities   -->  edge width = weight
                     -->  Cache as DataFrame   -->  Interactive explore
```

**This pattern solves three problems simultaneously:**
1. **Virtuoso timeouts**: Simple SELECT queries (no aggregation) run fast
2. **Network analysis**: NetworkX has all required algorithms built-in
3. **Caching**: Computed metrics are DataFrames, fitting existing cache pattern

## What NOT to Add

| Technology | Why Not |
|------------|---------|
| Dash / Dash Cytoscape | Framework conflict with Flask templates; would require migration or dual-framework complexity |
| Redis / Memcached | Adds operational complexity (second container) for a single-instance deployment |
| Celery / task queue | Over-engineering for pre-computed visualizations; ThreadPoolExecutor is sufficient |
| PostgreSQL / SQLite | No relational data to store; RDF data stays in Virtuoso |
| GraphQL | SPARQL is the native query language for RDF; adding GraphQL would be a translation layer with no benefit |
| Apache Jena / Fuseki | Project uses Virtuoso (controlled); no reason to add a second triplestore |
| Neo4j | Property graph DB is wrong paradigm for RDF data; Virtuoso already handles the graph storage |
| WebSocket (Flask-SocketIO) | Real-time updates are explicitly out of scope (PROJECT.md) |
| React / Vue frontend | Project uses server-rendered templates with Plotly; SPA framework adds complexity without clear benefit for this use case |

## Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| NetworkX for graph algorithms | HIGH | De facto standard, verified from training data, stable API for 20+ years, pure Python install |
| Cytoscape.js for network viz | HIGH | Already in codebase, biological network heritage, verified from existing implementation |
| Gunicorn for production | HIGH | Flask's documented recommendation, universal Python deployment pattern |
| python-louvain for communities | MEDIUM | Well-established but verify latest version; alternatives exist (cdlib) |
| Flask-Caching | MEDIUM | Stable library but verify API hasn't changed significantly since training data |
| Cytoscape.js exact version (3.30.4) | LOW | Based on training data cutoff; verify actual latest at implementation time |
| cytoscape-cola exact version | LOW | Plugin version may have changed; verify compatibility with Cytoscape.js version |
| ruff version | LOW | Fast-moving project; verify latest version at implementation time |

## Open Questions for Implementation

1. **VHP platform deployment requirements**: What container orchestration does VHP use? Kubernetes? Docker Compose? This affects Gunicorn worker configuration and health check expectations.
2. **Virtuoso version and configuration**: What Virtuoso version is running? Are there `ResultSetMaxRows`, `MaxQueryExecutionTime`, or `MaxQueryMem` settings that can be tuned? This affects whether query decomposition alone is sufficient.
3. **Graph size projection**: How many KEs/KERs/AOPs are in the latest version? If under 1,000 nodes, NetworkX will compute all metrics in under 1 second. If growing toward 10,000, consider pre-computation scheduling.
4. **Cytoscape.js CDN vs bundled**: For VHP deployment, should Cytoscape.js be bundled locally instead of loaded from unpkg CDN? Depends on whether the deployment environment has internet access.

## Sources

- Codebase analysis: `.planning/codebase/STACK.md`, `.planning/codebase/CONCERNS.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/INTEGRATIONS.md`
- Project requirements: `.planning/PROJECT.md`
- Existing implementation: `plots.py` (Cytoscape.js integration at line 3248+), `plots/shared.py` (SPARQL patterns), `requirements.txt` (current dependencies)
- Architecture constraints: `CLAUDE.md` (established Flask + Plotly stack, no migration)
- Library knowledge: Training data (May 2025 cutoff) -- versions marked LOW confidence where exact numbers may be stale

---

*Stack research: 2026-02-20*
