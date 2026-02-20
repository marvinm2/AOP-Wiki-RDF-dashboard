# Architecture Patterns

**Domain:** SPARQL-based data monitoring dashboard with interactive network visualization
**Researched:** 2026-02-20
**Confidence:** MEDIUM (training data only; WebSearch unavailable for verification)

## Current Architecture Assessment

The existing system is a layered Flask monolith with three tiers:

```
Browser (Plotly.js, Cytoscape.js) --> Flask (app.py routes) --> Plot Layer (plots/) --> SPARQL (Virtuoso)
```

**What works well:**
- Lazy loading via IntersectionObserver + `/api/plot/<name>` endpoints (50ms initial load)
- ThreadPoolExecutor parallel startup for 22 precomputed plots
- Version-aware plot regeneration through `?version=` query param
- Global DataFrame caching enables CSV/PNG/SVG exports without re-querying
- Graceful degradation via `safe_plot_execution()` wrapper

**What is strained:**
- `app.py` is 1,609 lines with 40+ route handlers, mostly duplicated download boilerplate
- Plot results stored in 20+ global variables after startup (fragile unpacking)
- No separation between data acquisition and visualization concerns in plot functions
- Complex SPARQL queries timeout (boxplot_by_status removed, completeness takes 75s)
- Legacy `plots.py` (4,194 lines) duplicates the refactored `plots/` package
- Network visualization exists only in legacy file, not migrated to modular package

## Recommended Architecture

### Target: Introduce Three New Layers Without Disrupting Existing

The architecture should evolve by **adding layers**, not replacing the working ones. Network analysis, entity deep-dives, and version comparison each need distinct data processing that does not fit the existing "SPARQL query -> Plotly chart" pipeline.

```
                         +-------------------+
                         |  Browser Client   |
                         |  (Plotly, Cyto.js)|
                         +--------+----------+
                                  |
                    +-------------+-------------+
                    |                           |
           +--------v--------+        +--------v--------+
           | Static Plots    |        | Interactive     |
           | (existing lazy) |        | Views (new)     |
           +---------+-------+        +--------+--------+
                     |                         |
           +---------v-------+        +--------v--------+
           | Plot Layer      |        | API Layer       |
           | (plots/)        |        | (new: api/)     |
           +---------+-------+        +--------+--------+
                     |                         |
           +---------v---------+------+--------v--------+
           |            Data Service Layer               |
           |  (new: services/ - query, cache, compute)   |
           +---------------------+-----------------------+
                                 |
                         +-------v--------+
                         |  Virtuoso      |
                         |  SPARQL 1.1    |
                         +----------------+
```

### Component Boundaries

| Component | Responsibility | Communicates With | New/Existing |
|-----------|---------------|-------------------|--------------|
| **Browser Client** | Render Plotly charts, Cytoscape.js graphs, handle user interaction | Flask API via fetch() | Existing (extend) |
| **Plot Layer** (`plots/`) | Generate static Plotly HTML from DataFrames | Data Service Layer for queries | Existing (refactor data access) |
| **API Layer** (`api/`) | REST endpoints for network data, entity details, version comparison | Data Service Layer, returns JSON | **New** |
| **Data Service Layer** (`services/`) | SPARQL query execution, result caching, graph computation | Virtuoso endpoint | **New** |
| **Network Analysis Module** (`services/network.py`) | Build NetworkX graph, compute centrality/clustering/PageRank | Data Service, returns computed metrics | **New** |
| **Entity Service** (`services/entities.py`) | Fetch individual entity details with related data | Data Service, returns entity dicts | **New** |
| **Version Comparison Service** (`services/comparison.py`) | Diff two versions, compute deltas | Data Service, returns diff structures | **New** |
| **Query Builder** (`services/sparql.py`) | Parameterized SPARQL query construction, Virtuoso-specific optimizations | Virtuoso endpoint via SPARQLWrapper | **New** (extract from shared.py) |
| **Cache Manager** (`services/cache.py`) | TTL-based cache with eviction, memory monitoring | In-memory dicts or optional Redis | **New** (extract from shared.py) |

### Data Flow

#### Flow 1: Network Analysis (New)

```
1. User navigates to /network page
2. Browser loads network.html template (skeleton + Cytoscape.js container)
3. JavaScript calls GET /api/network/graph?version=<v>
4. API route calls NetworkService.get_graph(version)
5. NetworkService checks cache -> miss -> calls QueryService
6. QueryService executes 3 SPARQL queries:
   a. All KEs with titles, MIE/AO classification
   b. All KERs with upstream/downstream links
   c. AOP-KE membership (for frequency calculation)
7. NetworkService builds NetworkX DiGraph from results
8. NetworkService computes metrics (degree, betweenness, PageRank)
9. NetworkService serializes to Cytoscape.js JSON format
10. Response: {nodes: [...], edges: [...], metrics: {...}}
11. Browser renders with Cytoscape.js, metrics in sidebar
```

**Key decision: Compute graph metrics server-side with NetworkX.**

Rationale: The AOP-Wiki network is moderate size (hundreds of KEs, hundreds of KERs) -- small enough that NetworkX handles it in < 1 second, but complex enough that computing centrality in the browser with JavaScript graph libraries would be fragile and slow. Server-side NetworkX gives access to proven algorithms (betweenness centrality, PageRank, connected components, clustering coefficient) without reimplementing them in JS.

The Cytoscape.js client receives pre-computed metrics as node/edge data attributes and uses them for visual encoding (node size = degree, color = betweenness, edge width = AOP frequency).

#### Flow 2: Entity Deep-Dive (New)

```
1. User clicks a KE node in network graph (or navigates to /entity/KE/<id>)
2. Browser calls GET /api/entity/KE/<id>?version=<v>
3. API route calls EntityService.get_entity("KE", id, version)
4. EntityService executes entity-specific SPARQL query:
   - KE properties (title, description, biological components)
   - Connected KERs (upstream and downstream)
   - Parent AOPs containing this KE
   - Property completeness score
5. Returns structured JSON: {entity: {...}, connections: [...], completeness: {...}}
6. Browser renders detail panel (side panel or new page)
```

**Key decision: Entity queries are single-entity, single-version -- always fast.**

Unlike aggregate queries that timeout on Virtuoso, entity-level queries target a single subject URI within a single named graph. These are guaranteed fast because Virtuoso indexes by subject. This makes entity deep-dives safe to implement without SPARQL optimization concerns.

#### Flow 3: Version Comparison (New)

```
1. User selects two versions from comparison UI
2. Browser calls GET /api/compare?v1=<a>&v2=<b>
3. API route calls ComparisonService.diff(v1, v2)
4. ComparisonService fetches entity counts for both versions (reuse existing queries)
5. Computes deltas: added entities, removed entities, changed properties
6. For network comparison: builds both graphs, computes structural diff
7. Returns: {summary: {...}, entities: {added: [...], removed: [...]}, metrics_delta: {...}}
8. Browser renders side-by-side comparison view
```

**Key decision: Version comparison reuses existing per-version queries, diffs in Python.**

Cross-version SPARQL queries (comparing two named graphs in a single query) are what cause Virtuoso timeouts. The safe pattern is: query version A, query version B, diff in Python. This transforms a dangerous O(n*m) SPARQL JOIN into two O(n) queries plus an O(n+m) Python set operation.

#### Flow 4: Existing Plots (Unchanged)

```
Startup -> ThreadPoolExecutor -> plot functions -> _plot_data_cache
User -> /api/plot/<name>?version=<v> -> cached HTML or regenerate -> JSON response
User -> /download/<name>?format=csv -> _plot_data_cache -> CSV Response
```

No changes to the existing lazy loading, version selector, or CSV export flows.

### Detailed Component Specifications

#### services/sparql.py - Query Builder

Extract SPARQL query execution from `plots/shared.py` into a dedicated service. The current `run_sparql_query_with_retry()` and `extract_counts()` become methods on a QueryService class.

```python
class QueryService:
    """Centralized SPARQL query execution with retry, timeout, and caching."""

    def __init__(self, endpoint: str, timeout: int, max_retries: int):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries

    def execute(self, query: str) -> list[dict]:
        """Execute SPARQL query with retry logic (existing logic from shared.py)."""

    def execute_for_version(self, query_template: str, version: str) -> list[dict]:
        """Execute a parameterized query for a specific graph version."""

    def get_entity(self, entity_type: str, entity_id: str, version: str) -> dict:
        """Fetch a single entity with all properties."""
```

**Build order implication:** This should be extracted first. All other new components depend on it. Existing plot functions can be gradually migrated to use it.

#### services/cache.py - Cache Manager

Replace unbounded global `_plot_data_cache` and `_plot_figure_cache` dicts with a managed cache that supports TTL expiry and memory limits.

```python
class CacheManager:
    """TTL-based cache with optional memory limit and eviction."""

    def __init__(self, max_memory_mb: int = 512, default_ttl: int = 3600):
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl: Optional[int] = None): ...
    def invalidate(self, pattern: str): ...  # e.g., invalidate("network:*")
    def stats(self) -> dict: ...  # memory usage, hit rate
```

**Build order implication:** Can be built in parallel with API layer. Integrate with existing cache first, then new services.

#### services/network.py - Network Analysis

The core new capability. Builds a NetworkX directed graph from SPARQL query results and computes graph-theoretic metrics.

```python
class NetworkService:
    """Build and analyze AOP-Wiki network graphs."""

    def __init__(self, query_service: QueryService, cache: CacheManager): ...

    def get_graph(self, version: str = None) -> dict:
        """Return Cytoscape.js-compatible JSON with computed metrics."""

    def get_metrics(self, version: str = None) -> dict:
        """Return network-level metrics (density, components, diameter)."""

    def get_node_metrics(self, version: str = None) -> pd.DataFrame:
        """Return per-node metrics (degree, betweenness, PageRank)."""

    def get_subgraph(self, aop_id: str, version: str = None) -> dict:
        """Return subgraph for a specific AOP."""

    def _build_graph(self, version: str) -> nx.DiGraph:
        """Internal: build NetworkX graph from SPARQL results."""

    def _compute_centrality(self, G: nx.DiGraph) -> dict:
        """Internal: compute all centrality measures."""

    def _to_cytoscape(self, G: nx.DiGraph, metrics: dict) -> dict:
        """Internal: serialize to Cytoscape.js JSON format."""
```

**Build order implication:** Depends on QueryService. This is the highest-value new component and should be built immediately after QueryService extraction.

#### api/ - REST API Layer

New Flask Blueprint for structured API endpoints. Separate from existing routes in `app.py`.

```python
# api/__init__.py
from flask import Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# api/network.py - Network analysis endpoints
@api_bp.route('/network/graph')         # Full network as Cytoscape.js JSON
@api_bp.route('/network/metrics')       # Network-level summary metrics
@api_bp.route('/network/node/<node_id>')  # Single node with neighbors

# api/entities.py - Entity detail endpoints
@api_bp.route('/entity/<type>/<id>')    # Full entity detail

# api/compare.py - Version comparison endpoints
@api_bp.route('/compare')               # Side-by-side version diff
```

**Build order implication:** Register as Blueprint on existing `app`. Does not modify existing routes. Can be built incrementally (network endpoints first, then entity, then comparison).

### Frontend Architecture Extension

#### Cytoscape.js Integration Pattern

The legacy `plots.py` contains a working Cytoscape.js implementation (lines 3248-4190) that embeds everything inline in a massive HTML string. This approach should be refactored into a proper JavaScript module:

```
static/js/
  lazy-loading.js          # Existing
  version-selector.js      # Existing
  network-viewer.js        # New: Cytoscape.js wrapper
  entity-detail.js         # New: Entity detail panel
  version-comparison.js    # New: Comparison views
```

`network-viewer.js` should:
1. Fetch network JSON from `/api/v1/network/graph?version=<v>`
2. Initialize Cytoscape.js with the response
3. Map computed metrics to visual encodings (size, color, width)
4. Handle user interactions (click node -> show detail, hover -> tooltip)
5. Provide controls (zoom, layout algorithm, filter by AOP, highlight path)

**Key pattern: Server computes, client renders.** No graph algorithms in JavaScript. The server returns pre-computed `degree`, `betweenness`, `pagerank` as node data attributes. The client only maps them to visual properties.

#### Entity Detail Pattern

Entity details should render as a slide-in panel (not a separate page) when clicking a node in the network graph or a row in a data table. This keeps context while showing detail.

```
+---------------------------+----------------+
|  Network Graph            |  Detail Panel  |
|  (Cytoscape.js)           |  (slide-in)    |
|                           |  KE Title      |
|    [node clicked] ------> |  Description   |
|                           |  Components    |
|                           |  Connected KEs |
|                           |  Parent AOPs   |
|                           |  Completeness  |
+---------------------------+----------------+
```

## Patterns to Follow

### Pattern 1: Service Layer Extraction

**What:** Move data acquisition and computation logic out of plot functions into reusable services.

**When:** Any time data fetching logic is needed by more than one consumer (plot function, API endpoint, comparison engine).

**Why:** Currently `plots/trends_plots.py` contains both SPARQL queries AND Plotly rendering in the same functions. Network analysis needs the same data but different rendering (Cytoscape.js instead of Plotly). Without extraction, queries get duplicated.

**Example:**
```python
# Before: Data and visualization coupled in plot function
def plot_network_density() -> str:
    query = """SELECT ?graph ..."""          # Data acquisition
    results = run_sparql_query(query)        # Data acquisition
    df = pd.DataFrame([...])                 # Processing
    fig = px.line(df, ...)                   # Visualization
    return pio.to_html(fig, ...)             # Rendering

# After: Service provides data, consumers choose rendering
class NetworkService:
    def get_density_data(self, version=None) -> pd.DataFrame:
        results = self.query.execute(DENSITY_QUERY, version)
        return pd.DataFrame([...])

# Plot consumer
def plot_network_density(network_svc: NetworkService) -> str:
    df = network_svc.get_density_data()
    fig = px.line(df, ...)
    return pio.to_html(fig, ...)

# API consumer
@api_bp.route('/network/density')
def api_network_density():
    df = network_svc.get_density_data(request.args.get('version'))
    return jsonify(df.to_dict('records'))
```

### Pattern 2: Blueprint-Based API Organization

**What:** Use Flask Blueprints to organize new API endpoints separately from existing routes.

**When:** Adding any new REST API endpoints.

**Why:** `app.py` is already 1,609 lines. Adding network, entity, and comparison endpoints inline would push it past 2,500 lines. Blueprints provide clean separation without changing the existing routing.

**Example:**
```python
# api/__init__.py
from flask import Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

from . import network, entities, compare  # noqa: register routes

# app.py (minimal change)
from api import api_bp
app.register_blueprint(api_bp)
```

### Pattern 3: Query-Once, Diff-in-Python

**What:** Never compare two named graphs in a single SPARQL query. Query each version separately and diff in Python.

**When:** Any cross-version analysis (version comparison, trend computation, delta calculations).

**Why:** The primary performance pain in this codebase comes from complex multi-graph SPARQL queries that timeout on Virtuoso. The removed `plot_aop_completeness_boxplot_by_status` and the 75-second completeness query both involve cross-version aggregations. Python set operations on two result sets are orders of magnitude faster than asking Virtuoso to JOIN across large named graphs.

**Example:**
```python
# DANGEROUS: Cross-version SPARQL (what caused timeouts)
"""SELECT ?v1_count ?v2_count WHERE {
    GRAPH <v1> { ... COUNT ... }
    GRAPH <v2> { ... COUNT ... }
}"""

# SAFE: Query each version, diff in Python
v1_data = query_service.execute_for_version(ENTITY_QUERY, "2024-07-01")
v2_data = query_service.execute_for_version(ENTITY_QUERY, "2024-10-01")

v1_entities = {r['entity']['value'] for r in v1_data}
v2_entities = {r['entity']['value'] for r in v2_data}

added = v2_entities - v1_entities
removed = v1_entities - v2_entities
```

### Pattern 4: Compute Server-Side, Render Client-Side

**What:** Graph algorithms run in Python (NetworkX). Visual rendering happens in the browser (Cytoscape.js or Plotly.js).

**When:** Any interactive visualization that needs computed properties (centrality, clustering, layout).

**Why:** NetworkX provides battle-tested implementations of betweenness centrality, PageRank, connected components, and dozens of other algorithms. Reimplementing these in JavaScript is unnecessary error-prone work. The network size (hundreds of nodes) means Python computation takes < 1 second. The browser receives pre-computed attributes and only handles rendering and interaction.

### Pattern 5: Generic Download Handler

**What:** Replace the 20+ duplicated download route handlers with a single generic handler.

**When:** Immediately, as tech debt cleanup before adding new download routes.

**Why:** `app.py` currently has near-identical route handlers for every download endpoint. Each is ~40 lines of copy-pasted code. Adding network analysis exports would require yet more copies. A single generic handler reduces code by 800+ lines and makes adding new exports trivial.

**Example:**
```python
# Before: 20+ copies of this pattern
@app.route("/download/latest_entity_counts")
def download_latest_entity_counts():
    plot_name = 'latest_entity_counts'
    # ... 35 lines of identical logic ...

# After: One generic handler
@app.route("/download/<path:plot_name>")
def download_plot(plot_name):
    export_format = request.args.get('format', 'csv').lower()
    # ... single implementation handles all plots ...
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Inline HTML Generation for Complex Visualizations

**What:** Building HTML strings with f-strings containing embedded JavaScript, CSS, and Cytoscape.js initialization (as done in legacy `plots.py` lines 3617-4190).

**Why bad:** The legacy network visualization is a 570-line f-string mixing Python variables with JavaScript. It is untestable, unmaintainable, and has no syntax highlighting or linting. Any change risks breaking string interpolation.

**Instead:** Serve a static HTML template with Cytoscape.js initialized from JSON data fetched via API. The template is a normal HTML file. The data is a normal JSON API response. They are tested independently.

### Anti-Pattern 2: Cross-Version SPARQL Aggregations

**What:** Writing a single SPARQL query that reads from multiple named graphs with GROUP BY, COUNT, and aggregation across all versions simultaneously.

**Why bad:** This is the root cause of all timeout issues in this codebase. Virtuoso's query planner handles multi-graph JOINs poorly and often exhausts execution limits. The removed boxplot_by_status and the 75-second completeness query are direct evidence.

**Instead:** Query one version at a time. Aggregate in Python. It is always faster and never times out.

### Anti-Pattern 3: Global Variable Plot Storage

**What:** Storing precomputed plot HTML in 20+ module-level global variables (`graph_main_abs`, `graph_avg_delta`, etc.) with manual tuple unpacking.

**Why bad:** Adding a new plot requires: modifying the task list, adding a new global, adding try/except unpacking, adding it to `plot_map`. Missing any step causes silent failures. The pattern does not scale.

**Instead:** Store all plot results in the existing `plot_results` dict and look up by name. The `plot_map` dict in `/api/plot/<name>` already does this for some plots. Extend it to all.

### Anti-Pattern 4: Computing Network Metrics in the Browser

**What:** Implementing graph algorithms (centrality, PageRank, clustering) in JavaScript client-side code.

**Why bad:** JavaScript graph libraries (e.g., graphology, sigma.js built-in algorithms) exist but are less tested, harder to debug, and create browser performance issues on larger graphs. Python's NetworkX is the standard for graph analysis with extensive documentation and proven correctness.

**Instead:** Compute all metrics server-side with NetworkX. Pass results as node/edge data attributes in the API response. Client only renders.

## Scalability Considerations

| Concern | Current (hundreds of entities) | At 5x data (thousands) | At 10x data (tens of thousands) |
|---------|------|------|------|
| **SPARQL query time** | 1-75 seconds | Timeouts likely on aggregate queries | Requires pre-computation or materialized views |
| **Network graph rendering** | Cytoscape.js handles well | Enable progressive loading, reduce initial viewport | Server-side subgraph extraction, only render visible region |
| **NetworkX computation** | < 1 second | 2-5 seconds (still acceptable) | Pre-compute and cache; consider igraph for large-scale |
| **Memory (cache)** | ~100MB for all plots | ~500MB, need eviction policy | Redis external cache, not in-process |
| **Startup time** | ~75 seconds (bottleneck: completeness query) | Would exceed 5 minutes | Must switch to lazy-only, no precompute |

## Suggested Build Order

The build order is driven by dependencies between components and value delivery:

```
Phase 1: Foundation (must come first)
  1a. Extract QueryService from plots/shared.py
  1b. Implement CacheManager with TTL
  1c. Consolidate download routes (tech debt)
  1d. Remove legacy plots.py

Phase 2: Network Analysis (highest value, depends on Phase 1)
  2a. Implement NetworkService with NetworkX
  2b. Create api/ Blueprint with network endpoints
  2c. Build network-viewer.js (Cytoscape.js client)
  2d. Create /network page template

Phase 3: Entity Deep-Dives (depends on Phase 1 QueryService)
  3a. Implement EntityService
  3b. Add entity API endpoints
  3c. Build entity detail panel (slide-in or page)
  3d. Wire click-through from network nodes

Phase 4: Version Comparison (depends on Phase 1 + 2)
  4a. Implement ComparisonService
  4b. Add comparison API endpoints
  4c. Build comparison UI (side-by-side view)
  4d. Wire version selector to comparison mode
```

**Ordering rationale:**
- Phase 1 must be first because all new features depend on the extracted services
- Phase 2 (network) is highest value per issue #11 and extends the existing connectivity analysis
- Phase 3 (entity details) provides natural click-through from network nodes
- Phase 4 (version comparison) builds on both entity details and network for structural diffs

**Each phase is independently deployable.** Phase 1 improves code quality without changing functionality. Phase 2 adds the network page. Phase 3 adds drill-down capability. Phase 4 adds cross-version analysis.

## Technology Decisions for New Components

| Component | Technology | Why | Confidence |
|-----------|-----------|-----|------------|
| Graph computation | NetworkX | Standard Python graph library, proven algorithms, moderate-size graphs | HIGH (well-known library) |
| Graph rendering | Cytoscape.js | Already prototyped in legacy code, excellent API, layout algorithms built-in | HIGH (existing precedent in codebase) |
| API structure | Flask Blueprint | Built into Flask, zero new dependencies, clean separation | HIGH |
| Caching | In-process dict with TTL | Simplest solution, no new infra, matches existing pattern | HIGH |
| Node metrics | NetworkX (degree_centrality, betweenness_centrality, pagerank) | Proven implementations, < 1s on this graph size | MEDIUM (need to verify on actual data) |

## Sources

- Codebase analysis: `app.py` (1,609 lines), `plots/shared.py` (1,024 lines), `plots/trends_plots.py`, `plots/latest_plots.py`
- Legacy network implementation: `plots.py` lines 3248-4190 (Cytoscape.js prototype)
- Existing architecture: `.claude/architecture.md`, `.planning/codebase/ARCHITECTURE.md`
- Project requirements: `.planning/PROJECT.md` (active requirements list)
- Known concerns: `.planning/codebase/CONCERNS.md` (performance bottlenecks, tech debt)
- NetworkX capabilities: Training data (MEDIUM confidence -- standard library, unlikely to have changed significantly)
- Cytoscape.js capabilities: Training data + codebase evidence (HIGH confidence -- already used in legacy code)

---

*Architecture research: 2026-02-20*
