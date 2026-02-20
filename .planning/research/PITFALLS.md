# Domain Pitfalls

**Domain:** SPARQL-based RDF data monitoring dashboard with network analysis
**Project:** AOP-Wiki RDF Dashboard
**Researched:** 2026-02-20
**Overall confidence:** HIGH (based on codebase analysis and established SPARQL/Virtuoso/Flask patterns)

## Critical Pitfalls

Mistakes that cause rewrites, feature removal, or major production incidents. These are the ones that have already bitten this project or will bite it in the next phase.

---

### Pitfall 1: Virtuoso Query Plan Explosion on Multi-Version Aggregations

**What goes wrong:** SPARQL queries that aggregate data across multiple named graphs with JOINs, GROUP BY, and FILTER combinations cause Virtuoso's query planner to generate exponentially large execution plans. The planner cannot efficiently optimize cross-graph operations combined with property path traversals.

**Why it happens:** Virtuoso's cost-based optimizer struggles with queries that combine: (a) iteration over many named graphs via `GRAPH ?graph`, (b) multi-way JOINs within each graph, (c) aggregate functions like COUNT, GROUP_CONCAT across the combined result set. The optimizer often materializes intermediate results that grow as O(graphs x entities x properties), which exhausts the `MaxQueryExecutionTime` and `MaxQueryMem` limits.

**Consequences:** Features get removed entirely. This project already lost `plot_aop_completeness_boxplot_by_status` to this exact problem, and `plot_aop_completeness_boxplot` still takes 75 seconds after aggressive optimization (down from timeout). Any new network analysis features (centrality, PageRank) that require cross-graph or multi-hop traversals will hit the same wall harder.

**Warning signs:**
- A query works fine on a single graph but times out when run across all versions
- Adding a single OPTIONAL clause doubles or triples execution time
- Virtuoso returns "Exceeded configured MaxQueryExecutionTime" or "Ran out of memory" errors
- Query works on the first 5 graph versions but fails at 10+
- EXPLAIN output shows nested loop joins instead of hash joins

**Prevention:**
1. Never write a single SPARQL query that aggregates across all versions with JOINs. Instead, query one version at a time in Python and aggregate DataFrames in pandas. This is what the project already does for entity counts (4 separate queries) but failed to do for completeness-by-status.
2. Set per-query timeouts at the SPARQLWrapper level (already done at 30s) AND in the Virtuoso INI (`MaxQueryExecutionTime`), but also implement a "complexity budget" -- if a query needs more than 2 JOINs across all graphs, split it.
3. Use the CSV-based property loading pattern (already adopted) instead of dynamic property discovery in SPARQL.
4. For network analysis: compute graph metrics in Python (NetworkX) from pre-fetched adjacency data, not via SPARQL path queries.

**Detection:** Monitor query execution time. Any query taking >10 seconds on the current dataset will likely timeout when the dataset grows by 2-3x. The `run_sparql_query_with_retry` function already logs slow queries (>10s) -- treat these as bugs, not warnings.

**Phase relevance:** Network analysis phase (Issue #11) and any new cross-version visualizations.

---

### Pitfall 2: NetworkX In-Process Memory Bomb on Flask Workers

**What goes wrong:** Computing graph centrality metrics (betweenness, closeness, PageRank) with NetworkX inside a Flask request handler or startup routine consumes memory proportional to O(V^2) or O(V*E) and blocks the worker thread for the entire computation duration. For the AOP-Wiki network (~400+ AOPs, ~1000+ KEs, ~1000+ KERs per version, times ~20+ versions), this means multi-GB memory spikes and multi-minute blocking computations.

**Why it happens:** NetworkX stores graphs as Python dictionaries of dictionaries -- extremely memory-inefficient compared to adjacency matrices. Betweenness centrality is O(VE) time and O(V+E) space. PageRank converges iteratively but still requires the full graph in memory. When these computations happen in-process alongside Flask's WSGI server, they compete for the same memory pool as the DataFrame caches, Plotly figure objects, and SPARQL result buffers.

**Consequences:** Server becomes unresponsive during computation. With the existing `_plot_data_cache` and `_plot_figure_cache` (already flagged as unbounded), adding NetworkX graphs on top can push a 512MB container into OOM kills. Multiple concurrent users triggering network analysis could cascade into complete service unavailability.

**Warning signs:**
- Container memory usage spikes above 80% during network computations
- Flask health check endpoint (`/health`) starts returning 503 during graph analysis
- Users see timeouts on other pages while network analysis is running
- Docker logs show OOM kill signals

**Prevention:**
1. Pre-compute network metrics as a background task at startup or on a schedule, NOT in request handlers. Store results as static JSON/DataFrames.
2. Use scipy sparse matrices (via `nx.to_scipy_sparse_array()`) for large graphs instead of the default dict-of-dict representation. This cuts memory usage by 5-10x.
3. Compute metrics for only the latest version by default. Historical network evolution should be opt-in, not computed at startup.
4. Set hard memory limits on the Docker container and monitor with `/health`. If NetworkX computation would exceed a memory threshold, refuse the computation and serve cached results.
5. Consider `graph-tool` or `igraph` instead of NetworkX for large graphs -- they use C backends and are 10-100x faster with lower memory overhead. However, they are harder to install in Docker.

**Detection:** Add memory monitoring to the health endpoint. Before any NetworkX computation, check `psutil.Process().memory_info().rss` and abort if above 70% of container limit.

**Phase relevance:** Network Analysis Expansion (Issue #11). This is the most likely pitfall to hit during that phase.

---

### Pitfall 3: Unbounded In-Memory Cache Leading to Gradual OOM

**What goes wrong:** The global `_plot_data_cache` and `_plot_figure_cache` dictionaries grow indefinitely as plots are generated, version-specific plots are requested, and network analysis results are added. Each Plotly figure object can be 500KB-5MB in memory. Each cached DataFrame can be 100KB-10MB. With 30+ plots, multiple versions, and no eviction policy, memory grows monotonically until the process is killed.

**Why it happens:** The caching system was designed for a simpler use case (startup-computed plots served for the application lifetime). But the version selector now generates new plot variants per version, each stored as a separate cache entry. Adding network analysis will add more entries. There is a `clear_plot_cache()` function defined in `plots/__init__.py` but it is never called anywhere.

**Consequences:** Slow memory leak that manifests as OOM after hours or days of operation. Particularly dangerous in production where the container runs with `restart: unless-stopped` -- it will restart, re-compute all plots (75+ seconds), then slowly leak again. Users experience periodic outages.

**Warning signs:**
- Container memory usage trends upward over time (hours/days)
- Response times gradually increase as Python garbage collector works harder
- After a restart, performance is good but degrades over 4-8 hours
- Version selector usage correlates with memory growth spikes

**Prevention:**
1. Implement an LRU eviction policy on both caches. Use `collections.OrderedDict` or `functools.lru_cache` with a maxsize. Keep the 10 most recently accessed versions in cache, evict older ones.
2. Separate "startup cache" (historical trends, always in memory) from "dynamic cache" (version-specific latest plots, evictable).
3. For figure objects in `_plot_figure_cache`, do NOT cache Plotly figure objects -- they hold references to the entire data payload. Instead, cache the serialized HTML string and regenerate figures on demand for image export.
4. Add a memory monitoring background thread that logs cache size every 5 minutes and force-clears if above threshold.
5. Pin requirements.txt versions. An unexpected pandas or plotly upgrade that changes memory characteristics could accelerate this.

**Detection:** Add cache size metrics to the `/health` endpoint: number of entries and estimated memory usage (sum of `sys.getsizeof()` on cached DataFrames).

**Phase relevance:** Immediate concern. Should be addressed BEFORE adding network analysis features.

---

### Pitfall 4: Legacy plots.py Creating Silent Divergence

**What goes wrong:** The legacy `plots.py` (4,194 lines) coexists with the refactored `plots/` package (5,685 lines total). The import in `app.py` (`from plots import ...`) resolves to the `plots/` package because Python prefers packages over modules. But the legacy file remains importable as `import plots` from other contexts (tests, scripts, one-off analysis). If anyone modifies the legacy file thinking it is authoritative, or if a deployment misconfiguration causes the monolithic file to be loaded instead, the dashboard silently serves stale or inconsistent visualizations.

**Why it happens:** The refactoring from monolith to package was done incrementally. The legacy file was kept as a safety net but never removed. Python's import system makes this a ticking time bomb -- the `plots/` directory shadows `plots.py` for `from plots import X`, but direct `import plots` as a module name is ambiguous depending on sys.path ordering.

**Consequences:** Stale data in visualizations, inconsistent query patterns (the legacy file may not have the latest SPARQL optimizations), wasted developer time debugging the wrong file, and confusion for any contributor.

**Warning signs:**
- Git diff shows changes to `plots.py` instead of `plots/` package files
- Two different query patterns for the same visualization
- A "fixed" bug reappears because the fix was applied to only one location
- `verify_properties.py` or other scripts importing from the wrong location

**Prevention:**
1. Delete `plots.py` now. Not "deprecate," not "mark as legacy" -- delete it. The package has been working, and the file is fully redundant.
2. After deletion, grep the entire codebase for `import plots` or `from plots` to verify all imports resolve correctly.
3. Add a `.gitignore` entry or pre-commit hook that prevents recreation of a top-level `plots.py`.

**Detection:** Run `python -c "import plots; print(plots.__file__)"` -- if it prints `plots.py` instead of `plots/__init__.py`, the wrong module is being loaded.

**Phase relevance:** Immediate. Should be the very first cleanup task before any feature development.

---

### Pitfall 5: No Test Suite Means Silent Regression on SPARQL Schema Changes

**What goes wrong:** AOP-Wiki's RDF schema evolves (new properties, changed URIs, removed predicates). Without tests that validate query results against expected schemas, a schema change in the upstream data will cause plots to silently show wrong data, empty results, or break with cryptic errors. The dashboard will appear to work but display incorrect information.

**Why it happens:** The project has zero automated tests. SPARQL queries are only validated by running the full application against a live endpoint. This means:
- No unit tests for query construction (could generate invalid SPARQL)
- No integration tests for expected result shapes
- No smoke tests for endpoint health
- No regression tests for CSV export data integrity
- Property labels in `property_labels.csv` may reference URIs that no longer exist in the data

**Consequences:** Researchers download CSV data that does not match reality. Plots show misleading trends because a property URI was renamed. The dashboard reports "0 stressors" because the NCI ontology URI changed. These failures are silent -- no error is thrown because the queries return valid (but empty) results.

**Warning signs:**
- A plot that used to show data now shows "No data available" without any code change
- Entity counts don't match between the dashboard and direct SPARQL queries
- Property presence percentages suddenly drop to 0% for categories that were previously well-populated
- `verify_properties.py` (the one-off validation script) fails but nobody runs it regularly

**Prevention:**
1. Add pytest with a minimal test suite BEFORE adding network analysis features. Priority tests:
   - Unit: Query string construction produces valid SPARQL syntax
   - Unit: `extract_counts()` handles edge cases (empty results, missing fields, non-numeric values)
   - Integration: Each plot function returns non-empty results against a test endpoint
   - Smoke: `/health` endpoint returns 200 with valid JSON
   - Contract: Known entity counts for a specific version match expected values
2. Run `verify_properties.py` as a scheduled check (weekly) or as part of a CI pipeline.
3. Add "data freshness" assertions -- if the latest version date is more than 90 days old, flag it on the status page.

**Detection:** Automated CI that runs tests on every commit or at minimum nightly against the live endpoint.

**Phase relevance:** Should be addressed in the first phase, BEFORE network analysis and deployment.

---

## Moderate Pitfalls

Issues that cause significant rework, poor UX, or operational headaches but do not require full rewrites.

---

### Pitfall 6: Virtuoso-Specific SPARQL Patterns Creating Vendor Lock-In

**What goes wrong:** The codebase uses Virtuoso-specific optimizations: `GROUP_CONCAT` with pipe separators, `STRSTARTS()` for graph filtering, `UNION` query structuring optimized for Virtuoso's query planner, and reliance on Virtuoso's specific timeout behavior. Migrating to another triplestore (GraphDB, Jena Fuseki, Blazegraph) would require rewriting many queries.

**Prevention:**
1. Document which query patterns are Virtuoso-specific in code comments (some already have optimization comments).
2. Abstract graph filtering into a shared function (partially done with `_build_graph_filter()`). Extend this pattern to all Virtuoso-specific constructs.
3. Keep a compatibility test suite that can run against different SPARQL endpoints.
4. For network analysis, do NOT use Virtuoso property paths (`^` inverse, `*` transitive) -- these are notoriously slow on Virtuoso and non-portable. Fetch raw edges and compute paths in Python.

**Phase relevance:** VHP4Safety deployment (Issue #3). The deployment target may use a different SPARQL endpoint.

---

### Pitfall 7: Startup-Time Plot Computation Blocking Container Readiness

**What goes wrong:** The application computes all 22 plots at startup in `compute_plots_parallel()` before Flask starts accepting requests. With the slowest plot at 75 seconds, this means the container is not ready for 75+ seconds after start. Health checks, load balancers, and orchestrators may time out and restart the container, creating a restart loop.

**Prevention:**
1. Move ALL trend plots to lazy loading (the infrastructure already exists for latest plots).
2. Have Flask start accepting requests immediately with a "loading" state.
3. Compute plots in a background thread after Flask is ready.
4. Return the health endpoint as "starting" (HTTP 503 with Retry-After header) during computation, then "healthy" (HTTP 200) when complete.
5. In Kubernetes/Docker deployment, set readiness probe timeout to at least 120 seconds, and use a separate liveness probe that just checks if Flask is responding.

**Warning signs:**
- Container restarts repeatedly in Kubernetes with "CrashLoopBackOff"
- Health check returns 503 for the first 2 minutes after deploy
- `docker-compose up` takes over a minute before the dashboard is accessible

**Phase relevance:** VHP4Safety deployment (Issue #3). This WILL block deployment if the platform has standard readiness probe timeouts (typically 30-60 seconds).

---

### Pitfall 8: Flask Development Server in Production

**What goes wrong:** The Dockerfile runs `flask run --host=0.0.0.0`, which uses Flask's built-in Werkzeug development server. This server is single-threaded, has no graceful shutdown, no worker management, no request queuing, and is explicitly documented as not suitable for production by the Flask project.

**Prevention:**
1. Replace `flask run` with a production WSGI server. Use `gunicorn` with the command: `gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:5000 app:app`.
2. Add `gunicorn` to `requirements.txt`.
3. Set worker timeout high enough to survive the 75-second startup computation (120 seconds).
4. Use `--preload` flag to share computed plots across workers, avoiding recomputation per worker.

**Warning signs:**
- Flask prints "WARNING: This is a development server. Do not use it in a production deployment." in Docker logs
- Under concurrent load (>2 users), requests queue and timeout
- Long-running SPARQL queries block all other requests

**Phase relevance:** VHP4Safety deployment (Issue #3). This is a deployment blocker.

---

### Pitfall 9: Unpinned Dependencies Causing Build Irreproducibility

**What goes wrong:** `requirements.txt` contains only package names without version pins (`flask`, `pandas`, `plotly`, `SPARQLWrapper`). A fresh `docker build` or `pip install` will pull the latest versions, which may have breaking changes, different behavior, or new dependencies that conflict.

**Prevention:**
1. Pin all direct dependencies with exact versions: `flask==3.1.0`, `pandas==2.2.3`, `plotly==5.24.1`, `SPARQLWrapper==2.0.0`.
2. Generate a lockfile: `pip freeze > requirements-lock.txt` and use that in the Dockerfile.
3. Add `requests` explicitly to requirements.txt -- it is imported in `plots/shared.py` but not listed (it comes transitively through SPARQLWrapper, which could change).
4. When adding NetworkX, pin it explicitly: `networkx==3.4.2`.

**Warning signs:**
- Docker build succeeds locally but fails in CI
- "Works on my machine" -- different team members get different behavior
- A deploy after weeks of no changes suddenly breaks because a dependency released a new major version

**Phase relevance:** Immediate. Should be fixed before any new dependencies are added.

---

### Pitfall 10: SPARQL Query String Injection via Version Parameter

**What goes wrong:** The `_build_graph_filter()` function in `latest_plots.py` interpolates the `version` parameter directly into SPARQL queries via f-string: `f'FILTER(?graph = <http://aopwiki.org/graph/{version}>)'`. While the API endpoint receives version from `request.args.get('version')`, there is no validation that the version string is a valid date format. A crafted version string could inject arbitrary SPARQL.

**Prevention:**
1. Validate the version parameter against a regex pattern: `^\d{4}-\d{2}-\d{2}$` (ISO date format matching the graph naming convention).
2. Reject any version not in the known versions list (available from `get_all_versions()`).
3. Use parameterized queries if SPARQLWrapper supports them (it partially does via `addParameter()`), or at minimum sanitize by stripping all characters except alphanumeric and hyphens.

**Warning signs:**
- Unusual entries in server logs showing version strings with angle brackets, quotes, or SPARQL keywords
- Unexpected SPARQL errors from the endpoint when processing version-specific requests

**Phase relevance:** Before VHP4Safety deployment. Public-facing dashboard must validate all user input.

---

### Pitfall 11: Network Analysis Metrics Without Domain Context Are Misleading

**What goes wrong:** Applying generic graph centrality metrics (betweenness, PageRank, clustering coefficient) to the AOP-Wiki network without understanding AOP semantics produces misleading results. A KE with high betweenness centrality may be an artifact of the AOP structure (e.g., "cell death" appears in many AOPs because it is a common downstream event, not because it is scientifically "central"). PageRank in a directed AOP graph has different semantics than in a web link graph.

**Why it happens:** Network analysis tools compute mathematical properties of graph topology. They do not know that AOPs have biological semantics, that KERs are directional cause-effect relationships, or that OECD assessment status should weight the analysis.

**Prevention:**
1. Work with domain experts (toxicologists, AOP curators) to define which metrics are meaningful for AOPs. Document these decisions.
2. Present metrics with biological context labels, not just "betweenness centrality = 0.42". Example: "This Key Event connects N upstream causes to M downstream adverse outcomes."
3. Weight edges by KER confidence/evidence level if available in the RDF data (e.g., weight of evidence).
4. Provide comparison baselines: "The average KE connects to X AOPs; this KE connects to Y."
5. Filter out trivially connected nodes before computing metrics -- common MIE (Molecular Initiating Events) and common AO (Adverse Outcomes) will dominate all centrality rankings.

**Warning signs:**
- Users report that "the most central KE" is something obvious like cell death
- PageRank results look nearly uniform (all nodes similar) or are dominated by 2-3 hub nodes
- Scientists say the analysis "doesn't tell them anything new"

**Phase relevance:** Network Analysis Expansion (Issue #11). Must be addressed during design, not after implementation.

---

## Minor Pitfalls

Issues that cause inconvenience or minor bugs but are easily recoverable.

---

### Pitfall 12: JavaScript-Backend Plot Name Synchronization

**What goes wrong:** The `versionedPlots` array in `version-selector.js` must exactly match the plot names in `latest_plots_with_version` dictionary in `app.py`. When a new plot is added to one but not the other, the version selector silently fails to update that plot, or attempts to load a non-existent plot and receives a 404.

**Prevention:**
1. Generate the plot list from a single source of truth. Add a Flask endpoint `/api/plot-names` that returns the list of available versioned plots as JSON, and have JavaScript fetch it instead of hardcoding.
2. Until then, add a startup assertion in `app.py` that compares the plot_map keys against a known list.

**Phase relevance:** Whenever new plots are added (network analysis, new visualizations).

---

### Pitfall 13: Duplicate Flask Imports in app.py

**What goes wrong:** Lines 41 and 63 of `app.py` both import from Flask. While Python handles this gracefully (the second import is a no-op), it signals copy-paste development and makes it easy to miss import-related issues during code review.

**Prevention:** Consolidate into a single import block at the top of the file during the legacy cleanup phase.

**Phase relevance:** Code cleanup phase (remove legacy plots.py and consolidate).

---

### Pitfall 14: CORS Blocking Cross-Domain Embedding

**What goes wrong:** If the VHP4Safety platform embeds the dashboard in an iframe or makes AJAX requests from a different domain, browser CORS policy will block the requests. The dashboard currently has no CORS headers configured.

**Prevention:**
1. Add `flask-cors` dependency and configure allowed origins for the VHP4Safety domain.
2. Alternatively, deploy the dashboard on the same domain/subdomain as the VHP platform.
3. Test cross-origin access before declaring deployment complete.

**Phase relevance:** VHP4Safety deployment (Issue #3).

---

### Pitfall 15: Docker Image Hardcoded to Host Bridge Network for SPARQL

**What goes wrong:** The `docker-compose.yml` uses `SPARQL_ENDPOINT=http://172.17.0.1:8890/sparql`, which is the default Docker bridge gateway IP on Linux. This does not work on macOS/Windows Docker Desktop (which uses `host.docker.internal`), nor in Kubernetes or cloud deployments where the SPARQL endpoint has a DNS name.

**Prevention:**
1. Do not hardcode the endpoint in docker-compose.yml. Use environment variable override or a `.env` file.
2. Document the required SPARQL_ENDPOINT configuration for different deployment environments.
3. For VHP platform deployment, use the platform's service discovery mechanism (e.g., Kubernetes DNS name).

**Phase relevance:** VHP4Safety deployment (Issue #3).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Severity | Mitigation |
|-------------|---------------|----------|------------|
| Legacy cleanup | Pitfall 4 (divergent plots.py) | HIGH | Delete `plots.py` first, grep for stale imports |
| Legacy cleanup | Pitfall 9 (unpinned deps) | MEDIUM | Pin before adding new deps |
| Legacy cleanup | Pitfall 13 (duplicate imports) | LOW | Consolidate during cleanup pass |
| SPARQL optimization | Pitfall 1 (query plan explosion) | CRITICAL | Per-version queries, aggregate in Python |
| SPARQL optimization | Pitfall 6 (Virtuoso lock-in) | MEDIUM | Abstract Virtuoso-specific patterns |
| Cache management | Pitfall 3 (unbounded cache) | HIGH | LRU eviction before adding more cache entries |
| Network analysis | Pitfall 2 (NetworkX memory bomb) | CRITICAL | Pre-compute, use sparse matrices, set memory limits |
| Network analysis | Pitfall 1 (Virtuoso query explosion) | CRITICAL | Fetch raw edges, compute metrics in Python |
| Network analysis | Pitfall 11 (meaningless metrics) | HIGH | Domain expert consultation, contextual labels |
| Testing | Pitfall 5 (silent regression) | HIGH | Pytest suite before new features |
| VHP deployment | Pitfall 7 (startup blocking) | HIGH | Lazy loading for all plots, background computation |
| VHP deployment | Pitfall 8 (dev server in prod) | HIGH | Switch to gunicorn |
| VHP deployment | Pitfall 10 (SPARQL injection) | MEDIUM | Validate version parameter format |
| VHP deployment | Pitfall 14 (CORS) | MEDIUM | flask-cors with VHP domain |
| VHP deployment | Pitfall 15 (Docker networking) | MEDIUM | Configurable endpoint, no hardcoded IP |

## Recommended Pitfall Mitigation Order

Based on dependency analysis and severity:

1. **Immediate (before any feature work):** Pitfall 4 (delete legacy plots.py), Pitfall 9 (pin dependencies)
2. **Foundation phase:** Pitfall 3 (cache eviction), Pitfall 5 (basic test suite), Pitfall 8 (gunicorn)
3. **Network analysis phase:** Pitfall 2 (NetworkX memory), Pitfall 1 (per-version queries), Pitfall 11 (domain context)
4. **Deployment phase:** Pitfall 7 (startup readiness), Pitfall 10 (input validation), Pitfall 14 (CORS), Pitfall 15 (Docker networking)
5. **Ongoing:** Pitfall 6 (vendor lock-in documentation), Pitfall 12 (JS-backend sync)

## Sources

- Codebase analysis of `app.py`, `plots/shared.py`, `plots/trends_plots.py`, `plots/latest_plots.py`, `config.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`
- Existing `.planning/codebase/CONCERNS.md` (documents known issues including removed features, memory growth, missing tests)
- Existing `.planning/codebase/TESTING.md` (documents 0% test coverage)
- Existing `.planning/codebase/INTEGRATIONS.md` (documents SPARQL integration patterns)
- Existing `.planning/PROJECT.md` (documents project constraints and active requirements)
- Virtuoso SPARQL optimization patterns (HIGH confidence -- training data verified against observed codebase behavior)
- NetworkX memory characteristics for centrality computation (HIGH confidence -- well-documented algorithmic properties)
- Flask production deployment best practices (HIGH confidence -- Flask official documentation)
- SPARQL injection risk patterns (MEDIUM confidence -- theoretical based on code pattern, no exploit verified)

---

*Pitfalls audit: 2026-02-20*
