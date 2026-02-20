# Project Research Summary

**Project:** AOP-Wiki RDF Dashboard — Performance, Network Analysis, and Platform Deployment
**Domain:** SPARQL-based RDF data monitoring dashboard with interactive network visualization
**Researched:** 2026-02-20
**Confidence:** MEDIUM

## Executive Summary

The AOP-Wiki RDF Dashboard is a mature Flask/Plotly application that needs to evolve from a "monitoring tool" into an "exploration and analysis platform." The recommended approach is evolutionary, not revolutionary: extend the existing stack by adding a service layer, NetworkX-based graph analysis, and a REST API, while simultaneously paying down critical technical debt that has already forced feature removal. The stack itself is not changing — Flask, Plotly, SPARQLWrapper against Virtuoso — but the architecture needs three new layers (services/, api/ Blueprint, frontend JS modules) to support network analysis, entity deep-dives, and version comparison without collapsing under SPARQL query timeouts.

The highest-risk element is Virtuoso's query planner behavior under aggregation load. This has already caused one feature removal (`plot_aop_completeness_boxplot_by_status`) and a 75-second load time for the completeness plot. Every new feature that involves cross-version joins or multi-hop graph traversals faces this same ceiling. The research is unanimous: the safe pattern is per-version SPARQL queries aggregated in Python, never cross-version SPARQL joins. Network analysis metrics must be computed server-side in Python (NetworkX), not in SPARQL property paths, and not in browser JavaScript.

The second critical constraint is that three immediate housekeeping items must precede any feature work: (1) delete the legacy `plots.py` monolith that shadows the refactored `plots/` package, (2) pin all dependencies in `requirements.txt`, and (3) implement LRU cache eviction on the unbounded in-memory caches. Without these, adding network analysis will push memory usage past container limits and create import ambiguity between the legacy and refactored code. The recommended phase order is: Foundation (cleanup + infrastructure) → Network Analysis → Entity Deep-Dives → Version Comparison → Production Deployment.

## Key Findings

### Recommended Stack

The project's existing stack (Flask, Plotly, Pandas, SPARQLWrapper, Cytoscape.js) is correct and should not change. The key additions are NetworkX for graph computation, Gunicorn for production serving, and pytest for a missing test suite. See `.planning/research/STACK.md` for full rationale and version recommendations.

**Core technologies:**
- **NetworkX >=3.2**: Graph algorithms (centrality, PageRank, community detection) — pure Python, no Docker build complications, proven on AOP-scale graphs (<10K nodes), all required algorithms built-in
- **python-louvain >=0.16**: Community detection for AOP network clustering — lightweight, implements Louvain method on top of NetworkX
- **Cytoscape.js (upgrade 3.26.0 → >=3.28)**: Interactive network rendering — already in codebase, purpose-built for biological network visualization, extend rather than replace
- **Gunicorn >=22.0**: Production WSGI server — Flask's dev server (`flask run`) is currently in the Dockerfile, which is a deployment blocker for VHP platform
- **Flask-Caching >=2.1**: Bounded cache with TTL eviction — replaces unbounded `_plot_data_cache` dict that grows until OOM
- **pytest >=8.0 + pytest-flask >=1.3**: Test framework — project has zero automated tests, required before adding features that compound existing SPARQL query risks

**Critical version pinning needed:** All packages in `requirements.txt` are currently unpinned. A fresh Docker build can pull breaking changes. This must be fixed before adding new dependencies.

### Expected Features

The dashboard already provides 30+ visualizations, lazy loading, CSV/PNG/SVG export, version selection, and health monitoring. Research identifies the next milestone's feature priorities. See `.planning/research/FEATURES.md` for full breakdown.

**Must have (table stakes — complete what already exists):**
- Reliable plot loading — the 75-second completeness plot and removed boxplot leave the dashboard feeling incomplete
- Consistent error messaging — standardize all fallback states to `create_fallback_plot()` with actionable messages
- Full export coverage — audit gaps between displayed plots and available download routes
- Plot methodology descriptions — scientists and regulators require transparency about what each visualization measures
- Direct links to AOP-Wiki — users cannot navigate to source data from entity references

**Should have (differentiators that define the next milestone):**
- Interactive AOP network graph — no existing tool visualizes the AOP-Wiki RDF network interactively; this is the flagship feature
- Node centrality metrics (degree, betweenness, closeness) and PageRank — identifies which Key Events are structurally critical; directly supports regulatory prioritization
- Individual entity detail views — enables drill-down from any aggregate visualization into KE/AOP/KER specifics
- Community/cluster detection — groups related AOPs, helps curators identify knowledge gaps
- Version-to-version diff summary — high curator demand; shows what changed between releases

**Defer to v2+:**
- Custom SPARQL query interface — high complexity, niche audience (bioinformaticians only)
- Report generation (PDF/HTML) — depends on having the analysis features to report on
- Annotation timeline per entity — computationally expensive cross-version analysis
- Path analysis between KEs — requires mature network implementation and careful UX for complex traversals

### Architecture Approach

The architecture should evolve by adding layers, not replacing working ones. Introduce a `services/` layer (QueryService, CacheManager, NetworkService, EntityService, ComparisonService) to decouple data acquisition from visualization, and a new `api/` Flask Blueprint to serve JSON endpoints for network data and entity details. The existing plot/route structure remains unchanged. See `.planning/research/ARCHITECTURE.md` for full component specifications.

**Major components:**
1. **services/sparql.py (QueryService)** — Extracted from `plots/shared.py`; centralized SPARQL execution with retry and version parameterization; foundation for all new components
2. **services/cache.py (CacheManager)** — Replaces unbounded global dicts with TTL-based LRU eviction; built in parallel with API layer
3. **services/network.py (NetworkService)** — Builds NetworkX DiGraph from SPARQL results, computes centrality/PageRank/communities, serializes to Cytoscape.js JSON format; highest-value new component
4. **api/ Blueprint** — Flask Blueprint (prefix `/api/v1`) with endpoints for network graph, entity details, and version comparison; does not modify existing routes in `app.py`
5. **static/js/ modules** — `network-viewer.js` (Cytoscape.js wrapper), `entity-detail.js` (slide-in panel), `version-comparison.js` — replaces the 570-line f-string Cytoscape.js implementation in legacy `plots.py`

**Key architectural pattern:** Query one version at a time in Python; aggregate/compute in Python; never compare named graphs in a single SPARQL query.

### Critical Pitfalls

See `.planning/research/PITFALLS.md` for all 15 pitfalls with detection signals and phase-specific warnings.

1. **Virtuoso multi-version query plan explosion** — Any SPARQL query that JOINs or aggregates across multiple named graphs will timeout or get removed. Prevention: always query one version at a time and diff in Python. This is the root cause of all existing performance problems.
2. **NetworkX in-process memory bomb** — Computing betweenness centrality in a Flask request handler can spike memory to multi-GB and block the WSGI worker. Prevention: pre-compute network metrics at startup in a background thread, store results as DataFrames, use `nx.to_scipy_sparse_array()` for large graphs.
3. **Unbounded in-memory cache leading to gradual OOM** — `_plot_data_cache` grows indefinitely with version-specific variants; `clear_plot_cache()` exists but is never called. Prevention: implement LRU eviction before adding network analysis cache entries.
4. **Legacy plots.py import divergence** — The 4,194-line `plots.py` monolith coexists with the `plots/` package; modifying the wrong file causes silent stale data. Prevention: delete `plots.py` immediately, before any feature work.
5. **Flask dev server in production** — Dockerfile currently runs `flask run`, which is single-threaded and not suitable for production. Prevention: switch to `gunicorn --workers 2 --threads 4 --timeout 120` before VHP deployment.

## Implications for Roadmap

Based on cross-research synthesis, the following phase structure is recommended. The ordering is driven by three hard dependencies: (1) cleanup must precede features that compound existing debt, (2) the service layer must be extracted before any new consumers can use it, (3) network analysis is highest-value and must precede entity drill-down (which clicks through from network nodes).

### Phase 1: Foundation and Cleanup

**Rationale:** Three critical pitfalls (legacy divergence, unpinned deps, unbounded cache) make any new feature work risky. Adding network analysis to the existing architecture will push memory past container limits and create import ambiguity. These are true blockers, not nice-to-haves. Additionally, the test suite must be established before adding SPARQL-heavy features that have already caused one feature removal.

**Delivers:**
- Reproducible Docker builds (pinned dependencies)
- Eliminated import ambiguity (legacy `plots.py` deleted)
- Bounded memory growth (LRU cache eviction)
- Production-ready WSGI server (Gunicorn in Dockerfile)
- Minimal test suite (smoke tests, query construction tests, CSV export integrity)
- Generic download route handler (eliminates 800+ lines of duplicated boilerplate)
- Consolidated `app.py` imports and standardized error messaging

**Addresses from FEATURES.md:** Reliable plot loading, consistent error messaging, full export coverage

**Avoids from PITFALLS.md:** Pitfall 4 (legacy divergence), Pitfall 9 (unpinned deps), Pitfall 3 (unbounded cache), Pitfall 8 (dev server), Pitfall 5 (silent regression)

**Research flag:** Standard patterns, no additional research needed. All tasks are cleanup of known issues.

### Phase 2: Service Layer Extraction

**Rationale:** QueryService and CacheManager are prerequisites for all new features (network analysis, entity detail, version comparison). They must exist before Phase 3 can build on them. Extracting them also enables Phase 1's generic download handler to use a unified data access pattern.

**Delivers:**
- `services/sparql.py` — QueryService with retry, timeout, version parameterization
- `services/cache.py` — CacheManager with TTL, LRU eviction, memory monitoring
- `api/` Blueprint skeleton registered on existing `app.py`
- SPARQL query validation using regex for version parameter (security fix for Pitfall 10)
- SPARQL injection prevention (version parameter validated against ISO date regex)

**Uses from STACK.md:** Flask Blueprint (no new dependencies), optional Flask-Caching for managed cache backend

**Implements from ARCHITECTURE.md:** QueryService, CacheManager, api/ Blueprint skeleton

**Avoids from PITFALLS.md:** Pitfall 10 (SPARQL injection), Pitfall 6 (Virtuoso vendor lock-in — abstract Virtuoso-specific patterns into QueryService)

**Research flag:** Standard patterns, no additional research needed. Flask Blueprint and service layer extraction are well-documented patterns.

### Phase 3: Network Analysis Expansion (Issue #11)

**Rationale:** Highest-value new feature. The interactive AOP network graph is the flagship differentiator — no existing tool provides this view of AOP-Wiki RDF data. Building it requires Phase 1 (stable foundation) and Phase 2 (QueryService for SPARQL data, CacheManager for pre-computed metrics). Network analysis metrics must be pre-computed, not computed on-demand in request handlers.

**Delivers:**
- `/network` page with interactive Cytoscape.js visualization
- `services/network.py` — NetworkService building NetworkX DiGraph from KE/KER SPARQL queries
- Pre-computed metrics: degree centrality, betweenness centrality, closeness centrality, PageRank, community detection (Louvain)
- `api/v1/network/graph` and `/api/v1/network/metrics` endpoints returning Cytoscape.js JSON
- `static/js/network-viewer.js` — replaces 570-line f-string implementation in deleted legacy file
- Shared KE heatmap (Plotly, cross-tabulation of AOP-KE membership)
- Network metrics sidebar with domain-contextualized labels (Pitfall 11 mitigation)
- Export: network metrics as CSV, subgraph as JSON

**Uses from STACK.md:** NetworkX >=3.2, python-louvain >=0.16, Cytoscape.js upgrade, cytoscape-cola layout plugin

**Implements from ARCHITECTURE.md:** NetworkService, network API endpoints, network-viewer.js, server-compute/client-render pattern

**Avoids from PITFALLS.md:** Pitfall 2 (NetworkX memory bomb — pre-compute at startup), Pitfall 1 (Virtuoso query explosion — fetch raw edges, compute in Python), Pitfall 4 (already deleted in Phase 1), Pitfall 11 (meaningless metrics — domain-contextualized labels and comparison baselines)

**Research flag:** Needs phase-level research before planning. Specifically: (a) AOP-Wiki graph topology and typical size (node/edge count per version), (b) which centrality metrics are scientifically meaningful for AOP networks (requires domain expert input), (c) memory profiling strategy for NetworkX on actual data size.

### Phase 4: Entity Deep-Dives and Data Quality

**Rationale:** Entity detail views are the natural click-through target from network nodes (Phase 3). They also address the "individual entity detail views" table-stakes gap identified in features research. Building this after network analysis means entity panels can be linked from network node clicks from day one.

**Delivers:**
- `services/entities.py` — EntityService for single-entity SPARQL queries (KE, AOP, KER, Stressor)
- `api/v1/entity/<type>/<id>` endpoints with completeness scores and connected entities
- Slide-in detail panel (JavaScript, renders alongside network graph without navigation)
- Direct links to AOP-Wiki entity pages (`https://aopwiki.org/aops/[ID]`, etc.)
- Responsive data table view beneath each plot (toggleable, shows cached DataFrame)
- Data quality scoring (composite per-AOP score combining completeness + annotation depth + connectivity)

**Addresses from FEATURES.md:** Individual entity detail views, direct links to AOP-Wiki, responsive table view, data quality scoring

**Avoids from PITFALLS.md:** Pitfall 1 is not a concern here (single-entity queries against a single named graph are guaranteed fast — Virtuoso indexes by subject)

**Research flag:** Standard patterns for entity detail APIs. Single-entity SPARQL queries are well-documented. No additional research needed.

### Phase 5: Version Comparison and Cross-Version Analysis

**Rationale:** Version comparison requires both entity services (Phase 4) and network analysis (Phase 3) to produce meaningful diffs that include structural changes. Cross-version queries are the highest SPARQL risk — must apply the "query each version separately, diff in Python" pattern rigorously. Curator demand is high; this enables audit trail workflows required by regulators.

**Delivers:**
- `services/comparison.py` — ComparisonService using per-version queries, diffing in Python
- `api/v1/compare` endpoint returning entity-level diffs and structural metric deltas
- Side-by-side version comparison UI (two-panel view)
- Entity changelog (filterable table: added/modified/removed entities between versions)
- Version-to-version diff summary (headline metrics: entity count deltas, completeness score changes)

**Addresses from FEATURES.md:** Version-to-version diff summary, entity changelog, side-by-side comparison, completeness trend per entity

**Avoids from PITFALLS.md:** Pitfall 1 (never cross-version SPARQL — query each version separately, set diff in Python), Pitfall 3 (each version comparison pair could expand cache — apply TTL eviction)

**Research flag:** Standard patterns for version diffing. The "query-once, diff-in-Python" pattern is established in this codebase. No additional research needed.

### Phase 6: Production Deployment (Issue #3)

**Rationale:** VHP4Safety platform deployment has specific infrastructure requirements that are separate from feature work. Gunicorn is already added in Phase 1, but deployment validation, CORS configuration, Docker networking, and health probe tuning require integration with the VHP platform environment.

**Delivers:**
- Production-validated Docker image (Gunicorn, correct worker/timeout config for VHP CPU allocation)
- CORS headers for VHP4Safety domain (flask-cors with allowed origins configured)
- Configurable SPARQL endpoint (replace hardcoded Docker bridge IP with environment variable)
- Health probe compatibility (startup vs readiness probe separation, 120-second readiness timeout)
- VHP platform integration testing

**Uses from STACK.md:** Gunicorn >=22.0, Flask-CORS >=4.0

**Avoids from PITFALLS.md:** Pitfall 7 (startup blocking readiness probe — all trend plots lazy-loaded by this phase), Pitfall 8 (dev server — already replaced in Phase 1), Pitfall 14 (CORS), Pitfall 15 (Docker networking)

**Research flag:** Needs deployment-specific research during planning. Open questions: (a) what container orchestration does VHP use (Kubernetes vs Docker Compose, affects Gunicorn worker config and probe format), (b) VHP4Safety subdomain configuration (affects CORS allowed origins), (c) whether Cytoscape.js should be bundled locally vs CDN for the deployment environment.

### Phase Ordering Rationale

- **Phase 1 before everything**: Three pitfalls are true blockers. Legacy `plots.py` divergence will cause mysterious bugs in new code. Unpinned deps will cause irreproducible builds when new dependencies are added. Unbounded cache will OOM when network analysis adds entries.
- **Phase 2 before Phase 3+**: QueryService and CacheManager are prerequisite services. NetworkService, EntityService, and ComparisonService all depend on them. Building them first means no duplication.
- **Phase 3 before Phase 4**: Entity detail panels make most sense as click-through targets from network nodes. Building Phase 4 after Phase 3 allows entity panels to be wired to the network graph from day one.
- **Phase 4 before Phase 5**: Version comparison diffs need entity-level data structures (Phase 4) and network metrics (Phase 3) to produce complete diffs.
- **Phase 6 last**: VHP deployment validation requires feature-complete functionality. Running deployment integration against an incomplete feature set wastes time.
- **All of Phase 1-2 avoids the SPARQL pitfall**: The pattern "query one version, aggregate in Python" is established before new complex features add new SPARQL query load.

### Research Flags

Phases needing deeper research during planning:

- **Phase 3 (Network Analysis):** AOP-Wiki graph topology needs profiling on actual data (node/edge counts per version). Which centrality metrics are scientifically meaningful for directed AOP networks needs domain expert input before UI labels are finalized. Memory usage of NetworkX on actual graph size needs measurement before committing to in-process pre-computation.
- **Phase 6 (VHP Deployment):** VHP4Safety infrastructure specifics (orchestration technology, probe configuration, CORS domain) are unknown. These must be resolved with VHP4Safety platform team before deployment planning.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** All tasks are cleanup of documented known issues. Patterns are established.
- **Phase 2 (Service Layer):** Flask Blueprint and service extraction are well-documented. Existing codebase has clear extraction targets.
- **Phase 4 (Entity Deep-Dives):** Single-entity SPARQL queries are straightforward. Pattern is well-documented.
- **Phase 5 (Version Comparison):** "Query-once, diff-in-Python" pattern is established in this codebase and confirmed safe by pitfalls research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core stack is verified from codebase. NetworkX, Gunicorn, and pytest are standard choices with no viable alternatives at this scale. Exact version numbers for Cytoscape.js plugins are LOW confidence — verify at implementation time. |
| Features | MEDIUM | Table stakes are HIGH confidence (derived from codebase gaps). Network analysis features are MEDIUM confidence — NetworkX computation approach is established but Virtuoso query performance for graph extraction on actual data is unproven. Competitor analysis (AOPXplorer, AOP-helpFinder) could not be verified (no web search). |
| Architecture | MEDIUM | Service layer extraction and Blueprint pattern are HIGH confidence (standard Flask patterns). NetworkX memory characteristics on AOP-Wiki's actual graph size are MEDIUM confidence — algorithmic analysis is correct but empirical profiling is needed. |
| Pitfalls | HIGH | SPARQL pitfalls are HIGH confidence — directly evidenced by removed features and 75-second query. NetworkX memory characteristics are HIGH confidence (algorithmic properties are documented). Flask dev server and dependency pinning are HIGH confidence (official documentation). SPARQL injection is MEDIUM confidence (theoretical, not exploited). |

**Overall confidence:** MEDIUM — research conclusions are well-grounded for planning purposes. Two open questions (AOP network actual size, VHP platform infrastructure) require empirical discovery during implementation.

### Gaps to Address

- **AOP-Wiki network graph size**: How many KEs, KERs, and AOPs exist in the current and projected dataset? This determines whether NetworkX in-process computation is safe or requires scipy sparse matrix optimization. Measure at Phase 3 planning time with a simple SPARQL COUNT query.
- **VHP4Safety platform specifics**: Container orchestration technology, readiness probe timeout, CORS domain configuration, and Cytoscape.js CDN availability in the deployment environment. Resolve with VHP4Safety team before Phase 6 planning.
- **Domain expert input on network metrics**: Which centrality measures are scientifically meaningful for directed AOP networks? This affects Phase 3 UI design and the risk of Pitfall 11 (misleading metrics). Consult AOP curators before finalizing network analysis feature design.
- **Virtuoso configuration**: Are `MaxQueryExecutionTime`, `MaxQueryMem`, and `ResultSetMaxRows` tunable in the deployment environment? Knowing these limits informs whether query decomposition alone is sufficient or if server-side configuration changes are also needed.
- **Cytoscape.js plugin versions**: `cytoscape-cola` and `cytoscape-popper` exact version compatibility with the upgraded Cytoscape.js version. Verify at implementation time; do not rely on training-data version numbers.

## Sources

### Primary (HIGH confidence)
- `app.py`, `plots/shared.py`, `plots/trends_plots.py`, `plots/latest_plots.py`, `plots/__init__.py` — direct codebase analysis
- `.planning/codebase/CONCERNS.md` — documented known issues (OOM risk, removed features, missing tests)
- `.planning/codebase/TESTING.md` — confirms 0% test coverage
- `.planning/codebase/ARCHITECTURE.md` — existing architecture documentation
- `.planning/PROJECT.md` — active requirements and project constraints
- `Dockerfile`, `docker-compose.yml`, `requirements.txt` — deployment configuration analysis
- `CLAUDE.md` — established patterns and constraints

### Secondary (MEDIUM confidence)
- Training data on NetworkX algorithms (centrality, PageRank, community detection) — stable library, API unlikely to have changed significantly
- Training data on Flask Blueprint patterns — standard Flask feature, well-documented
- Training data on Virtuoso query optimization patterns — verified against observed codebase behavior
- Domain knowledge of AOP-Wiki ecosystem and OECD AOP programme

### Tertiary (LOW confidence)
- Cytoscape.js exact version numbers (3.30.4) — training data only; verify at implementation
- `cytoscape-cola` and `cytoscape-popper` plugin version compatibility — verify at implementation
- Competitor tool current state (AOPXplorer, AOP-helpFinder, CompTox) — no web search available; differentiation claims are MEDIUM confidence

---
*Research completed: 2026-02-20*
*Ready for roadmap: yes*
