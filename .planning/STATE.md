# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-20)

**Core value:** Researchers, curators, and regulators can explore AOP-Wiki data across any version to understand how the knowledge base evolves over time — reliably and without timeouts.
**Current focus:** Phase 3: Network Analysis (All gap closures complete) -- Next: Phase 4: Entity Exploration

## Current Position

Phase: 3 of 6 (Network Analysis)
Plan: 5 of 5 in current phase (COMPLETE)
Status: Phase Complete (all gap closures done)
Last activity: 2026-02-23 — Completed 03-05-PLAN.md (Network metrics CSV cache fix)

Progress: [████████████████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: ~4.3 min
- Total execution time: ~1.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 4/4 | ~19 min | ~5 min |
| 02 | 7/7 | ~29 min | ~4 min |
| 03 | 5/5 | ~41 min | ~8 min |

**Recent Trend:**
- Last 5 plans: 03-01 (3m), 03-02 (4m), 03-03 (3m), 03-04 (30m), 03-05 (1m)
- Trend: 03-05 was a small targeted fix (cache re-population)

*Updated after each plan completion*
| Phase 02 P05 | 9min | 2 tasks | 2 files |
| Phase 02 P07 | 2min | 2 tasks | 3 files |
| Phase 03 P01 | 3min | 2 tasks | 3 files |
| Phase 03 P02 | 4min | 2 tasks | 6 files |
| Phase 03 P03 | 3min | 1 tasks | 1 files |
| Phase 03 P04 | 30min | 2 tasks | 7 files |
| Phase 03 P05 | 1min | 1 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Phases 2 and 3 can execute in parallel (both depend only on Phase 1)
- [Roadmap]: Service layer extraction folded into consuming phases rather than standalone phase (no user-observable outcome on its own)
- [Roadmap]: Version comparison deferred to v2 (user prefers trends across all versions over 1-to-1 comparisons)
- [Roadmap]: Network analysis Phase 3 includes performance benchmarking criteria (user flagged prior performance issues with interactive graphs)
- [01-02]: VersionedPlotCache uses TTL=1800s and max_versions=5 for bounded memory growth
- [01-02]: Health endpoint returns 503 when SPARQL is down (honest reporting, may trigger container restarts)
- [01-02]: JSON logging configured centrally in config.py before any other imports
- [01-04]: Generalized tuple fallback by counting str elements in annotation rather than hardcoding function names
- [01-04]: Detect 'Data Unavailable' sentinel string in fallback HTML to distinguish from legitimate content
- [02-02]: Used request.args.get('version') uniformly for all routes; trend routes get None, latest routes get version from version-selector.js
- [02-02]: 12 data cache keys were already present (not 20 as estimated); only 6 functions needed updates
- [Phase 02-01]: Used per-version parallel SPARQL queries (4 workers) instead of single cross-version query to avoid Virtuoso limits
- [Phase 02-01]: Line chart with marker shapes for OECD status visualization (aggregated means vs raw distributions)
- [02-03]: Methodology notes stored in single JSON file (28 entries) for centralized maintainability
- [02-03]: Used native HTML details/summary for zero-JS collapsibility of methodology sections
- [02-03]: Virtuoso tuning documented as recommendations only, not applied in Docker config
- [02-04]: All trend plot functions wrapped in try/except with create_fallback_plot for graceful degradation
- [02-04]: Used errors='coerce' universally on pd.to_datetime calls for malformed date safety
- [02-06]: Centered horizontal legends (xanchor=center) for better visual balance with multi-status legends
- [02-06]: Methodology note limitations should contain only methodological/data-quality caveats, no performance or implementation details
- [02-06]: Varied presence-only limitation text per entity type for contextual clarity
- [Phase 02]: Root cause of KE component failures was Virtuoso 400s execution time limit on triple-OPTIONAL cross-product, not wrong predicates
- [Phase 02]: Direct GRAPH <uri> targeting preferred over GRAPH ?graph + FILTER for latest-data single-version queries
- [03-01]: Used nx.Graph (undirected) for bipartite AOP-KE network since membership edges are inherently undirected
- [03-01]: Louvain community detection with seed=42 for deterministic results across requests
- [03-01]: Node shape stored in Cytoscape.js data (round-rectangle for AOP, ellipse for KE) for frontend styling
- [03-01]: Metrics DataFrame cached in _plot_data_cache for CSV export compatibility with existing infrastructure
- [03-02]: Network metrics CSV download triggers get_or_compute_network() to ensure cache is populated before export
- [03-02]: Info panel uses fixed positioning with CSS transform slide-in animation for non-intrusive node details
- [03-02]: Tab navigation separates Graph View (interactive) from Metrics & Communities (data tables)
- [03-03]: Export buttons handled as plain HTML anchor links (no additional JS) since download endpoints return files directly
- [03-03]: Search matches both node label and ID for broader discoverability
- [03-03]: Community dropdown populated dynamically from graph data after Cytoscape initialization
- [03-03]: XSS prevention via escapeHtml utility on all dynamic HTML interpolation from SPARQL data
- [03-04]: Simplified network to KE nodes + KER edges only (removed AOP nodes) per user feedback for clearer topology
- [03-04]: Added scipy as explicit dependency since NetworkX Louvain community detection requires it
- [03-04]: CDN dependency chain for fcose layout: layout-base -> cose-base -> cytoscape-fcose
- [03-05]: Used VersionedPlotCache.__contains__ TTL check to detect expired entries rather than manual timestamp comparison

### Pending Todos

None yet.

### Verification Issues (Phase 2)

User verification of Phase 2 delivery found 4 issues requiring follow-up:
1. ~~OECD plot title and legend overlap (cosmetic)~~ -- RESOLVED in 02-06
2. Composite AOP Completeness Distribution only has pre-2020 data (possible Virtuoso limit) -- addressed in 02-05
3. ~~Methodology notes incomplete on some plots; performance limitations unnecessary for average users~~ -- RESOLVED in 02-06
4. Some KE component annotation plots broken (not rendering) -- addressed in 02-05

### Blockers/Concerns

- Phase 6 needs research: VHP4Safety infrastructure specifics (orchestration, probes, CORS domain)

## Session Continuity

Last session: 2026-02-23
Stopped at: Completed 03-05-PLAN.md (Network metrics CSV cache fix) -- Phase 3 all plans complete
Resume file: .planning/phases/03-network-analysis/03-05-SUMMARY.md
