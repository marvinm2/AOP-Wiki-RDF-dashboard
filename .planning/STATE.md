---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Milestone archived, ready for next milestone
stopped_at: Completed 08-02-PLAN.md
last_updated: "2026-03-27T10:55:42.316Z"
last_activity: 2026-03-17 — Completed 06-02 (Docker Swarm Stack & Deployment Guide)
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 6
  completed_plans: 14
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-18)

**Core value:** Researchers, curators, and regulators can explore AOP-Wiki data across any version to understand how the knowledge base evolves over time — reliably and without timeouts.
**Current focus:** v1.0 shipped and deployed — planning next milestone

## Current Position

Milestone v1.0 shipped 2026-03-18
Status: Milestone archived, ready for next milestone
Last activity: 2026-03-17 — Completed 06-02 (Docker Swarm Stack & Deployment Guide)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 24
- Average duration: ~4.4 min
- Total execution time: ~1.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 4/4 | ~19 min | ~5 min |
| 02 | 7/7 | ~29 min | ~4 min |
| 03 | 5/5 | ~41 min | ~8 min |

**Recent Trend:**

- Last 5 plans: 04-03 (7m), 05-01 (2m), 05-02 (5m), 05-03 (13m), 05-04 (15m)
- Trend: 05-04 included human-verify checkpoint with post-verification fix commit

*Updated after each plan completion*
| Phase 02 P05 | 9min | 2 tasks | 2 files |
| Phase 02 P07 | 2min | 2 tasks | 3 files |
| Phase 03 P01 | 3min | 2 tasks | 3 files |
| Phase 03 P02 | 4min | 2 tasks | 6 files |
| Phase 03 P03 | 3min | 1 tasks | 1 files |
| Phase 03 P04 | 30min | 2 tasks | 7 files |
| Phase 03 P05 | 1min | 1 tasks | 1 files |
| Phase 04 P01 | 6min | 2 tasks | 8 files |
| Phase 04 P02 | 6min | 2 tasks | 6 files |
| Phase 04 P03 | 7min | 3 tasks | 8 files |
| Phase 05 P01 | 2min | 2 tasks | 3 files |
| Phase 05 P02 | 5min | 2 tasks | 8 files |
| Phase 05 P03 | 13min | 2 tasks | 2 files |
| Phase 05 P04 | 15min | 3 tasks | 9 files |
| Phase 06 P01 | 2min | 2 tasks | 3 files |
| Phase 06 P02 | 5min | 2 tasks | 2 files |
| Phase 08 P02 | 3min | 2 tasks | 4 files |

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
- [04-01]: Used absolute plot name as data source for dual-view (abs/delta) plot-boxes since underlying data is same
- [04-01]: URL state uses history.replaceState to avoid creating browser history entries on version change
- [04-01]: version-changed custom DOM event pattern for decoupled communication between version-selector and raw-data-tables
- [04-02]: Generic /download/latest/<plot_name> route instead of individual routes to reduce route proliferation
- [04-02]: MutationObserver for attaching Plotly click handler on lazy-loaded KE reuse plot
- [04-02]: OPTIONAL+BIND pattern for SPARQL queries to include unannotated entities as 'Not Annotated'
- [04-02]: Only KE reuse bar chart gets entity links (EXPL-04) since other plots show aggregate groupings
- [04-03]: Separate COUNT queries per (entity_type, property) for heatmap to avoid cross-product explosion
- [04-03]: Curation progress uses Title and Description as proxy annotations (simple, interpretable)
- [04-03]: Ontology term growth counts unique IRIs not concepts — Phase 5 adds per-term detail
- [05-01]: Kept legacy aliases (secondary, accent, light, content, config) for backward compatibility with existing plot code
- [05-01]: Template composited as plotly_white+vhp4safety to inherit plotly_white base and overlay brand styling
- [05-02]: SVG placeholder logo instead of PNG download (no official high-res logo downloadable via CLI)
- [05-02]: Navigation is static (scrolls with page), not fixed, to preserve screen real estate for data-dense pages
- [05-02]: Version selector in nav-version-selector block, visible only on Database Snapshot page
- [05-02]: Kept legacy CSS rules (.page-navigation, header, footer) for templates not yet migrated (landing.html, index.html)
- [05-03]: Removed all Plotly figure titles — HTML headings provide context for every plot
- [05-03]: Phase 5 right-side vertical legend decision supersedes Phase 02-06 centered horizontal legend
- [05-03]: Kept OECD legend title="OECD Status" as intentional per-plot customization (position from template)
- [05-03]: All rendering via render_plot_html(fig) — no direct pio.to_html or fig.to_html anywhere
- [05-04]: Version selector moved out of nav bar into /snapshot page content for nav consistency across pages
- [05-04]: Sticky footer via flexbox min-height: 100vh pattern
- [05-04]: OECD status colors centralized in BRAND_COLORS['oecd_status'] for consistency between latest and trend views
- [05-04]: SPARQL endpoint links set to href="#" with "coming soon" until multi-graph URL is provided
- [05-04]: Legacy routes /old-dashboard and /dashboard redirect to /snapshot
- [06-01]: Wildcard CORS (Access-Control-Allow-Origin: *) appropriate for public read-only dashboard
- [06-01]: _startup_complete flag set synchronously around compute_plots_parallel() for container probe safety
- [06-02]: Dual-network topology: external 'core' for Traefik routing, internal overlay for Virtuoso-dashboard SPARQL communication
- [06-02]: No ports exposed directly -- all traffic routed through Traefik with TLS termination
- [06-02]: Dashboard health probe start_period 120s to tolerate plot precomputation on cold start
- [06-02]: Virtuoso health probe uses wget (curl not available in Virtuoso image)
- [Phase 08]: Legacy color aliases replaced with explicit BRAND_COLORS keys in trends_plots.py; palette array frozen per D-09

### Pending Todos

None yet.

### Verification Issues (Phase 2)

User verification of Phase 2 delivery found 4 issues requiring follow-up:

1. ~~OECD plot title and legend overlap (cosmetic)~~ -- RESOLVED in 02-06
2. Composite AOP Completeness Distribution only has pre-2020 data (possible Virtuoso limit) -- addressed in 02-05
3. ~~Methodology notes incomplete on some plots; performance limitations unnecessary for average users~~ -- RESOLVED in 02-06
4. Some KE component annotation plots broken (not rendering) -- addressed in 02-05

### Blockers/Concerns

None -- all phases complete.

## Session Continuity

Last session: 2026-03-27T10:55:42.306Z
Stopped at: Completed 08-02-PLAN.md
Resume file: None
