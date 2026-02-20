# AOP-Wiki RDF Dashboard

## What This Is

A publicly available monitoring and data exploration dashboard for AOP-Wiki RDF data. It visualizes current and historical versions of AOP-Wiki through interactive plots, trend analysis, and data exports — powered by SPARQL queries against a Virtuoso endpoint with versioned named graphs. Built for the VHP4Safety project and targeting deployment on the VHP platform.

## Core Value

Researchers, curators, and regulators can explore AOP-Wiki data across any version to understand how the knowledge base evolves over time — reliably and without timeouts.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Version selector with dynamic plot regeneration — existing
- ✓ CSV export system with metadata headers — existing
- ✓ Lazy loading for fast page loads (~50ms initial) — existing
- ✓ 30+ interactive Plotly visualizations (latest snapshots + historical trends) — existing
- ✓ Property presence visualizations for AOPs, KEs, KERs, and Stressors with marker shape differentiation — existing
- ✓ Health monitoring endpoints (/health, /status) — existing
- ✓ Docker containerization with docker-compose — existing
- ✓ SPARQL query retry logic with exponential backoff — existing
- ✓ Parallel plot computation with ThreadPoolExecutor — existing
- ✓ Modular plots architecture (plots/ package) — existing
- ✓ VHP4Safety brand consistency — existing
- ✓ Configurable environment (SPARQL endpoint, timeouts, workers) — existing

### Active

<!-- Current scope. Building toward these. -->

- [ ] All plots load reliably without timeouts
- [ ] Consistent error handling with user-friendly fallback messages across all plots
- [ ] Interactive network analysis (centrality, clustering, PageRank) for AOP relationships
- [ ] Entity deep-dive views for individual AOPs, KEs, and stressors
- [ ] Version comparison views (side-by-side diffs between releases)
- [ ] Complete export coverage (CSV/PNG/SVG for all visualizations)
- [ ] VHP platform deployment (issue #3)
- [ ] Legacy code cleanup (remove monolithic plots.py, consolidate imports)
- [ ] SPARQL query optimization to eliminate remaining timeout-prone queries
- [ ] Cache management with eviction policy to prevent unbounded memory growth

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- User authentication / login — dashboard is publicly available, read-only
- Write operations to SPARQL endpoint — dashboard is read-only monitoring
- Real-time data updates / WebSocket streaming — version-based snapshots are sufficient
- Mobile-native app — web-first, responsive design is sufficient
- Multi-endpoint federation — single controlled Virtuoso instance

## Context

- AOP-Wiki is a knowledge base for Adverse Outcome Pathways used in toxicology and regulatory science
- RDF data is loaded into Virtuoso with named graphs per version (e.g., `http://aopwiki.org/graph/YYYY-MM-DD`)
- The SPARQL endpoint is controlled by the team — Virtuoso settings can be tuned
- Current pain: some complex queries (multi-way JOINs, cross-version aggregations) hit Virtuoso execution limits, causing timeouts or forced feature removal
- The `plot_aop_completeness_boxplot_by_status` was already removed because it exceeded Virtuoso limits
- The Composite AOP Completeness Distribution still takes ~75 seconds despite optimization
- A legacy monolithic `plots.py` (4,194 lines) duplicates the refactored `plots/` package and needs removal
- No test suite exists — queries and plots are only tested implicitly through execution
- Issue #11 tracks network analysis expansion; Issue #3 tracks VHP platform deployment

## Constraints

- **Tech stack**: Python/Flask with Plotly — established and working, no framework migration
- **Data source**: SPARQL 1.1 against Virtuoso — some query patterns are Virtuoso-specific
- **Deployment target**: VHP4Safety platform — must meet their deployment requirements
- **Public access**: No authentication required — dashboard is read-only and public
- **Performance**: Must handle Virtuoso query limits without timeouts for all user-facing plots

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep Flask + Plotly stack | Proven, team knows it, no reason to migrate | — Pending |
| Interactive network viz (not static) | Users need to explore connections, not just view them | — Pending |
| Optimize queries before adding features | Unreliable foundation undermines new features | — Pending |
| Remove legacy plots.py | Maintenance burden, code duplication, inconsistency risk | — Pending |

---
*Last updated: 2026-02-20 after initialization*
