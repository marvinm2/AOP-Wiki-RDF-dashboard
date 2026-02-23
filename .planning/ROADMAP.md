# Roadmap: AOP-Wiki RDF Dashboard

## Overview

This roadmap transforms the AOP-Wiki RDF Dashboard from a functioning but fragile monitoring tool into a reliable, feature-complete exploration platform deployed on VHP4Safety. The journey starts by stabilizing the existing foundation (removing legacy code, fixing unbounded caches, production-readying the server), then making all existing plots reliable and complete, then adding the flagship network analysis capability, then building entity drill-down views, then polishing the visual identity and consistency, and finally deploying to the VHP platform. Each phase delivers a coherent, verifiable capability that builds on the previous one.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation and Cleanup** - Eliminate technical debt, production-ready infrastructure, developer guardrails (completed 2026-02-20)
- [ ] **Phase 2: Reliability and Completeness** - All plots load reliably, full export coverage, methodology transparency
- [x] **Phase 3: Network Analysis** - Interactive AOP network graph with centrality metrics, PageRank, and community detection (completed 2026-02-23)
- [x] **Phase 4: Entity Exploration** - Deep-dive views for AOPs, KEs, and stressors with shareable URLs (completed 2026-02-23)
- [ ] **Phase 5: Polish & Consistency** - House style colors, landing page, plot uniformity, navigation redesign
- [ ] **Phase 6: VHP Platform Deployment** - Production deployment on VHP4Safety with CORS and health probe integration

## Phase Details

### Phase 1: Foundation and Cleanup
**Goal**: The dashboard runs on a clean, production-ready codebase with bounded memory, pinned dependencies, and developer documentation that enables confident feature additions
**Depends on**: Nothing (first phase)
**Requirements**: INFR-01, INFR-02, INFR-03, INFR-04, RELY-03, DEVX-01, DEVX-02
**Success Criteria** (what must be TRUE):
  1. Application starts with zero import warnings and no reference to legacy `plots.py` (file deleted, all imports use `plots/` package)
  2. `docker-compose up --build` produces identical results on consecutive runs (all dependencies pinned to tested versions)
  3. Application serves requests under Gunicorn with multiple workers (not Flask dev server)
  4. Memory usage stabilizes under sustained load — cache does not grow unboundedly when cycling through versions
  5. A developer can add a new plot by following documented templates and checklists without reading existing plot implementations
**Plans:** 4 plans (3 complete + 1 gap closure)

Plans:
- [x] 01-01-PLAN.md — Delete legacy files, clean dead code, pin dependencies (Wave 1)
- [x] 01-02-PLAN.md — Gunicorn, bounded cache, JSON logging, health endpoint, error cards (Wave 2)
- [x] 01-03-PLAN.md — Add-a-plot checklist, GitHub issue template, CLAUDE.md update (Wave 1)
- [ ] 01-04-PLAN.md — Gap closure: fix API content validation so failed plots show error cards (Wave 1)

### Phase 2: Reliability and Completeness
**Goal**: Every visualization on the dashboard loads reliably, exports in all formats, and explains its methodology to users
**Depends on**: Phase 1
**Requirements**: RELY-01, RELY-02, RELY-04, EXPL-07
**Success Criteria** (what must be TRUE):
  1. AOP Completeness Distribution plot loads in under 30 seconds (down from ~75s)
  2. AOP completeness by OECD status is visualized through a working alternative that does not hit Virtuoso limits
  3. Every displayed plot has CSV, PNG, and SVG download buttons that produce valid files with metadata
  4. Every plot has an expandable methodology note visible to the user explaining what it measures and how data is sourced
**Plans:** 7 plans (6 complete + 1 UAT gap closure)

Plans:
- [x] 02-01-PLAN.md — SPARQL optimization (boxplot to startup) and OECD completeness trend visualization (Wave 1)
- [x] 02-02-PLAN.md — Export infrastructure: add data caches for CSV export, versioned filenames (Wave 1)
- [x] 02-03-PLAN.md — Methodology notes: JSON data file, Jinja2 macro, CSS, applied to all plots (Wave 1)
- [x] 02-04-PLAN.md — Reliability audit and user verification checkpoint (Wave 2) -- issues found
- [x] 02-05-PLAN.md — Gap closure: fix KE component SPARQL predicates and boxplot data truncation (Wave 1)
- [x] 02-06-PLAN.md — Gap closure: OECD layout fix, missing methodology notes, limitations cleanup (Wave 1)
- [ ] 02-07-PLAN.md — UAT gap closure: OECD right-side legend, PNG/SVG export coverage, data scope caveat (Wave 1)

### Phase 3: Network Analysis
**Goal**: Users can explore the AOP-Wiki as an interactive network, identifying structurally important Key Events and discovering community groupings (Issue #11)
**Depends on**: Phase 1
**Requirements**: NETW-01, NETW-02, NETW-03, NETW-04
**Success Criteria** (what must be TRUE):
  1. User can view an interactive network graph of AOPs connected through shared KEs — with click-to-select nodes, property-based filtering, and zoom/pan navigation — that renders within 10 seconds for the full dataset
  2. User can view centrality metrics (degree, betweenness, closeness) in a sortable table, with high-centrality nodes visually highlighted on the network graph
  3. User can view PageRank scores ranking Key Events by structural importance, displayed alongside centrality metrics
  4. User can see community/cluster groupings of related AOPs with distinct visual coloring on the network graph
  5. Network graph performs acceptably under real data load — page remains responsive during interaction (no browser freezes or multi-second delays on node click/filter)
**Plans:** 5 plans (4 complete + 1 UAT gap closure)

Plans:
- [x] 03-01-PLAN.md — Backend graph data layer: SPARQL queries, NetworkX graph, metrics, JSON conversion (Wave 1)
- [x] 03-02-PLAN.md — Flask routes, network page template, CSS, landing page navigation (Wave 2)
- [x] 03-03-PLAN.md — Frontend Cytoscape.js interactivity: graph rendering, search, filters, metrics table (Wave 3)
- [x] 03-04-PLAN.md — Automated verification and user acceptance checkpoint (Wave 4)
- [x] 03-05-PLAN.md — UAT gap closure: fix network metrics CSV export cache desync (Wave 1)

### Phase 4: Dashboard Enrichment & Raw Data
**Goal**: The dashboard is enriched with new plot variations (breakdowns by biological level, taxonomy, OECD status; KE reuse; data quality insights) and raw data tables beneath all plots, with shareable URLs for version state
**Depends on**: Phase 2, Phase 3
**Requirements**: EXPL-04, EXPL-05, EXPL-06 (EXPL-01, EXPL-02, EXPL-03 deferred)
**Success Criteria** (what must be TRUE):
  1. User can toggle a raw data table beneath each plot showing the underlying data (existing + new plots)
  2. URL encodes version selection via ?version= parameter for shareable views
  3. 7 new plots render on the dashboard: biological level, taxonomy, OECD status, KE reuse (2 views), ontology diversity, ontology term growth
  4. Entity links in KE reuse plot open corresponding AOP-Wiki source pages
  5. All new plots have methodology notes, CSV downloads, lazy loading, and version selector support
**Plans:** 3/3 plans complete

Plans:
- [ ] 04-01-PLAN.md — Raw data table infrastructure + shareable URLs (Wave 1)
- [ ] 04-02-PLAN.md — Breakdown plots: bio level, taxonomy, OECD status, KE reuse (Wave 2)
- [ ] 04-03-PLAN.md — Data quality + trends: annotation heatmap, ontology diversity/growth, curation progress (Wave 3)

### Phase 5: Polish & Consistency
**Goal**: The dashboard has a cohesive visual identity with unified house style colors, a polished landing page, standardized plot styling, and redesigned navigation — making the existing feature set look and feel professional
**Depends on**: Phase 4
**Requirements**: Visual consistency, brand alignment, UX polish
**Success Criteria** (what must be TRUE):
  1. All plots use colors from a central color system (Python config + CSS variables) derived from VHP4Safety brand palette
  2. Landing page serves as a navigation hub with icon+description cards, light live data, expandable AOP-Wiki intro, and funding acknowledgment
  3. All existing plots have uniform white backgrounds, right-side legends, hover-visible toolbars, and consistent typography/spacing
  4. Header and footer are redesigned with house style; version selector is in the navigation bar
  5. About page is accessible from navigation with project info and contact/issue-reporting link
  6. Dashboard is readable on tablets (basic responsive)
**Plans:** 3/4 plans executed

Plans:
- [ ] 05-01-PLAN.md — Color system foundation: BRAND_COLORS, Plotly custom template, CSS custom properties (Wave 1)
- [ ] 05-02-PLAN.md — Base template + navigation redesign: Jinja2 inheritance, header/footer, version selector in nav (Wave 2)
- [ ] 05-03-PLAN.md — Plot uniformity: standardize all ~57 plot renderings with custom template and render helper (Wave 2)
- [ ] 05-04-PLAN.md — Landing page hub + About page + visual verification checkpoint (Wave 3)

**Deferred to v2 backlog:** Advanced Analytics (stressor-AOP mapping, data quality scoring, ontology coverage analysis — previously ANLY-01, ANLY-02, ANLY-03)

### Phase 6: VHP Platform Deployment
**Goal**: The dashboard is publicly accessible on the VHP4Safety platform with production-grade reliability
**Depends on**: Phase 1, Phase 2 (minimum); Phase 5 (polished UI)
**Requirements**: INFR-05, INFR-06
**Success Criteria** (what must be TRUE):
  1. Dashboard is accessible at its VHP4Safety URL and serves all pages without error
  2. CORS headers allow cross-origin requests from the VHP4Safety platform domain
  3. Health probes report accurate status — `/health` responds within 5 seconds, `/status` reflects actual SPARQL endpoint connectivity
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6
Note: Phase 2 and Phase 3 can execute in parallel (both depend only on Phase 1).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation and Cleanup | 4/4 | Complete | 2026-02-20 |
| 2. Reliability and Completeness | 6/7 | UAT gap closure | - |
| 3. Network Analysis | 5/5 | Complete | 2026-02-23 |
| 4. Dashboard Enrichment & Raw Data | 0/3 | Complete    | 2026-02-23 |
| 5. Polish & Consistency | 3/4 | In Progress|  |
| 6. VHP Platform Deployment | 0/2 | Not started | - |
