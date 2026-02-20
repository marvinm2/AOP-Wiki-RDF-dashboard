# Roadmap: AOP-Wiki RDF Dashboard

## Overview

This roadmap transforms the AOP-Wiki RDF Dashboard from a functioning but fragile monitoring tool into a reliable, feature-complete exploration platform deployed on VHP4Safety. The journey starts by stabilizing the existing foundation (removing legacy code, fixing unbounded caches, production-readying the server), then making all existing plots reliable and complete, then adding the flagship network analysis capability, then building entity drill-down views, then adding advanced analytics, and finally deploying to the VHP platform. Each phase delivers a coherent, verifiable capability that builds on the previous one.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation and Cleanup** - Eliminate technical debt, production-ready infrastructure, developer guardrails
- [ ] **Phase 2: Reliability and Completeness** - All plots load reliably, full export coverage, methodology transparency
- [ ] **Phase 3: Network Analysis** - Interactive AOP network graph with centrality metrics, PageRank, and community detection
- [ ] **Phase 4: Entity Exploration** - Deep-dive views for AOPs, KEs, and stressors with shareable URLs
- [ ] **Phase 5: Advanced Analytics** - Stressor-AOP mapping, data quality scoring, ontology coverage analysis
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
**Plans**: TBD

Plans:
- [ ] 01-01: TBD
- [ ] 01-02: TBD
- [ ] 01-03: TBD

### Phase 2: Reliability and Completeness
**Goal**: Every visualization on the dashboard loads reliably, exports in all formats, and explains its methodology to users
**Depends on**: Phase 1
**Requirements**: RELY-01, RELY-02, RELY-04, EXPL-07
**Success Criteria** (what must be TRUE):
  1. AOP Completeness Distribution plot loads in under 30 seconds (down from ~75s)
  2. AOP completeness by OECD status is visualized through a working alternative that does not hit Virtuoso limits
  3. Every displayed plot has CSV, PNG, and SVG download buttons that produce valid files with metadata
  4. Every plot has an expandable methodology note visible to the user explaining what it measures and how data is sourced
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD
- [ ] 02-03: TBD

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
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD
- [ ] 03-03: TBD

### Phase 4: Entity Exploration
**Goal**: Users can drill down from any aggregate visualization into individual AOP, KE, and stressor detail views, share specific views via URL, and see raw data behind plots
**Depends on**: Phase 2, Phase 3
**Requirements**: EXPL-01, EXPL-02, EXPL-03, EXPL-04, EXPL-05, EXPL-06
**Success Criteria** (what must be TRUE):
  1. User can navigate to an individual AOP detail page showing all properties, constituent KEs, KERs, and relationships
  2. User can navigate to individual KE and stressor detail pages showing all properties and connected AOPs
  3. Entity names and IDs displayed in plots and network graph are clickable links to the corresponding AOP-Wiki source pages
  4. User can toggle a raw data table beneath each plot showing the underlying data that produced the visualization
  5. URL encodes current version selection, active plot, and entity view — copying and sharing the URL restores the exact view
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Advanced Analytics
**Goal**: Users can explore higher-order relationships and data quality patterns across the AOP-Wiki knowledge base
**Depends on**: Phase 3, Phase 4
**Requirements**: ANLY-01, ANLY-02, ANLY-03
**Success Criteria** (what must be TRUE):
  1. User can view stressor-to-AOP mapping as an interactive network or Sankey diagram showing which stressors connect to which AOPs
  2. User can see a composite data quality score per AOP that combines completeness, annotation depth, and network connectivity into a single comparable metric
  3. User can view ontology coverage analysis showing which GO/CHEBI/UBERON terms are used in AOP-Wiki versus what is available, identifying annotation gaps
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: VHP Platform Deployment
**Goal**: The dashboard is publicly accessible on the VHP4Safety platform with production-grade reliability
**Depends on**: Phase 1, Phase 2 (minimum); Phase 5 (full feature set)
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
| 1. Foundation and Cleanup | 0/3 | Not started | - |
| 2. Reliability and Completeness | 0/3 | Not started | - |
| 3. Network Analysis | 0/3 | Not started | - |
| 4. Entity Exploration | 0/3 | Not started | - |
| 5. Advanced Analytics | 0/2 | Not started | - |
| 6. VHP Platform Deployment | 0/2 | Not started | - |
