# Requirements: AOP-Wiki RDF Dashboard

**Defined:** 2026-02-20
**Core Value:** Researchers, curators, and regulators can explore AOP-Wiki data across any version to understand how the knowledge base evolves over time — reliably and without timeouts.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Infrastructure

- [ ] **INFR-01**: Legacy `plots.py` monolith (4,194 lines) is deleted and all imports use the modular `plots/` package
- [ ] **INFR-02**: All dependency versions are pinned in requirements.txt with tested versions
- [ ] **INFR-03**: Application runs under Gunicorn in production (not Flask dev server)
- [ ] **INFR-04**: Cache has eviction policy preventing unbounded memory growth
- [ ] **INFR-05**: CORS headers are configured for cross-origin access on VHP platform
- [ ] **INFR-06**: Dashboard is deployed and accessible on VHP4Safety platform (Issue #3)

### Reliability

- [ ] **RELY-01**: AOP completeness distribution plot loads in under 30 seconds (currently ~75s)
- [ ] **RELY-02**: AOP completeness by OECD status visualization is restored or replaced with working alternative
- [ ] **RELY-03**: All plot error states use consistent fallback with actionable user-facing messages
- [ ] **RELY-04**: Every displayed visualization has CSV, PNG, and SVG export options

### Data Exploration

- [ ] **EXPL-01**: User can view individual AOP detail page showing all properties, KEs, KERs, and relationships
- [ ] **EXPL-02**: User can view individual KE detail page showing all properties and connected AOPs
- [ ] **EXPL-03**: User can view individual stressor detail page showing all properties and connected AOPs
- [ ] **EXPL-04**: Entity names and IDs in plots link directly to corresponding AOP-Wiki source pages
- [ ] **EXPL-05**: User can toggle a raw data table view beneath each plot showing the underlying data
- [ ] **EXPL-06**: URL encodes version and active plot state so users can share specific views
- [ ] **EXPL-07**: Each plot has an expandable methodology note explaining what it measures and how

### Network Analysis

- [ ] **NETW-01**: User can explore interactive AOP network graph showing AOPs connected through shared KEs (click nodes, filter by properties, zoom)
- [ ] **NETW-02**: User can view centrality metrics (degree, betweenness, closeness) for Key Events displayed as sortable table and highlighted on network graph
- [ ] **NETW-03**: User can view PageRank scores ranking Key Events by structural importance in the AOP network
- [ ] **NETW-04**: User can see community/cluster groupings of related AOPs with cluster visualization on network graph

### Advanced Analytics

- [ ] **ANLY-01**: User can view stressor-to-AOP mapping as interactive network or Sankey diagram
- [ ] **ANLY-02**: User can see composite data quality score per AOP combining completeness, annotation depth, and connectivity
- [ ] **ANLY-03**: User can view ontology coverage analysis showing which GO/CHEBI/UBERON terms are used vs available

### Developer Experience

- [ ] **DEVX-01**: Plot documentation and architecture docs are comprehensive enough for AI-assisted plot creation (CLAUDE.md, architecture docs, inline conventions)
- [ ] **DEVX-02**: Plot addition follows a standardized workflow with clear templates, naming conventions, and registration checklist

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Cross-Version Analysis

- **VERS-01**: User can see version-to-version diff summary showing what changed between two releases
- **VERS-02**: User can view entity changelog showing AOPs added/modified/removed between versions
- **VERS-03**: User can compare same metrics side-by-side for two selected versions
- **VERS-04**: User can track per-entity completeness trend over time

### Extended Network Analysis

- **NETW-05**: User can view shared KE heatmap matrix showing which AOPs share Key Events
- **NETW-06**: User can find all paths from a Molecular Initiating Event to an Adverse Outcome across the network

### Export & Integration

- **EXPO-01**: User can generate formatted PDF/HTML reports with selected plots and methodology
- **EXPO-02**: REST API returns JSON for entity counts, completeness scores, and network metrics
- **EXPO-03**: SPARQL results exportable as JSON-LD for linked data interoperability

### Advanced Features

- **ADVN-01**: Power users can run custom SPARQL queries against the endpoint via embedded editor
- **ADVN-02**: User can view annotation timeline showing when each property of an AOP was first added

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| User authentication / accounts | Dashboard is publicly available and read-only; auth handled at platform level if needed |
| Write/edit operations on RDF data | Dashboard is monitoring tool, not curation tool; AOP-Wiki has its own editing interface |
| Real-time streaming updates | AOP-Wiki releases are quarterly; version-based snapshots are sufficient |
| SPARQL federation across endpoints | Single controlled Virtuoso instance; federation adds complexity with no benefit |
| Mobile-native app | Scientists work at desktops; responsive web design covers mobile adequately |
| Machine learning / predictions | AOP data is curated expert knowledge; ML predictions would be scientifically inappropriate |
| Multi-language internationalization | AOP-Wiki is English-language scientific resource; all users read English |
| Role-based dashboards | Same data serves all audiences; use progressive disclosure instead |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFR-01 | - | Pending |
| INFR-02 | - | Pending |
| INFR-03 | - | Pending |
| INFR-04 | - | Pending |
| INFR-05 | - | Pending |
| INFR-06 | - | Pending |
| RELY-01 | - | Pending |
| RELY-02 | - | Pending |
| RELY-03 | - | Pending |
| RELY-04 | - | Pending |
| EXPL-01 | - | Pending |
| EXPL-02 | - | Pending |
| EXPL-03 | - | Pending |
| EXPL-04 | - | Pending |
| EXPL-05 | - | Pending |
| EXPL-06 | - | Pending |
| EXPL-07 | - | Pending |
| NETW-01 | - | Pending |
| NETW-02 | - | Pending |
| NETW-03 | - | Pending |
| NETW-04 | - | Pending |
| ANLY-01 | - | Pending |
| ANLY-02 | - | Pending |
| ANLY-03 | - | Pending |
| DEVX-01 | - | Pending |
| DEVX-02 | - | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 0
- Unmapped: 26 ⚠️

---
*Requirements defined: 2026-02-20*
*Last updated: 2026-02-20 after initial definition*
