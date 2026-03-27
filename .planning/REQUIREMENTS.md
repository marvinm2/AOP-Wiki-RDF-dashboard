# Requirements: AOP-Wiki RDF Dashboard

**Defined:** 2026-03-26
**Core Value:** Researchers, curators, and regulators can explore AOP-Wiki data across any version to understand how the knowledge base evolves over time -- reliably and without timeouts.

## v1.1 Requirements

Requirements for the Dashboard Maturation milestone. Each maps to roadmap phases.

### Plot Audit

- [x] **AUDIT-01**: User sees every plot evaluated against a binary rubric (color correctness, chart type appropriateness, data table usefulness, title quality)
- [x] **AUDIT-02**: User sees chart type recommendations for plots better served as tables, different chart types, or removed
- [x] **AUDIT-03**: User sees colorblind accessibility verification of the VHP4Safety palette with documented findings

### Color Consistency

- [x] **COLOR-01**: User sees non-categorical bar plots rendered in single brand color (#307BBF) instead of per-category rainbow colors
- [x] **COLOR-02**: User can reference a codified color decision framework defining when single vs multi-color is appropriate
- [x] **COLOR-03**: User sees all audit-flagged plots corrected to follow the color framework

### Network Graph

- [ ] **NET-01**: User sees the network graph render instantly with pre-computed server-side layout (deterministic positions across visits)
- [ ] **NET-02**: User sees network nodes colored by biological role (MIE, KE, AO) instead of community assignment by default
- [ ] **NET-03**: User sees a type legend in the graph showing MIE/KE/AO color meanings

### SPARQL Transparency

- [ ] **SPARQL-01**: User can view the exact SPARQL query that produced each plot via a collapsible panel
- [ ] **SPARQL-02**: User can click "Run on Endpoint" to open the Virtuoso SPARQL editor with the query pre-filled in a new tab

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Data Downloads

- **DL-01**: User can download raw SPARQL results as CSV via server-side query execution per plot

### Entity Exploration

- **EXPL-01**: User can view individual AOP detail page
- **EXPL-02**: User can view individual KE detail page
- **EXPL-03**: User can view individual stressor detail page

### Version Comparison

- **VERS-01**: User can view version-to-version diff summary
- **VERS-02**: User can view entity changelog between versions
- **VERS-03**: User can view side-by-side version comparison

### Advanced Analytics

- **ANLY-01**: User can view stressor-to-AOP mapping visualization
- **ANLY-02**: User can view composite data quality score per AOP
- **ANLY-03**: User can view ontology coverage analysis

## Out of Scope

| Feature | Reason |
|---------|--------|
| In-dashboard SPARQL query editor | YASGUI already does this; link to endpoint instead |
| Dynamic color theme picker | Single brand (VHP4Safety), no user customization needed |
| Animated plot transitions | Slows comprehension in scientific dashboards |
| Real-time live updating plots | Version-based snapshots; AOP-Wiki updates quarterly |
| Mobile-optimized plot interactions | Primary audience uses desktop workstations |
| Chart type conversions from audit | Implement in future milestone after audit report is complete |
| Plot title rewrites | Low priority, do incrementally in future milestones |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUDIT-01 | Phase 7 | Complete |
| AUDIT-02 | Phase 7 | Complete |
| AUDIT-03 | Phase 7 | Complete |
| COLOR-01 | Phase 8 | Complete |
| COLOR-02 | Phase 8 | Complete |
| COLOR-03 | Phase 8 | Complete |
| NET-01 | Phase 9 | Pending |
| NET-02 | Phase 9 | Pending |
| NET-03 | Phase 9 | Pending |
| SPARQL-01 | Phase 10 | Pending |
| SPARQL-02 | Phase 10 | Pending |

**Coverage:**
- v1.1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0

---
*Requirements defined: 2026-03-26*
*Last updated: 2026-03-26 after roadmap creation*
