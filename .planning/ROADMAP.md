# Roadmap: AOP-Wiki RDF Dashboard

## Milestones

- ✅ **v1.0 AOP-Wiki RDF Dashboard** — Phases 1-6 (shipped 2026-03-18)
- **v1.1 Dashboard Maturation** — Phases 7-10 (in progress)

## Phases

<details>
<summary>v1.0 AOP-Wiki RDF Dashboard (Phases 1-6) — SHIPPED 2026-03-18</summary>

- [x] Phase 1: Foundation and Cleanup (4/4 plans) — completed 2026-02-20
- [x] Phase 2: Reliability and Completeness (7/7 plans) — completed 2026-02-22
- [x] Phase 3: Network Analysis (5/5 plans) — completed 2026-02-23
- [x] Phase 4: Dashboard Enrichment & Raw Data (3/3 plans) — completed 2026-02-23
- [x] Phase 5: Polish & Consistency (4/4 plans) — completed 2026-02-23
- [x] Phase 6: VHP Platform Deployment (2/2 plans) — completed 2026-03-17

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### v1.1 Dashboard Maturation (In Progress)

**Milestone Goal:** Make the existing dashboard production-quality -- consistent colors, audited plots, SPARQL transparency, and a faster network graph.

- [x] **Phase 7: Plot Audit** - Systematic review of all 37+ plots with binary rubric and per-plot color/chart-type decisions (completed 2026-03-27)
- [ ] **Phase 8: Color Consistency** - Enforce single brand color for non-categorical plots and verify palette accessibility
- [ ] **Phase 9: Network Graph Overhaul** - Pre-computed layout with MIE/KE/AO type coloring for instant, meaningful rendering
- [ ] **Phase 10: SPARQL Transparency** - Show the exact query behind each plot with a link to run it on the endpoint

## Phase Details

### Phase 7: Plot Audit
**Goal**: Every plot has been systematically evaluated so that color, network, and SPARQL work proceeds from informed decisions rather than guesswork
**Depends on**: Phase 6 (v1.0 complete)
**Requirements**: AUDIT-01, AUDIT-02, AUDIT-03
**Success Criteria** (what must be TRUE):
  1. User can read an audit report that scores every plot on color correctness, chart type appropriateness, data table usefulness, and title quality
  2. User can see which plots are recommended for chart type changes, removal, or keeping as-is (FIX NOW / FIX LATER / SKIP classification)
  3. User can see a colorblind accessibility evaluation of the VHP4Safety palette with specific findings documented
  4. User can see a per-plot color decision (single #307BBF vs categorical palette) that Phase 8 will implement
**Plans**: 3 plans

Plans:
- [x] 07-01-PLAN.md — Colorblind accessibility analysis (deuteranopia simulation + delta E)
- [x] 07-02-PLAN.md — Audit all 20 latest_plots.py functions
- [x] 07-03-PLAN.md — Audit all 19 trends_plots.py functions + assemble AUDIT-REPORT.md

### Phase 8: Color Consistency
**Goal**: Users see a visually coherent dashboard where bar color encodes meaning (single color = one series, multiple colors = categorical distinction) instead of arbitrary rainbow assignment
**Depends on**: Phase 7 (audit determines which plots get single vs categorical color)
**Requirements**: COLOR-01, COLOR-02, COLOR-03
**Success Criteria** (what must be TRUE):
  1. User sees all non-categorical bar plots rendered in #307BBF instead of per-category rainbow colors
  2. User can reference a documented color decision framework that defines when single-color vs multi-color is appropriate
  3. User sees every audit-flagged plot corrected to follow the color framework (zero plots violating the framework remain)
**Plans**: 3 plans
**UI hint**: yes

Plans:
- [x] 08-01-PLAN.md — Color Decision Framework doc + fix 15 latest_plots.py color assignments
- [x] 08-02-PLAN.md — Fix 7 trends_plots.py color assignments + 3 wiring defects
- [x] 08-03-PLAN.md — Automated verification suite + visual spot-check

### Phase 9: Network Graph Overhaul
**Goal**: Users can explore the AOP network graph with biologically meaningful coloring and instant rendering instead of slow community-based visualization
**Depends on**: Phase 7 (audit may flag network-related findings; but architecturally isolated -- can proceed in parallel with Phase 8)
**Requirements**: NET-01, NET-02, NET-03
**Success Criteria** (what must be TRUE):
  1. User sees the network graph render instantly with pre-computed node positions that are consistent across page visits
  2. User sees nodes colored by biological role (MIE = one color, KE = another, AO = another) instead of community assignment
  3. User sees a type legend on the network graph showing MIE/KE/AO color meanings
**Plans**: TBD
**UI hint**: yes

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD

### Phase 10: SPARQL Transparency
**Goal**: Users can verify and reproduce any plot's data by viewing and running the exact SPARQL query that produced it
**Depends on**: Phase 8 and Phase 9 (queries must be finalized before building the registry)
**Requirements**: SPARQL-01, SPARQL-02
**Success Criteria** (what must be TRUE):
  1. User can expand a collapsible panel on any plot to see the exact SPARQL query that generated it
  2. User can click "Run on Endpoint" to open the Virtuoso SPARQL editor with the query pre-filled in a new tab
**Plans**: TBD
**UI hint**: yes

Plans:
- [ ] 10-01: TBD
- [ ] 10-02: TBD

## Progress

**Execution Order:**
Phases 8 and 9 can execute in parallel after Phase 7. Phase 10 follows after both are complete.
7 -> 8 (parallel with 9) -> 10

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation and Cleanup | v1.0 | 4/4 | Complete | 2026-02-20 |
| 2. Reliability and Completeness | v1.0 | 7/7 | Complete | 2026-02-22 |
| 3. Network Analysis | v1.0 | 5/5 | Complete | 2026-02-23 |
| 4. Dashboard Enrichment & Raw Data | v1.0 | 3/3 | Complete | 2026-02-23 |
| 5. Polish & Consistency | v1.0 | 4/4 | Complete | 2026-02-23 |
| 6. VHP Platform Deployment | v1.0 | 2/2 | Complete | 2026-03-17 |
| 7. Plot Audit | v1.1 | 3/3 | Complete   | 2026-03-27 |
| 8. Color Consistency | v1.1 | 0/3 | Planned | - |
| 9. Network Graph Overhaul | v1.1 | 0/0 | Not started | - |
| 10. SPARQL Transparency | v1.1 | 0/0 | Not started | - |
