# Phase 2: Reliability and Completeness - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Every existing visualization on the dashboard loads reliably, exports in all formats (CSV, PNG, SVG), and explains its methodology to users. This phase also restores the removed OECD status visualization with a working alternative and optimizes slow queries. No new analytical capabilities are added — this is about making existing features complete and reliable.

</domain>

<decisions>
## Implementation Decisions

### OECD Status Alternative Visualization
- Grouped bar chart replacing the removed boxplot (which hit Virtuoso limits)
- Shows mean completeness score per OECD status
- Include all OECD statuses found in the data (no grouping into "Other")
- Both a latest-data snapshot AND historical trend (completeness per status over time)
- Two visualizations: bar chart for selected version + trend lines across versions

### Export Buttons
- Export button UI design: Claude's discretion (icon row, dropdown, or modebar integration)
- PNG/SVG metadata inclusion: Claude's discretion (based on scientific visualization practices)
- Each sub-plot in multi-figure trend plots gets its own export buttons (per sub-plot, not combined)
- Export filenames include version and date for traceability (e.g., `aop-entity-counts_2025-12-01_v2025-12-01.csv`)

### Methodology Notes
- Presentation format: Claude's discretion (accordion, popover, etc.)
- Depth: moderate — paragraph explaining what it measures and how, plus SPARQL reference
- Always include known limitations for every plot (builds trust with researchers)
- SPARQL reference: viewable query snippet in an expandable code block (maximum transparency for technical users)

### Query Optimization
- Target: AOP Completeness Distribution under 30 seconds (down from ~75s)
- Pre-compute and cache heavy queries at startup (trade boot time for user experience)
- Keep full completeness scoring — no property simplification even if slower
- Audit ALL plots for reliability, with specific attention to AOP Lifetime plot and KE component trends
- Graceful degradation: error card with retry (current behavior) — no stale data or fallback plots
- SPARQL timeout policy: Claude's discretion based on query complexity research
- Include Virtuoso-side tuning recommendations, documented only (not applied in Docker config)

### Claude's Discretion
- Export button UI pattern (icon row, dropdown, or modebar)
- PNG/SVG metadata approach
- Methodology note presentation component (accordion, popover, etc.)
- Per-plot timeout configuration
- Exact spacing and styling details

</decisions>

<specifics>
## Specific Ideas

- Researchers need to trust the data — methodology notes with limitations and viewable SPARQL queries enable reproducibility
- Export filenames should be self-documenting so downloaded files are identifiable months later
- The AOP Lifetime plot and KE component trends were flagged as broken in previous UAT — these must be fixed or properly error-handled in this phase

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-reliability-and-completeness*
*Context gathered: 2026-02-21*
