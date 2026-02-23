# Phase 5: Polish & Consistency - Context

**Gathered:** 2026-02-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish a cohesive visual identity and polished user experience across all existing dashboard pages and plots. Create a proper landing page, unify the color system, standardize plot styling, and redesign navigation. No new analytics or plots — this phase is about making what exists look and feel professional and consistent.

</domain>

<decisions>
## Implementation Decisions

### House style & colors
- VHP4Safety brand colors for page chrome (header, footer, nav) — extended palette for chart data series
- Central color system with both layers: CSS variables for page chrome + Python config for Plotly charts, kept in sync from one source of truth
- Professional look aligned with VHP4Safety branding, no specific external reference

### Landing page
- Navigation hub as primary purpose — guide users to the right section
- Icon + description cards for each section (Latest Data, Trends, Network)
- Light live data: show latest version number and headline entity counts (minimal SPARQL)
- Expandable intro explaining AOP-Wiki — collapsed by default for regulars, expandable for newcomers
- About section with project info accessible from landing (same page or separate — Claude's discretion)
- SPARQL endpoint link only (no example queries)
- VHP4Safety logo + funding/project acknowledgment

### Plot uniformity — full visual audit
- Unify colors, typography, labels, legends, margins, and sizing across ALL existing plots
- White background on all plots (clean, printable, consistent)
- Plotly toolbar hidden by default, appears on hover
- Legends on right side consistently
- Standardized hover tooltips (Claude determines format per plot type)
- Grid lines and axes standardized (Claude determines what aids readability)
- Plot heights determined by Claude (prevent clipping, look good)

### Page layout & navigation
- Add About page to navigation alongside existing Latest Data, Trends, Network
- Header and footer redesigned (not just refined) to match house style
- Version selector moved to navigation bar — always accessible
- Basic responsive: readable on tablets, not optimized for phones
- Footer includes VHP4Safety branding, funding note, and contact/issue-reporting link
- Page transitions: Claude's discretion

### Claude's Discretion
- Plot title approach (Plotly internal vs HTML heading vs both)
- About section placement (same page vs separate /about page)
- Navigation bar behavior (fixed vs static)
- Page transition style
- Tooltip format per plot type
- Grid line styling per plot type
- Plot height per plot type
- Number of extended chart colors (based on actual data series counts)

</decisions>

<specifics>
## Specific Ideas

- Color system should have a single source of truth that both CSS and Python reference
- Version selector in nav bar is a key UX improvement — currently buried in page content
- Landing page should work as a wayfinding hub, not a data dashboard itself
- Desktop/tablet focused audience (researchers, curators, regulators)

</specifics>

<deferred>
## Deferred Ideas

- Advanced Analytics (stressor-AOP mapping, data quality scoring, ontology coverage) — deferred to v2 backlog
- Documentation/help page with SPARQL examples and data dictionary — potential future addition
- Bulk data downloads page — potential future addition
- Full mobile responsiveness — not needed for target audience

</deferred>

---

*Phase: 05-polish-consistency*
*Context gathered: 2026-02-23*
