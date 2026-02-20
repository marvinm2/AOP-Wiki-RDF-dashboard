# Feature Landscape

**Domain:** RDF/SPARQL data monitoring dashboard for scientific knowledge base (AOP-Wiki)
**Researched:** 2026-02-20
**Overall Confidence:** MEDIUM (based on codebase analysis, domain knowledge of AOP-Wiki ecosystem, RDF tooling landscape, and scientific data dashboard patterns; web search unavailable for verification of current competitor state)

## Context: What Already Exists

The dashboard currently provides 30+ Plotly visualizations across two main views:

- **Database Snapshot** (13 plots): Entity counts, KE components, AOP connectivity, completeness by OECD status, ontology usage, annotation depth -- all version-selectable
- **Historical Trends** (17+ plots): Entity evolution, delta changes, property presence over time, completeness trends, author patterns, AOP lifetime analysis
- **Export**: CSV with metadata, PNG, SVG, bulk ZIP downloads
- **Infrastructure**: Lazy loading, parallel computation, health monitoring, Docker deployment

What follows categorizes features for the **next milestone** -- where the dashboard goes from "monitoring tool" to "exploration and analysis platform."

---

## Table Stakes

Features users expect given what the dashboard already provides. Missing these creates friction or makes the dashboard feel incomplete for its stated purpose.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Version-to-version diff summary** | Users already select versions; natural next question is "what changed?" | Medium | Show entity count deltas, new/removed AOPs, changed completeness scores between any two selected versions. The version selector infrastructure already exists. |
| **Individual entity detail views** | Dashboard shows aggregate counts but users need to drill into specific AOPs, KEs, or KERs | Medium | Click on an entity in a plot to see its properties, relationships, completeness. Requires new SPARQL queries and detail page template. |
| **Reliable loading for all plots** | Some plots take 75+ seconds or were removed for timeouts. Users expect everything works. | High | `plot_aop_completeness_boxplot_by_status` was removed; entity completeness trends takes 75s. Core reliability before new features. |
| **Consistent error messaging** | Some plots return `<p>No data available</p>`, others use `create_fallback_plot()`. Users need uniform experience. | Low | Standardize all error states to use `create_fallback_plot()` with actionable messages. |
| **Full export coverage** | Users can export most but not all plots. Every visible chart should be downloadable. | Low | Audit for gaps between displayed plots and available download routes. Some trend plots may lack PNG/SVG endpoints. |
| **Plot descriptions/methodology** | Scientists need to understand what each visualization measures and how. Regulatory context demands transparency. | Low | Add expandable methodology notes under each plot explaining SPARQL query logic, what "completeness" means, etc. Currently only some plots have `plot-description` paragraphs. |
| **Direct links to AOP-Wiki** | Users see entity counts but cannot navigate to source data. Standard for any dashboard over external data. | Low | Link entity names/IDs to `https://aopwiki.org/aops/[ID]`, `https://aopwiki.org/events/[ID]`, etc. |
| **Responsive table view for data** | Scientists often want to see raw data, not just charts. CSV download requires leaving the page. | Medium | Add toggleable data table view beneath each plot showing the cached DataFrame. Plotly has built-in table support. |

## Differentiators

Features that set this dashboard apart from the standard AOP-Wiki interface and other toxicology tools. Not expected, but create significant value for researchers, curators, and regulators.

### Network Analysis (Issue #11)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Interactive AOP network graph** | Visualize how AOPs connect through shared KEs. No existing tool provides this view of the AOP-Wiki RDF data. | High | Use Plotly's network graph or Cytoscape.js. Show AOPs as nodes, shared KEs as edges. Color by OECD status. Must handle Virtuoso query limits -- pre-aggregate in Python. |
| **Node centrality metrics** (degree, betweenness, closeness) | Identify which Key Events are most critical to the AOP network. Directly supports regulatory prioritization. | Medium | Compute with NetworkX in Python after fetching the graph structure via SPARQL. Display as sortable table + highlight on network graph. |
| **Community/cluster detection** | Find groups of closely related AOPs. Helps curators identify knowledge gaps and researchers find related work. | Medium | Apply Louvain or label propagation from NetworkX. Color clusters on the network graph. Export cluster membership. |
| **PageRank for KE importance** | Rank Key Events by structural importance in the AOP network, analogous to how Google ranks web pages. Novel analytical lens for toxicology. | Medium | Straightforward NetworkX computation once graph is built. Display as ranked list with percentile scores. |
| **Shared KE heatmap** | Matrix showing which AOPs share Key Events with which. Powerful for identifying convergent pathways. | Medium | Cross-tabulation query, rendered as Plotly heatmap. May need pagination for large AOP counts. |
| **Path analysis between KEs** | Find all paths from a Molecular Initiating Event to an Adverse Outcome across the network. Core AOP reasoning task. | High | Graph traversal with NetworkX. Must handle cyclic paths and branching. Display as interactive flow diagram. |

### Cross-Version Analysis

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Side-by-side version comparison** | Curators need to verify what changed between releases. Regulators need audit trails. | Medium | Two-panel layout showing same metrics for two selected versions. Highlight differences. Reuses existing version-selectable plot infrastructure. |
| **Entity changelog** | "What AOPs were added/modified/removed between version X and Y?" Critical for curation workflows. | Medium | SPARQL diff query between two named graphs. Display as filterable table with entity links. |
| **Completeness trend per entity** | Track how a specific AOP's completeness improves over time. Curators want to see their work reflected. | Medium | Requires cross-version query for a single entity. Display as sparkline or small multiples chart. |

### Advanced Analytics

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Stressor-AOP mapping** | Show which stressors connect to which AOPs. Critical for chemical risk assessment -- the primary regulatory use case. | Medium | SPARQL query for stressor-AOP relationships. Render as searchable network or Sankey diagram. |
| **Ontology coverage analysis** | Which GO/CHEBI/UBERON terms are used vs available? Identifies annotation gaps. | Medium | Compare used terms against ontology term counts. Display as coverage percentage bars. |
| **Data quality scoring** | Composite quality score per AOP combining completeness, annotation depth, and structural connectivity. | Medium | Weighted formula using existing completeness data + new metrics. Display as ranked table with drill-down. |
| **Custom SPARQL query interface** | Power users (bioinformaticians) want to run their own queries against the endpoint. | High | Embedded SPARQL editor with syntax highlighting, query templates, and result visualization. Security: read-only queries only, query timeout enforcement. |
| **Annotation timeline per entity** | When was each property of an AOP first added? Track curation effort over time. | High | Requires comparing property presence across all versions for a specific entity. Computationally expensive cross-version analysis. |

### Export and Integration

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Report generation** | Curators and regulators need formatted reports, not just dashboards. | Medium | Generate PDF/HTML reports with selected plots, version info, and methodology notes. Use ReportLab or WeasyPrint. |
| **API for programmatic access** | Other VHP4Safety tools need dashboard data without scraping HTML. | Medium | REST API returning JSON for entity counts, completeness scores, network metrics. Extends existing `/api/` endpoints. |
| **RDF/JSON-LD export** | Scientists working with linked data want results in semantic format. | Low | Serialize SPARQL results as JSON-LD. Adds interoperability with other RDF tools. |
| **Bookmark/share specific views** | Users want to share a specific version + plot combination with colleagues. | Low | Encode version and active tab/plot in URL parameters. Already partially works via `/snapshot?version=X`. |

## Anti-Features

Features to explicitly NOT build. Each has been considered and rejected for stated reasons.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **User authentication / accounts** | Dashboard is publicly available and read-only. Auth adds complexity with zero value for the use case. Users are researchers accessing public data. | Keep it public. If VHP platform requires auth, handle at the reverse proxy / platform level, not in the Flask app. |
| **Write/edit operations on RDF data** | Dashboard is a monitoring tool, not a curation tool. AOP-Wiki already has its own editing interface. Mixing read/write creates data integrity risks. | Link to AOP-Wiki's editing interface for entities that need updates. |
| **Real-time streaming updates** | AOP-Wiki releases are quarterly. Real-time monitoring of a quarterly-updated dataset is meaningless overhead. | Keep the version-based snapshot model. Add a "last updated" indicator. |
| **Full SPARQL federation across endpoints** | The project uses a single controlled Virtuoso instance. Federation adds network latency, error surface, and query plan complexity for no benefit. | Keep single-endpoint queries. If needed, pre-load data from other sources into the same Virtuoso. |
| **Mobile-native app** | User base is scientists at desktops. Responsive web design covers the rare mobile use case adequately. | Ensure CSS is responsive (already implemented). Test on tablets for conference demos. |
| **Machine learning / prediction features** | AOP data is curated expert knowledge, not statistical training data. ML predictions on AOP completeness would be misleading and scientifically inappropriate. | Focus on descriptive analytics and network analysis. Leave predictive modeling to domain-specific research tools. |
| **Multi-language internationalization** | AOP-Wiki is an English-language scientific resource. All users read English. i18n adds significant maintenance burden. | Keep English-only. Ensure technical terms are defined in methodology notes. |
| **Embedding in iframes / widget mode** | Creates maintenance burden for a rarely-needed feature. Complicates CSP headers and breaks interactive Plotly features. | Provide API endpoints instead. Let consuming tools fetch data and render their own UI. |
| **Complex role-based dashboards** | Different views for "researcher" vs "curator" vs "regulator" adds navigation complexity. The same data serves all audiences. | Use a single clean interface with progressive disclosure (expandable details, methodology notes). Let users self-serve what they need. |

## Feature Dependencies

```
Reliable plot loading ──────────────────┐
                                        ├──> Version-to-version diff
Full export coverage ───────────────────┘

Individual entity detail views ─────────┐
                                        ├──> Entity changelog
Direct links to AOP-Wiki ──────────────┘

Interactive AOP network graph ──────────┐
                                        ├──> Centrality metrics
                                        ├──> Community detection
                                        ├──> PageRank for KE importance
                                        ├──> Shared KE heatmap
                                        └──> Path analysis between KEs

Stressor-AOP mapping ──────────────────> Path analysis (stressor to AO)

Data quality scoring ──────────────────> Report generation

API for programmatic access ───────────> RDF/JSON-LD export

Responsive table view ─────────────────> Custom SPARQL query interface
```

Key dependency chains:
1. **Network analysis** requires the interactive graph as foundation -- all centrality, clustering, and path features build on it
2. **Entity detail views** are prerequisite for entity-level changelog and cross-version tracking
3. **Reliability fixes** must precede new features that add more SPARQL query load

## MVP Recommendation

For the next milestone, prioritize in this order:

### Must Ship (Phase 1 foundation)

1. **Reliable plot loading** -- fix the 75-second completeness plot and unblock the removed boxplot. Without this, new SPARQL-heavy features will compound the problem.
2. **Consistent error messaging** -- quick win that professionalizes the dashboard.
3. **Full export coverage** -- close the gap on existing functionality.
4. **Plot descriptions/methodology** -- low effort, high trust with scientific users.

### Core Differentiators (Phase 2)

5. **Interactive AOP network graph** -- the flagship differentiator. No other tool visualizes the AOP network from RDF data interactively.
6. **Node centrality metrics + PageRank** -- builds directly on the network graph with NetworkX.
7. **Individual entity detail views** -- enables drill-down from any aggregate visualization.
8. **Direct links to AOP-Wiki** -- connects the dashboard to its source of truth.

### Extended Analysis (Phase 3)

9. **Community detection / clustering** -- adds analytical depth to the network view.
10. **Version-to-version diff summary** -- high demand from curators.
11. **Stressor-AOP mapping** -- directly serves the regulatory use case.
12. **Side-by-side version comparison** -- enables the audit trail regulators need.

### Defer

- **Custom SPARQL interface**: High complexity, niche audience. Revisit after core features ship.
- **Report generation**: Valuable but depends on having the analysis features to report on.
- **Annotation timeline per entity**: Computationally expensive cross-version analysis. Defer until query performance is proven stable.
- **Path analysis between KEs**: Requires mature network graph implementation and careful UX for complex results.

## User Persona Feature Mapping

| Feature | Researcher | Curator | Regulator |
|---------|:----------:|:-------:|:---------:|
| Version diff | Medium | HIGH | HIGH |
| Entity detail views | HIGH | HIGH | Medium |
| Network graph | HIGH | Medium | Medium |
| Centrality/PageRank | HIGH | Low | Medium |
| Community detection | HIGH | Medium | Low |
| Stressor-AOP mapping | Medium | Low | HIGH |
| Completeness scoring | Medium | HIGH | HIGH |
| Export/reports | Medium | Medium | HIGH |
| SPARQL interface | HIGH | Low | Low |
| Methodology notes | Medium | Low | HIGH |

## Confidence Notes

- **Table stakes**: HIGH confidence -- derived directly from codebase gaps and standard scientific dashboard expectations
- **Network analysis features**: MEDIUM confidence -- NetworkX computation approach is well-established but Virtuoso query performance for graph extraction is the key risk (same root cause that already forced feature removal)
- **Cross-version analysis**: MEDIUM confidence -- feasible with existing version infrastructure but query performance for cross-graph comparisons is unproven
- **Anti-features**: HIGH confidence -- clearly out of scope per PROJECT.md and architectural constraints
- **Complexity estimates**: MEDIUM confidence -- based on codebase structure assessment; actual SPARQL query complexity may shift estimates upward

## Sources

- Codebase analysis: `app.py`, `plots/__init__.py`, `plots/latest_plots.py`, `plots/trends_plots.py`, `plots/shared.py`
- Project planning: `.planning/PROJECT.md`, `.planning/codebase/CONCERNS.md`, `.planning/codebase/INTEGRATIONS.md`
- Architecture docs: `.claude/architecture.md`, `CLAUDE.md`
- Domain knowledge: AOP-Wiki ecosystem (aopwiki.org), OECD AOP programme, VHP4Safety project context
- Note: Web search was unavailable during this research session. Competitor analysis of tools like AOPXplorer, AOP-helpFinder, CompTox, and similar AOP visualization tools could not be verified against current state. Confidence on differentiator uniqueness is MEDIUM as a result.
