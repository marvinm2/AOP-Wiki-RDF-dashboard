# Phase 4: Dashboard Enrichment & Raw Data - Research

**Researched:** 2026-02-23
**Domain:** Flask/Plotly dashboard enrichment, SPARQL data exploration, raw data tables, URL state management
**Confidence:** HIGH

## Summary

Phase 4 enriches the existing AOP-Wiki RDF Dashboard with new plot variations (breakdowns by biological level, taxonomy, OECD status; KE reuse across AOPs; annotation completeness heatmaps; curation progress; ontology term coverage) and adds toggle-able raw data tables beneath all plots (existing and new). Two optional requirements (EXPL-04: entity links to AOP-Wiki, EXPL-06: shareable URLs) are at Claude's discretion.

The existing codebase provides a well-established pattern for adding plots (documented in `.claude/add-a-plot.md`), a working lazy-loading system, comprehensive CSV export infrastructure via `_plot_data_cache`, and a consistent VHP4Safety brand styling system. The main technical challenges are: (1) crafting SPARQL queries for new data dimensions (biological level, taxonomy, KE reuse) that perform within Virtuoso's limits; (2) implementing a raw data table toggle system that works across 40+ existing and new plot containers; (3) managing page length as the main page grows significantly.

**Primary recommendation:** Implement the raw data table toggle as a reusable JavaScript component that reads from the existing `_plot_data_cache` via a new API endpoint, and add new plots following the established `.claude/add-a-plot.md` checklist. Keep all content on the existing pages (snapshot + trends) without creating new pages.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

#### New plot variations -- breakdowns
- Break down existing metrics by **biological level** (molecular, cellular, tissue, organ, organism) -- show which levels are most studied across KEs
- Break down by **taxonomic group** -- which organisms/species are most represented in AOPs
- More detailed views by **OECD status** -- what distinguishes approved vs under-development AOPs

#### New plot variations -- cross-entity relationships
- **KE reuse across AOPs** -- which Key Events appear in multiple AOPs, showing shared biology across pathways
- Stressor-AOP connections and KER pathway chains were discussed but NOT selected -- focus on KE reuse

#### New plot variations -- data quality
- **Annotation completeness** -- which AOPs/KEs are missing key annotations (descriptions, ontology terms, evidence); heatmap of gaps
- **Curation progress over time** -- how is curation effort trending? Are new versions filling gaps or just adding entities?
- **Ontology term coverage** -- which GO/CHEBI/UBERON terms are used in AOP-Wiki vs available; identify annotation opportunities

#### Page organization
- All new plots go on the **existing main page** -- extend, don't create new pages

#### Raw data tables
- Add toggle-able raw data tables beneath **all plots** (new + existing)
- Static display -- no sort/search/filter needed
- Columns and row limits: Claude's discretion based on data size per plot

### Claude's Discretion
- Entity detail page layout (if entity links are implemented)
- Whether to include EXPL-04 (clickable entity links to AOP-Wiki) -- fit naturally if feasible
- Whether to include EXPL-06 (shareable URLs) -- fit naturally if feasible
- Number of new plots (scope based on what data supports and what's feasible)
- Raw data table toggle mechanism (expandable accordion vs button, etc.)
- Raw data table column selection (match CSV export vs simplified view)
- Raw data table row limits for large datasets

### Deferred Ideas (OUT OF SCOPE)
- **Individual AOP detail pages** (EXPL-01) -- deferred from original Phase 4 scope
- **Individual KE detail pages** (EXPL-02) -- deferred from original Phase 4 scope
- **Individual stressor detail pages** (EXPL-03) -- deferred from original Phase 4 scope
- **Stressor-AOP mapping** (ANLY-01, originally Phase 5) -- user chose KE reuse over stressor connections
- **KER pathway chain visualization** -- discussed but not selected for this phase

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EXPL-04 | Entity names and IDs in plots link directly to corresponding AOP-Wiki source pages | Claude's discretion. AOP-Wiki URLs follow pattern `https://aopwiki.org/aops/{id}`, `https://aopwiki.org/events/{id}`, `https://aopwiki.org/relationships/{id}`. Plotly `customdata` + click events can open external links. RDF data already contains `foaf:page` URIs for entities (see network.py KE query). |
| EXPL-05 | User can toggle a raw data table view beneath each plot showing the underlying data | Locked decision. `_plot_data_cache` already stores DataFrames for every plot. Need new `/api/plot-data/<plot_name>` endpoint to serve JSON, plus JavaScript toggle component. Currently 40+ plot containers across latest.html and trends.html. |
| EXPL-06 | URL encodes version and active plot state so users can share specific views | Claude's discretion. No URL state management exists currently. Would use `URLSearchParams` and `history.replaceState()` to encode `?version=2024-10-01&plot=latest_entity_counts`. Moderate implementation effort, natural fit with version-selector.js. |

</phase_requirements>

## Standard Stack

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Flask | (pinned in requirements.txt) | Web framework, API endpoints | Already the project foundation |
| Plotly | 2.32.0 (CDN) | Interactive visualizations | Already used for all 40+ plots |
| pandas | (pinned) | Data processing, DataFrame cache | Already used extensively |
| SPARQLWrapper | (pinned) | SPARQL endpoint communication | Already used for all queries |
| NetworkX | (pinned) | Graph analysis | Already used for network module |

### Supporting (No New Dependencies Needed)
| Library | Purpose | When to Use |
|---------|---------|-------------|
| plotly.express | High-level Plotly API | New plot functions (px.bar, px.heatmap, px.treemap) |
| plotly.graph_objects | Low-level Plotly API | Complex heatmaps, annotations needing fine control |
| Jinja2 | Template rendering | HTML for raw data table components |

### No New Dependencies Required
This phase can be implemented entirely with existing dependencies. The raw data tables use plain HTML `<table>` elements styled with CSS. No DataTables.js or similar library is needed (user explicitly requested static display with no sort/search/filter).

## Architecture Patterns

### Recommended Project Structure Changes
```
plots/
  __init__.py        # Add new exports
  shared.py          # Add get_plot_data_as_json() utility
  latest_plots.py    # Add ~5 new latest-data plot functions
  trends_plots.py    # Add ~3 new trends plot functions (curation progress)
  network.py         # No changes
static/
  js/
    lazy-loading.js  # Minor: emit event after plot load
    version-selector.js  # Minor: add URL state if EXPL-06
    raw-data-tables.js   # NEW: toggle component for data tables
  css/
    main.css         # Add raw data table styles
    lazy-loading.css # No changes
templates/
  latest.html        # Add new plot sections + data-table containers
  trends_page.html   # No changes (includes trends.html)
  trends.html        # Add new plot sections + data-table containers
  macros/
    methodology.html # No changes
    data_table.html  # NEW: reusable macro for data table toggle
app.py               # Add new API endpoint, download routes, plot registrations
static/data/
  methodology_notes.json  # Add entries for new plots
```

### Pattern 1: New Plot Addition (Established)
**What:** Follow the existing checklist in `.claude/add-a-plot.md`
**When to use:** Every new plot
**Key files to touch per plot type:**
- Latest Data: `plots/latest_plots.py`, `plots/__init__.py`, `app.py` (API + download), `templates/latest.html`, `static/js/version-selector.js`
- Historical Trends: `plots/trends_plots.py`, `plots/__init__.py`, `app.py` (tasks + unpack + plot_map + download), `templates/trends.html`

### Pattern 2: Raw Data Table Toggle
**What:** A reusable component that shows/hides a data table beneath any plot
**When to use:** Every plot container (existing + new)
**Design:**
```html
<!-- Template macro: data_table.html -->
{% macro data_table_toggle(plot_name) %}
<div class="data-table-container" data-table-for="{{ plot_name }}">
    <button class="data-table-toggle" onclick="toggleDataTable('{{ plot_name }}')">
        Show Raw Data
    </button>
    <div class="data-table-content" style="display:none;">
        <div class="data-table-loading">Loading data...</div>
    </div>
</div>
{% endmacro %}
```

```javascript
// raw-data-tables.js
async function toggleDataTable(plotName) {
    const container = document.querySelector(`[data-table-for="${plotName}"]`);
    const content = container.querySelector('.data-table-content');
    const button = container.querySelector('.data-table-toggle');

    if (content.style.display === 'none') {
        content.style.display = 'block';
        button.textContent = 'Hide Raw Data';

        // Fetch data only on first open
        if (!content.dataset.loaded) {
            const response = await fetch(`/api/plot-data/${plotName}`);
            const data = await response.json();
            content.innerHTML = buildTable(data.columns, data.rows);
            content.dataset.loaded = 'true';
        }
    } else {
        content.style.display = 'none';
        button.textContent = 'Show Raw Data';
    }
}

function buildTable(columns, rows) {
    let html = '<table class="raw-data-table"><thead><tr>';
    columns.forEach(col => html += `<th>${col}</th>`);
    html += '</tr></thead><tbody>';
    rows.forEach(row => {
        html += '<tr>';
        columns.forEach(col => html += `<td>${row[col] ?? ''}</td>`);
        html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
}
```

```python
# In app.py - new API endpoint
@app.route("/api/plot-data/<plot_name>")
def api_plot_data(plot_name):
    """API endpoint to serve raw plot data as JSON for data table display."""
    version = request.args.get('version')
    # Build cache key considering version
    if version and plot_name.startswith('latest_'):
        cache_key = f"{plot_name}_{version}"
    else:
        cache_key = plot_name

    if cache_key not in _plot_data_cache:
        # Try without version suffix
        if plot_name not in _plot_data_cache:
            return jsonify({'error': 'Data not available'}), 404
        cache_key = plot_name

    df = _plot_data_cache[cache_key]
    # Limit rows for large datasets
    max_rows = 100
    truncated = len(df) > max_rows
    display_df = df.head(max_rows)

    return jsonify({
        'columns': list(display_df.columns),
        'rows': display_df.to_dict('records'),
        'total_rows': len(df),
        'truncated': truncated,
        'plot_name': plot_name
    })
```

### Pattern 3: SPARQL Queries for New Data Dimensions
**What:** Optimized SPARQL queries for biological level, taxonomy, and KE reuse
**When to use:** New plot functions

**Biological Level of Organization:**
```sparql
# KEs by biological level - uses property http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25664
SELECT ?level (COUNT(DISTINCT ?ke) AS ?count)
WHERE {
    GRAPH <http://aopwiki.org/graph/{version}> {
        ?ke a aopo:KeyEvent .
        OPTIONAL {
            ?ke <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25664> ?level_obj .
            BIND(STR(?level_obj) AS ?level)
        }
    }
}
GROUP BY ?level
```

**Taxonomic Applicability:**
```sparql
# Taxonomic groups across AOPs - uses property http://purl.bioontology.org/ontology/NCBITAXON/131567
SELECT ?taxon (COUNT(DISTINCT ?aop) AS ?count)
WHERE {
    GRAPH <http://aopwiki.org/graph/{version}> {
        ?aop a aopo:AdverseOutcomePathway .
        ?aop <http://purl.bioontology.org/ontology/NCBITAXON/131567> ?taxon_obj .
        BIND(STR(?taxon_obj) AS ?taxon)
    }
}
GROUP BY ?taxon
ORDER BY DESC(?count)
```

**KE Reuse Across AOPs:**
```sparql
# KEs shared across multiple AOPs
SELECT ?ke ?title (COUNT(DISTINCT ?aop) AS ?aop_count)
WHERE {
    GRAPH <http://aopwiki.org/graph/{version}> {
        ?aop a aopo:AdverseOutcomePathway ;
             aopo:has_key_event ?ke .
        ?ke a aopo:KeyEvent .
        OPTIONAL { ?ke dc:title ?title }
    }
}
GROUP BY ?ke ?title
HAVING (COUNT(DISTINCT ?aop) > 1)
ORDER BY DESC(?aop_count)
```

### Pattern 4: URL State Management (EXPL-06, if implemented)
**What:** Encode version and scroll position in URL for shareability
**When to use:** On version change and plot scroll into view
```javascript
// In version-selector.js
function updateURLState() {
    const params = new URLSearchParams(window.location.search);
    if (selectedVersion) {
        params.set('version', selectedVersion);
    } else {
        params.delete('version');
    }
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    history.replaceState(null, '', newUrl);
}

// On page load, restore from URL
function restoreFromURL() {
    const params = new URLSearchParams(window.location.search);
    const version = params.get('version');
    if (version) {
        document.getElementById('version-selector').value = version;
        // Trigger change event
    }
}
```

### Anti-Patterns to Avoid
- **Separate pages for new plots:** User explicitly said "extend, don't create new pages." All plots go on existing snapshot and trends pages.
- **Interactive DataTables:** User explicitly said "static display, no sort/search/filter." Use plain HTML tables.
- **Hand-rolled plot infrastructure:** Follow `.claude/add-a-plot.md` exactly. Do not create alternative registration patterns.
- **Unbounded raw data tables:** Large datasets (1000+ rows) must be truncated with a "showing first N of M rows" note.
- **New SPARQL patterns:** Use established `run_sparql_query()` and `_build_graph_filter()` patterns. Do not create parallel query infrastructure.
- **Duplicate download route boilerplate:** Use the existing generic `/download/trend/<plot_name>` route instead of creating individual routes for each new plot.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Plot registration | Custom discovery system | `.claude/add-a-plot.md` checklist | Established pattern with 40+ plots already using it |
| CSV export | Custom CSV generation | `_plot_data_cache` + `get_csv_with_metadata()` | Already handles metadata, version info, export filenames |
| Error handling | Per-plot try/catch | `safe_plot_execution()` wrapper | Provides fallback plots, timing, consistent logging |
| Image export | Custom Plotly-to-image | `export_figure_as_image()` + `_plot_figure_cache` | Handles Kaleido engine, format negotiation, error recovery |
| DataTable UI | jQuery DataTables or AG Grid | Plain HTML `<table>` with CSS | User requested static display with no interactivity |
| Version filtering | Custom SPARQL WHERE clauses | `_build_graph_filter(version)` | Returns consistent (where_filter, order_limit) tuple |

**Key insight:** The codebase has 1,100+ lines of shared infrastructure in `plots/shared.py` specifically designed to prevent hand-rolling. Every new plot should use this infrastructure, not work around it.

## Common Pitfalls

### Pitfall 1: SPARQL Query Performance with New Data Dimensions
**What goes wrong:** Queries that join across multiple entity types (AOP -> KE -> biological level) or count distinct values across large sets can timeout on Virtuoso (30-second default).
**Why it happens:** Virtuoso struggles with complex OPTIONAL chains, cross-graph queries, and large intermediate result sets. The project has already encountered this -- the original "AOP Completeness by OECD Status" boxplot was removed because it hit Virtuoso limits.
**How to avoid:**
1. Always target a specific graph with `GRAPH <{target_graph}>` instead of cross-graph queries
2. Use separate simple queries and join results in Python rather than complex SPARQL JOINs
3. Limit result sets: use COUNT/GROUP BY in SPARQL, not SELECT * followed by Python counting
4. Test each new query against the actual SPARQL endpoint before committing
5. Follow the CSV-based property loading pattern: load property URIs from CSV, build FILTER clauses
**Warning signs:** Query execution >10 seconds logged as slow by `run_sparql_query_with_retry()`

### Pitfall 2: Page Performance with 50+ Plots and Data Tables
**What goes wrong:** Adding ~10 new plots plus data table toggles for all 50+ plots could degrade page load and scroll performance.
**Why it happens:** Even with lazy loading, the DOM grows significantly. Data table fetches for 50+ endpoints could create server load.
**How to avoid:**
1. Keep lazy loading for all new plots (already established pattern)
2. Data tables should fetch on-demand (only when user clicks "Show Raw Data"), not on page load
3. Truncate data tables to first 100 rows for large datasets
4. Use lightweight HTML tables, not complex JavaScript-driven components
5. Consider grouping navigation buttons to help users jump to sections
**Warning signs:** Initial page load >3 seconds, scroll jank, browser memory warnings

### Pitfall 3: Raw Data Table Cache Key Mismatches
**What goes wrong:** The `_plot_data_cache` uses specific keys (e.g., `'latest_entity_counts'`, `'aop_entity_counts_absolute'`). The data table API endpoint must match these exactly, including version suffixes for latest_ plots.
**Why it happens:** Latest plots use versioned cache keys like `latest_entity_counts_2024-10-01`. Trend plots use fixed keys like `aop_entity_counts_absolute`. The naming conventions differ.
**How to avoid:**
1. Document exact cache key for every plot in the API endpoint
2. For latest plots, try both `{plot_name}_{version}` and `{plot_name}` as fallback
3. Use `_plot_data_cache.keys()` to debug which keys actually exist
4. Test data table toggle for both current and historical versions
**Warning signs:** Data tables showing "Data not available" for plots that clearly have data

### Pitfall 4: Biological Level Data May Be Sparse
**What goes wrong:** The `C25664` (Level of Biological Organisation) property appears in property_labels.csv for KEs, but not all KEs have this annotation. The plot could show mostly "No level assigned."
**Why it happens:** AOP-Wiki annotations are heterogeneous -- some properties are well-populated, others sparse. The existing entity completeness plots already show this variation.
**How to avoid:**
1. First run a SPARQL query to check data availability before building the plot function
2. Design the plot to handle sparse data gracefully (show "Not Annotated" as a category)
3. Consider making this a "latest data" plot initially, not a trend, to reduce query complexity
4. Use the existing completeness infrastructure to understand expected population rates
**Warning signs:** More than 80% of values falling into "Not Annotated" category, making the plot uninformative

### Pitfall 5: Phase 5 Overlap with Data Quality Plots
**What goes wrong:** Phase 5 (ANLY-02) defines "composite data quality score per AOP" and Phase 5 (ANLY-03) defines "ontology coverage analysis." Both overlap with Phase 4's annotation completeness and ontology term coverage decisions.
**Why it happens:** The user redirected Phase 4's scope from entity exploration to dashboard enrichment, creating natural overlap with Phase 5's advanced analytics.
**How to avoid:**
1. Phase 4 should implement **dashboard-level aggregate views**: heatmaps of annotation gaps, curation progress line charts, overall ontology coverage pie charts
2. Phase 5 should implement **per-entity drill-down views**: individual AOP quality scores, per-term coverage detail, interactive explorations
3. Document which aspects each phase covers in PLAN.md so Phase 5 planner can adjust
4. Phase 4 plots should be "overview" quality; Phase 5 adds "deep dive" capability
**Warning signs:** Phase 5 planner finding nothing new to implement because Phase 4 already did everything

### Pitfall 6: Download Route Proliferation
**What goes wrong:** Adding ~10 new plots could mean ~10-20 new download routes (CSV + image per variant), bloating app.py further.
**Why it happens:** The existing pattern has individual download routes for each plot, resulting in highly repetitive code.
**How to avoid:** Use the existing generic `/download/trend/<plot_name>` route which already works for any plot cached in `_plot_data_cache`. For latest plots, either add to the generic handler or create a single `/download/latest/<plot_name>` generic route. Do not add individual routes.
**Warning signs:** app.py growing beyond 2000 lines, copy-paste download route code

## Code Examples

### Example 1: KE Reuse Across AOPs (Latest Data Plot)
```python
def plot_latest_ke_reuse(version: str = None) -> str:
    """Show which Key Events appear in multiple AOPs."""
    global _plot_data_cache

    if version:
        target_graph = f"http://aopwiki.org/graph/{version}"
        latest_version = version
    else:
        version_query = """
        SELECT ?graph WHERE {
            GRAPH ?graph { ?s a aopo:KeyEvent . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ORDER BY DESC(?graph) LIMIT 1
        """
        version_results = run_sparql_query(version_query)
        if not version_results:
            return create_fallback_plot("KE Reuse", "No graphs available")
        target_graph = version_results[0]["graph"]["value"]
        latest_version = target_graph.split("/")[-1]

    query = f"""
    SELECT ?ke (SAMPLE(?t) AS ?title) (COUNT(DISTINCT ?aop) AS ?aop_count)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway ;
                 aopo:has_key_event ?ke .
            ?ke a aopo:KeyEvent .
            OPTIONAL {{ ?ke dc:title ?t }}
        }}
    }}
    GROUP BY ?ke
    ORDER BY DESC(?aop_count)
    LIMIT 30
    """

    results = run_sparql_query(query)
    if not results:
        return create_fallback_plot("KE Reuse", "No data available")

    data = []
    for r in results:
        ke_id = r["ke"]["value"].split("/")[-1]
        title = r.get("title", {}).get("value", ke_id)
        count = int(r["aop_count"]["value"])
        data.append({
            "KE": f"KE {ke_id}: {title[:40]}",
            "AOP Count": count,
            "KE_ID": ke_id
        })

    df = pd.DataFrame(data)
    df["Version"] = latest_version
    _plot_data_cache[f'latest_ke_reuse_{latest_version}'] = df

    fig = px.bar(
        df, x="KE", y="AOP Count",
        title=f"Most Reused Key Events Across AOPs ({latest_version})",
        color="AOP Count",
        color_continuous_scale=[BRAND_COLORS['light'], BRAND_COLORS['primary']],
        text="AOP Count"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white", autosize=True,
        margin=dict(l=50, r=20, t=50, b=120),
        xaxis=dict(tickangle=45),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    _plot_figure_cache[f'latest_ke_reuse_{latest_version}'] = fig
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})
```

### Example 2: Annotation Completeness Heatmap (Latest Data Plot)
```python
def plot_latest_annotation_heatmap(version: str = None) -> str:
    """Heatmap showing annotation gaps across entity types and property categories."""
    # Query completeness for each entity type x property type combination
    # Uses the existing property_labels.csv categories
    # Reuses existing completeness query patterns from plot_latest_aop_completeness_by_status

    entity_types = [
        ("AOP", "aopo:AdverseOutcomePathway"),
        ("KE", "aopo:KeyEvent"),
        ("KER", "aopo:KeyEventRelationship"),
    ]

    # ... query each entity type's property completeness ...
    # ... build a matrix: rows=property categories, columns=entity types ...

    fig = px.imshow(
        matrix, x=entity_labels, y=property_categories,
        color_continuous_scale=[BRAND_COLORS['light'], BRAND_COLORS['primary']],
        text_auto='.1f',
        title=f"Annotation Completeness Heatmap ({version})",
        labels={'color': 'Completeness %'}
    )
    # ...
```

### Example 3: Raw Data Table API Endpoint
```python
@app.route("/api/plot-data/<plot_name>")
def api_plot_data(plot_name):
    """Serve cached plot data as JSON for raw data table display."""
    version = request.args.get('version')

    # Try version-specific key first for latest_ plots
    cache_key = plot_name
    if version and plot_name.startswith('latest_'):
        versioned_key = f"{plot_name}_{version}"
        if versioned_key in _plot_data_cache:
            cache_key = versioned_key

    if cache_key not in _plot_data_cache:
        return jsonify({'error': f'No data available for {plot_name}'}), 404

    df = _plot_data_cache[cache_key]
    max_rows = 100
    truncated = len(df) > max_rows
    display_df = df.head(max_rows)

    return jsonify({
        'columns': list(display_df.columns),
        'rows': display_df.to_dict('records'),
        'total_rows': len(df),
        'truncated': truncated,
        'plot_name': plot_name,
        'success': True
    })
```

### Example 4: Biological Level Breakdown
```python
def plot_latest_ke_by_bio_level(version: str = None) -> str:
    """Show KE distribution by biological level of organization."""
    # Uses property: http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25664
    # This is the "Level of Biological Organisation" property in property_labels.csv

    target_graph = _resolve_target_graph(version)

    query = f"""
    SELECT ?level (COUNT(DISTINCT ?ke) AS ?count)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?ke a aopo:KeyEvent .
            OPTIONAL {{
                ?ke <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25664> ?level_obj .
            }}
        }}
        BIND(IF(BOUND(?level_obj), STR(?level_obj), "Not Annotated") AS ?level)
    }}
    GROUP BY ?level
    ORDER BY DESC(?count)
    """
    # ... process results and create bar/treemap chart ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Individual download routes per plot | Generic `/download/trend/<plot_name>` route | Phase 2 | Reduces boilerplate; new plots should use generic route |
| Pre-computed all plots at startup | Lazy loading via IntersectionObserver | Phase 1 | ~50ms initial page load; new plots get this automatically |
| Single monolithic plots.py | Modular plots/ package (shared, latest, trends, network) | Phase 1 | New plots go in appropriate module |
| No version filtering | `_build_graph_filter(version)` helper | Phase 2 | All latest plots support version selection |
| Simple dict cache | VersionedPlotCache with TTL + LRU + pinning | Phase 3 | Memory-safe caching for version-specific data |

## Feasibility Assessment for Requested Plot Types

### Breakdowns by Biological Level (HIGH confidence)
- **Data availability:** Property `C25664` (Level of Biological Organisation) exists in property_labels.csv, applies to KEs
- **Query pattern:** Simple `SELECT ?level COUNT(DISTINCT ?ke)` with OPTIONAL for unannotated KEs
- **Risk:** Data may be sparse. Must test against actual endpoint to see population rate.
- **Recommendation:** Implement as latest-data plot first. Add trend variant only if data is consistently available across versions.

### Breakdowns by Taxonomic Group (HIGH confidence)
- **Data availability:** Property `NCBITAXON/131567` (Taxonomic Applicability) exists in property_labels.csv, applies to AOP|KE|KER
- **Query pattern:** Simple `SELECT ?taxon COUNT(DISTINCT ?aop)` GROUP BY
- **Risk:** Taxon values may be URIs requiring label lookup. Could fall back to URI suffix parsing.
- **Recommendation:** Implement for AOPs first (most interesting angle). Use BIND to extract species name from URI.

### Breakdowns by OECD Status (HIGH confidence)
- **Data availability:** Already implemented in `plot_latest_aop_completeness_by_status()`. OECD status property `C25688` well-populated.
- **Query pattern:** Existing pattern in latest_plots.py lines 1246-1431 provides proven approach.
- **Risk:** None -- this is an extension of existing proven functionality.
- **Recommendation:** Add new angles: entity count breakdown by status, property presence by status, average completeness by status over time.

### KE Reuse Across AOPs (HIGH confidence)
- **Data availability:** `aopo:has_key_event` relationship is fundamental AOP-Wiki structure. The connectivity analysis already queries this.
- **Query pattern:** `COUNT(DISTINCT ?aop)` per KE, filtered to HAVING count > 1
- **Risk:** Minimal. This query is structurally simple.
- **Recommendation:** Two views: (1) bar chart of top 20 most reused KEs, (2) histogram of reuse distribution.

### Annotation Completeness Heatmap (HIGH confidence)
- **Data availability:** All completeness data infrastructure exists. Property labels, completeness queries, and by-status grouping are proven.
- **Query pattern:** Extend existing completeness patterns to create a matrix across entity types.
- **Risk:** Heatmap may need px.imshow or go.Heatmap rather than standard bar charts.
- **Recommendation:** Use plotly.express.imshow with the existing brand color scale.

### Curation Progress Over Time (MEDIUM confidence)
- **Data availability:** Entity completeness trends already exist (`plot_entity_completeness_trends`). Need to decompose into "new entities added" vs "existing entities improved."
- **Query pattern:** Compare entity counts AND completeness across consecutive versions.
- **Risk:** Requires version-to-version comparison. The query pattern exists in trends_plots.py (delta calculations) but this specific decomposition is new.
- **Recommendation:** Implement as: (1) new entities added per version (from main_graph delta), (2) average completeness change for entities that existed in previous version. May need two separate queries per version.

### Ontology Term Coverage (MEDIUM confidence)
- **Data availability:** Process and object ontology usage already tracked. Current plots show ontology source distribution (GO, CHEBI, etc.).
- **Query pattern:** Count unique terms used vs count of unique terms in reference ontology. The "reference" side is the challenge.
- **Risk:** We don't have a reference ontology loaded into the SPARQL endpoint to compare against. Can only show "what's used" not "what's available."
- **Phase 5 overlap:** ANLY-03 is explicitly "which GO/CHEBI/UBERON terms are used vs available." Phase 4 should show usage diversity and growth trends. Phase 5 should compare against reference ontologies.
- **Recommendation:** Focus on (1) number of unique ontology terms used over time, (2) term diversity (ratio of unique terms to total annotations), (3) new terms appearing per version. Do NOT attempt "used vs available" comparison -- that's Phase 5 territory.

## Discretion Recommendations

### EXPL-04: Entity Links to AOP-Wiki -- RECOMMEND YES
- **Effort:** LOW. Plotly supports `customdata` arrays that can carry URLs. A `plotly_click` event handler can open `window.open(url)`.
- **Value:** HIGH. Transforms plots from read-only dashboards into navigation tools.
- **Implementation:** AOP-Wiki URLs follow predictable patterns (`https://aopwiki.org/aops/{id}`). Entity IDs are already available in cached DataFrames. Add wiki_url column to DataFrames and configure Plotly click handlers.
- **Constraint:** Only feasible for plots that show individual entities (KE reuse, entity counts). Not feasible for aggregate metrics (averages, percentages).

### EXPL-06: Shareable URLs -- RECOMMEND YES
- **Effort:** MEDIUM. Requires modifying version-selector.js to read/write URL parameters.
- **Value:** HIGH. Researchers frequently share specific views.
- **Implementation:** Use `URLSearchParams` API (supported in all modern browsers). Encode `?version=2024-10-01` on the snapshot page. Optionally add `#section-id` for scroll position.
- **Constraint:** Only encode version selection, not individual plot states (too many plots).

### Raw Data Table Toggle Mechanism -- RECOMMEND: Expandable section with button
- **Rationale:** Consistent with the existing `<details>` elements used for property details and methodology notes. Users understand collapsible sections.
- **Design:** A styled `<details>` element beneath each plot's download dropdown area, or a dedicated button styled like the existing download buttons.
- **Row limit:** 100 rows for standard plots, 50 rows for plots with many columns (>8 columns). Show "Showing first N of M rows. Download CSV for full data." message when truncated.

### Number of New Plots -- RECOMMEND: 8-10 new plots
Based on data feasibility:
1. **KE by Biological Level** (latest data) -- bar chart
2. **Taxonomic Group Distribution** (latest data) -- bar/treemap chart
3. **Entity Counts by OECD Status** (latest data) -- grouped bar chart
4. **KE Reuse Across AOPs** (latest data) -- bar chart of top reused KEs
5. **KE Reuse Distribution** (latest data) -- histogram of reuse frequency
6. **Annotation Completeness Heatmap** (latest data) -- heatmap matrix
7. **Curation Progress** (trends) -- new entities + completeness improvement line chart
8. **Ontology Term Diversity** (latest data) -- unique terms per ontology
9. **Ontology Term Growth** (trends) -- unique terms over time (builds on existing bio_processes/bio_objects)

## Open Questions

1. **Biological level data population rate**
   - What we know: The property `C25664` exists in property_labels.csv and applies to KEs
   - What's unclear: What percentage of KEs actually have this annotation filled in
   - Recommendation: Run a test SPARQL query against the endpoint before implementing. If <20% populated, consider showing this as an "annotation gap" example rather than a standalone breakdown.

2. **Taxonomic data format**
   - What we know: `NCBITAXON/131567` property exists and applies to AOP|KE|KER
   - What's unclear: Whether taxon values are URIs (needing label lookup) or literal strings
   - Recommendation: Run test query, inspect value format. Use BIND(STR(?taxon_obj)) to normalize.

3. **Ontology coverage boundary with Phase 5**
   - What we know: Phase 5 ANLY-03 is "ontology coverage vs available." Phase 4 should avoid duplicating this.
   - What's unclear: Exact boundary between "usage overview" (Phase 4) and "coverage analysis" (Phase 5)
   - Recommendation: Phase 4 shows aggregate counts, diversity ratios, and growth trends. Phase 5 adds per-term detail and external reference comparison. Document this in PLAN.md.

4. **App.py file size management**
   - What we know: app.py is 1,735 lines. Adding ~10 download routes + 1 API endpoint could push to 2,000+.
   - What's unclear: Whether the download route proliferation pattern should be refactored
   - Recommendation: Use generic download routes (`/download/trend/<plot_name>` and a new `/download/latest/<plot_name>`) instead of individual routes. Defer full refactoring to a future phase.

## Sources

### Primary (HIGH confidence)
- **Codebase analysis:** Direct reading of all 6,955 lines across `app.py`, `plots/latest_plots.py`, `plots/trends_plots.py`, `plots/shared.py`, `plots/__init__.py`, `plots/network.py`
- **Property labels:** `property_labels.csv` -- 45 properties with URI, label, type, applies_to columns
- **Templates:** `templates/latest.html`, `templates/trends.html`, `templates/trends_page.html` -- current HTML structure
- **JavaScript:** `static/js/lazy-loading.js`, `static/js/version-selector.js` -- current frontend infrastructure
- **Methodology notes:** `static/data/methodology_notes.json` -- documentation for each plot
- **Add-a-plot checklist:** `.claude/add-a-plot.md` -- verified step-by-step process
- **Architecture reference:** `.claude/architecture.md` -- confirmed current stack and data flows
- **Phase 5 roadmap:** `.planning/ROADMAP.md` lines 98-106 -- confirmed ANLY-01/02/03 overlap areas

### Secondary (MEDIUM confidence)
- **Plotly heatmap capabilities:** Confirmed via Plotly documentation that `px.imshow()` supports custom color scales and text annotations
- **URL state management:** Standard browser API (`URLSearchParams`, `history.replaceState`) -- well-established patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing tools proven
- Architecture: HIGH -- well-established patterns documented in `.claude/add-a-plot.md`
- SPARQL queries: HIGH for entity counts/reuse, MEDIUM for biological level/taxonomy (untested against endpoint)
- Raw data tables: HIGH -- straightforward HTML tables with API backing
- Pitfalls: HIGH -- based on documented project history (Virtuoso limits, cache key mismatches)

**Research date:** 2026-02-23
**Valid until:** 2026-03-23 (stable codebase, no fast-moving dependencies)
