# Phase 2: Reliability and Completeness - Research

**Researched:** 2026-02-21
**Domain:** SPARQL query optimization, Plotly export infrastructure, frontend UI components, scientific visualization metadata
**Confidence:** HIGH

## Summary

Phase 2 makes the existing dashboard reliable and complete. Four requirements must be addressed: optimizing the AOP Completeness Distribution boxplot to load in under 30 seconds (RELY-01), restoring the OECD status visualization as a grouped bar chart (RELY-02), adding CSV/PNG/SVG export buttons to every displayed plot (RELY-04), and adding expandable methodology notes to every plot (EXPL-07).

The codebase already has substantial export infrastructure (download routes, figure caching, Plotly-to-image via Kaleido), but coverage is incomplete: many trend plots cache figures but NOT data (preventing CSV export), and kaleido is not in requirements.txt (preventing PNG/SVG export in Docker). The OECD status visualization already exists as working latest-data plots (`plot_latest_aop_completeness_by_status`, `plot_latest_ke_completeness_by_status`, `plot_latest_ker_completeness_by_status`) but the historical trend version was removed because its boxplot-per-version-per-status query hit Virtuoso limits. The user wants a grouped bar chart (mean completeness per status) as the replacement, which requires far less data than a boxplot.

**Primary recommendation:** Structure work into four plans: (1) SPARQL optimization and OECD visualization, (2) complete export infrastructure gap closure, (3) methodology notes UI component, (4) audit and verification. The OECD replacement should reuse the existing `plot_latest_aop_completeness_by_status` SPARQL pattern (aggregate in SPARQL, transfer means not raw scores) rather than trying to fix the removed boxplot approach.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Grouped bar chart replacing the removed boxplot (which hit Virtuoso limits)
- Shows mean completeness score per OECD status
- Include all OECD statuses found in the data (no grouping into "Other")
- Both a latest-data snapshot AND historical trend (completeness per status over time)
- Two visualizations: bar chart for selected version + trend lines across versions
- Export filenames include version and date for traceability (e.g., `aop-entity-counts_2025-12-01_v2025-12-01.csv`)
- Each sub-plot in multi-figure trend plots gets its own export buttons (per sub-plot, not combined)
- Depth: moderate -- paragraph explaining what it measures and how, plus SPARQL reference
- Always include known limitations for every plot (builds trust with researchers)
- SPARQL reference: viewable query snippet in an expandable code block (maximum transparency for technical users)
- Target: AOP Completeness Distribution under 30 seconds (down from ~75s)
- Pre-compute and cache heavy queries at startup (trade boot time for user experience)
- Keep full completeness scoring -- no property simplification even if slower
- Audit ALL plots for reliability, with specific attention to AOP Lifetime plot and KE component trends
- Graceful degradation: error card with retry (current behavior) -- no stale data or fallback plots
- Include Virtuoso-side tuning recommendations, documented only (not applied in Docker config)

### Claude's Discretion
- Export button UI pattern (icon row, dropdown, or modebar)
- PNG/SVG metadata approach
- Methodology note presentation component (accordion, popover, etc.)
- Per-plot timeout configuration
- Exact spacing and styling details
- SPARQL timeout policy based on query complexity research

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RELY-01 | AOP completeness distribution plot loads in under 30 seconds (currently ~75s) | Optimization analysis of `plot_aop_completeness_boxplot` shows 4 SPARQL queries; pre-computation at startup + potential query batching can reduce wall-clock time. See "SPARQL Optimization Patterns" section. |
| RELY-02 | AOP completeness by OECD status visualization is restored or replaced with working alternative | Working latest-data pattern exists in `plot_latest_aop_completeness_by_status`. Historical trend requires new `plot_oecd_completeness_trend()` function using aggregated means per status per version. See "OECD Status Alternative" section. |
| RELY-04 | Every displayed visualization has CSV, PNG, and SVG export options | Export infrastructure exists but has gaps: 8 trend plot functions missing data cache, kaleido not in requirements.txt, some download routes missing. See "Export Gap Analysis" section. |
| EXPL-07 | Each plot has an expandable methodology note explaining what it measures and how | No existing methodology note infrastructure. Must build reusable HTML/CSS component. See "Methodology Notes Component" section. |
</phase_requirements>

## Standard Stack

### Core (already in codebase)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Flask | ~=3.1 | Web framework | Already the application framework |
| Plotly | ~=5.22 | Visualization | Already generates all charts |
| pandas | ~=2.2 | Data processing | Already used for all DataFrames |
| SPARQLWrapper | ~=2.0 | SPARQL queries | Already the query interface |
| python-json-logger | ~=2.0 | Structured logging | Already configured for JSON output |

### Supporting (needs addition)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| kaleido | ~=0.2 | Static image export (PNG/SVG) | MUST add to requirements.txt -- used by `export_figure_as_image()` but missing from deps |

### No New Libraries Required
All phase work can be accomplished with existing stack. The methodology notes component uses plain HTML/CSS/JS. Export buttons use existing dropdown patterns. SPARQL optimization is query-level, not library-level.

**Installation:**
```bash
# Add to requirements.txt
kaleido~=0.2
```

## Architecture Patterns

### Current Plot Registration Flow
```
plots/trends_plots.py or plots/latest_plots.py
    -> plots/__init__.py (export)
    -> app.py (plot_tasks + plot_map + download routes)
    -> templates/*.html (lazy-plot containers + download buttons)
    -> static/js/lazy-loading.js (IntersectionObserver)
    -> static/js/version-selector.js (latest_* plots only)
```

### Pattern 1: Existing Download Button UI (Dropdown)
**What:** Hover-activated dropdown with CSV/PNG/SVG options
**When to use:** Already established on latest.html and trends.html
**Example:**
```html
<div class="download-dropdown">
    <button class="download-button dropdown-toggle">Download ▼</button>
    <div class="dropdown-menu">
        <a href="/download/trend/{cache_key}?format=csv" class="dropdown-item">CSV</a>
        <a href="/download/trend/{cache_key}?format=png" class="dropdown-item">PNG</a>
        <a href="/download/trend/{cache_key}?format=svg" class="dropdown-item">SVG</a>
    </div>
</div>
```
**Recommendation:** Keep this exact pattern. It is consistent, already styled, and users on latest.html are familiar with it. **Do NOT switch to modebar integration** (Plotly modebar is too hidden for non-technical users).

### Pattern 2: Methodology Note Component (Recommended: `<details>` element)
**What:** Native HTML `<details>`/`<summary>` for collapsible methodology notes
**When to use:** Every plot container
**Why `<details>` over custom accordion:**
- Zero JS required (native browser behavior)
- Already used in latest.html for property details (lines 245-250, 275-280, 304-309)
- Accessible by default (screen readers, keyboard navigation)
- No library dependency
- Consistent with existing codebase pattern

**Example (per-plot methodology note):**
```html
<details class="methodology-note">
    <summary>Methodology</summary>
    <div class="methodology-content">
        <p><strong>What this measures:</strong> [Description]</p>
        <p><strong>Data source:</strong> [SPARQL endpoint details]</p>
        <p><strong>Known limitations:</strong> [Honest limitations]</p>
        <details class="sparql-query">
            <summary>View SPARQL Query</summary>
            <pre><code>[query text]</code></pre>
        </details>
    </div>
</details>
```

### Pattern 3: Generic Download Route (already exists)
**What:** `/download/trend/<plot_name>` catches all trend plots generically
**Where:** app.py line 1148-1196
**Key insight:** This route already handles any `plot_name` from `_plot_data_cache` or `_plot_figure_cache`. Many plots are already covered by this generic route -- they just need their data/figure caches populated.

### Pattern 4: Export Filename with Version and Date
**What:** Self-documenting filenames for downloaded files
**User decision:** Filenames include version and date, e.g., `aop-entity-counts_2025-12-01_v2025-12-01.csv`
**Implementation:** Modify `get_csv_with_metadata()` and download routes to include version in `Content-Disposition` header filename. For trend plots (multi-version), use the export date. For latest plots (single version), include both version and date.

### Anti-Patterns to Avoid
- **Don't create per-plot download routes for trend plots:** The generic `/download/trend/<plot_name>` handles all of them. Only latest-data plots need individual routes (because they have version query params and different cache key patterns).
- **Don't embed methodology text in Python plot functions:** Keep methodology content in templates (HTML) or a separate data file, not in Python. This allows updating text without restarting the server.
- **Don't try to fix the removed boxplot-by-status approach:** The boxplot requires transferring ~240K rows from SPARQL per version. The user chose grouped bar chart (mean per status) specifically because it aggregates in SPARQL.
- **Don't add kaleido as a runtime import guard:** It is needed for all PNG/SVG exports. Add it to requirements.txt as a hard dependency.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Collapsible UI sections | Custom accordion with JS | HTML `<details>` element | Native browser behavior, zero JS, accessible, already used in codebase |
| Image export | Custom canvas-to-image | `plotly.io.to_image()` with kaleido | Already implemented in `export_figure_as_image()` in shared.py |
| Download dropdowns | New component library | Existing `.download-dropdown` CSS class | Already styled and working on both pages |
| SPARQL retry logic | Custom HTTP retry | `run_sparql_query_with_retry()` | Already handles exponential backoff, error classification |
| CSV metadata headers | Custom file builder | `get_csv_with_metadata()` | Already generates metadata-prefixed CSV strings |

**Key insight:** The export infrastructure is 90% built. The remaining work is gap closure (adding data caches to plots that only cache figures, adding kaleido to requirements.txt), not new system design.

## Common Pitfalls

### Pitfall 1: Missing Data Cache Prevents CSV Export
**What goes wrong:** A plot caches its Plotly figure (`_plot_figure_cache`) for PNG/SVG export but never caches the DataFrame (`_plot_data_cache`). The download route returns 404 for CSV.
**Why it happens:** The add-a-plot checklist mentions caching, but several early plots were written before the CSV export system existed.
**How to avoid:** Audit every plot function. For each `_plot_figure_cache[key] = fig`, verify a corresponding `_plot_data_cache[key] = df` exists.
**Affected plots (trend plots missing data cache):**
- `plot_avg_per_aop()` -- caches figures as `average_components_per_aop_absolute/delta` but no data cache
- `plot_network_density()` -- caches figure as `aop_network_density` but no data cache
- `plot_author_counts()` -- caches figures as `aop_authors_absolute/delta` but no data cache
- `plot_aop_lifetime()` -- caches figures as `aops_created_over_time/modified/scatter` but no data cache
- `plot_ke_components()` -- caches figures as `ke_component_annotations_absolute/delta` but no data cache
- `plot_ke_components_percentage()` -- caches figures as `ke_components_percentage_absolute/delta` but no data cache
- `plot_unique_ke_components()` -- caches figures as `unique_ke_components_absolute/delta` but no data cache
- `plot_bio_processes()` -- caches figures as `biological_process_annotations_absolute/delta` but no data cache
- `plot_bio_objects()` -- caches figures as `biological_object_annotations_absolute/delta` but no data cache
- `plot_kes_by_kec_count()` -- caches figures as `kes_by_kec_count_absolute/delta` but no data cache

### Pitfall 2: Kaleido Not in requirements.txt
**What goes wrong:** `export_figure_as_image()` calls `pio.to_image(..., engine='kaleido')` but kaleido is not in requirements.txt. Works locally (installed manually) but fails in Docker build.
**Why it happens:** Kaleido was added as a runtime dependency but never added to the package manifest.
**How to avoid:** Add `kaleido~=0.2` to requirements.txt before any export testing.
**Warning signs:** PNG/SVG download routes return 404 in Docker; work fine in local development.

### Pitfall 3: Virtuoso Query Execution Limits
**What goes wrong:** Complex SPARQL queries that join across all versions with large intermediate result sets hit Virtuoso's `MaxQueryExecutionTime` or `MaxQueryMem` limits and return empty results or timeouts.
**Why it happens:** The removed `plot_aop_completeness_boxplot_by_status()` tried to query network structure + property counts + OECD status for ALL AOPs across ALL versions, producing ~240K intermediate rows before grouping.
**How to avoid:** Aggregate in SPARQL (use GROUP BY to compute means in the query, not in Python). For the OECD alternative, query mean completeness per status per version, not raw per-AOP scores.
**Warning signs:** Query returns 0 results after a long wait; works on one version but fails across all versions.

### Pitfall 4: Inconsistent Cache Key Naming
**What goes wrong:** Download buttons use a cache key that doesn't match what the plot function stored, resulting in 404 on download.
**Why it happens:** Cache keys are string-based with no central registry. Template HTML hard-codes one key, Python code uses another.
**How to avoid:** When adding data caches to existing plots, use the EXACT same keys already used for figure caches (they match the download URLs in templates).

### Pitfall 5: Methodology Notes Bloat Template Files
**What goes wrong:** Adding ~30 methodology note blocks (one per plot) directly in HTML makes templates enormous and hard to maintain.
**Why it happens:** Each note has 3-5 paragraphs plus a SPARQL query snippet.
**How to avoid:** Store methodology content in a structured data file (JSON or YAML) and render via Jinja2 template includes or a Flask context processor. Alternatively, serve methodology content from the API and load it dynamically (similar to how property details are loaded on latest.html).

### Pitfall 6: SPARQL Query Strings in Methodology Notes Require Escaping
**What goes wrong:** SPARQL queries contain `{`, `}`, `<`, `>` which conflict with HTML and Jinja2 template syntax.
**Why it happens:** Raw SPARQL inserted into HTML templates without proper escaping.
**How to avoid:** If using Jinja2, pipe through `|e` filter or use `<pre><code>` with escaped content. If using JSON data source, the JSON encoding handles escaping. If loading from API, JSON serialization handles it automatically.

## Code Examples

### Example 1: Adding Data Cache to Existing Plot (Gap Closure Pattern)
```python
# In plot_avg_per_aop() -- add these lines BEFORE the return statement:

# Cache data for CSV export
_plot_data_cache['average_components_per_aop_absolute'] = df_melted.copy()
_plot_data_cache['average_components_per_aop_delta'] = df_delta_melted.copy()
```
This pattern applies to all 10 trend plots missing data caches. The cache key MUST match the figure cache key (which matches the download URL).

### Example 2: OECD Mean Completeness Trend (Historical Version)
```python
def plot_oecd_completeness_trend() -> tuple[str, str]:
    """Mean AOP completeness per OECD status across all versions.

    Returns:
        tuple: (bar_chart_latest_html, trend_lines_html)
    """
    # Strategy: For each version, query aggregated mean completeness per status
    # This avoids the boxplot approach that required per-AOP raw scores (240K+ rows)

    # Single query: aggregate completeness by status and version
    query = f"""
    SELECT ?graph ?status (AVG(?completeness_pct) AS ?mean_completeness)
    WHERE {{
      GRAPH ?graph {{
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {{
            ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
            BIND(STR(?status_obj) AS ?status)
        }}
        # Count properties per AOP
        {{
          SELECT ?aop (COUNT(DISTINCT ?p) AS ?prop_count)
          WHERE {{
            ?aop ?p ?o .
            FILTER(?p IN ({property_filter}))
          }}
          GROUP BY ?aop
        }}
        BIND((?prop_count / {total_props}) * 100 AS ?completeness_pct)
      }}
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }}
    GROUP BY ?graph ?status
    ORDER BY ?graph ?status
    """
    # NOTE: This is conceptual. The actual implementation may need to compute
    # completeness in Python like the existing functions do, but aggregate
    # to means before creating the plot. The key optimization is: only store
    # one mean value per (version, status) pair, not one score per AOP.
```

### Example 3: Methodology Note Data Structure
```python
# methodology.py or methodology.json approach
METHODOLOGY = {
    "latest_entity_counts": {
        "title": "Entity Counts",
        "description": "Total number of each entity type (AOPs, Key Events, Key Event Relationships, Stressors) and unique authors in the selected version of the AOP-Wiki database.",
        "data_source": "Counts are obtained by querying the SPARQL endpoint for instances of each entity class (aopo:AdverseOutcomePathway, aopo:KeyEvent, aopo:KeyEventRelationship, aopo:Stressor) within the selected version graph.",
        "limitations": "Author counts reflect unique creator URIs in the RDF data, which may not exactly match AOP-Wiki user accounts due to data transformation.",
        "sparql_summary": "SELECT (COUNT(DISTINCT ?entity) AS ?count) WHERE { GRAPH ?g { ?entity a aopo:AdverseOutcomePathway } }"
    },
    # ... one entry per plot
}
```

### Example 4: Export Filename with Version and Date
```python
# Modify download route to include version and date in filename
from datetime import datetime

def build_export_filename(plot_name: str, format: str, version: str = None) -> str:
    """Build self-documenting export filename.

    Examples:
        aop-entity-counts_2026-02-21_v2025-12-01.csv
        ke-components-absolute_2026-02-21.png  (trend plots, no single version)
    """
    date_str = datetime.now().strftime('%Y-%m-%d')
    clean_name = plot_name.replace('_', '-')

    if version:
        return f"{clean_name}_{date_str}_v{version}.{format}"
    return f"{clean_name}_{date_str}.{format}"
```

### Example 5: Methodology Note CSS
```css
.methodology-note {
    margin-top: 8px;
    margin-bottom: 8px;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    font-size: 13px;
}

.methodology-note > summary {
    cursor: pointer;
    padding: 8px 12px;
    color: #307BBF;
    font-weight: 500;
    user-select: none;
}

.methodology-note > summary:hover {
    background-color: #f9f9f9;
}

.methodology-content {
    padding: 12px 16px;
    background-color: #fafafa;
    border-top: 1px solid #e0e0e0;
    line-height: 1.6;
}

.methodology-content p {
    margin: 6px 0;
}

.sparql-query > summary {
    cursor: pointer;
    color: #666;
    font-size: 12px;
    margin-top: 8px;
}

.sparql-query pre {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 11px;
    line-height: 1.4;
    max-height: 300px;
    overflow-y: auto;
}
```

## Detailed Findings

### RELY-01: AOP Completeness Distribution Optimization

**Current state:** `plot_aop_completeness_boxplot()` (trends_plots.py:2161) currently takes ~75 seconds. It executes 4 SPARQL queries:
1. `query_combined_totals` -- entity totals per version (fast, ~1s)
2. `query_combined_presence` -- property presence per entity type per version (moderate, ~10s)
3. `query_network` -- AOP-KE-KER network structure with GROUP_CONCAT (slow, ~30s)
4. `query_combined` -- property counts per entity per version with UNION (slow, ~30s)

**Optimization approaches (ordered by impact):**

1. **Pre-compute at startup and cache result** (HIGH impact, user-approved):
   - The boxplot is currently lazy-loaded (on-demand via lambda in plot_map)
   - Move it to startup computation in `compute_plots_parallel()` like other heavy plots
   - User explicitly approved trading boot time for user experience
   - Expected impact: 0s user-facing load time (pre-computed)

2. **Batch queries in parallel** (MEDIUM impact):
   - Queries 1+2 have no dependency on 3+4
   - Use ThreadPoolExecutor to run (1,2) and (3,4) in parallel
   - Expected: reduce from ~75s to ~40s (before caching)

3. **Limit historical versions** (MEDIUM impact, may not be needed with caching):
   - Instead of querying ALL historical versions, only query the last N (e.g., 10)
   - The boxplot currently shows all ~15 versions; trimming older ones reduces data volume
   - Tradeoff: loses long-term history

4. **Optimize GROUP_CONCAT query** (LOW impact):
   - The network query already uses GROUP_CONCAT (optimized in earlier commit)
   - Further optimization would require Virtuoso-specific tuning (indices, query plans)

**Recommendation:** Approach 1 (pre-compute at startup) alone meets the 30-second requirement for user-facing load time. Approach 2 can additionally reduce boot time. The function should also use `_plot_data_cache` global declaration which it already has.

**Current code location:** `app.py` line 256 shows `graph_aop_completeness_boxplot = ""` (skipped at startup, generated on-demand). The fix is to include it in `plot_tasks` (line 156-185) and remove the lambda from `plot_map`.

### RELY-02: OECD Status Alternative Visualization

**Current state:**
- **Removed:** `plot_aop_completeness_boxplot_by_status()` (trends_plots.py:2563-2889) -- boxplot approach, hits Virtuoso limits
- **Working:** `plot_latest_aop_completeness_by_status()` (latest_plots.py:1216-1401) -- grouped bar chart for single version, works well
- **Working:** `plot_latest_ke_completeness_by_status()` and `plot_latest_ker_completeness_by_status()` -- same pattern for KE/KER

**User decision:** Two visualizations:
1. **Latest snapshot (bar chart):** Already exists and works. Already on latest.html. No changes needed.
2. **Historical trend (new):** Mean completeness per OECD status over time as trend lines.

**Implementation approach for historical trend:**
- New function `plot_oecd_completeness_trend()` in trends_plots.py
- For each version: compute mean completeness per OECD status (reuse the aggregation pattern from `plot_latest_aop_completeness_by_status`)
- Plot as line chart: x=version, y=mean_completeness, color=OECD_status
- This transfers ~(versions * statuses) rows (~15 * ~5 = 75 data points), not 240K raw scores
- Returns `(absolute_html, data)` or just `str` depending on whether delta view makes sense for means

**SPARQL query strategy:**
- The latest version already uses an efficient aggregated query (latest_plots.py:1277-1291)
- For the historical version, run the same aggregated query but without filtering to a single graph
- Key: aggregate in SPARQL using `AVG()` or compute means in Python from per-version batched queries

**Where the removed code commented out in template:** trends.html lines 308-334 (HTML comment block)

### RELY-04: Export Gap Analysis

**Export infrastructure assessment:**

| Component | Status | Gap |
|-----------|--------|-----|
| `export_figure_as_image()` | EXISTS in shared.py | Works but kaleido not in requirements.txt |
| `get_csv_with_metadata()` | EXISTS in shared.py | Works; filename needs version/date enhancement |
| `create_bulk_download()` | EXISTS in shared.py | Works for ZIP downloads |
| Generic trend download route | EXISTS `/download/trend/<plot_name>` | Works for any cache key |
| Download dropdown CSS | EXISTS in main.css | Fully styled |
| Download dropdown HTML pattern | EXISTS on both pages | Established pattern |

**Plots with COMPLETE export support (data + figure cache):**
- `plot_main_graph()` -- data: absolute/delta, figure: absolute/delta
- `plot_aop_property_presence()` -- data: absolute/percentage, figure: absolute/percentage
- `plot_ke_property_presence()` -- same pattern
- `plot_ker_property_presence()` -- same pattern
- `plot_stressor_property_presence()` -- same pattern
- `plot_entity_completeness_trends()` -- data + figure
- `plot_aop_completeness_boxplot()` -- data + figure
- `plot_kes_by_kec_count()` -- figure only; data cache MISSING
- All `plot_latest_*()` functions -- data + figure

**Plots with FIGURE cache only (PNG/SVG works, CSV fails):**
- `plot_avg_per_aop()` -- figures: `average_components_per_aop_absolute/delta`
- `plot_network_density()` -- figure: `aop_network_density`
- `plot_author_counts()` -- figures: `aop_authors_absolute/delta`
- `plot_aop_lifetime()` -- figures: `aops_created_over_time/modified/scatter`
- `plot_ke_components()` -- figures: `ke_component_annotations_absolute/delta`
- `plot_ke_components_percentage()` -- figures: `ke_components_percentage_absolute/delta`
- `plot_unique_ke_components()` -- figures: `unique_ke_components_absolute/delta`
- `plot_bio_processes()` -- figures: `biological_process_annotations_absolute/delta`
- `plot_bio_objects()` -- figures: `biological_object_annotations_absolute/delta`
- `plot_kes_by_kec_count()` -- figures: `kes_by_kec_count_absolute/delta`

**Total: 10 trend plot functions need data cache additions (affecting ~22 cache keys).**

**Download button HTML coverage:**
- Latest page (latest.html): ALL plots have download dropdowns with CSV/PNG/SVG -- COMPLETE
- Trends page (trends.html): ALL plot boxes have download dropdowns -- COMPLETE for existing cache keys
- But: many download links will 404 for CSV because data caches are missing

**Export filename enhancement:**
- Current: `Content-Disposition: attachment; filename={plot_name}.csv`
- Required: `Content-Disposition: attachment; filename={plot_name}_{date}_v{version}.{format}`
- Affects: `get_csv_with_metadata()` and all download route handlers

### EXPL-07: Methodology Notes

**Current state:** No methodology infrastructure exists. Plots have `<p class="plot-description">` with 1-2 sentences, but no methodology depth, no SPARQL queries, no limitations.

**Content requirements per note:**
1. **What this measures** -- plain language description (~1 paragraph)
2. **How data is sourced** -- data pipeline description
3. **Known limitations** -- honest disclosure (required by user)
4. **SPARQL query** -- expandable code block with actual query

**Total plots needing methodology notes:**
- Latest page: 10 plot containers (entity_counts, aop_connectivity, avg_per_aop, ke_components, ke_annotation_depth, aop_completeness, aop_completeness_by_status, ke_completeness_by_status, ker_completeness_by_status, process_usage, object_usage)
- Trends page: ~16 plot boxes (main_graph, aop_property_presence, ke_property_presence, ker_property_presence, stressor_property_presence, entity_completeness, aop_completeness_boxplot, avg_per_aop, network_density, authors, aops_created, aops_modified, aop_creation_scatter, ke_components, ke_components_pct, unique_ke_components, bio_processes, bio_objects, kes_by_kec)
- Total: ~26 unique methodology note blocks

**Storage strategy recommendation:**
- Option A: Inline in templates -- simplest, but bloats HTML
- Option B: JSON data file + Jinja2 macro -- separates content from presentation
- Option C: API endpoint + dynamic loading -- like property details on latest.html

**Recommendation: Option B (JSON + Jinja2 macro)**
- Create `methodology_notes.json` with one entry per plot
- Create a Jinja2 macro `{% macro methodology_note(plot_name) %}` that reads from the data
- Pass methodology data to templates via Flask's `render_template()` context
- Keeps template files clean; methodology content is editable without touching HTML structure
- SPARQL queries are stored as strings in JSON (properly escaped)

### SPARQL Timeout Policy Recommendation

**Current settings:**
- `SPARQL_TIMEOUT = 30` (seconds, from Config)
- `PLOT_TIMEOUT = 60` (seconds, for entire plot function execution)

**Query complexity tiers observed in codebase:**
| Tier | Query Pattern | Examples | Typical Duration |
|------|--------------|----------|-----------------|
| Fast | Single entity count per graph | entity counts, author counts | 1-5s |
| Medium | Multi-entity join per graph | KE components, bio processes | 5-15s |
| Slow | Cross-entity network + property counts | completeness boxplot, OECD status | 30-75s |

**Recommendation:**
- Keep default `SPARQL_TIMEOUT=30` for most queries
- Add per-query timeout override capability: `run_sparql_query_with_retry(query, timeout=60)` for known-slow queries
- Set `PLOT_TIMEOUT=120` (up from 60) because the completeness boxplot legitimately takes >60s during startup
- Add `SPARQL_SLOW_TIMEOUT` env var (default: 60) for complex queries

### Virtuoso-Side Tuning Recommendations (Documentation Only)

**User decision:** Document tuning recommendations but do NOT apply them in Docker config.

**Recommended Virtuoso settings for this workload:**
```ini
; virtuoso.ini tuning for AOP-Wiki dashboard queries
[SPARQL]
MaxQueryExecutionTime = 120      ; seconds (default: 0 = unlimited, but cloud hosts often set 30)
MaxQueryCostEstimationTime = 30  ; seconds for query plan estimation
ResultSetMaxRows = 1000000       ; allow large result sets for completeness queries

[Parameters]
NumberOfBuffers = 340000         ; ~2.5GB with 8KB page size
MaxDirtyBuffers = 250000         ; ~1.9GB
MaxCheckpointRemap = 2000

[HTTPServer]
MaxClientConnections = 10        ; dashboard uses 5 parallel workers
ServerThreads = 10
```

**Key insight:** The Virtuoso instance the dashboard queries is an external service (configured via `SPARQL_ENDPOINT` env var). The dashboard Docker container does not run Virtuoso. These tuning recommendations are for the operator of the SPARQL endpoint.

### Reliability Audit: AOP Lifetime Plot

**Current state:** `plot_aop_lifetime()` (trends_plots.py:528-590) returns `tuple[str, str, str]` -- three HTML strings for created histogram, modified histogram, and scatter plot.

**Known issues from Phase 1 UAT:**
- Plot was flagged as "broken" during previous testing
- Phase 1 (01-04) added return type annotation `-> tuple[str, str, str]` to fix fallback generation
- The plot function itself has no error handling around empty DataFrames
- If no AOPs have `dcterms:created` or `dcterms:modified`, the entire function crashes

**Reliability fix needed:**
- Add try/except wrapper with `create_fallback_plot()` for each of the 3 sub-plots
- Handle empty DataFrame from SPARQL query
- Handle `pd.to_datetime` failures on malformed date strings
- Cache data for CSV export (currently missing)

### Reliability Audit: KE Component Trends

**Current state:** `plot_ke_components()` (trends_plots.py:593-696) works correctly but:
- No data cache for CSV export
- No error handling for empty SPARQL results (will crash on empty DataFrame)

**Fix needed:** Add data cache + error handling.

### Plot Inventory: All Displayed Plots

**Latest page (10 containers):**
1. `latest_entity_counts` -- export: COMPLETE
2. `latest_aop_connectivity` -- export: COMPLETE
3. `latest_avg_per_aop` -- export: COMPLETE
4. `latest_ke_components` -- export: COMPLETE
5. `latest_ke_annotation_depth` -- export: COMPLETE
6. `latest_aop_completeness` -- export: COMPLETE
7. `latest_aop_completeness_by_status` -- export: COMPLETE
8. `latest_ke_completeness_by_status` -- export: COMPLETE
9. `latest_ker_completeness_by_status` -- export: COMPLETE
10. `latest_process_usage` -- export: COMPLETE
11. `latest_object_usage` -- export: COMPLETE

**Trends page (33 sub-plots across 16 boxes):**
1. `aop_entity_counts_absolute` -- data: YES, figure: YES
2. `aop_entity_counts_delta` -- data: YES, figure: YES
3. `aop_property_presence_absolute` -- data: YES, figure: YES
4. `aop_property_presence_percentage` -- data: YES, figure: YES
5. `ke_property_presence_absolute` -- data: YES, figure: YES
6. `ke_property_presence_percentage` -- data: YES, figure: YES
7. `ker_property_presence_absolute` -- data: YES, figure: YES
8. `ker_property_presence_percentage` -- data: YES, figure: YES
9. `stressor_property_presence_absolute` -- data: YES, figure: YES
10. `stressor_property_presence_percentage` -- data: YES, figure: YES
11. `entity_completeness_trends` -- data: YES, figure: YES
12. `aop_completeness_boxplot` -- data: YES, figure: YES
13. `average_components_per_aop_absolute` -- data: NO, figure: YES
14. `average_components_per_aop_delta` -- data: NO, figure: YES
15. `aop_network_density` -- data: NO, figure: YES
16. `aop_authors_absolute` -- data: NO, figure: YES
17. `aop_authors_delta` -- data: NO, figure: YES
18. `aops_created_over_time` -- data: NO, figure: YES
19. `aops_modified_over_time` -- data: NO, figure: YES
20. `aop_creation_vs_modification_timeline` -- data: NO, figure: YES
21. `ke_component_annotations_absolute` -- data: NO, figure: YES
22. `ke_component_annotations_delta` -- data: NO, figure: YES
23. `ke_components_percentage_absolute` -- data: NO, figure: YES
24. `ke_components_percentage_delta` -- data: NO, figure: YES
25. `unique_ke_components_absolute` -- data: NO, figure: YES
26. `unique_ke_components_delta` -- data: NO, figure: YES
27. `biological_process_annotations_absolute` -- data: NO, figure: YES
28. `biological_process_annotations_delta` -- data: NO, figure: YES
29. `biological_object_annotations_absolute` -- data: NO, figure: YES
30. `biological_object_annotations_delta` -- data: NO, figure: YES
31. `kes_by_kec_count_absolute` -- data: NO, figure: YES
32. `kes_by_kec_count_delta` -- data: NO, figure: YES

**Gap: 20 trend sub-plots missing data cache (items 13-32).**

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single monolithic plots.py (4,194 LOC) | Modular plots/ package (3,407 LOC) | Phase 1 (2026-02-20) | Clean imports, maintainable |
| Unbounded dict cache | VersionedPlotCache with TTL + LRU + pinning | Phase 1 (2026-02-20) | Bounded memory |
| Flask dev server | Gunicorn with gthread workers | Phase 1 (2026-02-20) | Production-grade serving |
| Boxplot per-AOP per-status across all versions | Grouped bar chart with mean per status | Phase 2 (planned) | Avoids Virtuoso limits |
| On-demand completeness boxplot | Pre-computed at startup | Phase 2 (planned) | 0s user-facing load |

## Open Questions

1. **OECD Status Historical Trend: Aggregate in SPARQL or Python?**
   - What we know: The latest-data version aggregates in SPARQL (works well). The historical version needs the same across all versions.
   - What's unclear: Whether a single cross-version aggregated SPARQL query will be fast enough, or if we need to loop over versions and query each one.
   - Recommendation: Start with a single cross-version query. If it hits Virtuoso limits, fall back to per-version batched queries (like the existing completeness boxplot pattern).

2. **Methodology Notes: How to handle SPARQL query text for dynamically-generated queries?**
   - What we know: Some queries are static strings in the code. Others are dynamically generated with f-strings based on property_labels.csv content.
   - What's unclear: For dynamic queries, do we show the template with placeholders, or capture the actual generated query at runtime?
   - Recommendation: Show representative static versions of queries (with `{property_list}` placeholders where dynamic). The exact query varies by version and property config -- showing a template is more honest than a snapshot.

3. **Per-Sub-Plot Export Buttons: Where to place for trend plots?**
   - What we know: Trend plots use abs/delta toggle. Each variant is a separate sub-plot. User wants per-sub-plot export buttons (not combined).
   - What's unclear: Should download buttons be inside the `.abs` and `.delta` divs (hidden/shown with toggle), or always visible with clear labels?
   - Recommendation: Place download dropdowns inside the `.abs` and `.delta` divs respectively, so they toggle with the content. Each shows its own download URLs.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection of all files listed above (architecture, patterns, gaps identified from actual code)
- Phase 1 summary documents and UAT results

### Secondary (MEDIUM confidence)
- Plotly documentation for `to_image()` with kaleido engine
- HTML `<details>` element specification (native browser support)

### Tertiary (LOW confidence)
- Virtuoso tuning parameters (based on general Virtuoso documentation, not verified against specific deployment version)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in codebase, only kaleido addition needed
- Architecture: HIGH - all patterns already established, this is gap closure work
- Pitfalls: HIGH - identified from direct code inspection, not speculation
- SPARQL optimization: MEDIUM - pre-computation strategy is certain; exact query timing for OECD trend is estimated

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (stable codebase, no fast-moving dependencies)
