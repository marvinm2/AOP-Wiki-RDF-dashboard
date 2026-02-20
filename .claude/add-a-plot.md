# Adding a New Plot

Step-by-step checklist for adding a new visualization to the AOP-Wiki RDF Dashboard. Follow the section that matches your plot type. Every step includes the exact file path and code pattern to use.

Before you start, decide which type of plot you are adding:

- **Latest Data Plot** (Section A) -- Shows a snapshot of the database for a single selected version. Users can switch versions via the version selector.
- **Historical Trends Plot** (Section B) -- Shows how a metric changes across all RDF graph versions over time. Displayed on the trends page.

---

## Section A: Adding a Latest Data Plot

These plots live on the "Latest Data" page and respond to the version selector dropdown.

### A1. Implement the plot function

**File:** `plots/latest_plots.py`

Create your function with this signature and structure:

```python
def plot_latest_<name>(version: str = None) -> str:
    """One-line description of what this plot shows.

    Args:
        version: Optional version string (e.g., "2024-10-01"). If None, uses latest.

    Returns:
        str: Plotly HTML string for embedding.
    """
    where_filter, order_limit = _build_graph_filter(version)

    query = f"""
        SELECT ...
        WHERE {{
            GRAPH ?graph {{
                ...
            }}
            {where_filter}
        }}
        {order_limit}
    """

    results = run_sparql_query(query)
    df = pd.DataFrame(results)

    # Cache for CSV export
    version_key = version or "latest"
    _plot_data_cache[f'latest_<name>_{version_key}'] = df

    fig = px.bar(df, ...)  # or px.line, px.scatter, etc.

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color=BRAND_COLORS['primary_dark'],
    )

    # Cache figure for image export
    _plot_figure_cache[f'latest_<name>_{version_key}'] = fig

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
```

Key imports already available at the top of the file:
- `BRAND_COLORS`, `config`, `_plot_data_cache`, `_plot_figure_cache` from `plots.shared`
- `run_sparql_query` from `plots.shared`
- `pd` (pandas), `px` (plotly.express), `pio` (plotly.io)

Use `_build_graph_filter(version)` to generate the SPARQL WHERE filter and ORDER/LIMIT clauses. This handles both "specific version" and "latest version" modes.

### A2. Export from the package

**File:** `plots/__init__.py`

1. Add your function to the `from .latest_plots import (...)` block:
   ```python
   from .latest_plots import (
       ...
       plot_latest_<name>,
   )
   ```

2. Add to the `__all__` list under the "Current snapshot functions" comment:
   ```python
   'plot_latest_<name>',
   ```

3. Add to the `get_available_functions()` dictionary under `'current_snapshots'`:
   ```python
   'plot_latest_<name>',
   ```

### A3. Add lazy-loading API route

**File:** `app.py`

Find the `plot_map` dictionary inside the `/api/plot/<plot_name>` route handler (around line 1412). This dictionary is only for historical trend plots. Latest plots are handled separately in the same endpoint via the `latest_plots_precomputed` dictionary.

For latest plots, add an entry to `latest_plots_precomputed` inside the `compute_latest_plots()` function, or register the plot so the lazy-loading endpoint can call it on demand. The pattern for latest plots that support versioning:

The `/api/plot/<plot_name>` endpoint already handles latest plots by detecting the `latest_` prefix. It calls the function from the imported module. Check how existing latest plots are served in that endpoint and follow the same pattern.

### A4. Add CSV download route

**File:** `app.py`

Add a route for CSV download. Follow this pattern (search for existing `/download/latest_*` routes as reference):

```python
@app.route("/download/latest_<name>")
def download_latest_<name>():
    """Download latest <name> data as CSV."""
    version = request.args.get('version')
    include_metadata = request.args.get('metadata', 'true').lower() == 'true'
    version_key = version or "latest"
    cache_key = f'latest_<name>_{version_key}'

    return get_csv_with_metadata(cache_key, include_metadata=include_metadata)
```

### A5. Add HTML container

**File:** `templates/latest.html`

Add a plot container inside the appropriate section. Follow the existing pattern:

```html
<div class="plot-container">
    <h3>Plot Title</h3>
    <div class="lazy-plot" data-plot-name="latest_<name>">
        <div class="skeleton-loader">
            <div class="skeleton-chart"></div>
        </div>
    </div>
    <a href="/download/latest_<name>" class="download-btn">Download CSV</a>
</div>
```

The `data-plot-name` attribute must match the key used by the lazy-loading JavaScript.

### A6. Register for version selector

**File:** `static/js/version-selector.js`

Add your plot name to the `versionedPlots` array (around line 16):

```javascript
const versionedPlots = [
    'latest_entity_counts',
    ...
    'latest_<name>',  // <-- add here
];
```

This ensures the plot reloads when users select a different version.

### A7. Verify

- [ ] Plot renders with sample data
- [ ] Version selector triggers plot reload
- [ ] CSV download returns data with metadata
- [ ] Brand colors applied (no default Plotly colors)

---

## Section B: Adding a Historical Trends Plot

These plots live on the "Trends" page and show data across all RDF graph versions.

### B1. Implement the plot function

**File:** `plots/trends_plots.py`

Choose the appropriate return type based on what the plot shows:

**For standard metric trends** (absolute count + delta change):
```python
def plot_<name>() -> tuple[str, str, pd.DataFrame]:
    """One-line description.

    Returns:
        tuple: (absolute_html, delta_html, dataframe)
    """
    query = """
        SELECT ?graph (COUNT(...) AS ?count)
        WHERE {
            GRAPH ?graph { ... }
        }
        GROUP BY ?graph
        ORDER BY ?graph
    """

    results = run_sparql_query(query)
    df = pd.DataFrame(results)

    # Cache for CSV export
    _plot_data_cache['<name>_absolute'] = df
    _plot_data_cache['<name>_delta'] = delta_df  # computed delta

    fig_abs = px.line(df, ...)
    fig_delta = px.bar(delta_df, ...)

    # Apply brand styling to both
    for fig in [fig_abs, fig_delta]:
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color=BRAND_COLORS['primary_dark'],
        )

    _plot_figure_cache['<name>_absolute'] = fig_abs
    _plot_figure_cache['<name>_delta'] = fig_delta

    abs_html = pio.to_html(fig_abs, full_html=False, include_plotlyjs='cdn')
    delta_html = pio.to_html(fig_delta, full_html=False, include_plotlyjs='cdn')

    return abs_html, delta_html, df
```

**For property presence trends** (absolute count + percentage):
```python
def plot_<name>_property_presence() -> tuple[str, str]:
    """One-line description.

    Returns:
        tuple: (absolute_html, percentage_html)
    """
    # ... build query, compute data ...

    _plot_data_cache['<name>_property_presence_absolute'] = df
    _plot_data_cache['<name>_property_presence_percentage'] = pct_df

    return abs_html, pct_html
```

Key imports already available at the top of the file:
- `BRAND_COLORS`, `config`, `_plot_data_cache`, `_plot_figure_cache` from `plots.shared`
- `run_sparql_query`, `extract_counts`, `safe_read_csv`, `create_fallback_plot`, `get_properties_for_entity` from `plots.shared`
- `pd`, `px`, `pio`, `logging`, `reduce` (from functools)

### B2. Export from the package

**File:** `plots/__init__.py`

1. Add to `from .trends_plots import (...)`:
   ```python
   plot_<name>,
   ```

2. Add to `__all__` under "Historical trends functions":
   ```python
   'plot_<name>',
   ```

3. Add to `get_available_functions()` under `'historical_trends'`:
   ```python
   'plot_<name>',
   ```

### B3. Add to startup computation

**File:** `app.py`

Find the `plot_tasks` list inside `compute_plots_parallel()` (around line 164). Add your task:

```python
plot_tasks = [
    ...
    ('<name>', lambda: safe_plot_execution(plot_<name>)),
]
```

### B4. Unpack results

**File:** `app.py`

After `compute_plots_parallel()` returns, results are unpacked into global variables. Find the unpacking section (around line 230+) and add:

**For 3-tuple (absolute + delta + data):**
```python
graph_<name>_abs, graph_<name>_delta, _ = plot_results.get('<name>', ("", "", None))
```

**For 2-tuple (absolute + percentage):**
```python
graph_<name>_abs, graph_<name>_pct = plot_results.get('<name>', ("", ""))
```

### B5. Add to plot_map for lazy loading

**File:** `app.py`

Find the `plot_map` dictionary inside the `/api/plot/<plot_name>` endpoint (around line 1412). Add entries for both variants:

**For 3-tuple:**
```python
'<name>_absolute': graph_<name>_abs,
'<name>_delta': graph_<name>_delta,
```

**For 2-tuple:**
```python
'<name>_absolute': graph_<name>_abs,
'<name>_percentage': graph_<name>_pct,
```

### B6. Add download routes

**File:** `app.py`

Add CSV/image download routes. For trends, the generic route `/download/trend/<plot_name>` (around line 1140) handles most cases automatically if the cache key matches. If you need explicit routes:

```python
@app.route("/download/trend/<name>_absolute")
def download_<name>_absolute():
    """Download <name> absolute trend data."""
    return get_csv_with_metadata('<name>_absolute', include_metadata=True)

@app.route("/download/trend/<name>_delta")
def download_<name>_delta():
    """Download <name> delta trend data."""
    return get_csv_with_metadata('<name>_delta', include_metadata=True)
```

### B7. Add HTML containers

**File:** `templates/trends.html`

Add a section with side-by-side plot containers. Follow the existing pattern:

```html
<!-- <Name> Section -->
<div class="plot-row">
    <div class="plot-container half-width">
        <h3>Plot Title (Absolute)</h3>
        <div class="lazy-plot" data-plot-name="<name>_absolute">
            <div class="skeleton-loader">
                <div class="skeleton-chart"></div>
            </div>
        </div>
        <a href="/download/trend/<name>_absolute" class="download-btn">Download CSV</a>
    </div>
    <div class="plot-container half-width">
        <h3>Plot Title (Delta / Percentage)</h3>
        <div class="lazy-plot" data-plot-name="<name>_delta">
            <div class="skeleton-loader">
                <div class="skeleton-chart"></div>
            </div>
        </div>
        <a href="/download/trend/<name>_delta" class="download-btn">Download CSV</a>
    </div>
</div>
```

### B8. Verify

- [ ] Both plot variants render on the trends page
- [ ] Lazy loading works (plots load on scroll)
- [ ] CSV download returns data for both variants
- [ ] Brand colors applied consistently

---

## Section C: Naming Conventions

| Element             | Latest Data Pattern                  | Historical Trends Pattern                            |
|---------------------|--------------------------------------|------------------------------------------------------|
| Function name       | `plot_latest_<name>(version=None)`   | `plot_<name>()`                                      |
| Cache key (data)    | `latest_<name>_<version>`            | `<name>_absolute`, `<name>_delta`                    |
| Cache key (figure)  | `latest_<name>_<version>`            | `<name>_absolute`, `<name>_delta`                    |
| Download route      | `/download/latest_<name>`            | `/download/trend/<name>_absolute`                    |
| Template div        | `data-plot-name="latest_<name>"`     | `data-plot-name="<name>_absolute"`                   |
| Version selector    | Added to `versionedPlots` array      | Not applicable (shows all versions by design)        |
| Startup computation | Not in `plot_tasks` (on-demand)      | Added to `plot_tasks` in `compute_plots_parallel()`  |

---

## Section D: Common Patterns

### SPARQL Queries

All queries use `run_sparql_query()` or `run_sparql_query_with_retry()` from `plots/shared.py`. The retry variant adds exponential backoff for unreliable endpoints.

```python
from plots.shared import run_sparql_query_with_retry

results = run_sparql_query_with_retry(query, retries=3)
```

### Error Handling

Wrap plot functions with `safe_plot_execution()` for graceful degradation:

```python
from plots import safe_plot_execution

result = safe_plot_execution(plot_latest_<name>, version)
```

If the function raises an exception, `safe_plot_execution` returns a fallback error plot instead of crashing the dashboard.

For creating your own fallback within a function:

```python
from plots.shared import create_fallback_plot

if df.empty:
    return create_fallback_plot("No data available for <name>")
```

### Brand Colors

Import from `plots/shared.py`:

```python
from plots.shared import BRAND_COLORS
```

The `BRAND_COLORS` dictionary contains:
- `'primary_dark'`: `#29235C` -- headers, titles, axis labels
- `'primary_magenta'`: `#E6007E` -- CTAs, highlights
- `'primary_blue'`: `#307BBF` -- navigation, links
- Plus secondary colors. See `.claude/colors.md` for the full palette.

### Plotly Layout Defaults

Apply these layout settings to every figure for visual consistency:

```python
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',      # transparent plot area
    paper_bgcolor='rgba(0,0,0,0)',     # transparent paper
    title_font_color=BRAND_COLORS['primary_dark'],
    font=dict(color=BRAND_COLORS['primary_dark']),
    hovermode='x unified',            # unified hover tooltip
    margin=dict(l=60, r=30, t=60, b=60),
)
```

### CSV Export Cache

Every plot should cache its DataFrame so users can download the underlying data:

```python
_plot_data_cache['your_cache_key'] = df
```

The `get_csv_with_metadata()` function retrieves from this cache and attaches version/timestamp metadata to the CSV output.

---

## Quick Reference: Files to Touch

| Step | Latest Data         | Historical Trends     |
|------|---------------------|-----------------------|
| 1    | `plots/latest_plots.py`       | `plots/trends_plots.py`        |
| 2    | `plots/__init__.py`           | `plots/__init__.py`            |
| 3    | `app.py` (API endpoint)       | `app.py` (`plot_tasks` list)   |
| 4    | `app.py` (download route)     | `app.py` (unpack results)      |
| 5    | `templates/latest.html`       | `app.py` (`plot_map` dict)     |
| 6    | `static/js/version-selector.js` | `app.py` (download routes)   |
| 7    | --                            | `templates/trends.html`        |
