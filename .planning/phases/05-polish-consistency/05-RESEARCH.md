# Phase 5: Polish & Consistency - Research

**Researched:** 2026-02-23
**Domain:** Visual identity, UX polish, Plotly theming, Flask template architecture
**Confidence:** HIGH

## Summary

Phase 5 transforms the AOP-Wiki RDF Dashboard from a functional data tool into a professionally branded, visually coherent product. The codebase currently has ~50 Plotly plots spread across 3,400+ LOC in two modules, 6 HTML templates with no shared base template (no Jinja2 `extends`/`block` pattern), and hardcoded colors throughout CSS and Python. There is no CSS variable system, no Plotly custom template, and no VHP4Safety logo file in the static assets.

The primary technical strategy is a three-layer color system: (1) a single Python config dict as the source of truth, (2) CSS variables generated or manually synced from that dict, and (3) a Plotly custom template registered via `pio.templates` that applies brand defaults to every figure automatically. This eliminates the need to touch every individual `update_layout()` call -- instead, the template provides defaults that individual plots can override only when needed.

The second major change is introducing Jinja2 template inheritance with a `base.html` that contains the redesigned header, navigation bar (with version selector), and footer. All 6 page templates inherit from this base, eliminating the current copy-paste pattern for headers, footers, and navigation. The landing page gets a redesign as a navigation hub with icon+description cards.

**Primary recommendation:** Create a Plotly custom template (`vhp4safety`) registered as default via `pio.templates.default`, introduce a `base.html` with Jinja2 template inheritance, and implement CSS custom properties (`--var`) for all brand colors. This approach minimizes per-plot changes while maximizing consistency.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### House style & colors
- VHP4Safety brand colors for page chrome (header, footer, nav) -- extended palette for chart data series
- Central color system with both layers: CSS variables for page chrome + Python config for Plotly charts, kept in sync from one source of truth
- Professional look aligned with VHP4Safety branding, no specific external reference

#### Landing page
- Navigation hub as primary purpose -- guide users to the right section
- Icon + description cards for each section (Latest Data, Trends, Network)
- Light live data: show latest version number and headline entity counts (minimal SPARQL)
- Expandable intro explaining AOP-Wiki -- collapsed by default for regulars, expandable for newcomers
- About section with project info accessible from landing (same page or separate -- Claude's discretion)
- SPARQL endpoint link only (no example queries)
- VHP4Safety logo + funding/project acknowledgment

#### Plot uniformity -- full visual audit
- Unify colors, typography, labels, legends, margins, and sizing across ALL existing plots
- White background on all plots (clean, printable, consistent)
- Plotly toolbar hidden by default, appears on hover
- Legends on right side consistently
- Standardized hover tooltips (Claude determines format per plot type)
- Grid lines and axes standardized (Claude determines what aids readability)
- Plot heights determined by Claude (prevent clipping, look good)

#### Page layout & navigation
- Add About page to navigation alongside existing Latest Data, Trends, Network
- Header and footer redesigned (not just refined) to match house style
- Version selector moved to navigation bar -- always accessible
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

### Deferred Ideas (OUT OF SCOPE)
- Advanced Analytics (stressor-AOP mapping, data quality scoring, ontology coverage) -- deferred to v2 backlog
- Documentation/help page with SPARQL examples and data dictionary -- potential future addition
- Bulk data downloads page -- potential future addition
- Full mobile responsiveness -- not needed for target audience
</user_constraints>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Flask | ~3.1 | Web framework, Jinja2 template engine | Already used; template inheritance is built in |
| Plotly | ~5.22 | Interactive visualizations | Already used; has built-in custom template system |
| Kaleido | ~0.2 | Static image export | Already used for PNG/SVG export |

### Supporting (no new dependencies needed)
| Library | Purpose | When to Use |
|---------|---------|-------------|
| Jinja2 (via Flask) | Template inheritance, macros | Base template, shared nav/footer |
| CSS Custom Properties | CSS variable system | Page chrome colors, consistent theming |
| `plotly.graph_objects.layout.Template` | Plotly custom template | Register `vhp4safety` theme as default |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| CSS Custom Properties | SASS/LESS variables | SASS adds build step; CSS vars work natively, no tooling needed |
| Plotly custom template | Per-plot `update_layout()` | Custom template is set-once; per-plot is 50+ edits with ongoing maintenance |
| Jinja2 extends/block | Jinja2 include (current) | extends gives proper inheritance + block overrides; includes are just paste-in |
| Separate About page | About section on landing | Separate route is cleaner for navigation but adds a template; recommended for consistency |

**Installation:** No new packages needed. All capabilities exist in current dependencies.

## Architecture Patterns

### Recommended Project Structure Changes
```
static/
  css/
    main.css              # Refactored with CSS custom properties
    lazy-loading.css      # Unchanged
    network.css           # Unchanged
  images/
    vhp4safety-logo.png   # NEW: VHP4Safety logo asset
  js/
    lazy-loading.js       # Unchanged
    version-selector.js   # MODIFIED: version selector now in nav bar
    raw-data-tables.js    # Unchanged
    network-graph.js      # Unchanged

templates/
  base.html               # NEW: Base template with nav, header, footer
  landing.html            # MODIFIED: extends base.html, redesigned
  latest.html             # MODIFIED: extends base.html
  trends_page.html        # MODIFIED: extends base.html
  network.html            # MODIFIED: extends base.html
  about.html              # NEW: About page (recommended)
  status.html             # MODIFIED: extends base.html
  index.html              # MODIFIED: extends base.html (legacy)
  macros/                 # Unchanged
  trends.html             # Unchanged (included partial)

plots/
  shared.py               # MODIFIED: add PLOT_DEFAULTS dict, register custom template
  latest_plots.py         # MODIFIED: use custom template, standardize layouts
  trends_plots.py         # MODIFIED: use custom template, standardize layouts
  __init__.py             # Minor: export new constants

config.py                 # MODIFIED: add color config section
```

### Pattern 1: Plotly Custom Template (Source of Truth for Chart Styling)
**What:** Register a custom Plotly template that sets all shared layout defaults
**When to use:** At module import time in `plots/shared.py`
**Example:**
```python
# Source: https://plotly.com/python/templates/
import plotly.graph_objects as go
import plotly.io as pio

# Define the VHP4Safety brand template
vhp4safety_template = go.layout.Template(
    layout=go.Layout(
        colorway=BRAND_COLORS['palette'],
        font=dict(family="Arial, sans-serif", size=13, color="#29235C"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
        ),
        xaxis=dict(
            gridcolor="#e0e0e0",
            gridwidth=1,
            linecolor="#cccccc",
            linewidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#e0e0e0",
            gridwidth=1,
            linecolor="#cccccc",
            linewidth=1,
            zeroline=False,
        ),
    )
)

pio.templates["vhp4safety"] = vhp4safety_template
pio.templates.default = "plotly_white+vhp4safety"
```

### Pattern 2: CSS Custom Properties for Page Chrome
**What:** Define brand colors as CSS variables in `:root`, use throughout stylesheets
**When to use:** In `main.css`, referenced by all page elements
**Example:**
```css
:root {
    /* VHP4Safety Brand Colors -- synced with plots/shared.py BRAND_COLORS */
    --color-primary: #29235C;
    --color-magenta: #E6007E;
    --color-blue: #307BBF;
    --color-light-blue: #009FE3;
    --color-orange: #EB5B25;
    --color-sky-blue: #93D5F6;
    --color-deep-magenta: #9A1C57;
    --color-teal: #45A6B2;
    --color-dark-teal: #005A6C;
    --color-violet: #64358C;

    /* Semantic aliases */
    --color-bg: #f5f5f5;
    --color-surface: white;
    --color-text: #29235C;
    --color-text-secondary: #555;
    --color-border: #e0e0e0;
    --color-nav-bg: #29235C;
    --color-nav-text: white;
    --color-footer-bg: #29235C;
    --color-cta: #E6007E;

    /* Typography */
    --font-family: 'Arial', 'Helvetica Neue', sans-serif;
    --font-size-base: 16px;

    /* Spacing */
    --nav-height: 64px;
}

body {
    font-family: var(--font-family);
    color: var(--color-text);
    background-color: var(--color-bg);
}

header {
    background-color: var(--color-nav-bg);
    color: var(--color-nav-text);
}
```

### Pattern 3: Jinja2 Template Inheritance
**What:** Base template with blocks for head, nav, content, footer; child templates extend it
**When to use:** Every page template
**Example:**
```html
{# templates/base.html #}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}AOP-Wiki RDF Dashboard{% endblock %}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/lazy-loading.css') }}">
    {% block head_extra %}{% endblock %}
</head>
<body>
    <nav class="main-nav">
        {# Redesigned navigation with version selector #}
        {% block navigation %}
        <div class="nav-brand">
            <a href="/">AOP-Wiki RDF Dashboard</a>
        </div>
        <div class="nav-links">
            <a href="/snapshot" class="nav-link {% if active_page == 'snapshot' %}active{% endif %}">
                Database Snapshot
            </a>
            <a href="/trends" class="nav-link {% if active_page == 'trends' %}active{% endif %}">
                Historical Trends
            </a>
            <a href="/network" class="nav-link {% if active_page == 'network' %}active{% endif %}">
                Network Analysis
            </a>
            <a href="/about" class="nav-link {% if active_page == 'about' %}active{% endif %}">
                About
            </a>
        </div>
        <div class="nav-version">
            {% block version_selector %}{% endblock %}
        </div>
        {% endblock %}
    </nav>

    <main>
        {% block content %}{% endblock %}
    </main>

    <footer>
        {% block footer %}
        <div class="footer-content">
            <div class="footer-branding">
                <img src="{{ url_for('static', filename='images/vhp4safety-logo.png') }}"
                     alt="VHP4Safety" class="footer-logo">
                <span>Built for VHP4Safety</span>
            </div>
            <div class="footer-links">
                <a href="https://aopwiki.cloud.vhp4safety.nl" target="_blank">
                    SPARQL Endpoint
                </a>
                <a href="https://github.com/marvinm2/AOP-Wiki-RDF-dashboard/issues"
                   target="_blank">Report an Issue</a>
            </div>
            <div class="footer-funding">
                Funded by NWO (VHP4Safety project)
            </div>
        </div>
        {% endblock %}
    </footer>

    <script src="{{ url_for('static', filename='js/lazy-loading.js') }}"></script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
```

### Pattern 4: Standardized `to_html` Config
**What:** Shared config dict passed to every `pio.to_html()` call
**When to use:** Every plot rendering call
**Example:**
```python
# In shared.py
PLOTLY_HTML_CONFIG = {
    "responsive": True,
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "aopwiki_plot",
        "height": 500,
        "width": 800,
        "scale": 4
    }
}

def render_plot_html(fig, include_plotlyjs=False):
    """Standardized HTML rendering for all plots."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config=PLOTLY_HTML_CONFIG
    )
```

### Anti-Patterns to Avoid
- **Inline styles in templates:** The current codebase has many `style="..."` attributes on HTML elements. These should be moved to CSS classes. However, migrating all of them in this phase is a large effort -- focus on structural elements (header, footer, nav, section headers) and leave plot-internal inline styles for a later cleanup.
- **Copy-paste headers/footers:** The current approach copies the same header/footer/nav HTML into every template. Template inheritance eliminates this.
- **Per-plot color overrides:** With a custom Plotly template, individual plots should not need to specify `colorway` or `template` unless they truly need a deviation.
- **Mixed `include_plotlyjs` patterns:** Currently some plots use `"cdn"`, some use `False`. With Plotly loaded in the `<head>` tag, all should use `False` except the first loaded on the page (or just always use `False` since it is loaded in `<head>`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Chart theming | Per-plot color/font/margin code | `pio.templates` custom template | Set once, applies everywhere; 50+ plots would need individual updates otherwise |
| CSS color system | Hardcoded hex values scattered across CSS | CSS Custom Properties (`:root` vars) | Single source, IDE autocomplete, easy theme switching |
| Template shared elements | Copy-paste header/footer/nav | Jinja2 `extends`/`block` | Flask's built-in feature; changes propagate automatically |
| Plotly toolbar behavior | Per-plot config dicts | Shared `PLOTLY_HTML_CONFIG` dict | One config, consistent behavior |
| Responsive navigation | Custom JS hamburger menu | CSS flexbox + media queries | CSS-only solution is simpler and more reliable for tablet support |

**Key insight:** The codebase has ~50 plots each with their own `update_layout()` calls. A custom Plotly template eliminates the need to touch most of them -- the template provides the defaults, and individual plots only override what is truly unique (like specific margin needs for horizontal bar charts with long labels).

## Common Pitfalls

### Pitfall 1: Plotly Template Override Order
**What goes wrong:** Custom template properties get overridden by per-plot `update_layout()` calls that set the same properties
**Why it happens:** Plotly applies templates first, then layout updates on top. If existing code explicitly sets `template="plotly_white"` in `update_layout()`, it overrides the default template.
**How to avoid:** Remove all explicit `template="plotly_white"` from individual `update_layout()` calls. The default template (`plotly_white+vhp4safety`) handles this. Search for `template=` in both `latest_plots.py` and `trends_plots.py`.
**Warning signs:** Plots that look different from the standard after migration -- check that explicit template overrides were removed.

### Pitfall 2: include_plotlyjs Inconsistency
**What goes wrong:** Plotly JS gets loaded multiple times on a page, or not loaded at all
**Why it happens:** Currently some plots use `include_plotlyjs="cdn"` and others use `False`. The CDN script tag is already in the HTML `<head>`.
**How to avoid:** Standardize all `to_html()` calls to use `include_plotlyjs=False` since the CDN script is loaded in `base.html`. Use the shared `render_plot_html()` helper.
**Warning signs:** Console errors about `Plotly` being undefined, or very large HTML payloads containing embedded Plotly JS.

### Pitfall 3: Version Selector in Nav Bar -- Scope Issue
**What goes wrong:** Version selector appears on pages where it should not (trends, network, about)
**Why it happens:** Moving the version selector to the global nav bar makes it visible on all pages, but it only applies to the `/snapshot` (latest data) page.
**How to avoid:** Use a Jinja2 block override: `base.html` defines an empty `{% block version_selector %}` that only `latest.html` fills. Or use conditional rendering based on `active_page`.
**Warning signs:** Version selector appears on trends page with no effect, confusing users.

### Pitfall 4: Legend Position Breaking Pie/Donut Charts
**What goes wrong:** Default right-side legend placement works for line/bar charts but clips or overlaps for pie charts
**Why it happens:** Pie charts have the legend inside the plot area by default
**How to avoid:** The custom template sets `legend.orientation="v"` and right-side positioning as default, but pie chart plots override to `showlegend=False` (with labels on slices) or use bottom horizontal legends. This is an acceptable per-plot override.
**Warning signs:** Pie chart legends overlapping the chart area.

### Pitfall 5: Margin Overrides for Long Labels
**What goes wrong:** Horizontal bar charts with long y-axis labels get clipped
**Why it happens:** The custom template sets a standard margin, but some plots need wider left margins for long labels (e.g., KE reuse plot with `margin=dict(l=300)`)
**How to avoid:** The custom template sets reasonable defaults (`l=60`), and individual plots with long labels explicitly override just the left margin. Do not try to make one margin work for all.
**Warning signs:** Truncated labels on horizontal bar charts.

### Pitfall 6: Breaking Existing Functionality During Refactor
**What goes wrong:** Lazy loading, version selector, CSV export, or bulk download breaks
**Why it happens:** Template restructuring changes element IDs or DOM structure that JavaScript depends on
**How to avoid:** Keep all `data-plot-name` attributes, element IDs, and CSS classes that JavaScript targets. Test lazy loading, version switching, and CSV download after every template change.
**Warning signs:** Plots not loading, version selector not working, 404 on download links.

## Code Examples

### Creating and Registering the Custom Plotly Template
```python
# Source: https://plotly.com/python/templates/
import plotly.graph_objects as go
import plotly.io as pio

BRAND_COLORS = {
    'primary': '#29235C',
    'secondary': '#E6007E',
    'accent': '#307BBF',
    'palette': [
        '#29235C', '#E6007E', '#307BBF', '#009FE3', '#EB5B25',
        '#93D5F6', '#9A1C57', '#45A6B2', '#B81178', '#005A6C', '#64358C'
    ],
}

vhp4safety_template = go.layout.Template(
    layout=go.Layout(
        colorway=BRAND_COLORS['palette'],
        font=dict(family="Arial, sans-serif", size=13, color=BRAND_COLORS['primary']),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="#e0e0e0",
            linecolor="#cccccc",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="#e0e0e0",
            linecolor="#cccccc",
            zeroline=False,
            tickfont=dict(size=11),
        ),
    )
)

pio.templates["vhp4safety"] = vhp4safety_template
pio.templates.default = "plotly_white+vhp4safety"
```

### Simplified Plot Function After Template
```python
# BEFORE (current pattern -- every plot sets template, margin, hovermode):
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    autosize=True,
    margin=dict(l=50, r=20, t=50, b=50)
)

# AFTER (with custom template as default -- only set what differs):
# No update_layout needed for standard plots!
# Only override for specific needs:
fig.update_layout(
    margin=dict(l=300)  # Only for horizontal bar charts with long labels
)
```

### Standardized to_html Rendering
```python
PLOTLY_HTML_CONFIG = {
    "responsive": True,
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "aopwiki_plot",
        "height": 500,
        "width": 800,
        "scale": 4,
    }
}

def render_plot_html(fig, include_plotlyjs=False):
    """Render a Plotly figure to HTML with standardized config."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config=PLOTLY_HTML_CONFIG,
    )
```

### Jinja2 Base Template Pattern
```html
{# templates/base.html -- Flask docs: https://flask.palletsprojects.com/en/stable/patterns/templateinheritance/ #}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}AOP-Wiki RDF Dashboard{% endblock %}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
    {% block head_extra %}{% endblock %}
</head>
<body>
    {% include 'partials/nav.html' %}
    <main>{% block content %}{% endblock %}</main>
    {% include 'partials/footer.html' %}
    {% block scripts_extra %}{% endblock %}
</body>
</html>
```

## Discretion Recommendations

Based on research into the codebase structure, plot types, and UX patterns:

### Plot Title Approach: HTML Heading Only (Not Plotly Internal)
**Recommendation:** Remove Plotly `title=` from all plots; use the existing `<h3>` HTML headings in plot containers.
**Rationale:** HTML headings are already present in every plot box (`<div class="plot-header"><h3>...</h3>`). Plotly titles add redundant text, consume vertical space, and are harder to style consistently. Removing them saves ~30px of vertical space per plot and ensures typography matches the page. The current codebase already has HTML titles for every plot.
**Impact:** Remove `title=` parameter from all `px.line()`, `px.bar()`, `px.scatter()`, `px.pie()` calls. This is a search-and-replace operation.

### About Section: Separate `/about` Route
**Recommendation:** Create a dedicated `/about` page rather than embedding in the landing page.
**Rationale:** The landing page should be a clean navigation hub. An about section with project info, funding details, and methodology notes is substantial enough to warrant its own page. This also makes the nav bar cleaner -- "About" is a standard nav item users expect.
**Content:** Project description, VHP4Safety acknowledgment, SPARQL endpoint link, methodology overview, contact/issue link, data sources.

### Navigation Bar: Static (Not Fixed)
**Recommendation:** Use a static navigation bar that scrolls with the page.
**Rationale:** Fixed navbars consume screen real estate, which is valuable on the data-dense snapshot and trends pages. The pages already have in-page navigation buttons for jumping between sections. A static nav at the top is sufficient for a desktop/tablet audience.

### Page Transitions: None
**Recommendation:** No page transition animations.
**Rationale:** This is a data dashboard for researchers and regulators. Transitions add perceived latency. Clean, instant page loads are more professional for this audience.

### Tooltip Format Per Plot Type
**Recommendation:**
- **Line charts (trends):** Use `hovermode="x unified"` (current pattern, works well)
- **Bar charts:** Use `hovermode="closest"` with individual bar hover showing value + percentage where applicable
- **Pie charts:** Use `textinfo='percent+label'` on slices, disable separate hover
- **Box plots:** Use default Plotly hover (shows Q1, median, Q3, whiskers)
- **Scatter plots:** Use `hovermode="closest"` with point-specific data
- **Horizontal bar charts:** Use `hovermode="closest"` showing bar value

### Grid Line Styling
**Recommendation:**
- Light gray grid lines (`#e0e0e0`) on both axes for all chart types
- No zero line (already the case for most plots)
- Grid lines on y-axis always; grid lines on x-axis for line/scatter charts, hidden for bar charts
- No box/frame around the plot area (the `plotly_white` base template handles this)

### Plot Heights
**Recommendation:**
- Standard plots (bar, line, pie): Let autosize handle it (default ~450px in container)
- Full-width trend plots: Default autosize
- Horizontal bar charts with many items: Dynamic height based on item count (`max(400, n_items * 25 + 100)`)
- Box plots: Default autosize
- Scatter plots: Default autosize
- Keep the `autosize=True` from the template; remove explicit `height=` from most plots

### Number of Extended Chart Colors
**Recommendation:** 11 colors (current palette) is sufficient.
**Rationale:** The maximum number of data series across all plots:
- Property presence plots: 8-12 properties per entity type (covered by 11 colors)
- Entity count trends: 4 series (AOPs, KEs, KERs, Stressors)
- Component annotations: 3 series (Process, Object, Action)
- OECD status: 4-5 categories
The existing 11-color `BRAND_COLORS['palette']` covers all cases. No extension needed.

## Current State Audit

### Inconsistencies Found in Codebase

| Issue | Current State | Count | Fix Strategy |
|-------|--------------|-------|-------------|
| `template=` in update_layout | All plots set `template="plotly_white"` explicitly | ~50 calls | Remove; handled by default template |
| `include_plotlyjs` variation | Mix of `"cdn"`, `False` | ~40 calls | Standardize to `False`; CDN in base.html |
| `margin=dict(...)` variation | ~8 different margin configurations | ~50 calls | Custom template default + per-plot overrides only when needed |
| Legend position | Some right, some bottom horizontal, some hidden | ~15 plots with legends | Template default right-side; pie charts override |
| Missing `hovermode` | Some plots have it, some don't | ~20 without | Template default `"x unified"` |
| No CSS variables | All colors hardcoded in CSS | ~30 color references | Migrate to `var()` references |
| Duplicated header/footer | Copy-pasted across 6 templates | 6 files | `base.html` with `extends` |
| No `displayModeBar` config | Toolbar always visible | ~50 plots | Shared config with `"hover"` |
| `plot_bgcolor`/`paper_bgcolor` variation | Some transparent, some not set | ~20 plots | Template default white |
| No VHP4Safety logo file | No image assets in static/images/ | 0 files | Add logo asset |

### Template File Analysis
| Template | Has Header | Has Footer | Has Nav | Uses base.html |
|----------|-----------|-----------|---------|----------------|
| landing.html | Custom hero | Yes | No nav bar | No |
| latest.html | Yes | Yes | Yes | No |
| trends_page.html | Yes | Yes | Yes | No |
| network.html | Yes | Yes | Yes | No |
| index.html | Yes | No (in included) | Tab-based | No |
| status.html | Via inline | Partial | Partial | No |

## Open Questions

1. **VHP4Safety Logo Asset**
   - What we know: The GitHub organization has an avatar image. The project references VHP4Safety branding but has no logo file in the repository.
   - What's unclear: Whether a high-resolution official logo is available, or if the GitHub avatar should be used.
   - Recommendation: Download the VHP4Safety logo from their GitHub organization profile or official sources and add to `static/images/`. If no vector logo is available, use the GitHub avatar as a fallback and note it for future upgrade.

2. **Funding Acknowledgment Text**
   - What we know: The footer says "Built for VHP4Safety". The project is funded by NWO.
   - What's unclear: The exact required funding acknowledgment text.
   - Recommendation: Use "Developed within the VHP4Safety project, funded by NWO" as placeholder text. The user can adjust the exact wording.

3. **Prior Decision Conflict: Legend Position**
   - What we know: Phase 02-06 decided "Centered horizontal legends (xanchor=center) for better visual balance with multi-status legends". Phase 5 context says "Legends on right side consistently".
   - What's unclear: Whether the Phase 5 decision supersedes Phase 02-06 for the specific plots that were changed.
   - Recommendation: Phase 5 decision takes precedence as the latest. Use right-side vertical legends as the default. For plots with many legend items (>6), consider using a right-side scrollable legend or keeping it right-side with smaller font. The property presence plots with many series can use the right-side legend since they have full-width containers.

## Sources

### Primary (HIGH confidence)
- Plotly Templates Documentation: https://plotly.com/python/templates/ -- custom template creation, `pio.templates.default`
- Plotly Configuration Options: https://plotly.com/python/configuration-options/ -- `displayModeBar`, toolbar config
- Flask Template Inheritance: https://flask.palletsprojects.com/en/stable/patterns/templateinheritance/ -- `extends`/`block` pattern
- Codebase analysis of `plots/shared.py`, `plots/latest_plots.py`, `plots/trends_plots.py` -- current state audit

### Secondary (MEDIUM confidence)
- VHP4Safety GitHub Organization: https://github.com/VHP4Safety -- branding/logo reference
- Plotly Theming Guide (Medium article): https://hi-artemii.medium.com/quick-plotly-py-theming-guide-4b38ead662ba -- practical template examples
- CSS Custom Properties (MDN): https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties -- CSS variables spec

### Tertiary (LOW confidence)
- VHP4Safety funding text -- inferred from codebase footer text "Built for VHP4Safety"; exact required acknowledgment text needs verification with stakeholder

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools already in the project; no new dependencies
- Architecture: HIGH -- Plotly templates and Jinja2 inheritance are well-documented, widely used patterns
- Pitfalls: HIGH -- identified from direct codebase analysis of existing inconsistencies
- Discretion recommendations: MEDIUM -- based on UX best practices for data dashboards and codebase constraints; user may prefer different choices

**Research date:** 2026-02-23
**Valid until:** 2026-03-23 (stable technologies, long-lived patterns)
