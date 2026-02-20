# Codebase Structure

**Analysis Date:** 2026-02-20

## Directory Layout

```
AOP-Wiki-RDF-dashboard/
├── app.py                          # Main Flask application (1,608 lines)
├── config.py                       # Configuration management (178 lines)
├── requirements.txt                # Python dependencies
├── property_labels.csv             # Property metadata for entity types
├── verify_properties.py             # Utility script for property validation
├── docker-compose.yml              # Docker orchestration
├── Dockerfile                      # Container image definition
├── CLAUDE.md                       # Development guidance
├── README.md                       # Setup and technical docs
├── plots/                          # Core visualization package
│   ├── __init__.py                 # Module exports (322 lines)
│   ├── shared.py                   # SPARQL queries, utilities, constants (1,023 lines)
│   ├── trends_plots.py             # Historical trend visualizations (2,888 lines)
│   └── latest_plots.py             # Current state visualizations (1,774 lines)
├── static/                         # Client-side assets
│   ├── css/
│   │   ├── main.css                # Primary styling (15,008 bytes)
│   │   └── lazy-loading.css        # Lazy load placeholder styles (3,081 bytes)
│   └── js/
│       ├── lazy-loading.js         # Lazy plot loading logic (7,055 bytes)
│       └── version-selector.js     # Database version selection UI (9,593 bytes)
├── templates/                      # Jinja2 HTML templates
│   ├── landing.html                # Navigation/intro page
│   ├── latest.html                 # Database snapshot with version selector
│   ├── trends.html                 # Historical trends visualization page (old)
│   ├── trends_page.html            # Historical trends page (current)
│   ├── index.html                  # Legacy tabbed dashboard
│   └── status.html                 # Health monitoring dashboard
├── docs/                           # Sphinx documentation
│   └── source/
│       ├── conf.py                 # Sphinx configuration
│       └── _static/                # Sphinx static files
└── .planning/
    └── codebase/                   # GSD codebase analysis documents
        ├── ARCHITECTURE.md         # This file
        └── STRUCTURE.md            # This file
```

## Directory Purposes

**plots/ Package:**
- Purpose: Modular visualization and data processing layer
- Contains: SPARQL query functions, plot generation, data caching
- Key files: shared.py (utilities), trends_plots.py (30+ historical plots), latest_plots.py (14+ current state plots)
- Import pattern: from plots import plot_function_name or from plots.shared import utility_function

**static/ Directory:**
- Purpose: Client-side JavaScript and CSS assets served directly to browsers
- Contains: Responsive styling, lazy-loading handlers, version selector UI
- Served via: Flask static_folder at /static/* routes
- Auto-minification: No; files served as-is

**templates/ Directory:**
- Purpose: Jinja2 templates for server-side HTML rendering
- Contains: Master layouts, plot containers, section markup
- Rendering: Flask render_template() in app.py routes
- Dynamic content: Version lists, plot HTML injection, version parameter passing

**docs/ Directory:**
- Purpose: Sphinx-based documentation (not actively used in production)
- Contains: Configuration, API docs generation configuration
- Build target: Rarely used; maintained for future documentation builds

## Key File Locations

**Entry Points:**
- `app.py`: Main Flask application with 22 routes and startup orchestration
- `config.py`: Configuration class with 13 environment-driven parameters
- `plots/__init__.py`: Module initialization with function exports and re-exports

**Configuration:**
- `config.py`: All application settings (SPARQL endpoint, timeouts, parallelism, Flask host/port)
- `requirements.txt`: Python package dependencies (Flask, SPARQLWrapper, pandas, plotly)
- `.env` or environment variables: Runtime configuration (see config.py for all accepted vars)

**Core Logic:**
- `plots/shared.py`: SPARQL execution, retry logic, health checks, brand colors, data caching
- `plots/trends_plots.py`: 15+ historical trend visualization functions with delta/percentage calculations
- `plots/latest_plots.py`: 14+ current state visualization functions with version parameter support

**Testing:**
- No test files found in repository
- Test utilities: `verify_properties.py` (standalone script for property CSV validation)

## Naming Conventions

**Files:**
- Python modules: snake_case (e.g., trends_plots.py, version_selector.js)
- HTML templates: kebab-case-style names (e.g., trends_page.html, lazy-loading.css)
- Static assets: kebab-case for multi-word names (lazy-loading.js)
- Entry point: app.py (convention for Flask applications)

**Directories:**
- Package directories: lowercase plural (plots/, templates/, static/)
- Asset subdirectories: lowercase plural (css/, js/)
- Documentation: singular (docs/)

**Variables & Functions:**
- Python functions: snake_case (e.g., plot_main_graph, safe_plot_execution, run_sparql_query_with_retry)
- Configuration constants: UPPER_SNAKE_CASE (e.g., SPARQL_ENDPOINT, BRAND_COLORS)
- Plot function naming pattern: plot_[latest_|][entity/concept]_[dimension] (e.g., plot_latest_entity_counts, plot_aop_property_presence)
- Cache keys: snake_case matching plot names (e.g., _plot_data_cache['latest_entity_counts'])

**Route naming pattern:**
- Page routes: /snapshot, /trends, / (leading slash implied)
- API routes: /api/[resource]/[id] (e.g., /api/plot/latest_entity_counts, /api/latest-version)
- Download routes: /download/[plot_name], /download/bulk, /download/trend/[plot_name]

## Where to Add New Code

**New Latest Data Plot (Current State Visualization):**
1. **Implementation:** Add function to `plots/latest_plots.py` (follows pattern: plot_latest_[name]())
   - Include optional `version: str = None` parameter for snapshot selection
   - Use `_build_graph_filter(version)` to generate SPARQL WHERE/ORDER clauses
   - Cache data to `_plot_data_cache['plot_name']` during execution
   - Return HTML string from pio.to_html(fig)
2. **Export:** Add to `plots/__init__.py` in latest_plots import list and __all__
3. **Web Route:** Add download endpoint to `app.py` (e.g., @app.route("/download/latest_[name]"))
4. **Template:** Add plot container to `templates/latest.html` with lazy-plot div and download link
5. **Version Selector:** Register plot in `static/js/version-selector.js` versionedPlots array

**New Historical Trends Plot (Time-Series Visualization):**
1. **Implementation:** Add function to `plots/trends_plots.py` (follows pattern: plot_[name]())
   - Return tuple: (absolute_html, delta_html, data_df) or (absolute_html, percentage_html) for property presence
   - Execute SPARQL queries across all versions
   - Cache data to `_plot_data_cache['absolute_plot_name']` and `_plot_data_cache['delta_plot_name']`
   - Use safe_plot_execution() wrapper in app.py for startup computation
2. **Export:** Add to `plots/__init__.py` in trends_plots import and __all__
3. **Startup Computation:** Add task tuple to plot_tasks list in `app.py:compute_plots_parallel()` (line ~164)
4. **Variable Unpacking:** Extract results with pattern: abs_plot, delta_plot = plot_results.get('[plot_name]', ("", "")) (line ~225)
5. **Web Routes:** Add download endpoints for absolute and delta views (e.g., /download/trend/[name]_absolute)
6. **Template:** Add plot containers to `templates/trends_page.html` with side-by-side layout
7. **Bulk Download:** Register in categories dict in `app.py:/download/bulk` endpoint if applicable

**New Utility Function:**
- Add to `plots/shared.py` under appropriate section (SPARQL functions, data processing, etc.)
- Export from `plots/__init__.py` in shared utilities import and __all__
- Document with comprehensive docstring following existing pattern

**Modify Configuration:**
- Add environment variable to `config.py` as class attribute with default value
- Pattern: `NAME = os.getenv("NAME", "default_value")`
- Include in Config.get_config_dict() automatically (returns all uppercase attributes)
- Add validation logic to Config.validate_config() if bounds/format matter

**New CSS Styling:**
- Add to `static/css/main.css` for general styling
- Add to `static/css/lazy-loading.css` for lazy-load-specific styles
- Use VHP4Safety brand colors from `plots/shared.py:BRAND_COLORS` dict
- Follow existing responsive design patterns (@media queries)

**New JavaScript Functionality:**
- Core lazy-loading logic: `static/js/lazy-loading.js`
- Version selector logic: `static/js/version-selector.js`
- Follow fetch API pattern for /api/ endpoints
- Include error handling and fallback UI updates

**New HTML Template:**
- Location: `templates/[name].html`
- Base structure: Inherit from common header/footer if applicable
- Plot containers: Use data-plot-name attribute for lazy loading
- Links: Reference CSS via {{ url_for('static', filename='css/[name].css') }}
- Scripts: Load after body for performance

## Special Directories

**plots/__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Yes (automatic by Python)
- Committed: No (.gitignore entry)
- Action: Ignore; will regenerate on runtime

**__pycache__/ (root level):**
- Purpose: Root-level Python bytecode cache
- Generated: Yes (from config.py, app.py imports)
- Committed: No (.gitignore entry)
- Action: Ignore; will regenerate on runtime

**.planning/codebase/:**
- Purpose: GSD codebase analysis documents
- Generated: Yes (by /gsd:map-codebase command)
- Committed: Yes (checked into git)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md, STACK.md, INTEGRATIONS.md

**docs/:**
- Purpose: Sphinx documentation source
- Generated: No (manually maintained, rarely updated)
- Committed: Yes
- Build output: Not committed (conf.py only)

**.claude/:**
- Purpose: Modular reference documentation
- Generated: No (manually maintained)
- Committed: Yes
- Contents: architecture.md, colors.md (referenced in CLAUDE.md)

## Structural Patterns

**Plot Function Organization:**
- All plot functions in plots/ package for modularity
- Latest plots grouped by theme (database state, components, connectivity, ontology, quality)
- Trend plots follow parallel grouping for consistency
- Shared utilities centralized in plots/shared.py to prevent duplication

**Data Flow Organization:**
- SPARQL queries isolated in plots/shared.py
- Plotting logic (data processing + visualization) in trends_plots.py and latest_plots.py
- Web routing and serving in app.py
- Configuration centralized in config.py

**Asset Organization:**
- CSS files organized by purpose (main styling, lazy-load styles)
- JavaScript files single-purpose (lazy-loading.js, version-selector.js)
- Static folder follows web convention (css/, js/)
- Templates folder follows Flask convention

---

*Structure analysis: 2026-02-20*
