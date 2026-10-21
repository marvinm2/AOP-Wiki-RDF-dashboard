# AOP-Wiki RDF Dashboard

A comprehensive web dashboard for monitoring and analyzing the AOP-Wiki RDF knowledge base evolution over time. Built for the VHP4Safety project to provide insights into Adverse Outcome Pathways (AOPs) data trends, quality metrics, and structural analysis.

## Features

### **Interactive Visualizations**
- **Latest Data Overview**: Current state snapshots of the AOP-Wiki database
- **Historical Trends**: Time-series analysis of database evolution
- **Version Diff** (`/diff`): What changed between two versions — entities added, removed or edited, the fields that changed, OECD stage moves, and a worklist with AOP-Wiki links
- **Network Analysis**: AOP connectivity and structural relationships
- **Network Evolution** (`/network`, Evolution tab): every snapshot's KE/KER network on one fixed layout, picked with the version scrubber; new and gone elements marked
- **Data Quality Metrics**: Annotation completeness and ontology usage patterns

### **Comprehensive Analytics**
- Entity counts (AOPs, Key Events, Key Event Relationships, Stressors)
- Key Event component distribution and annotation depth
- Biological process and object ontology usage patterns
- AOP completeness analysis using configurable property types
- Network density and connectivity analysis
- Author contribution tracking over time

### **Data Export Capabilities**
- **CSV Download**: Export underlying data from any plot for further analysis
- **10+ Download Options**: Entity counts, components, usage patterns, trends, and more
- **Multiple Formats**: Absolute values and delta (change) datasets available
- **Rich Metadata**: All exports include version info and contextual data

### **Performance & Reliability**
- **Parallel Processing**: Concurrent plot generation for fast loading
- **Health Monitoring**: Real-time SPARQL endpoint status tracking
- **Error Handling**: Graceful degradation with fallback visualizations
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Architecture

### **Backend Components**
- **Flask Web Framework**: Lightweight and fast web server
- **SPARQL Integration**: Direct queries to AOP-Wiki RDF endpoint
- **Plotly Visualizations**: Interactive, publication-quality charts
- **Pandas Data Processing**: Efficient data manipulation and caching
- **Concurrent Execution**: Multi-threaded plot generation

### **Data Sources**
- **Primary**: AOP-Wiki SPARQL endpoint (`http://localhost:8890/sparql`)
- **Configuration**: Property metadata from `property_labels.csv`
- **Ontology Sources**: GO, CHEBI, PR, CL, UBERON, HP, MP, and others

### **Visualization Types**
- Time-series line plots for trends
- Bar charts for entity counts and metrics
- Pie charts for distributions and proportions
- Interactive hover details and responsive layouts

## Quick Start

### Prerequisites
```bash
# Python 3.11 (the version in .python-version and the Docker base image)
python3 --version

# Install the pinned dependencies (+ test tools)
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                     # offline suite; live-endpoint tests are marked `slow`
```

Direct dependencies are declared in `requirements.in`; `requirements.txt` is the
pip-compile lock built for Python 3.11. To change a dependency, edit
`requirements.in` and recompile under 3.11 (a 3.12 compile can pin wheels that
don't exist for 3.11):

```bash
uv run --no-project --python 3.11 --with pip-tools \
  pip-compile --strip-extras --no-emit-index-url requirements.in -o requirements.txt
```

Dependabot proposes monthly updates to the lock, the GitHub Actions and the base
image. CI runs the offline suite on every PR and push; the image is only built
when it passes.

### Configuration
1. **SPARQL Endpoint**: Ensure AOP-Wiki RDF endpoint is accessible
2. **Property Labels**: Update `property_labels.csv` with desired AOP properties
3. **Environment**: Configure settings in `config.py`

### Running the Dashboard
```bash
# Start the application
python app.py

# Access the dashboard
open http://localhost:5000
```

### Health Check
```bash
# Verify system status
curl http://localhost:5000/health

# Monitor real-time status
open http://localhost:5000/status
```

## Methodology Query CI Lint

Every plot in the dashboard exposes its underlying SPARQL query via `static/data/methodology_notes.json`. The lint in `scripts/lint_methodology.py` guards that contract against drift. It runs six checks: **LINT-01** catches SPARQL syntax / SP031-class compiler errors by wrapping each entry's query and posting it to the live endpoint; **LINT-02** flags entries whose query returns zero rows (or a single all-zero aggregate row) against the latest graph, unless the entry opts out with `may_be_empty: true`; **LINT-03** statically rejects any `latest_*` entry that lacks a latest-snapshot constraint (it must contain either `__GRAPH_URI__` or the `ORDER BY DESC(?graph) LIMIT 1` + `STRSTARTS(STR(?graph), "http://aopwiki.org/graph/")` trio); **LINT-04** is a soft warning that fires when a plot function in `plots/*.py` issues more SPARQL calls than its methodology entry discloses (detected via AST scan only — the lint never imports `plots`); **LINT-05** is an endpoint-call-budget guard that refuses to start when the JSON has more than ~50 entries; **LINT-06** renders the deterministic markdown + JSON reports that the CI job posts back as a PR comment.

### Full local run (LINT-01 + LINT-02 + LINT-03 + LINT-04, hits live endpoint)

```bash
python scripts/lint_methodology.py \
  --json static/data/methodology_notes.json \
  --plots-dir plots \
  --report lint-report.md \
  --json-report lint-report.json
```

Exit codes: `0` = pass (warns allowed), `1` = at least one fail-severity finding, `2` = infrastructure error (DNS, 5xx, timeout), `3` = pre-flight refusal because the entry count exceeds the budget.

### Fast offline run (LINT-03 + LINT-04 only, no endpoint calls)

```bash
python scripts/lint_methodology.py \
  --json static/data/methodology_notes.json \
  --plots-dir plots \
  --no-network \
  --report lint-report.md \
  --json-report lint-report.json
```

`--no-network` skips LINT-01 + LINT-02 (no live endpoint round-trip and no latest-graph resolution). LINT-03 (static-only) and LINT-04 (AST-only) still run. This is the right path for quickly iterating on a query you just edited.

### Marking an entry as legitimately empty

Some queries can return zero rows in the latest snapshot for a real reason — e.g. a brand-new predicate not yet populated, or an opt-out group with no current members. Add `"may_be_empty": true` to the methodology entry to suppress LINT-02 for it:

```json
"latest_some_empty_thing": {
  "title": "…",
  "sparql": "…",
  "may_be_empty": true
}
```

The opt-out is intentionally explicit so the next maintainer sees the marker when reviewing the JSON. Today no entries opt out; the lock-list grows only as legitimate empties are identified.

### Why this exists

Failures are tracked via GitHub issues (e.g. #35 SP031 in `latest_ontology_usage`, #36 bogus `aopo:has_biological_process` predicates, #37 completeness scanning all 33 graphs instead of the latest one). Running the lint locally before opening a PR avoids surprise CI failures and gives you a per-check fix hint directly in the markdown report.

## Usage Guide

### **Navigation**
- **Latest Data**: Current database snapshots and quality metrics
- **Historical Trends**: Time-series analysis and evolution patterns
- **Version Diff**: `/diff?version=B&against=A` (or `?a=&b=`); A defaults to the version before B

### **Interactive Features**
- **Toggle Views**: Switch between absolute values and deltas
- **CSV Downloads**: Export data from any plot using download buttons
- **Responsive Charts**: Hover for details, zoom, and pan capabilities

### **Data Export**
1. Navigate to any plot with a "Download CSV" button
2. Click to download the underlying dataset
3. For historical trends: choose "CSV (Abs)" or "CSV (Δ)"
4. Files include version metadata and contextual information

### **Data API (v2)**
Every plot's data and figure are available as JSON, for notebooks, other tools and
custom views:

| Endpoint | Returns |
|---|---|
| `GET /api/v2/manifest` | every plot (id, kind, settings), the available versions, the store build time |
| `GET /api/v2/data/<id>?version=YYYY-MM-DD` | the rows of the plot's CSV download, as columnar JSON: `{"meta": {..., "columns": [{"name", "type"}], "queries": [...]}, "data": {column: [values]}}` |
| `GET /api/v2/figure/<id>?version=YYYY-MM-DD` | the plot as Plotly JSON (layout template removed) |

- `version` applies to snapshot plots (default: the latest); trend plots cover every version.
- A plot's settings are query parameters too, e.g. `?min_jaccard=0.4` or `?domain=Cardiovascular`
  (see `params` in the manifest).
- `meta.queries` holds the SPARQL behind the plot, as shown in its methodology panel.
- Responses carry an `ETag`; send it back as `If-None-Match` to get `304 Not Modified`.

```python
import pandas as pd, requests
r = requests.get("https://aopwiki-dashboard.vhp4safety.nl/api/v2/data/latest_entity_counts",
                 params={"version": "2024-01-01"}).json()
df = pd.DataFrame(r["data"])[[c["name"] for c in r["meta"]["columns"]]]
```

## Development

### **File Structure**
```
├── app.py                 # Main Flask application
├── plots/                 # Plot generation package (shared, trends, latest)
├── config.py              # Configuration settings
├── property_labels.csv    # AOP property metadata
├── templates/             # HTML templates
│   ├── index.html        # Main dashboard layout
│   ├── trends.html       # Historical trends section
│   └── status.html       # System monitoring page
├── static/
│   └── css/main.css      # Styling and responsive design
├── docs/                  # Comprehensive Sphinx documentation
│   ├── source/           # Documentation source files
│   │   ├── conf.py      # Sphinx configuration
│   │   ├── index.rst    # Main documentation page
│   │   ├── modules.rst  # API reference
│   │   ├── quickstart.rst # Quick start guide
│   │   ├── configuration.rst # Configuration guide
│   │   └── api.rst      # Detailed API documentation
│   └── build/            # Generated documentation
├── requirements.in       # Direct Python dependencies
└── requirements.txt      # pip-compile lock (Python 3.11)
```

### **Frontend runtime and its tests**
All plot loading goes through `static/js/runtime.js`. It fetches through a queue of 4, aborts requests the
page has moved past, never draws a response for an older selection, and dispatches `plot:rendered` after
each draw. Page state (version, trends range, plot settings) lives in the URL via `static/js/state.js`, so
every view can be bookmarked or shared. Controls only change that state.

Once a figure is drawn, `static/js/viz/export.js` adds "PNG (as shown)" and "SVG (as shown)" to its
download menu. These save the figure with its current zoom, range and settings, under a file name that
records the version or range and the settings. `static/js/viz/a11y.js` makes each figure a labelled,
focusable element (arrow keys, Home and End step through its values and read them out) and loads the
"Show Raw Data" tables from `/api/v2/data` for the same selection as the figure.

Views that Plotly has no form for live in `static/js/viz/` and register a renderer with
`PlotRuntime.registerRenderer(name, renderer)`; a container picks one with `data-renderer`. The version
scrubber (`scrubber.js`), the entity lifeline (`lifeline.js`, renderer `d3-lifeline`, drawn on canvas
from `/api/v2/data/entity_lifeline`) and the organ-system UpSet chart (`upset.js`, renderer `d3-upset`,
from `/api/v2/data/latest_organ_upset`) use D3, vendored in `static/vendor/`. The AOP-AOP overlap
(`overlap.js`) shows the server's cluster-ordered matrix in one tab and the same pairs as a Cytoscape
graph in another (renderer `cytoscape-overlap`, from `/api/v2/data/latest_aop_aop_overlap`; Cytoscape
loads when the tab is first opened), with one cluster picker and pair table for both. The AOP
completeness beeswarm (`beeswarm.js`, renderer `plotly-beeswarm`, from
`/api/v2/data/aop_completeness_beeswarm`) is Plotly: one dot per AOP for the snapshot at the end of the
/trends range, with the AOP IRI as the dot's id so `Plotly.animate` slides each AOP to its new score
when the range end moves (it jumps instead under `prefers-reduced-motion`). The property fill-in
heatmaps (`property-heatmap.js`) are server Plotly figures, one per entity tab (`?props=`); the module
adds the tabs and a cell-by-cell keyboard path, since `a11y.js` steps along one axis only. The ontology
coverage map (`coverage-map.js`) is a server icicle drawn as horizontal bars, one row per level of the
organ-system branch (Plotly's icicle trace is not in the cartesian bundle); the module zooms to a block on
click or Enter and walks the tree with the arrow keys. The organ-system bump chart (`bump.js`) is a server
Plotly figure; the module adds the Highlight select (`?organ=`), which dims the other lines, and a keyboard
path that moves along a system's line with left and right and between ranks with up and down. The OECD
status journey (`status-journey.js`, renderer `plotly-full`) is a server sankey, a trace the cartesian bundle
lacks: the renderer loads the full Plotly bundle (`static/vendor/plotly@2.32.0`, same version) the first time
the card comes into view, through a one-shot AMD `define` so `window.Plotly` stays the cartesian one, and
keeps it on the drawn div as `div._dashPlotly` for the "as shown" downloads. The keyboard walks the bands
and darkens the focused one on its SVG path, since a Plotly restyle of a sankey fades it in again. Cohort
survival is a plain server Plotly figure whose x axis is years since first seen, not a snapshot: it is in the
range selector's `EXCLUDED_PLOTS`, and `trend-links.js` only shares a zoom between plots that have snapshots on
x. Its three panels repeat the cohort names, so each trace carries `meta` ("KEs, First seen 2018–2019"), which
`a11y.js` speaks instead of the name; for an `x unified` figure the keyboard hovers by x value, so Plotly draws
one box rather than a label per trace. The KE reuse rank race (`race.js`, renderer `plotly-race`, from
`/api/v2/data/ke_reuse_race`) follows the end of the /trends range like the beeswarm: one bar per KE that is
ever in the top 15, keyed by its IRI, so `Plotly.animate` slides the bars to their new ranks as Play moves
the range end. With transitions (redraw off) Plotly neither fills in text that was empty nor interpolates
from a zero-length bar, and it stacks bars that share a y, so every bar keeps its label and a sliver of
length, KEs outside the top wait off the canvas, and `barmode` is `overlay`. The KER curation calendar is a
server Plotly figure, one heatmap strip per year with a cell per day (customdata "Tue 2023-10-17: 33"); like
survival it is in `EXCLUDED_PLOTS`, and `calendar.js` gives it a day-by-day keyboard path (up/down a day,
left/right a week, Page Up/Down a year), hovering the cell with `Fx.hover` on that strip's subplot. A view that draws
one SVG with its colours as attributes marks it `svg[data-export="svg"]`, or builds one on request with
`container.exportSvg()`, and its download menu gets "PNG/SVG (as shown)" like the Plotly figures.

Snapshot downloads (`/download/latest/<plot>`) take the plot's settings as query parameters, as
`/api/v2/data` does, and the page adds them to every download link, so a file holds what is shown.

```bash
npm ci && npx playwright install chromium
npm run test:js     # node --test: state, figure, export and a11y modules
npm run test:e2e    # Playwright: pages served by the app, data requests stubbed (no endpoint needed)
```

### **Adding New Plots**
1. Create plot function in the `plots/` package with data caching
2. Add Flask route for CSV download in `app.py`
3. Include plot variable in template rendering
4. Add download button to appropriate HTML template

### **Customization**
- **Colors**: Modify `BRAND_COLORS` in `plots/shared.py`, then run `python scripts/build_tokens.py` to refresh `static/data/tokens.json` (the copy browser code reads; a test fails while it is stale)
- **Properties**: Update `property_labels.csv` with new AOP properties
- **Endpoints**: Configure SPARQL endpoint in `config.py`
- **Layout**: Customize templates and CSS for different layouts

## Configuration

### **Environment Variables**
```bash
SPARQL_ENDPOINT=http://localhost:8890/sparql
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
LOG_LEVEL=INFO
PARALLEL_WORKERS=5
PLOT_TIMEOUT=60
STARTUP_READY_TIMEOUT=300   # wait this long for the endpoint to serve AOP-Wiki data before precomputing plots
STARTUP_READY_INTERVAL=5    # seconds between those readiness checks
PLOT_RECOVERY_COOLDOWN=60   # minimum seconds between recomputes of a trend plot that failed
```

### **Property Configuration**
Edit `property_labels.csv` to customize AOP completeness analysis:
```csv
uri,label,type
http://aopkb.org/aop_ontology#has_key_event,Has Key Event,Content
http://purl.org/dc/elements/1.1/title,Title,Essential
```

## Troubleshooting

### **Common Issues**
- **Plots not loading**: Check SPARQL endpoint connectivity
- **Trend plots show "data unavailable" after a redeploy**: the startup run probably hit the endpoint before it was ready. Each failed plot is recomputed on its next request (at most once per `PLOT_RECOVERY_COOLDOWN`). `python scripts/check_live_trends.py [BASE_URL]` checks every trend plot on a deployed instance.
- **Slow performance**: Adjust `PARALLEL_WORKERS` and `PLOT_TIMEOUT`
- **CSV downloads failing**: Verify data caching is working correctly
- **Memory issues**: Reduce concurrent workers or increase system resources

### **Logging**
Monitor application logs for detailed error information:
```bash
tail -f app_new.log
```

### **Health Monitoring**
Use the built-in monitoring endpoints:
- `/health` - System health check
- `/status` - Real-time monitoring dashboard

## Requesting a New Plot

You don't need to write any code to ask for a new visualization. Pick whichever route suits you — both create an issue from the same **New Plot Proposal** template:

- **From the web (one click):** open a [pre-filled plot-request form](https://github.com/marvinm2/AOP-Wiki-RDF-dashboard/issues/new?template=new-plot.md). The dashboard's [About page](https://aopwiki-dashboard.vhp4safety.nl/about) links to the same form.
- **On GitHub:** go to the [issue tracker](https://github.com/marvinm2/AOP-Wiki-RDF-dashboard/issues), click **New issue**, and choose **New Plot Proposal**.

A GitHub account is required to submit. Describe the question the plot should answer, whether it's a latest-snapshot or historical-trend view, the RDF properties involved, and the chart type you'd expect — the more concrete, the faster it can be built.

## Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit a pull request

### **Code Standards**
- Follow PEP 8 for Python code
- Include comprehensive Google/Sphinx style docstrings
- Use type hints for all functions and methods
- Test all SPARQL queries and visualizations
- Ensure responsive design for new UI elements

### **Documentation**
The project includes comprehensive Google/Sphinx format documentation:

#### **Building Documentation**
```bash
# Install Sphinx dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs && make html

# View documentation
open docs/build/html/index.html
```

#### **Documentation Features**
- **Professional Sphinx Documentation**: Complete API reference with VHP4Safety branding
- **Google-Style Docstrings**: All functions documented with comprehensive examples
- **Interactive API Reference**: Auto-generated from docstrings with cross-references
- **Configuration Guides**: Detailed environment variable and deployment documentation
- **Quick Start Guides**: Step-by-step setup and usage instructions

## Maintainers

- **Lead maintainer:** Marvin Martens — Department of Translational Genomics, Maastricht University — [ORCID 0000-0003-2230-0840](https://orcid.org/0000-0003-2230-0840)
- **Backup maintainer:** Egon Willighagen — Department of Translational Genomics, Maastricht University — [ORCID 0000-0001-7542-0286](https://orcid.org/0000-0001-7542-0286)

For questions, bug reports, and feature requests please open a [GitHub Issue](https://github.com/marvinm2/AOP-Wiki-RDF-dashboard/issues).

## License

- **This dashboard's code** (Flask application, plot modules, templates, deployment files): MIT — see [`LICENSE`](LICENSE).
- **Visualised AOP-Wiki RDF dataset**: Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0) — see the [AOPWikiRDF data licence](https://github.com/marvinm2/AOPWikiRDF/blob/master/data/LICENSE-DATA). Matches the upstream [AOP-Wiki](https://aopwiki.org/) content licence.

## Citation

If you use this dashboard, please cite the underlying AOP-Wiki RDF paper, the software, and the dataset DOI. See [`CITATION.cff`](CITATION.cff).

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21384259.svg)](https://doi.org/10.5281/zenodo.21384259)

- Paper: Martens M., Evelo C.T., Willighagen E.L. (2022). *Providing Adverse Outcome Pathways from the AOP-Wiki in a Semantic Web Format to Increase Usability and Accessibility of the Content.* Applied In Vitro Toxicology 8(1):2–13. [doi:10.1089/aivt.2021.0010](https://doi.org/10.1089/aivt.2021.0010)
- Software (this dashboard, concept DOI): [10.5281/zenodo.21384259](https://doi.org/10.5281/zenodo.21384259)
- Dataset releases (concept DOI): [10.5281/zenodo.13353286](https://doi.org/10.5281/zenodo.13353286)

## Support

For questions, issues, or feature requests:
1. Check the troubleshooting section above
2. Review existing issues in the project repository
3. Contact the VHP4Safety development team

## Related Projects

- **VHP4Safety**: Virtual Human Platform for safety assessment
- **AOP-Wiki**: Adverse Outcome Pathways knowledge base

