# AOP-Wiki RDF Dashboard

A comprehensive web dashboard for monitoring and analyzing the AOP-Wiki RDF knowledge base evolution over time. Built for the VHP4Safety project to provide insights into Adverse Outcome Pathways (AOPs) data trends, quality metrics, and structural analysis.

## Features

### **Interactive Visualizations**
- **Latest Data Overview**: Current state snapshots of the AOP-Wiki database
- **Historical Trends**: Time-series analysis of database evolution
- **Network Analysis**: AOP connectivity and structural relationships
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
# Python 3.8+ required
python3 --version

# Install dependencies
pip install -r requirements.txt
```

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

### **Interactive Features**
- **Toggle Views**: Switch between absolute values and deltas
- **CSV Downloads**: Export data from any plot using download buttons
- **Responsive Charts**: Hover for details, zoom, and pan capabilities

### **Data Export**
1. Navigate to any plot with a "Download CSV" button
2. Click to download the underlying dataset
3. For historical trends: choose "CSV (Abs)" or "CSV (Δ)"
4. Files include version metadata and contextual information

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
└── requirements.txt      # Python dependencies
```

### **Adding New Plots**
1. Create plot function in the `plots/` package with data caching
2. Add Flask route for CSV download in `app.py`
3. Include plot variable in template rendering
4. Add download button to appropriate HTML template

### **Customization**
- **Colors**: Modify `BRAND_COLORS` in `plots/shared.py`
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

If you use this dashboard, please cite the underlying AOP-Wiki RDF paper and the dataset DOI. See [`CITATION.cff`](CITATION.cff).

- Paper: Martens M., Evelo C.T., Willighagen E.L. (2022). *Providing Adverse Outcome Pathways from the AOP-Wiki in a Semantic Web Format to Increase Usability and Accessibility of the Content.* Applied In Vitro Toxicology 8(1):2–13. [doi:10.1089/aivt.2021.0010](https://doi.org/10.1089/aivt.2021.0010)
- Dataset releases (concept DOI): [10.5281/zenodo.13353286](https://doi.org/10.5281/zenodo.13353286)

## Support

For questions, issues, or feature requests:
1. Check the troubleshooting section above
2. Review existing issues in the project repository
3. Contact the VHP4Safety development team

## Related Projects

- **VHP4Safety**: Virtual Human Platform for safety assessment
- **AOP-Wiki**: Adverse Outcome Pathways knowledge base

