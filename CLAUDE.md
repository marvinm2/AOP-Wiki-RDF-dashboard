# CLAUDE.md

Project guidance for Claude Code when working with this repository.

## 🎯 Current Status & Roadmap

### Completed Features
- ✅ Version selector with dynamic plot regeneration
- ✅ CSV export system with metadata
- ✅ Lazy loading for 99.98% faster page loads
- ✅ Modular plots architecture (3,407 LOC)
- ✅ VHP4Safety brand consistency
- ✅ Property presence visualizations for AOPs, KEs, KERs, and Stressors with marker shape differentiation

### Open Issues
- **#11**: Network Analysis Expansion (centrality, clustering, PageRank) - MEDIUM priority
- **#3**: VHP platform deployment - MEDIUM priority

### Recent Work
- ✅ Optimized "Composite AOP Completeness Distribution Over Time" plot (e58a7e3)
  - Reduced from 10 queries to 4 (60% reduction)
  - Data transfer reduced from ~410K to ~20K rows (95% reduction)
  - Load time: timeout → ~75 seconds
- ✅ Removed "Composite AOP Completeness by OECD Status" plot (hits Virtuoso limits)
- ✅ Applied SPARQL optimization patterns: CSV-based property loading, GROUP_CONCAT aggregation, UNION queries

## Quick Commands

```bash
# Run
python app.py              # or: flask run

# Docker
docker-compose up --build

# Health
curl http://localhost:5000/health
curl http://localhost:5000/status
```

## Architecture

Flask dashboard for AOP-Wiki RDF data with SPARQL queries, Plotly visualizations, and CSV exports.

**See `.claude/architecture.md` for detailed architecture reference.**
**See `.claude/colors.md` for complete VHP4Safety brand colors.**

### Key Colors
- Headers/Footers: `#29235C` | Background: `#f5f5f5` | CTA: `#E6007E` | Navigation: `#307BBF`

## Configuration

Key environment variables:
```bash
SPARQL_ENDPOINT=http://localhost:8890/sparql
FLASK_PORT=5000
PARALLEL_WORKERS=5
SPARQL_TIMEOUT=30
```

## Plot Functions

Located in `plots/` package. All cache data for CSV export.

### Latest Data (plots/latest_plots.py)
All accept `version: str = None` for historical snapshots:
- `plot_latest_entity_counts()`, `plot_latest_ke_components()`, `plot_latest_network_density()`
- `plot_latest_avg_per_aop()`, `plot_latest_process_usage()`, `plot_latest_object_usage()`
- `plot_latest_aop_completeness()`, `plot_latest_ontology_usage()`, `plot_latest_database_summary()`
- `plot_latest_ke_annotation_depth()`

### Historical Trends (plots/trends_plots.py)
Return `(absolute_plot, percentage_plot)` tuples for property presence, `(absolute_plot, delta_plot, data)` for others:
- **Core Evolution**: `plot_main_graph()`, `plot_avg_per_aop()`, `plot_network_density()`, `plot_author_counts()`
- **Component Analysis**: `plot_ke_components()`, `plot_ke_components_percentage()`, `plot_unique_ke_components()`, `plot_kes_by_kec_count()`
- **Ontology Usage**: `plot_bio_processes()`, `plot_bio_objects()`
- **Property Presence** (with marker shape differentiation):
  - `plot_aop_property_presence()` - AOP metadata property presence over time
  - `plot_ke_property_presence()` - Key Event property presence over time
  - `plot_ker_property_presence()` - Key Event Relationship property presence over time
  - `plot_stressor_property_presence()` - Stressor property presence over time
- **Temporal Analysis**: `plot_aop_lifetime()`

### Shared Utilities (plots/shared.py)
- `run_sparql_query_with_retry()`, `get_latest_version()`, `get_all_versions()`
- `check_sparql_endpoint_health()`, `safe_plot_execution()`, `create_fallback_plot()`

## Adding New Features

### Adding New Plots

**See `.claude/add-a-plot.md` for the complete step-by-step checklist.**

Quick reference:
- **Latest Data Plot**: Add to `plots/latest_plots.py`, export from `plots/__init__.py`, register in `app.py` plot_map, add template container, register in version-selector.js
- **Historical Trends Plot**: Add to `plots/trends_plots.py`, export, add to startup computation in `app.py`, add to plot_map, add template containers

### New Utility
Add to `plots/shared.py`, export from `plots/__init__.py`

## Performance
- Lazy loading: ~50ms initial load (99.98% faster)
- SPARQL optimization: 85% faster queries
- Parallel processing: 5 workers
- Caching: Global DataFrame cache

## Documentation

- **README.md**: Setup and technical docs
- **CLAUDE.md**: This file - development guidance
- **docs/**: Sphinx documentation
- **.claude/**: Modular reference docs (architecture, colors)
- **.claude/add-a-plot.md**: Complete checklist for adding new plots

## Git Guidelines

- Never mention Claude/Anthropic in commits unless requested
- Focus on technical changes and impact
- Use conventional commit format
