---
phase: 04-entity-exploration
plan: 03
subsystem: ui
tags: [flask, plotly, sparql, jinja2, heatmap, trends, ontology]

# Dependency graph
requires:
  - phase: 04-entity-exploration
    provides: Raw data table toggle macro, URL state encoding, /api/plot-data endpoint
  - phase: 04-entity-exploration
    provides: 5 new latest-data plot functions, generic download route pattern
  - phase: 02-reliability-and-completeness
    provides: _plot_data_cache infrastructure, methodology notes JSON, brand colors, safe_plot_execution pattern
provides:
  - 2 new latest-data plots (annotation heatmap, ontology diversity)
  - 2 new trend plots (curation progress, ontology term growth) with absolute+delta variants
  - 4 methodology note entries for all new plots
  - Startup computation registration for trend plots
affects: [05-advanced-analytics]

# Tech tracking
tech-stack:
  added: []
  patterns: [ThreadPoolExecutor per-version parallel queries for trends, plotly.graph_objects.Heatmap for annotation matrix, URI prefix parsing for ontology classification]

key-files:
  created: []
  modified:
    - plots/latest_plots.py
    - plots/trends_plots.py
    - plots/__init__.py
    - app.py
    - templates/latest.html
    - templates/trends.html
    - static/js/version-selector.js
    - static/data/methodology_notes.json
---

## Summary

Added 4 data quality and trends plots completing Phase 4's dashboard enrichment vision: annotation completeness heatmap (property coverage matrix across entity types), ontology term diversity (unique terms per ontology source), curation progress over time (entity count vs annotation coverage growth), and ontology term growth (expanding annotation vocabulary across versions).

## Tasks Completed

| # | Task | Files | Status |
|---|------|-------|--------|
| 1 | Annotation heatmap + ontology diversity latest-data plots | plots/latest_plots.py, plots/__init__.py | ✓ |
| 2 | Curation progress + ontology term growth trend plots | plots/trends_plots.py, plots/__init__.py | ✓ |
| 3 | Register all in app.py, templates, methodology notes | app.py, templates/*.html, version-selector.js, methodology_notes.json | ✓ |

## Decisions

- Used separate COUNT queries per (entity_type, property) pair for heatmap to avoid cross-product explosion from multiple OPTIONALs
- Curation progress uses Title and Description as proxy annotations (simple, interpretable)
- Ontology term growth counts unique IRIs (not concepts) — Phase 5 adds per-term detail
- Methodology notes added for all 4 new plots

## Deviations

None.

## Self-Check: PASSED
- [x] All 3 tasks completed
- [x] 2 new latest-data plot functions implemented
- [x] 2 new trend plot functions implemented
- [x] All plots registered in app.py, templates, version selector
- [x] Methodology notes added for all 4 plots
- [x] Data table toggles present on all new plot containers
