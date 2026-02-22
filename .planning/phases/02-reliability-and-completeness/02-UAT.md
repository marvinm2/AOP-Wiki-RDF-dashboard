---
status: resolved
phase: 02-reliability-and-completeness
source: [02-01-SUMMARY.md, 02-02-SUMMARY.md, 02-03-SUMMARY.md, 02-04-SUMMARY.md, 02-05-SUMMARY.md, 02-06-SUMMARY.md]
started: 2026-02-22T10:45:00Z
updated: 2026-02-22T11:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Boxplot loads instantly and shows all versions
expected: Navigate to /trends, scroll to "Composite AOP Completeness Distribution". The boxplot should load without delay (pre-computed at startup). The x-axis should show versions from 2018 through 2025, including post-2020 versions — not just pre-2020 data.
result: pass

### 2. OECD completeness trend chart
expected: On /trends, scroll to the OECD section. A line chart titled "Mean AOP Completeness by OECD Status Over Time" should display with multiple colored lines (one per OECD status) and marker shapes differentiating each status. The title and legend should be clearly separated without overlap.
result: issue
reported: "Still overlapping. Maybe add the legend on the right side just like other plots?"
severity: cosmetic

### 3. KE component annotation trends
expected: On /trends, scroll to "KE Component Annotations". All three plot pairs (KE Component Annotations absolute/percentage, KE Components as %, Unique KE Components) should show actual chart data with colored traces — not grey error cards saying "Data Unavailable".
result: pass

### 4. KE component latest-page plots
expected: On /latest (with any version selected), the KE component pie chart and ontology usage chart should display actual data — not error cards.
result: pass

### 5. CSV download for trend plots
expected: On /trends, click the download dropdown on any trend plot (e.g., Main Graph) and select CSV. The browser should download a CSV file with actual data rows and headers. The filename should be descriptive with a date, like `main-graph-absolute_2026-02-22.csv`.
result: pass

### 6. PNG/SVG export
expected: On /trends or /latest, click the download dropdown on any plot and select PNG (or SVG). The browser should download a valid image file of the chart. This requires the kaleido dependency to be installed.
result: issue
reported: "pass but not every plot seems to work"
severity: minor

### 7. Methodology notes on latest page
expected: On /latest, each plot box should have an expandable "Methodology" section (a clickable disclosure triangle). Expanding it should show: a description of what the plot measures, the data source, limitations, and an expandable SPARQL query viewer.
result: pass

### 8. Methodology notes on trends page
expected: On /trends, each plot box should have an expandable "Methodology" section. All three AOP Lifetime sub-plots (AOPs Created Over Time, AOPs Modified Over Time, AOP Creation vs Modification) should each have their own methodology note.
result: pass

### 9. Methodology limitations quality
expected: Expand any methodology note and read the "Limitations" text. It should contain researcher-relevant caveats about data quality or methodology (e.g., "AOPs with no linked KEs or KERs use only their own completeness"). It should NOT contain performance text like "Query may take 30-75 seconds" or implementation details about query generation.
result: issue
reported: "pass, but also the trend plots go back to the first release version, but AOP-Wiki existed before"
severity: minor

### 10. Error resilience
expected: All plots on /trends and /latest should either show chart data or a clean error card with "Data Unavailable" message — never a Python traceback, blank white space, or broken page layout. If the SPARQL endpoint is slow, plots should degrade gracefully.
result: pass

## Summary

total: 10
passed: 7
issues: 3
pending: 0
skipped: 0

## Gaps

- truth: "OECD completeness trend plot title and legend are clearly separated without overlap"
  status: resolved
  reason: "User reported: Still overlapping. Maybe add the legend on the right side just like other plots?"
  severity: cosmetic
  test: 2
  root_cause: "Horizontal legend at y=1.15 collides with title regardless of top margin — Plotly renders title inside margin so increasing t just pushes both down. With 5 OECD categories the legend wraps to 2 rows. Sibling plot_aop_completeness_boxplot_by_status uses right-side vertical legend with zero overlap."
  artifacts:
    - path: "plots/trends_plots.py"
      issue: "OECD plot uses orientation='h' legend at y=1.15 which overlaps title"
  missing:
    - "Change to orientation='v' right-side legend matching sibling plot pattern (x=1.02, yanchor='top', y=1)"

- truth: "PNG/SVG export works for every plot on the dashboard"
  status: resolved
  reason: "User reported: pass but not every plot seems to work"
  severity: minor
  test: 6
  root_cause: "3 latest plots (aop_completeness_by_status, ke_completeness_by_status, ker_completeness_by_status) are missing from compute_plots_parallel() in app.py. Their figures are never cached at startup, so PNG/SVG export returns 404. Additionally, lazy-loaded plots have a race condition where download buttons are clickable before the plot has loaded."
  artifacts:
    - path: "app.py"
      issue: "3 plots missing from plot_tasks list in compute_plots_parallel()"
  missing:
    - "Add 3 missing plots to plot_tasks and extract results"

- truth: "Methodology limitations mention that trend data only covers RDF release versions, not the full AOP-Wiki history"
  status: resolved
  reason: "User reported: pass, but also the trend plots go back to the first release version, but AOP-Wiki existed before"
  severity: minor
  test: 9
  root_cause: "All 16 trend-over-time methodology note entries in methodology_notes.json lack a data scope caveat explaining that trends only cover published RDF release versions, not the full AOP-Wiki history which predates the first RDF publication."
  artifacts:
    - path: "static/data/methodology_notes.json"
      issue: "16 trend entries missing data scope limitation"
  missing:
    - "Append to each trend entry: 'Trend data covers only the RDF release versions available in the SPARQL endpoint; AOP-Wiki predates the first RDF publication, so activity before the initial release is not reflected.'"
