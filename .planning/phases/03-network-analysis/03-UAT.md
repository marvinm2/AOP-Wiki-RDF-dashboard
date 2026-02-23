---
status: diagnosed
phase: 03-network-analysis
source: [03-01-SUMMARY.md, 03-02-SUMMARY.md, 03-03-SUMMARY.md, 03-04-SUMMARY.md]
started: 2026-02-23T10:00:00Z
updated: 2026-02-23T10:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Navigate to Network Page
expected: Landing page shows a "Network Analysis" button alongside existing navigation. Clicking it loads the /network page.
result: pass

### 2. Network Graph Renders
expected: The /network page shows an interactive graph with KE (Key Event) nodes and KER (Key Event Relationship) edges. Graph renders within ~10 seconds. Nodes are colored by community and sized by centrality. A stats bar shows node/edge counts.
result: pass

### 3. Node Click Info Panel
expected: Clicking a node opens a slide-in info panel on the right showing: node name, centrality metrics (degree, betweenness, closeness, PageRank), community assignment, list of neighbors, and a link to the AOP-Wiki page.
result: pass

### 4. Type-Ahead Search
expected: Typing in the search box shows matching nodes (by name or ID). Selecting a result centers and highlights that node on the graph.
result: pass

### 5. Filter Controls
expected: Community dropdown filters the graph to show only nodes in the selected community. A reset button clears all active filters and restores the full graph.
result: pass

### 6. Sortable Metrics Table
expected: Switching to the "Metrics & Communities" tab shows a table with columns for node name, degree, betweenness, closeness, and PageRank. Clicking a column header sorts the table. Clicking a row centers the graph on that node.
result: pass

### 7. Community Summary Cards
expected: The Metrics & Communities tab also shows community summary cards with distinct color swatches, member counts, and top members listed. Clicking a card filters the graph to that community.
result: pass

### 8. Export Downloads
expected: "Download CSV" produces a valid CSV file with network metrics. "Download JSON" produces a valid JSON file with graph data. Both files download with meaningful filenames.
result: issue
reported: "does not work: No network metrics data available for download"
severity: major

### 9. Cross-Page Navigation
expected: The Snapshot (latest data) and Trends pages both have a navigation link to the /network page.
result: pass

## Summary

total: 9
passed: 8
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Download CSV produces a valid CSV file with network metrics. Download JSON produces a valid JSON file with graph data."
  status: failed
  reason: "User reported: does not work: No network metrics data available for download"
  severity: major
  test: 8
  root_cause: "get_or_compute_network() only populates _plot_data_cache['network_metrics'] on first call. Subsequent calls return from _network_cache (no TTL) without re-populating _plot_data_cache (TTL=1800s). After 30 min, the CSV export cache entry expires and download returns 404."
  artifacts:
    - path: "plots/network.py"
      issue: "Cache-hit fast path (lines 322-324) skips _plot_data_cache re-population"
    - path: "app.py"
      issue: "Download endpoint depends on _plot_data_cache which expires independently"
  missing:
    - "Re-populate _plot_data_cache['network_metrics'] from _network_cache when cache-hit path is taken and entry has expired"
  debug_session: ".planning/debug/network-export-download.md"
