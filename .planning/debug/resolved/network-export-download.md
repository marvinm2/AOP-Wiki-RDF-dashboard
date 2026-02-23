---
status: diagnosed
trigger: "Network metrics CSV download returns 'No network metrics data available for download' instead of producing a CSV file."
created: 2026-02-23T00:00:00Z
updated: 2026-02-23T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED - TTL expiration of _plot_data_cache['network_metrics'] combined with get_or_compute_network() not re-populating the cache on subsequent calls
test: Traced full data flow from download endpoint through cache layers
expecting: Cache miss after TTL expiration (30 min) because re-populate path is skipped
next_action: Document root cause and suggest fix

## Symptoms

expected: "Download CSV" produces a valid CSV file with network metrics (degree, betweenness, closeness, PageRank for each node)
actual: Returns "No network metrics data available for download"
errors: "No network metrics data available for download" (HTTP 404)
reproduction: Visit /network, wait for graph to render, click "Download CSV" (especially after >30 min)
started: Unknown - user reported it does not work

## Eliminated

- hypothesis: Cache key mismatch between store and lookup
  evidence: Both use the exact key 'network_metrics' - store in network.py:360, lookup in app.py:1693
  timestamp: 2026-02-23

- hypothesis: Network computation fails silently
  evidence: If get_or_compute_network() raised, the except block returns jsonify({'error': ...}), not "No network metrics data available for download"
  timestamp: 2026-02-23

- hypothesis: _plot_data_cache eviction by version cap
  evidence: Key 'network_metrics' rsplit on '_' gives 'metrics' which has no '-', so it is not counted as a version in _evict_if_needed()
  timestamp: 2026-02-23

## Evidence

- timestamp: 2026-02-23
  checked: /download/network/metrics route (app.py:1686-1703)
  found: Route calls get_or_compute_network() then get_csv_with_metadata('network_metrics'). If csv_data is None/falsy, returns 404 with "No network metrics data available for download"
  implication: The 404 message means get_csv_with_metadata returned None, meaning 'network_metrics' was not found in _plot_data_cache

- timestamp: 2026-02-23
  checked: get_or_compute_network() in plots/network.py:293-371
  found: On FIRST call, builds graph, computes metrics, populates _network_cache AND _plot_data_cache['network_metrics'] (line 360). On SUBSEQUENT calls, returns immediately from _network_cache (line 322-324) WITHOUT touching _plot_data_cache
  implication: _plot_data_cache['network_metrics'] is only populated once, on first computation

- timestamp: 2026-02-23
  checked: VersionedPlotCache (shared.py:71-174)
  found: TTL is 1800 seconds (30 minutes). _is_pinned() checks if self._pinned_prefix is contained in the key string. Pinned prefix is the latest version (e.g., "2025-07-01"). Key "network_metrics" does NOT contain any version string, so _is_pinned returns False. Therefore the entry is subject to TTL expiration.
  implication: After 30 minutes, __contains__ check in get_csv_with_metadata will find the entry expired, delete it, and return False

- timestamp: 2026-02-23
  checked: Cache re-population path
  found: When _plot_data_cache['network_metrics'] expires after TTL, the download endpoint calls get_or_compute_network() which finds _network_cache populated (plain dict, never expires) and returns immediately. The _plot_data_cache entry is NEVER re-populated.
  implication: This is a permanent failure after 30 minutes - the cache entry is gone forever and cannot be restored without an app restart

- timestamp: 2026-02-23
  checked: JSON export endpoint (/download/network/graph, app.py:1706-1726)
  found: This endpoint does NOT use _plot_data_cache at all - it reads directly from get_or_compute_network() result dict. Therefore JSON export is NOT affected by this bug.
  implication: Only CSV export is broken; JSON export works correctly

## Resolution

root_cause: |
  Two-layer cache inconsistency causes permanent CSV export failure after 30 minutes.

  1. `_plot_data_cache` (VersionedPlotCache) has a TTL of 1800 seconds (30 min).
     The key 'network_metrics' does not contain a version string, so it is NOT pinned
     and IS subject to TTL expiration.

  2. `_network_cache` (plain dict) has no TTL and persists forever.

  3. `get_or_compute_network()` only populates `_plot_data_cache['network_metrics']`
     on its FIRST call (network.py:360). On subsequent calls, it short-circuits via
     `_network_cache` (network.py:322-324) and never touches `_plot_data_cache` again.

  4. After 30 minutes, `_plot_data_cache['network_metrics']` expires. The download
     endpoint calls `get_or_compute_network()` which returns from `_network_cache`
     without re-populating `_plot_data_cache`. `get_csv_with_metadata()` finds no
     data and returns None, producing the 404 error.

  The metrics DataFrame is available in `_network_cache` (via the 'metrics' key as
  a list of dicts) but is never refreshed in `_plot_data_cache` after initial computation.

fix: |
  Two changes needed in plots/network.py get_or_compute_network():

  Option A (Recommended): Re-populate _plot_data_cache on every call, not just the first.
  Move the _plot_data_cache['network_metrics'] assignment into the cache-hit path too:

  ```python
  if _network_cache:
      logger.info("Returning cached network data")
      # Ensure CSV export cache stays populated (survives TTL expiration)
      if 'network_metrics' not in _plot_data_cache:
          _plot_data_cache['network_metrics'] = pd.DataFrame(_network_cache['metrics'])
      return _network_cache
  ```

  Option B (Alternative): Change the download endpoint to bypass _plot_data_cache
  and build CSV directly from _network_cache['metrics']. This avoids the cache
  indirection entirely.

  Option C (Belt-and-suspenders): Do both A and B.

verification:
files_changed: []
