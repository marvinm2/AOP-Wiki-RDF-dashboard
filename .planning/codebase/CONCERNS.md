# Codebase Concerns

**Analysis Date:** 2026-02-20

## Tech Debt

**Duplicate/Legacy Code - plots.py:**
- Issue: `plots.py` (4,194 lines) is a legacy monolithic file that duplicates the refactored plots package functionality
- Files: `plots.py` vs `plots/trends_plots.py` (2,888 lines), `plots/latest_plots.py` (1,774 lines), `plots/shared.py` (1,023 lines)
- Impact: Code duplication creates maintenance burden and inconsistent implementations. Changes must be made in two places. Total package = 5,685 lines vs monolithic 4,194 lines redundantly
- Fix approach: Remove or deprecate `plots.py` entirely. Verify all imports have been migrated to the modular `plots/` package structure. Update any remaining direct imports from `plots.py` to use the package instead

**Duplicate Import Statements - app.py:**
- Issue: Lines 41 and 63 both import from flask, creating redundant imports
- Files: `app.py` lines 41 and 63
- Impact: Code style violation, minor memory overhead from duplicate imports
- Fix approach: Consolidate all Flask imports into a single import statement at the top of the file

**Broad Exception Handling Throughout Codebase:**
- Issue: Widespread use of `except Exception as e:` without specific exception type filtering
- Files: `app.py` (20+ locations), `plots/trends_plots.py` (5+ locations), `plots/shared.py` (5+ locations), `plots.py` (5+ locations)
- Impact: Makes error diagnosis difficult, masks different failure modes that should be handled differently. Network errors, syntax errors, and timeouts are all treated identically
- Fix approach: Replace broad `Exception` catches with specific exception types (SPARQLExceptions, requests.RequestException, TimeoutError, etc.) appropriate to the operation. Use narrow exception handling for specific known errors

## Known Bugs

**Removed SPARQL Query - plot_aop_completeness_boxplot_by_status:**
- Symptoms: Plot function removed from dashboard, all references commented out, dashboard incomplete for OECD status analysis
- Files: `app.py` line 88 (commented import), line 182 (commented code), line 256 (commented variable); `plots/__init__.py` line 163 (commented), line 235 (commented), line 279 (commented)
- Trigger: Any request for OECD status completeness analysis will fail with 404 since the plot endpoint doesn't exist
- Workaround: Use the OECD status filtering in other plots or view data via CSV downloads
- Root Cause: "Hits Virtuoso execution limits" (CLAUDE.md line 24) - SPARQL query times out with network query complexity. Virtuoso triplestore has resource constraints that prevent this specific complex aggregation

**Long Load Times - Composite AOP Completeness Distribution:**
- Symptoms: Dashboard page load takes ~75 seconds despite optimization (CLAUDE.md line 23)
- Files: `plots/trends_plots.py` (the `plot_entity_completeness_trends()` function)
- Trigger: Initial page load or refresh, any version selector change
- Impact: Poor user experience, potential timeout issues in CI/CD pipelines or monitoring
- Workaround: Lazy loading (implemented) reduces perceived latency to ~50ms for initial page, but full computation still takes 75 seconds in background
- Root Cause: Even after 60% query reduction (from 10 to 4 SPARQL queries) and 95% data transfer reduction, query complexity is still high for Virtuoso endpoint

## Security Considerations

**No Input Validation on plot_name Parameter:**
- Risk: Potential path traversal or injection attacks via the plot_name parameter in `/api/plot/<plot_name>` endpoint
- Files: `app.py` line 1429-1492 (the `/api/plot/<plot_name>` endpoint)
- Current mitigation: Implicit whitelist via hardcoded `plot_map` dictionary lookup prevents unknown plots
- Recommendations: Add explicit validation that plot_name is in the allowed set before processing; sanitize the plot_name parameter; document that only defined plots are allowed

**No CORS Headers Defined:**
- Risk: If dashboard is deployed behind a different domain, cross-origin requests from other applications will be blocked by browser CORS policy
- Files: `app.py` (missing CORS configuration)
- Current mitigation: Likely running on same domain during development/deployment
- Recommendations: Add Flask-CORS configuration with appropriate allowed origins if multi-domain deployment is planned

**Debug Mode Configuration:**
- Risk: If `FLASK_DEBUG=true` is accidentally set in production environment, Flask debug page exposes full source code and stack traces
- Files: `config.py` line 93 (FLASK_DEBUG configuration)
- Current mitigation: Defaults to False, documentation warns against enabling in production (docs/source/configuration.rst line 150)
- Recommendations: Add environment-based override to force FLASK_DEBUG=False in production regardless of env var; add startup warning if debug mode detected in non-development environment

## Performance Bottlenecks

**SPARQL Query Complexity - Network Analysis Queries:**
- Problem: Virtuoso endpoint times out on complex network queries, forcing removal of features
- Files: `plots/trends_plots.py` (commented network queries), `app.py` (commented boxplot_by_status)
- Cause: Multi-way JOINs across large graphs with GROUP_CONCAT aggregations exceed Virtuoso's execution time limits. The removed `plot_aop_completeness_boxplot_by_status` likely involves cross-product queries on all versions
- Improvement path:
  1. Break single complex query into smaller federated queries (query once per version)
  2. Cache results and do cross-version aggregation in Python instead of SPARQL
  3. Use graph snapshots with pre-aggregated data if available
  4. Investigate Virtuoso query plan optimization or contact endpoint administrators for performance tuning

**Global Cache Not Garbage-Collected:**
- Problem: `_plot_data_cache` and `_plot_figure_cache` grow indefinitely and are never cleared
- Files: `plots/shared.py` line 130-131 (cache definition), all plot functions that add to caches
- Cause: No eviction policy, cache clearing mechanism, or memory monitoring
- Improvement path:
  1. Implement cache size limits with LRU eviction when threshold exceeded
  2. Add periodic cleanup routine (e.g., clear caches older than 24 hours)
  3. Monitor memory usage and add warnings when cache exceeds threshold
  4. Consider using functools.lru_cache with maxsize for automatic management

**Parallel Plot Generation Bounded by Slowest Plot:**
- Problem: `compute_plots_parallel()` uses ThreadPoolExecutor but slowest plot (75 seconds) determines startup time
- Files: `app.py` line 115-220
- Cause: All 22 plots must complete before dashboard becomes available. If one takes 75 seconds, that's the minimum startup time regardless of parallelism
- Improvement path:
  1. Move plot generation to async background tasks (implement startup without waiting for plots)
  2. Lazy load all non-critical plots only when requested
  3. Pre-generate highest-priority plots only during startup
  4. Implement incremental startup progress reporting to user

## Fragile Areas

**SPARQL Query String Concatenation:**
- Files: `plots/latest_plots.py` (lines 183-230+), `plots/trends_plots.py` (multiple query builders)
- Why fragile: Queries are built by string concatenation with f-strings and .format(). No parameterized queries means SQL/SPARQL injection is theoretically possible if user input reaches query builders (currently unlikely but risky pattern)
- Safe modification: Use parameterized queries if SPARQLWrapper supports them; validate all version/filter parameters before string interpolation; add unit tests that verify queries don't contain unexpected syntax
- Test coverage: No visible unit tests for SPARQL query generation. Queries only tested implicitly through plot generation

**Version Selector JavaScript Integration:**
- Files: `static/js/version-selector.js`, `templates/trends.html`, `templates/latest.html`
- Why fragile: JavaScript arrays (versionedPlots) in `version-selector.js` must exactly match backend plot names. Manual synchronization required when adding new plots
- Safe modification:
  1. Generate versionedPlots array from Flask endpoint instead of hardcoding in JavaScript
  2. Add integration tests that verify JavaScript array matches available backend plots
  3. Add assertions in Flask that verify all versioned plots have corresponding lazy-load endpoints
- Test coverage: No tests verify JavaScript plot array matches backend availability

**CSV Export Data Cache Desynchronization:**
- Files: `plots/shared.py` (`_plot_data_cache`), all plot functions, `app.py` (download routes)
- Why fragile: Each plot manually adds its dataframe to the cache with hardcoded key names. If plot name changes or cache key is misspelled, download route will serve old/wrong data silently
- Safe modification:
  1. Create a decorator `@cache_plot_data(key_name)` that automatically manages cache entries
  2. Add assertions in download routes that verify requested key exists in cache before serving
  3. Return 404 if cache key missing rather than serving wrong data
- Test coverage: No tests verify that CSV downloads actually match displayed plots

## Scaling Limits

**Single SPARQL Endpoint Dependency:**
- Current capacity: All data queries go to single endpoint at Config.SPARQL_ENDPOINT
- Limit: If endpoint becomes unavailable or overloaded, entire dashboard becomes unavailable (health check only), no fallback. Dashboard degrades gracefully but cannot serve data
- Scaling path:
  1. Implement endpoint failover (multiple SPARQL_ENDPOINT URLs, try primary then secondary)
  2. Add read-only replica endpoints for load distribution
  3. Implement query caching layer (Redis) to handle endpoint outages
  4. Add bulk data export to static files updated nightly as fallback

**Memory Growth with Large Datasets:**
- Current capacity: Caches store entire DataFrames in memory for all plots. With many versions/entities this can grow unbounded
- Limit: With 10+ years of historical data and growing dataset, memory usage could exceed available RAM
- Scaling path:
  1. Implement DataFrame pagination or lazy-loading from cache
  2. Use SQLite in-memory database instead of DataFrames for large datasets
  3. Compress/serialize cached data when not actively used
  4. Monitor memory usage and add warnings at 70%/85% thresholds

**Parallel Worker Pool Fixed at Runtime:**
- Current capacity: PARALLEL_WORKERS fixed at 5 (default)
- Limit: Cannot dynamically adjust to system load or server capacity. Not suitable for containerized environments with variable resources
- Scaling path:
  1. Auto-detect CPU count and scale workers proportionally
  2. Implement dynamic worker scaling based on system load
  3. Use process pools instead of thread pools for CPU-bound query processing

## Dependencies at Risk

**SPARQLWrapper Version Not Pinned:**
- Risk: `requirements.txt` likely doesn't pin SPARQLWrapper version (typical for Flask projects). Breaking changes in newer versions could cause failures
- Impact: New installations might pull incompatible version with different timeout behavior or query syntax requirements
- Migration plan: Pin SPARQLWrapper to specific tested version in requirements.txt; add integration tests against that version

**Virtuoso-Specific Query Patterns:**
- Risk: Code contains Virtuoso-specific optimizations (GROUP_CONCAT, UNION query patterns mentioned in CLAUDE.md line 25). Switching to different triplestore would require query rewrites
- Impact: Vendor lock-in to Virtuoso. Migration to standard SPARQL-only patterns would lose performance optimizations
- Migration plan: Extract query patterns into a query builder class per SPARQL dialect; document which optimizations are Virtuoso-specific

## Missing Critical Features

**Network Analysis Expansion (#11):**
- Problem: Issue #11 identified need for centrality measures, clustering, and PageRank analysis
- Blocks: Cannot perform deep structural analysis of AOP networks, identify critical pathways
- Priority: MEDIUM (noted in CLAUDE.md)
- Status: Blocked by Virtuoso query performance limits - same root cause as removed boxplot_by_status

**VHP Platform Deployment (#3):**
- Problem: Dashboard exists but cannot be deployed to VHP platform yet
- Blocks: Cannot reach intended users, integration with other VHP tools not possible
- Priority: MEDIUM (noted in CLAUDE.md)
- Current barrier: Likely deployment configuration, authentication, or infrastructure requirements not yet finalized

## Test Coverage Gaps

**No Unit Tests for SPARQL Query Builders:**
- What's not tested: The complex SPARQL query construction in `plots/latest_plots.py` and `plots/trends_plots.py`
- Files: `plots/latest_plots.py` (lines 183-230+), `plots/trends_plots.py` (multiple query builders)
- Risk: Query syntax errors, version filtering errors, or aggregation bugs won't be caught until runtime. Changes to RDF schema could break queries without warning
- Priority: HIGH - queries are core functionality

**No Integration Tests Against Real SPARQL Endpoint:**
- What's not tested: All plot functions assume SPARQL endpoint returns valid data in expected format. No tests verify contracts with actual endpoint
- Files: All plot functions in `plots/` package
- Risk: Endpoint schema changes, data format changes, or version discovery bugs won't be caught until users report issues
- Priority: HIGH - endpoint is external dependency

**No End-to-End Tests for Version Selector:**
- What's not tested: Version selector JavaScript integration with backend endpoints
- Files: `static/js/version-selector.js`, `app.py` (version endpoints), templates
- Risk: Version selection could silently fail or show wrong data in browser without testing
- Priority: MEDIUM - critical user-facing feature

**No Tests for CSV Export Data Accuracy:**
- What's not tested: CSV downloads match displayed plots, metadata is correct, no data corruption during serialization
- Files: `plots/shared.py` (cache management), `app.py` (download routes)
- Risk: Users download CSV containing wrong/stale data without knowing it doesn't match displayed plot
- Priority: MEDIUM - data integrity issue

**No Performance/Load Tests:**
- What's not tested: Dashboard behavior under load, concurrent user requests, memory limits
- Files: Entire application
- Risk: Cannot detect performance regressions, identify scaling limits, or plan capacity
- Priority: MEDIUM - performance is known bottleneck

---

*Concerns audit: 2026-02-20*
