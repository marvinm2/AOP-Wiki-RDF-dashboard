---
phase: 04-entity-exploration
verified: 2026-02-23T10:15:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 4: Entity Exploration Verification Report

**Phase Goal:** The dashboard is enriched with new plot variations (breakdowns by biological level, taxonomy, OECD status; KE reuse; data quality insights) and raw data tables beneath all plots, with shareable URLs for version state
**Verified:** 2026-02-23T10:15:00Z
**Status:** PASSED
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | User can toggle a raw data table beneath each plot showing the underlying data (existing + new plots) | VERIFIED | `data_table_toggle` macro applied to all 18 plots in latest.html (1:1 match), 22 toggles in trends.html covering all 37 plot divs (1 toggle per dual-view pair by design). `raw-data-tables.js` fetches `/api/plot-data/<name>` on click. |
| 2 | URL encodes version selection via ?version= parameter for shareable views | VERIFIED | `version-selector.js` has `updateURLState()` using `URLSearchParams` + `history.replaceState()` at line 40-48, `restoreVersionFromURL()` at line 54, and `version-changed` event dispatch at line 186-187. |
| 3 | 9 new plots render on the dashboard: biological level, taxonomy, OECD status, KE reuse (2 views), annotation heatmap, ontology diversity, curation progress, ontology term growth | VERIFIED | All 9 functions exist in source (7 in `plots/latest_plots.py`, 2 in `plots/trends_plots.py`). All registered in `app.py` plot_map. All have lazy-plot containers in templates. All pass `from plots import ...` test. |
| 4 | Entity links in KE reuse plot open corresponding AOP-Wiki source pages | VERIFIED | `plot_latest_ke_reuse()` populates `wiki_url` column with `https://aopwiki.org/events/{ke_id}` and uses `custom_data=['wiki_url']` in Plotly. MutationObserver click handler in `templates/latest.html` (lines 559-588) opens `customdata[0]` URL in new tab on `plotly_click`. |
| 5 | All new plots have methodology notes, CSV downloads, lazy loading, and version selector support | VERIFIED | All 9 keys present in `methodology_notes.json` (37 total entries). Generic `/download/latest/<plot_name>` route covers 7 latest plots; `/download/trend/<plot_name>` covers trend plots. All 7 latest plots in `versionedPlots` array in `version-selector.js`. All plots have `lazy-plot` skeleton containers in templates. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `static/js/raw-data-tables.js` | Toggle component that fetches data and builds HTML tables, min 30 lines | VERIFIED | 84 lines. Contains `toggleDataTable()`, `buildDataTable()`, `escapeHtml()`, `version-changed` event listener. |
| `templates/macros/data_table.html` | Reusable Jinja2 macro with `data_table_toggle` | VERIFIED | 8 lines. Macro `data_table_toggle(plot_name)` defined with button calling `toggleDataTable('...')` and hidden content div. |
| `app.py` (plan 01) | `/api/plot-data/<plot_name>` endpoint | VERIFIED | Route at line 1494, function `api_plot_data()`. Returns JSON with `columns`, `rows`, `total_rows`, `truncated`, `success`. Handles 100-row truncation, NaN filling, version-aware cache lookup. |
| `plots/latest_plots.py` | 5 new plot functions from plan 02, 2 from plan 03 - min 250 lines additional | VERIFIED | 7 new functions found at lines 1439, 1549, 1651, 1800, 1911, 2389, 2603. All have SPARQL queries, DataFrame caching, fallback handling, brand styling. File substantially over min_lines. |
| `plots/trends_plots.py` | 2 new functions: `plot_curation_progress`, `plot_ontology_term_growth` | VERIFIED | Both found at lines 3419 and 3618. Both use ThreadPoolExecutor per-version parallel queries, return `(absolute_html, delta_html, DataFrame)` tuples. |
| `templates/latest.html` | 7 new plot containers (5 from plan 02, 2 from plan 03) with lazy-loading, download, methodology, data table toggles | VERIFIED | All 7 new lazy-plot divs present (lines 238, 259, 280, 308, 329, 358, 380). Each has download dropdown, methodology_note, and data_table_toggle calls. |
| `templates/trends.html` | New plot containers for curation progress and ontology term growth (4 divs for 2 trend x 2 variants) | VERIFIED | 4 lazy-plot divs at lines 818, 823, 859, 864. Each pair has download dropdowns, methodology_note, and one data_table_toggle. |
| `static/data/methodology_notes.json` | 9 new methodology entries for all new plots | VERIFIED | Keys confirmed present: `latest_ke_by_bio_level`, `latest_taxonomic_groups`, `latest_entity_by_oecd_status`, `latest_ke_reuse`, `latest_ke_reuse_distribution`, `latest_annotation_heatmap`, `latest_ontology_diversity`, `curation_progress`, `ontology_term_growth`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `static/js/raw-data-tables.js` | `/api/plot-data/<plot_name>` | `fetch()` call on button click | WIRED | `var url = '/api/plot-data/' + plotName ...` + `var response = await fetch(url)` at lines 26-27. Response parsed and rendered into table. |
| `templates/macros/data_table.html` | `static/js/raw-data-tables.js` | `onclick="toggleDataTable('...')"` | WIRED | Button calls `toggleDataTable(plot_name)` which is defined in `raw-data-tables.js` loaded in templates. |
| `static/js/version-selector.js` | `window.location.search` | `URLSearchParams` read/write | WIRED | `updateURLState()` uses `new URLSearchParams(window.location.search)` and `history.replaceState()`. `restoreVersionFromURL()` reads `params.get('version')`. |
| `plots/latest_plots.py` | `_plot_data_cache` | DataFrame caching for all 7 new functions | WIRED | All 7 functions call `_plot_data_cache[f'latest_{name}_{version_key}'] = df`. |
| `templates/latest.html` | `static/js/version-selector.js` | new plot names in versionedPlots array | WIRED | All 7 new latest plot names found in `versionedPlots` array at lines 28-34. |
| `plots/latest_plots.py` | AOP-Wiki entity links | `custom_data=['wiki_url']` in Plotly for KE reuse | WIRED | `wiki_url = f"https://aopwiki.org/events/{ke_id}"` in `plot_latest_ke_reuse()` at line 1862, passed as `custom_data` at line 1886. |
| `plots/trends_plots.py` | `app.py compute_plots_parallel()` | `plot_tasks` list registration | WIRED | Lines 208-209: `('curation_progress', lambda: safe_plot_execution(plot_curation_progress))` and `('ontology_term_growth', ...)`. |
| `app.py` | `templates/trends.html` | `plot_map` dict for lazy loading | WIRED | Lines 1608-1611: `curation_progress_absolute`, `curation_progress_delta`, `ontology_term_growth_absolute`, `ontology_term_growth_delta` in plot_map. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXPL-04 | 04-02-PLAN, 04-03-PLAN | Entity names and IDs link directly to AOP-Wiki source pages | SATISFIED | KE reuse plot embeds `wiki_url` via Plotly `custom_data`; MutationObserver handler opens URL on click. REQUIREMENTS.md marks as Complete. |
| EXPL-05 | 04-01-PLAN | User can toggle raw data table beneath each plot | SATISFIED | Toggle macro applied to all plots in both templates. `/api/plot-data/` endpoint serves JSON. `raw-data-tables.js` builds HTML tables. REQUIREMENTS.md marks as Complete. |
| EXPL-06 | 04-01-PLAN | URL encodes version and active plot state for shareable views | SATISFIED | `?version=` parameter written on change and read on load via `URLSearchParams` + `history.replaceState()`. REQUIREMENTS.md marks as Complete. |

No orphaned requirements found. All three Phase 4 requirements (EXPL-04, EXPL-05, EXPL-06) are claimed and verified.

### Anti-Patterns Found

No blocker or warning anti-patterns detected in new files.

- No TODO/FIXME/PLACEHOLDER comments in `raw-data-tables.js`, `data_table.html`, or new plot functions
- No stub return values (`return null`, `return {}`, `return []`) in new plot functions — all have real SPARQL queries and fallback to `create_fallback_plot()` on empty results
- No console.log-only handlers
- No forms with only `preventDefault()`

### Human Verification Required

The following items cannot be verified programmatically and require a live application session:

#### 1. Raw Data Table Toggle Functionality

**Test:** Open the dashboard at `/latest`, click "Show Raw Data" beneath the "Entity Counts" plot, then a new plot like "Key Events by Biological Level of Organization."
**Expected:** A table appears with column headers and data rows. Click "Hide Raw Data" — table disappears. Click again — table appears instantly without a new network request. Change version — all open tables reset.
**Why human:** Requires a running SPARQL endpoint; can't test fetch response or cache-hit behavior programmatically.

#### 2. Shareable URL Round-Trip

**Test:** Open `/latest`, select a historical version from the dropdown. Copy the URL (should contain `?version=YYYY-MM-DD`). Paste URL in a new browser tab.
**Expected:** The new tab loads with that version pre-selected in the dropdown and all plots showing data for that version.
**Why human:** Requires a running application and a live browser session to verify URL round-trip state.

#### 3. KE Reuse Entity Links

**Test:** Navigate to the "Most Reused Key Events Across AOPs" chart. Click on any bar.
**Expected:** A new browser tab opens to `https://aopwiki.org/events/<id>` for the clicked KE.
**Why human:** Plotly click events cannot be tested without a rendered browser; MutationObserver behavior requires actual lazy-load completion.

#### 4. All 9 New Plots Render Visually

**Test:** Navigate to `/latest` and `/trends`. Scroll through all sections including "Breakdown Analysis," "KE Reuse Analysis," "Data Quality," and "Curation & Ontology Trends."
**Expected:** All 9 new plots render with data (not empty fallbacks), use VHP4Safety brand colors, have readable labels, and respond to version selector changes.
**Why human:** Requires live SPARQL endpoint connection; rendered output depends on actual data in the triplestore.

### Gaps Summary

No gaps. All automated checks pass across all three levels (exists, substantive, wired) for all artifacts in all three plans.

The 37-plot-div vs 22-toggle discrepancy in trends.html is intentional by design: dual-view plots (absolute + delta) share one underlying dataset, so one data_table_toggle per pair is architecturally correct and consistent with the plan decision "Used absolute plot name as data source for dual-view plot-boxes."

---

_Verified: 2026-02-23T10:15:00Z_
_Verifier: Claude (gsd-verifier)_
