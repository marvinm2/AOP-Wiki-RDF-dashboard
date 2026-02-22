---
phase: 02-reliability-and-completeness
verified: 2026-02-22T11:45:00Z
status: passed
score: 11/11 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 7/11
  gaps_closed:
    - "OECD completeness trend plot title and legend no longer overlap (t=120, y=1.15)"
    - "Composite AOP Completeness Distribution boxplot shows data from all available versions including post-2020 via per-version parallel queries"
    - "AOPs Modified Over Time and AOP Creation vs Modification plot boxes each have methodology notes (trends.html now has 19 calls)"
    - "Methodology note limitations contain only researcher-relevant information — performance text removed from all 28 entries"
    - "KE component annotation plots use per-version parallel queries that circumvent Virtuoso execution time limits"
  gaps_remaining: []
  regressions: []
---

# Phase 2: Reliability and Completeness — Re-Verification Report

**Phase Goal:** Every visualization on the dashboard loads reliably, exports in all formats, and explains its methodology to users
**Verified:** 2026-02-22
**Status:** PASSED — all five gaps from initial verification are closed
**Re-verification:** Yes — after gap closure plans 02-05 and 02-06

## Goal Achievement

All five gaps identified in the 2026-02-21 verification have been closed by plans 02-05 and 02-06. The two blocker gaps (KE component plots broken, boxplot incomplete data) are resolved by the per-version parallel SPARQL query pattern. The three warning gaps (OECD layout, missing methodology notes, stale limitations text) are resolved by targeted layout changes, template additions, and JSON cleanup. All 11 observable truths now verify at code level.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | AOP Completeness Distribution boxplot loads instantly (pre-computed at startup) | VERIFIED | `plot_results.get('aop_completeness_boxplot')` line 265 app.py; plot_map uses pre-computed variable at line 1460 |
| 2 | OECD completeness trend is visible on the trends page as a line chart | VERIFIED | `plot_oecd_completeness_trend` defined at line 2894 trends_plots.py; registered in app.py startup line 182; exported from `__init__.py` |
| 3 | OECD completeness trend plot renders clearly without title/legend overlap | VERIFIED | `margin=dict(l=50, r=20, t=120, b=50)` at line 3065; `legend=dict(..., y=1.15, xanchor="center", x=0.5)` at line 3068; commit 5c84ae2 |
| 4 | Composite AOP Completeness Distribution shows data across all available versions | VERIFIED | `_query_boxplot_version()` helper at line 2411; `ThreadPoolExecutor(max_workers=4)` at line 2659 queries all versions individually; commit 09400b0 |
| 5 | All 20 previously-missing data cache entries now exist after startup computation | VERIFIED | Regression check: `_plot_data_cache` keys confirmed present; 104 `create_fallback_plot` uses as safety net |
| 6 | Download filenames include the export date and version for traceability | VERIFIED | Regression check: `build_export_filename` used in 41 download route Content-Disposition headers in app.py |
| 7 | Every plot on the latest page has an expandable methodology note | VERIFIED | Regression check: 11 methodology_note() calls on 11 plot containers in latest.html (line 13 is an import, not a call) |
| 8 | Every plot on the trends page has an expandable methodology note | VERIFIED | 19 calls in trends.html confirmed; AOPs Modified Over Time at line 493 and AOP Creation vs Modification at line 515 now have notes; commit 5c84ae2 |
| 9 | Methodology notes contain accurate, researcher-appropriate content | VERIFIED | "30-75 seconds" text not present in methodology_notes.json; Python check confirms no performance/implementation text in any of the 28 entries; commit 250a8f3 |
| 10 | KE component annotation plots all load without errors | VERIFIED | `_query_ke_components_version()` helper at line 719 uses per-graph targeting (`GRAPH <{graph_uri}>`); `ThreadPoolExecutor(max_workers=4)` at line 780; root cause was Virtuoso 400s execution time limit on triple-OPTIONAL cross-product, not wrong predicates; commit cb3ff28 |
| 11 | kaleido is listed in requirements.txt so PNG/SVG export works in Docker | VERIFIED | Regression check: `kaleido~=0.2` present in requirements.txt |

**Score:** 11/11 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `requirements.txt` | kaleido dependency | VERIFIED | `kaleido~=0.2` present |
| `plots/trends_plots.py` | `plot_oecd_completeness_trend` function | VERIFIED | Defined at line 2894 |
| `plots/trends_plots.py` | `_query_ke_components_version()` per-version helper | VERIFIED | Defined at line 719; used by plot_ke_components, plot_ke_components_percentage, plot_unique_ke_components |
| `plots/trends_plots.py` | `_query_boxplot_version()` per-version helper | VERIFIED | Defined at line 2411; called via ThreadPoolExecutor at line 2659 |
| `plots/trends_plots.py` | OECD plot layout with adequate spacing | VERIFIED | `t=120` at line 3065; `y=1.15` at line 3072 |
| `plots/trends_plots.py` | Entity completeness plot layout fixed | VERIFIED | `t=100` at line 2397; `y=1.10` at line 2400 |
| `plots/trends_plots.py` | `_plot_data_cache` for all trend functions | VERIFIED | Regression: 104 uses of `create_fallback_plot` confirms safety net intact |
| `plots/shared.py` | `build_export_filename` helper function | VERIFIED | Regression: used in 41 download Content-Disposition headers in app.py |
| `app.py` | Boxplot pre-computed at startup, plot_map uses variable | VERIFIED | Regression: line 265 extracts result; line 1460 uses pre-computed variable |
| `app.py` | OECD trend integration | VERIFIED | Regression: exported from `__init__.py`, registered in startup, passed to plot_map |
| `app.py` | `build_export_filename` used in download routes | VERIFIED | Regression: 41 headers confirmed |
| `app.py` | methodology_notes loaded and passed to templates | VERIFIED | Regression: loaded at lines 67-69; passed to latest.html at line 1601, trends_page.html at line 1617 |
| `config.py` | `SPARQL_SLOW_TIMEOUT` and `PLOT_TIMEOUT=120` | VERIFIED | Regression: both confirmed in config.py |
| `static/data/methodology_notes.json` | Structured notes for all 28 plots, researcher-appropriate | VERIFIED | 28 entries, valid JSON; no performance/implementation text in any limitations or data_source fields |
| `templates/macros/methodology.html` | Jinja2 macro | VERIFIED | Regression: `{% macro methodology_note(notes, plot_key) %}` at line 1 |
| `static/css/main.css` | Methodology note styling | VERIFIED | Regression: `.methodology-note` class present |
| `templates/latest.html` | Methodology notes on all 11 plots | VERIFIED | Regression: 11 calls confirmed |
| `templates/trends.html` | Methodology notes on all trends plots | VERIFIED | 19 calls; aops_modified_over_time (line 493) and aop_creation_vs_modification_timeline (line 515) now have notes |
| `docs/virtuoso-tuning.md` | Virtuoso tuning documentation | VERIFIED | Regression: file present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `plots/trends_plots.py` | `plots/__init__.py` | export of `plot_oecd_completeness_trend` | VERIFIED | Regression: found in `__init__.py` |
| `app.py` | `plots/__init__.py` | import of `plot_oecd_completeness_trend` | VERIFIED | Regression: line 87 app.py |
| `app.py` | `compute_plots_parallel` | boxplot result extracted (not skipped) | VERIFIED | Regression: line 265 uses `plot_results.get('aop_completeness_boxplot')` |
| `plots/trends_plots.py` | SPARQL endpoint | per-version boxplot queries via `_query_boxplot_version` | VERIFIED | ThreadPoolExecutor submits per-graph queries; avoids MaxResultRows truncation |
| `plots/trends_plots.py` | SPARQL endpoint | per-version KE queries via `_query_ke_components_version` | VERIFIED | Direct `GRAPH <{graph_uri}>` targeting bypasses Virtuoso cross-graph execution limit |
| `plots/trends_plots.py` | `_plot_data_cache` | CSV export wiring | VERIFIED | Regression: all expected keys populated |
| `app.py` | `plots/shared.py` | import and use of `build_export_filename` | VERIFIED | Regression: used in 41 headers |
| `templates/latest.html` | `methodology_notes.json` | Flask template context + macro | VERIFIED | Regression: macro imported, `methodology_notes` passed from app.py |
| `templates/trends.html` | `methodology_notes.json` | Flask template context + macro | VERIFIED | 19 calls; was 17; both missing AOP Lifetime boxes now wired |
| `plots/trends_plots.py` | `plots/shared.py` | `create_fallback_plot` in error handlers | VERIFIED | Regression: 104 uses confirmed |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| RELY-01 | 02-01, 02-04 | AOP completeness distribution loads in under 30 seconds | SATISFIED | Pre-computed at startup; zero user wait time; REQUIREMENTS.md marked complete |
| RELY-02 | 02-01, 02-04, 02-05 | AOP completeness by OECD status visualized without hitting Virtuoso limits; full version history in boxplot | SATISFIED | OECD trend plot renders; boxplot uses per-version parallel queries for all 30 versions (was truncated to ~7 pre-2020); REQUIREMENTS.md marked complete |
| RELY-04 | 02-02, 02-04 | Every displayed visualization has CSV, PNG, and SVG export | SATISFIED | 41 download routes with `build_export_filename`; kaleido in requirements.txt; REQUIREMENTS.md marked complete |
| EXPL-07 | 02-03, 02-04, 02-06 | Each plot has an expandable methodology note | SATISFIED | 11 notes in latest.html; 19 notes in trends.html (was 17); all 28 JSON entries researcher-appropriate; REQUIREMENTS.md marked complete |

No orphaned requirements — all Phase 2 requirement IDs (RELY-01, RELY-02, RELY-04, EXPL-07) are claimed by plans and verified above.

### Anti-Patterns Found

None. The previous blocker anti-patterns have been resolved:
- SPARQL predicate issue (KE components): resolved by per-version parallel queries
- Virtuoso truncation (boxplot): resolved by per-version parallel queries
- Missing methodology notes (trends.html): resolved; 19 calls now present
- Stale performance text (methodology_notes.json): resolved; no performance/implementation text remains

### Human Verification Recommended

The following items confirmed by code analysis but benefit from visual/live confirmation:

#### 1. OECD Plot Visual Separation

**Test:** Navigate to /trends, scroll to "OECD Completeness Trend". Inspect title and horizontal legend for visual separation.
**Expected:** "Mean AOP Completeness by OECD Status Over Time" title sits below the horizontal legend with clear whitespace; no text overlap.
**Code evidence:** `t=120` top margin, legend at `y=1.15` centered — substantially larger than the original `t=50, y=1.02` that caused overlap.

#### 2. KE Component Plots Rendering with Live Data

**Test:** Start the application and navigate to /trends, scroll to "KE Component Annotations" section.
**Expected:** Three plot sections each show rendered Plotly charts (not "Data Unavailable" error cards).
**Code evidence:** Per-version parallel queries using correct `aopo:hasBiologicalEvent` intermediate pattern (confirmed 27,568 results in diagnostic); direct `GRAPH <uri>` targeting eliminates Virtuoso 400s execution time limit.

#### 3. Boxplot Version Coverage

**Test:** Navigate to /trends, scroll to "Composite AOP Completeness Distribution". Check x-axis includes versions from 2020 onward.
**Expected:** Labels include 2020, 2021, 2022, 2023, 2024, and 2025 versions.
**Code evidence:** Per-version parallel queries confirmed in code; commit message states "All 30 versions now included (was ~7 pre-2020 only)".

### Closure Summary

All five gaps from the 2026-02-21 verification are closed:

**CLOSED — KE component plots (Blocker):** Root cause diagnosed as Virtuoso estimated execution time (864s) exceeding 400s limit for triple-OPTIONAL cross-product queries. The `aopo:hasBiologicalEvent` predicate chain IS correct (27,568 results confirmed). Fixed with `_query_ke_components_version()` helper querying each version's graph directly via `GRAPH <{graph_uri}>`. Commit cb3ff28.

**CLOSED — Boxplot incomplete data (Blocker):** Combined UNION query across all graphs and entity types exceeded Virtuoso MaxResultRows and silently truncated post-2020 versions. Fixed with `_query_boxplot_version()` per-version helper and `ThreadPoolExecutor(max_workers=4)`. All 30 versions included, execution ~8 seconds. Commit 09400b0.

**CLOSED — OECD plot layout defect (Warning):** Top margin increased to 120px, legend repositioned to `y=1.15` centered. Entity completeness trends plot also fixed (`t=100, y=1.10`). Commit 5c84ae2.

**CLOSED — Missing methodology notes (Warning):** `{{ methodology_note(methodology_notes, 'aop_lifetime') }}` added to AOPs Modified Over Time (trends.html line 493) and AOP Creation vs Modification (line 515). Total calls now 19. Commit 5c84ae2.

**CLOSED — Stale methodology limitations text (Warning):** "Query may take 30-75 seconds on large datasets" removed from `aop_completeness_boxplot`. All 28 entries cleaned of performance/implementation text. All limitations remain non-empty with researcher-relevant caveats. Commit 250a8f3.

---

_Verified: 2026-02-22_
_Verifier: Claude (gsd-verifier)_
_Mode: Re-verification after gap closure plans 02-05 and 02-06_
