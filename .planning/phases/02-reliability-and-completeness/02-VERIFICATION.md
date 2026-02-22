---
phase: 02-reliability-and-completeness
verified: 2026-02-22T16:30:00Z
status: passed
score: 14/14 must-haves verified
re_verification:
  previous_status: passed
  previous_score: 11/11
  gaps_closed:
    - "OECD completeness trend plot legend changed to right-side vertical (orientation=v, x=1.02) — no title overlap possible"
    - "PNG/SVG export now works for all 3 completeness-by-status plots (aop/ke/ker) via startup cache population"
    - "All 17 trend methodology notes include data scope caveat about RDF release version coverage"
  gaps_remaining: []
  regressions: []
  note: "Previous VERIFICATION.md (2026-02-22T11:45:00Z) predated plan 02-07. This re-verification incorporates all 02-07 changes from commits f470edc and 3c6b351."
---

# Phase 2: Reliability and Completeness — Re-Verification Report

**Phase Goal:** Every visualization on the dashboard loads reliably, exports in all formats, and explains its methodology to users
**Verified:** 2026-02-22T16:30:00Z
**Status:** PASSED — all 14 observable truths verified, all 4 requirement IDs satisfied
**Re-verification:** Yes — after UAT gap closure plan 02-07 (commits f470edc, 3c6b351)

## Context: What Changed Since Previous Verification

The previous VERIFICATION.md (2026-02-22T11:45:00Z, score 11/11) was produced before plan 02-07 existed. UAT testing (02-UAT.md) revealed 3 additional issues: OECD legend overlap, incomplete PNG/SVG export for 3 plots, and missing data scope caveat in trend methodology notes. Plan 02-07 was created and executed, adding 3 new must-haves (from 02-07-PLAN.md frontmatter) to the verification scope. This report covers all 14 truths (11 prior + 3 from 02-07).

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | AOP Completeness Distribution boxplot loads instantly (pre-computed at startup) | VERIFIED | `('aop_completeness_boxplot', ...)` at app.py line 182; result extracted at line 268; plot_map uses pre-computed variable at line 1466 |
| 2 | OECD completeness trend is visible on the trends page as a line chart | VERIFIED | `plot_oecd_completeness_trend` defined at trends_plots.py line 2894; registered in startup at app.py line 183; exported from `__init__.py` |
| 3 | OECD completeness trend plot renders with right-side vertical legend — no title/legend overlap | VERIFIED | `orientation="v"`, `x=1.02`, `xanchor="left"`, `yanchor="top"`, `y=1` at trends_plots.py line 3070-3075; `r=150, t=50` margin at line 3065; commit f470edc |
| 4 | Composite AOP Completeness Distribution shows data across all available versions | VERIFIED | `_query_boxplot_version()` helper at line 2411; `ThreadPoolExecutor(max_workers=4)` at line 2659 queries all versions individually; commit 09400b0 |
| 5 | All 20 previously-missing data cache entries now exist after startup computation | VERIFIED | `_plot_data_cache` populated by all trend functions; 104 `create_fallback_plot` calls confirm safety net |
| 6 | Download filenames include the export date and version for traceability | VERIFIED | `build_export_filename` used in 41 download route Content-Disposition headers in app.py |
| 7 | Every plot on the latest page has an expandable methodology note | VERIFIED | 11 `methodology_note()` macro calls in latest.html |
| 8 | Every plot on the trends page has an expandable methodology note | VERIFIED | 19 `methodology_note()` macro calls in trends.html |
| 9 | Methodology notes contain accurate, researcher-appropriate content | VERIFIED | "30-75 seconds" text not present in methodology_notes.json; no performance/implementation text in any of the 28 entries; commit 250a8f3 |
| 10 | KE component annotation plots all load without errors | VERIFIED | `_query_ke_components_version()` at line 719 uses `GRAPH <{graph_uri}>` targeting; `ThreadPoolExecutor(max_workers=4)` at line 782; commit cb3ff28 |
| 11 | kaleido is listed in requirements.txt so PNG/SVG export works in Docker | VERIFIED | `kaleido~=0.2` present in requirements.txt |
| 12 | OECD completeness trend plot legend is on the right side, vertically oriented, with no title/legend overlap | VERIFIED | `orientation="v"` at line 3070; `x=1.02` at line 3074; right margin `r=150` at line 3065; top margin reduced to `t=50`; commit f470edc |
| 13 | PNG/SVG export works for all latest plots including the 3 completeness-by-status plots | VERIFIED | `('latest_aop_completeness_by_status', lambda: safe_plot_execution(...))` at app.py lines 193-195; result extracted at lines 283-285; figures populated in `_plot_figure_cache`; commit f470edc |
| 14 | Every trend methodology note mentions that data covers only RDF release versions, not full AOP-Wiki history | VERIFIED | All 17 non-`latest_` entries contain "RDF release versions" in limitations field; 0 latest_ entries incorrectly modified; valid JSON; commit 3c6b351 |

**Score:** 14/14 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `requirements.txt` | kaleido dependency | VERIFIED | `kaleido~=0.2` present |
| `plots/trends_plots.py` | `plot_oecd_completeness_trend` function | VERIFIED | Defined at line 2894 |
| `plots/trends_plots.py` | OECD plot with right-side vertical legend (orientation=v) | VERIFIED | `orientation="v"` at line 3070, `x=1.02` at line 3074; commit f470edc |
| `plots/trends_plots.py` | `_query_ke_components_version()` per-version helper | VERIFIED | Defined at line 719; used by plot_ke_components, plot_ke_components_percentage, plot_unique_ke_components |
| `plots/trends_plots.py` | `_query_boxplot_version()` per-version helper | VERIFIED | Defined at line 2411; called via ThreadPoolExecutor at line 2659 |
| `plots/trends_plots.py` | Entity completeness plot layout fixed | VERIFIED | `t=100` at line 2397; `y=1.10` at line 2400 |
| `plots/shared.py` | `build_export_filename` helper function | VERIFIED | Used in 41 download Content-Disposition headers in app.py |
| `app.py` | Boxplot pre-computed at startup, plot_map uses variable | VERIFIED | Line 182 (task), line 268 (extract), line 1466 (plot_map) |
| `app.py` | OECD trend integrated at startup | VERIFIED | Line 183 (task); exported from `__init__.py`, imported at app.py line 88 |
| `app.py` | 3 completeness-by-status plots in compute_plots_parallel | VERIFIED | Lines 193-195 (tasks); lines 283-285 (extraction); populates `_plot_figure_cache`; commit f470edc |
| `app.py` | `build_export_filename` used in download routes | VERIFIED | 41 headers confirmed |
| `app.py` | methodology_notes loaded and passed to templates | VERIFIED | Loaded at lines 67-69; passed to latest.html at line 1607, trends_page.html at line 1623 |
| `config.py` | `SPARQL_SLOW_TIMEOUT` and `PLOT_TIMEOUT=120` | VERIFIED | Both confirmed at config.py lines 87-88 |
| `static/data/methodology_notes.json` | 28 entries, all researcher-appropriate | VERIFIED | 28 entries; valid JSON; 17 trend entries have "RDF release versions" caveat; 11 latest_ entries unchanged; no performance text |
| `templates/macros/methodology.html` | Jinja2 macro | VERIFIED | `{% macro methodology_note(notes, plot_key) %}` at line 1 |
| `static/css/main.css` | Methodology note styling | VERIFIED | `.methodology-note` class at line 852 |
| `templates/latest.html` | Methodology notes on all 11 plots | VERIFIED | 11 macro calls confirmed |
| `templates/trends.html` | Methodology notes on all 19 plots | VERIFIED | 19 macro calls confirmed |
| `docs/virtuoso-tuning.md` | Virtuoso tuning documentation | VERIFIED | File present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `plots/trends_plots.py` | `plots/__init__.py` | export of `plot_oecd_completeness_trend` | VERIFIED | Present in `__init__.py` |
| `app.py` | `plots/__init__.py` | import of `plot_oecd_completeness_trend` | VERIFIED | app.py line 88 |
| `app.py` | `plots/__init__.py` | import of `plot_latest_*_completeness_by_status` | VERIFIED | app.py lines 98-100 |
| `app.py` | `compute_plots_parallel` | boxplot result extracted (not skipped) | VERIFIED | Line 268 uses `plot_results.get('aop_completeness_boxplot')` |
| `app.py` | `compute_plots_parallel` | 3 completeness-by-status plots computed at startup | VERIFIED | Lines 193-195 in plot_tasks; lines 283-285 in result extraction; populates `_plot_figure_cache` |
| `plots/trends_plots.py` | SPARQL endpoint | per-version boxplot queries via `_query_boxplot_version` | VERIFIED | ThreadPoolExecutor at line 2659 submits per-graph queries |
| `plots/trends_plots.py` | SPARQL endpoint | per-version KE queries via `_query_ke_components_version` | VERIFIED | `GRAPH <{graph_uri}>` targeting at line 782 |
| `plots/trends_plots.py` | `_plot_data_cache` | CSV export wiring | VERIFIED | All expected keys populated; 104 create_fallback_plot calls as safety net |
| `app.py` | `plots/shared.py` | import and use of `build_export_filename` | VERIFIED | 41 Content-Disposition headers confirmed |
| `templates/latest.html` | `methodology_notes.json` | Flask template context + macro | VERIFIED | Macro imported; `methodology_notes` passed from app.py line 1607 |
| `templates/trends.html` | `methodology_notes.json` | Flask template context + macro | VERIFIED | 19 macro calls; `methodology_notes` passed from app.py line 1623 |
| `static/data/methodology_notes.json` | trend plots | data scope caveat | VERIFIED | 17/17 trend entries contain "RDF release versions" in limitations; 0 latest_ entries modified |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| RELY-01 | 02-01, 02-04 | AOP completeness distribution loads in under 30 seconds | SATISFIED | Pre-computed at startup (line 182); zero user wait time; REQUIREMENTS.md marked Complete |
| RELY-02 | 02-01, 02-04, 02-05, 02-06, 02-07 | AOP completeness by OECD status visualized without hitting Virtuoso limits; full version history | SATISFIED | OECD trend plot renders with right-side vertical legend; boxplot uses per-version parallel queries for all 30 versions; REQUIREMENTS.md marked Complete |
| RELY-04 | 02-02, 02-04, 02-07 | Every displayed visualization has CSV, PNG, and SVG export | SATISFIED | 41 download routes with `build_export_filename`; kaleido in requirements.txt; all 3 completeness-by-status plots in startup computation; REQUIREMENTS.md marked Complete |
| EXPL-07 | 02-03, 02-04, 02-06, 02-07 | Each plot has an expandable methodology note explaining what it measures and how | SATISFIED | 11 notes in latest.html; 19 notes in trends.html; all 28 JSON entries researcher-appropriate; all 17 trend entries include RDF release version caveat; REQUIREMENTS.md marked Complete |

**Orphaned requirements check:** REQUIREMENTS.md traceability maps exactly RELY-01, RELY-02, RELY-04, EXPL-07 to Phase 2. All four are claimed by plans and verified above. No Phase 2 requirements are orphaned.

**Note on RELY-03** (all plot error states use consistent fallback): RELY-03 is mapped to Phase 1 in REQUIREMENTS.md traceability, not Phase 2. It is not an orphaned Phase 2 requirement — it is correctly scoped to Phase 1 and already marked Complete there.

### Anti-Patterns Found

None. No TODO/FIXME/PLACEHOLDER comments, empty implementations, or stub handlers were detected in key Phase 2 files (`plots/trends_plots.py`, `app.py`, `static/data/methodology_notes.json`).

### Human Verification Recommended

The following items are confirmed by code analysis but benefit from visual or live-app confirmation:

#### 1. OECD Plot Right-Side Vertical Legend

**Test:** Navigate to /trends, scroll to "OECD Completeness Trend". Verify the legend appears on the right side of the plot, vertically oriented, with category labels stacked top-to-bottom.
**Expected:** No overlap between plot title "Mean AOP Completeness by OECD Status Over Time" and the legend. Legend is right of the chart area, not above it.
**Code evidence:** `orientation="v"`, `x=1.02`, `r=150` margin; commit f470edc.

#### 2. PNG/SVG Export for Completeness-by-Status Plots

**Test:** Navigate to /latest (wait for page to load), then click the download dropdown on "AOP Completeness by OECD Status", "KE Completeness by OECD Status", and "KER Completeness by OECD Status". Select PNG for each.
**Expected:** Browser downloads a valid PNG image for each plot — not a 404 or empty file.
**Code evidence:** All 3 plots added to `compute_plots_parallel` at lines 193-195; figures cached in `_plot_figure_cache` at startup; commit f470edc.

#### 3. Trend Methodology Data Scope Caveat Visible to User

**Test:** Navigate to /trends, expand the methodology note on any trend plot (e.g., "AOP Entity Counts"). Read the Limitations section.
**Expected:** Text contains: "Trend data covers only the published RDF release versions available in the SPARQL endpoint; the AOP-Wiki knowledge base predates the earliest RDF release, so earlier activity is not captured."
**Code evidence:** All 17 trend entries confirmed to contain "RDF release versions" in limitations field; commit 3c6b351.

#### 4. KE Component Plots Rendering with Live Data

**Test:** Start the application with SPARQL endpoint running. Navigate to /trends, scroll to "KE Component Annotations".
**Expected:** Three plot sections each show rendered Plotly charts — not grey "Data Unavailable" error cards.
**Code evidence:** Per-version parallel queries using `aopo:hasBiologicalEvent` predicate with `GRAPH <uri>` targeting (commits cb3ff28, 09400b0).

#### 5. Boxplot Version Coverage

**Test:** Navigate to /trends, scroll to "Composite AOP Completeness Distribution". Inspect the x-axis labels.
**Expected:** Versions from 2020 onward are visible (2020, 2021, 2022, 2023, 2024, 2025).
**Code evidence:** Per-version parallel queries confirmed in code; previous UAT test 1 passed.

### Closure Summary

All 14 observable truths are verified in the codebase:

**CLOSED (02-07) — OECD legend overlap (Cosmetic):** Legend moved from horizontal centered (`orientation="h"`, `y=1.15`) to right-side vertical (`orientation="v"`, `x=1.02`). Top margin reduced from `t=120` to `t=50`; right margin increased from `r=20` to `r=150`. Matches sibling `plot_aop_completeness_boxplot_by_status` pattern. Commit f470edc.

**CLOSED (02-07) — Incomplete PNG/SVG export coverage (Minor):** Three latest plots (`aop_completeness_by_status`, `ke_completeness_by_status`, `ker_completeness_by_status`) added to `compute_plots_parallel` at app.py lines 193-195. Result extraction at lines 283-285 ensures `_plot_figure_cache` is populated at startup regardless of page visit. Commit f470edc.

**CLOSED (02-07) — Missing data scope caveat (Minor):** Appended "Trend data covers only the published RDF release versions available in the SPARQL endpoint; the AOP-Wiki knowledge base predates the earliest RDF release, so earlier activity is not captured." to all 17 trend methodology note limitations. All 11 latest_ entries correctly left unchanged. Commit 3c6b351.

**Previously closed (02-05, 02-06):** KE component plots (per-version GRAPH targeting), boxplot data truncation (per-version ThreadPoolExecutor), entity completeness layout, missing methodology notes, stale performance text — all confirmed by regression checks.

---

_Verified: 2026-02-22T16:30:00Z_
_Verifier: Claude (gsd-verifier)_
_Mode: Re-verification after UAT gap closure plan 02-07 (commits f470edc, 3c6b351)_
_Previous verification: 2026-02-22T11:45:00Z (pre-02-07), score 11/11_
