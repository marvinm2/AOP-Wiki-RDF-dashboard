---
phase: 02-reliability-and-completeness
plan: 03
subsystem: ui
tags: [jinja2, methodology, sparql, documentation, html, css]

# Dependency graph
requires:
  - phase: 01-foundation-and-cleanup
    provides: Plot infrastructure, lazy loading, template structure
provides:
  - Expandable methodology notes on all ~27 plots across latest and trends pages
  - Centralized methodology data in JSON format for all plot documentation
  - Jinja2 macro for consistent methodology note rendering
  - Virtuoso tuning recommendations documentation
affects: [any-new-plots, template-modifications]

# Tech tracking
tech-stack:
  added: []
  patterns: [jinja2-macros-for-reusable-components, json-driven-content]

key-files:
  created:
    - static/data/methodology_notes.json
    - templates/macros/methodology.html
    - docs/virtuoso-tuning.md
  modified:
    - static/css/main.css
    - app.py
    - templates/latest.html
    - templates/trends.html
    - templates/trends_page.html

key-decisions:
  - "Methodology content stored in single JSON file (28 entries) rather than scattered across templates"
  - "Used native HTML details/summary elements for zero-JS collapsibility"
  - "SPARQL queries shown as representative static templates with placeholders for dynamic parts"
  - "Bio object annotations and AOPs modified/scatter plots share the aop_lifetime methodology key since they are part of the same logical section"
  - "Virtuoso tuning documented as recommendations only, not applied in Docker config per user decision"

patterns-established:
  - "Jinja2 macro pattern: create reusable macros in templates/macros/ and import in page templates"
  - "Content-driven templates: store structured content in static/data/ JSON files, pass via render_template"

requirements-completed: [EXPL-07]

# Metrics
duration: 8min
completed: 2026-02-21
---

# Phase 02 Plan 03: Methodology Notes Summary

**Expandable methodology notes with descriptions, data sources, limitations, and SPARQL queries on all 27 plots using Jinja2 macros and centralized JSON content**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-21T19:12:09Z
- **Completed:** 2026-02-21T19:20:47Z
- **Tasks:** 2
- **Files modified:** 8 (5 modified, 3 created)

## Accomplishments
- Created centralized methodology_notes.json with 28 entries covering all plots across both pages
- Built reusable Jinja2 macro for rendering collapsible methodology notes with nested SPARQL query viewer
- Added methodology notes to all 11 latest-page plots and 16 trends-page plot containers
- Created Virtuoso tuning recommendations documentation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create methodology data file, Jinja2 macro, CSS, and apply to latest.html** - `21ccef4` (feat)
2. **Task 2: Apply methodology notes to trends.html and add Virtuoso tuning docs** - `8a9a8d0` (feat)

## Files Created/Modified
- `static/data/methodology_notes.json` - Structured methodology content for all 28 plots (descriptions, data sources, limitations, SPARQL queries)
- `templates/macros/methodology.html` - Jinja2 macro rendering expandable methodology notes with nested SPARQL query viewer
- `static/css/main.css` - Added methodology note CSS styles (collapsible sections, VHP4Safety-branded styling)
- `app.py` - Added JSON loading at startup and methodology_notes context passing to both page templates
- `templates/latest.html` - Added macro import and 11 methodology note calls for all latest-page plots
- `templates/trends.html` - Added 16 methodology note macro calls for all trend plots (with conditional for OECD trend)
- `templates/trends_page.html` - Added macro import before include of trends.html partial
- `docs/virtuoso-tuning.md` - Virtuoso SPARQL endpoint tuning recommendations (documentation only)

## Decisions Made
- Stored all methodology content in a single JSON file (28 entries) for maintainability rather than scattering across templates
- Used native HTML `<details>/<summary>` elements requiring zero JavaScript for collapsibility
- Showed SPARQL queries as representative static templates with `{placeholders}` where queries are dynamically generated
- Used conditional rendering for OECD completeness trend methodology note since it depends on Plan 02-01 execution
- Documented Virtuoso tuning as recommendations only per user decision, not applied in Docker configuration

## Deviations from Plan

None - plan executed exactly as written.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All plots now have expandable methodology documentation for researcher trust and reproducibility
- The Jinja2 macro pattern is established for reuse in any future plot additions
- Adding methodology for new plots requires only adding an entry to the JSON file and a macro call in the template

## Self-Check: PASSED

All created files verified present. All commit hashes verified in git log.

---
*Phase: 02-reliability-and-completeness*
*Completed: 2026-02-21*
