---
phase: 05-polish-consistency
plan: 01
subsystem: ui
tags: [plotly, css-custom-properties, brand-colors, plotly-template]

# Dependency graph
requires: []
provides:
  - BRAND_COLORS dict with 11-color VHP4Safety palette in plots/shared.py
  - vhp4safety Plotly custom template registered as default
  - PLOTLY_HTML_CONFIG dict for standardized toolbar behavior
  - render_plot_html() helper for consistent Plotly HTML rendering
  - CSS custom properties (:root) for all brand colors and semantic aliases
affects: [05-02, 05-03, 05-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [plotly-custom-template, css-custom-properties, centralized-brand-config]

key-files:
  created: []
  modified:
    - plots/shared.py
    - plots/__init__.py
    - static/css/main.css

key-decisions:
  - "Kept legacy aliases (secondary, accent, light, content, config) for backward compatibility with existing plot code"
  - "Template composited as plotly_white+vhp4safety to inherit plotly_white base and overlay brand styling"

patterns-established:
  - "BRAND_COLORS dict: single source of truth for all chart colors in Python"
  - "CSS custom properties: single source of truth for all brand colors in CSS, synced with BRAND_COLORS hex values"
  - "render_plot_html(): standard helper for converting Plotly figures to HTML with consistent config"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-02-23
---

# Phase 5 Plan 1: Color System & Plotly Template Summary

**VHP4Safety brand color system with BRAND_COLORS dict, registered Plotly custom template (plotly_white+vhp4safety), PLOTLY_HTML_CONFIG, render_plot_html() helper, and CSS custom properties in :root**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-23T18:55:10Z
- **Completed:** 2026-02-23T18:57:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created centralized BRAND_COLORS dict with 11-color VHP4Safety palette and semantic keys (primary, magenta, blue, etc.)
- Registered `vhp4safety` Plotly custom template as default, automatically applying brand styling to all new figures
- Added PLOTLY_HTML_CONFIG with standardized toolbar behavior (hover display, no Plotly logo, lasso/select removed)
- Added render_plot_html() helper to wrap pio.to_html() with shared config
- Defined CSS custom properties in :root block at top of main.css, covering brand colors, semantic aliases, navigation/chrome, typography, and spacing
- Updated body, header, footer CSS rules to use var() references instead of hardcoded hex values

## Task Commits

Each task was committed atomically:

1. **Task 1: Create BRAND_COLORS config, Plotly custom template, and render helper** - `d232c73` (feat)
2. **Task 2: Add CSS custom properties for VHP4Safety brand colors** - `0c811d9` (feat)

## Files Created/Modified
- `plots/shared.py` - Updated BRAND_COLORS dict, added Plotly template registration, PLOTLY_HTML_CONFIG, render_plot_html()
- `plots/__init__.py` - Added exports for BRAND_COLORS, PLOTLY_HTML_CONFIG, render_plot_html
- `static/css/main.css` - Added :root block with CSS custom properties, updated body/header/footer to use var()

## Decisions Made
- Kept legacy aliases (secondary, accent, light, content, config) in BRAND_COLORS and module scope for backward compatibility with existing plot code across latest_plots.py and trends_plots.py
- Used template composition `plotly_white+vhp4safety` to inherit plotly_white as base and overlay brand styling, avoiding need to redefine all plotly_white properties

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Color system and Plotly template are in place for Plans 02-04 to consume
- Plan 02 can begin migrating individual plot functions to use the template and render_plot_html()
- Plan 04 can incrementally migrate remaining CSS hardcoded values to use the :root custom properties

## Self-Check: PASSED

All files exist, all commit hashes verified.

---
*Phase: 05-polish-consistency*
*Completed: 2026-02-23*
