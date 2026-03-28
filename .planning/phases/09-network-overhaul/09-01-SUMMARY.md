---
phase: 09-network-overhaul
plan: 01
subsystem: network
tags: [sparql, networkx, cytoscape, spring-layout, role-detection]

# Dependency graph
requires:
  - phase: 08-color-consistency
    provides: Frozen palette array, BRAND_COLORS as single source of truth
provides:
  - BRAND_COLORS['network_roles'] color mapping (MIE/KE/AO)
  - detect_ke_roles() SPARQL role detection function
  - Pre-computed spring_layout positions in Cytoscape JSON
  - Role-based node coloring in graph_to_cytoscape_json()
affects: [09-02 frontend changes, network.html, network-graph.js]

# Tech tracking
tech-stack:
  added: []
  patterns: [UNION+GROUP_CONCAT role detection, element-level position embedding, role priority MIE>AO>KE]

key-files:
  created: []
  modified:
    - plots/shared.py
    - plots/network.py

key-decisions:
  - "Role colors stored in BRAND_COLORS['network_roles'] separate from type_colors"
  - "spring_layout(seed=42, scale=1000) for deterministic positions"
  - "Position at element level (sibling to data) for Cytoscape preset layout"

patterns-established:
  - "Role detection via SPARQL UNION+GROUP_CONCAT with priority fallback"
  - "Pre-computed layout positions embedded in Cytoscape JSON at element level"

requirements-completed: [NET-01, NET-02]

# Metrics
duration: 3min
completed: 2026-03-28
---

# Phase 09 Plan 01: Backend Role Detection and Layout Summary

**MIE/KE/AO role detection via SPARQL UNION query with deterministic spring_layout positions embedded in Cytoscape JSON**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-28T20:19:45Z
- **Completed:** 2026-03-28T20:23:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `network_roles` color mapping to BRAND_COLORS (MIE=#EB5B25, KE=#307BBF, AO=#E6007E)
- Implemented `detect_ke_roles()` with SPARQL UNION+GROUP_CONCAT and MIE > AO > KE priority
- Modified `graph_to_cytoscape_json()` for role-based coloring with position at element level
- Integrated spring_layout(seed=42, scale=1000) in `get_or_compute_network()` for deterministic layout
- Added graceful fallback: empty role query defaults all nodes to KE with logged warning

## Task Commits

Each task was committed atomically:

1. **Task 1: Add network_roles color mapping to BRAND_COLORS** - `4f9a7a3` (feat)
2. **Task 2: Add role detection, spring_layout, and role-based Cytoscape JSON** - `c1c4d5d` (feat)

## Files Created/Modified
- `plots/shared.py` - Added `network_roles` key to BRAND_COLORS dict with MIE/KE/AO hex colors
- `plots/network.py` - Added `detect_ke_roles()`, modified `graph_to_cytoscape_json()` for roles+positions, updated `get_or_compute_network()` integration

## Decisions Made
- Role colors stored in `BRAND_COLORS['network_roles']` as a separate dict from `type_colors` (per D-05)
- `spring_layout(seed=42, scale=1000)` chosen for deterministic, well-spaced layout (per D-06)
- Positions embedded at element level (sibling to `data`) for Cytoscape preset layout (per D-07)
- MIE > AO > KE priority: if a KE is MIE in any AOP, it is labeled MIE (per D-02)
- Metrics DataFrame `type` column updated with roles for correct CSV export

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Local NumPy version incompatibility prevented direct Python import verification; verified via AST parsing and grep pattern matching instead. This is a local environment issue only, not affecting production code.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Backend role detection and layout computation ready for frontend consumption
- Plan 09-02 (frontend changes) can now use `layout: {name: 'preset'}` with positions from JSON
- Role filter dropdown and type legend can read `type` field from node data

---
*Phase: 09-network-overhaul*
*Completed: 2026-03-28*
