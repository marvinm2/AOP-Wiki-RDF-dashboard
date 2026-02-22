# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-20)

**Core value:** Researchers, curators, and regulators can explore AOP-Wiki data across any version to understand how the knowledge base evolves over time — reliably and without timeouts.
**Current focus:** Phase 2: Reliability and Completeness

## Current Position

Phase: 2 of 6 (Reliability and Completeness)
Plan: 7 of 7 in current phase
Status: Phase Complete
Last activity: 2026-02-22 — Completed 02-07-PLAN.md (UAT gap closure: OECD legend, PNG export, data scope caveat)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: ~4 min
- Total execution time: ~0.8 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 4/4 | ~19 min | ~5 min |
| 02 | 7/7 | ~29 min | ~4 min |

**Recent Trend:**
- Last 5 plans: 02-03 (8m), 02-04 (3m), 02-05 (3m), 02-06 (3m), 02-07 (2m)
- Trend: Stable

*Updated after each plan completion*
| Phase 02 P05 | 9min | 2 tasks | 2 files |
| Phase 02 P07 | 2min | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Phases 2 and 3 can execute in parallel (both depend only on Phase 1)
- [Roadmap]: Service layer extraction folded into consuming phases rather than standalone phase (no user-observable outcome on its own)
- [Roadmap]: Version comparison deferred to v2 (user prefers trends across all versions over 1-to-1 comparisons)
- [Roadmap]: Network analysis Phase 3 includes performance benchmarking criteria (user flagged prior performance issues with interactive graphs)
- [01-02]: VersionedPlotCache uses TTL=1800s and max_versions=5 for bounded memory growth
- [01-02]: Health endpoint returns 503 when SPARQL is down (honest reporting, may trigger container restarts)
- [01-02]: JSON logging configured centrally in config.py before any other imports
- [01-04]: Generalized tuple fallback by counting str elements in annotation rather than hardcoding function names
- [01-04]: Detect 'Data Unavailable' sentinel string in fallback HTML to distinguish from legitimate content
- [02-02]: Used request.args.get('version') uniformly for all routes; trend routes get None, latest routes get version from version-selector.js
- [02-02]: 12 data cache keys were already present (not 20 as estimated); only 6 functions needed updates
- [Phase 02-01]: Used per-version parallel SPARQL queries (4 workers) instead of single cross-version query to avoid Virtuoso limits
- [Phase 02-01]: Line chart with marker shapes for OECD status visualization (aggregated means vs raw distributions)
- [02-03]: Methodology notes stored in single JSON file (28 entries) for centralized maintainability
- [02-03]: Used native HTML details/summary for zero-JS collapsibility of methodology sections
- [02-03]: Virtuoso tuning documented as recommendations only, not applied in Docker config
- [02-04]: All trend plot functions wrapped in try/except with create_fallback_plot for graceful degradation
- [02-04]: Used errors='coerce' universally on pd.to_datetime calls for malformed date safety
- [02-06]: Centered horizontal legends (xanchor=center) for better visual balance with multi-status legends
- [02-06]: Methodology note limitations should contain only methodological/data-quality caveats, no performance or implementation details
- [02-06]: Varied presence-only limitation text per entity type for contextual clarity
- [Phase 02]: Root cause of KE component failures was Virtuoso 400s execution time limit on triple-OPTIONAL cross-product, not wrong predicates
- [Phase 02]: Direct GRAPH <uri> targeting preferred over GRAPH ?graph + FILTER for latest-data single-version queries

### Pending Todos

None yet.

### Verification Issues (Phase 2)

User verification of Phase 2 delivery found 4 issues requiring follow-up:
1. ~~OECD plot title and legend overlap (cosmetic)~~ -- RESOLVED in 02-06
2. Composite AOP Completeness Distribution only has pre-2020 data (possible Virtuoso limit) -- addressed in 02-05
3. ~~Methodology notes incomplete on some plots; performance limitations unnecessary for average users~~ -- RESOLVED in 02-06
4. Some KE component annotation plots broken (not rendering) -- addressed in 02-05

### Blockers/Concerns

- Phase 3 needs research: AOP-Wiki graph topology profiling, domain expert input on meaningful centrality metrics, memory profiling for NetworkX
- Phase 6 needs research: VHP4Safety infrastructure specifics (orchestration, probes, CORS domain)

## Session Continuity

Last session: 2026-02-22
Stopped at: Completed 02-07-PLAN.md (UAT gap closure: OECD legend, PNG export, data scope caveat) -- Phase 2 complete
Resume file: .planning/phases/02-reliability-and-completeness/02-07-SUMMARY.md
