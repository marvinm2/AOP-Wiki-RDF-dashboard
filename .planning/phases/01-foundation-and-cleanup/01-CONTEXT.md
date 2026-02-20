# Phase 1: Foundation and Cleanup - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Stabilize the existing codebase into a production-ready state: remove legacy code, bound memory usage, serve under Gunicorn, pin dependencies, and document the developer workflow for adding new plots. No new features — this phase makes the existing dashboard reliable and maintainable.

</domain>

<decisions>
## Implementation Decisions

### Data freshness & caching
- Cache entries expire after a time limit (TTL-based eviction)
- Hard cap on the number of versions cached simultaneously — evict oldest when cap is reached
- Latest/current version is pinned in cache and never evicted; only historical versions follow TTL + cap rules
- When a user views a version that was evicted, show a loading indicator while re-fetching (not silent)

### Developer documentation
- Create a `.claude/` instruction file with the add-a-plot checklist, referenced from CLAUDE.md
- Create a GitHub issue template (markdown format, not YAML form) for proposing new plots
- Checklist format only — no worked example walkthrough needed
- Documentation aimed at both human contributors and Claude equally — precise file paths and patterns, but also readable narrative

### Production error behavior
- Friendly error card with message and retry button when a SPARQL query times out or endpoint is unreachable
- Graceful degradation: each plot loads independently; failed ones show error cards, successful ones display normally
- `/health` endpoint reports unhealthy when SPARQL endpoint is completely down (honest reporting, may trigger container restarts)
- Production logging in structured JSON format (JSON lines) for log aggregation tools

### Cleanup boundaries
- General dead code sweep beyond legacy `plots.py` — includes unused imports, dead routes, commented-out code, orphaned templates
- Careful commit trail: separate commits for each removed file/section with explanatory messages
- Dependencies pinned to minor range (e.g., `Flask~=3.0`) — allows patch updates, balances reproducibility with security
- Linting/formatting (ruff etc.) is NOT in scope for this phase — save for later

### Claude's Discretion
- Specific TTL duration and cache cap number (based on memory profiling)
- Gunicorn worker count, timeout settings, and bind configuration
- Exact error card design and copy
- JSON log format structure and log levels
- Order and grouping of cleanup commits

</decisions>

<specifics>
## Specific Ideas

- Developer docs should live as a `.claude/` instruction file so Claude can follow it directly when asked to add a plot
- GitHub issue template gives contributors a structured way to propose new visualizations
- "Honest health reporting" — user prefers the orchestrator to know the truth even if it means restarts

</specifics>

<deferred>
## Deferred Ideas

- Linting/formatting setup (ruff or similar) — future phase or standalone task
- Code style enforcement — not this phase

</deferred>

---

*Phase: 01-foundation-and-cleanup*
*Context gathered: 2026-02-20*
