---
status: complete
phase: 01-foundation-and-cleanup
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md, 01-04-SUMMARY.md]
started: 2026-02-21T10:00:00Z
updated: 2026-02-21T10:05:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Docker build and Gunicorn startup
expected: Run `docker-compose up --build`. The build completes with no "version is obsolete" warning. Container starts with Gunicorn "Booting worker" messages (not Flask dev server).
result: pass

### 2. Health endpoint returns JSON status
expected: With the app running, `curl http://localhost:5000/health` returns JSON with fields: status ("healthy" or "unhealthy"), sparql_endpoint ("up" or "down"), plots_loaded (number), and timestamp.
result: pass

### 3. Health endpoint returns 503 when SPARQL is down
expected: With the SPARQL endpoint unreachable, `curl -w "\n%{http_code}" http://localhost:5000/health` returns HTTP 503 with `"status": "unhealthy"` and `"sparql_endpoint": "down"`.
result: pass

### 4. Dashboard loads with all plots
expected: Open `http://localhost:5000` in a browser. The dashboard renders with the version selector and all plot containers load via lazy loading (plots appear as they finish loading).
result: pass

### 5. Failed plot shows error card with retry
expected: If any plot fails to load (e.g., due to timeout or unreachable SPARQL), the error card shows a specific title ("Query Timed Out", "Service Unreachable", or "Unable to Load Plot"), a descriptive suggestion paragraph, and a styled "Retry" button. The retry button re-attempts loading the plot.
result: pass

### 6. Add-a-plot checklist is complete
expected: Open `.claude/add-a-plot.md`. It contains separate sections for "Latest Data Plot" and "Historical Trends Plot", each with precise file paths, function signatures, cache key patterns, and naming conventions. A developer can follow it without reading existing code.
result: pass

### 7. GitHub issue template exists
expected: Open `.github/ISSUE_TEMPLATE/new-plot.md`. It contains a structured template with sections for Plot Description, Plot Type, Data Source, Visualization Type, and Acceptance Criteria.
result: pass

## Summary

total: 7
passed: 7
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
