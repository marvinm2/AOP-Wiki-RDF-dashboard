---
status: diagnosed
phase: 01-foundation-and-cleanup
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md]
started: 2026-02-20T14:30:00Z
updated: 2026-02-20T14:55:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Docker build completes without deprecation warnings
expected: Run `docker-compose up --build`. The build completes with no "version is obsolete" warning. Container starts with Gunicorn "Booting worker" messages (not Flask dev server).
result: pass

### 2. Health endpoint returns JSON status
expected: With the app running, `curl http://localhost:5000/health` returns JSON with fields: status ("healthy" or "unhealthy"), sparql_endpoint ("up" or "down"), plots_loaded (number), and timestamp.
result: pass

### 3. Health endpoint returns 503 when SPARQL is down
expected: With the SPARQL endpoint unreachable, `curl -w "\n%{http_code}" http://localhost:5000/health` returns HTTP 503 with `"status": "unhealthy"` and `"sparql_endpoint": "down"`.
result: pass

### 4. Dashboard loads with all plots
expected: Open `http://localhost:5000` in a browser. The dashboard renders with the version selector and all plots load via lazy loading.
result: pass

### 5. Failed plot shows differentiated error card with retry
expected: If any plot fails to load (e.g., due to timeout or unreachable SPARQL), the error card shows a specific title ("Query Timed Out", "Service Unreachable", or "Unable to Load Plot"), a descriptive suggestion paragraph, and a styled "Retry" button. The retry button re-attempts loading the plot.
result: issue
reported: "I see some failed plots, but not with a good error message. E.g. AOP Creation vs Modification plot and KE component plots in the trends page do not work"
severity: major

### 6. Add-a-plot checklist is complete and self-contained
expected: Open `.claude/add-a-plot.md`. It should contain separate sections for "Latest Data Plot" and "Historical Trends Plot", each with precise file paths, function signatures, cache key patterns, and naming conventions. A developer should be able to follow it without reading existing code.
result: pass

### 7. GitHub issue template for new plot proposals
expected: Open `.github/ISSUE_TEMPLATE/new-plot.md`. It should contain a structured template with sections for Plot Description, Plot Type, Data Source, Visualization Type, Value to Users, Acceptance Criteria, and Additional Context.
result: pass

## Summary

total: 7
passed: 6
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Failed plots show differentiated error cards with specific title, suggestion, and retry button"
  status: failed
  reason: "User reported: I see some failed plots, but not with a good error message. E.g. AOP Creation vs Modification plot and KE component plots in the trends page do not work"
  severity: major
  test: 5
  root_cause: "API handler at app.py:1491-1496 returns success:true unconditionally for all plot_map lookups. Two failure paths: (1) plot_aop_lifetime has no return annotation so safe_plot_execution returns single fallback string, unpacking fails silently, empty strings stored — API returns {html:'', success:true}. (2) Annotated plots get fallback Plotly charts with 'Data Unavailable' text — API returns {html:fallback_html, success:true}. In both cases showErrorState() never fires because success is true."
  artifacts:
    - path: "app.py"
      issue: "Lines 1491-1496: no content check before returning success:true"
    - path: "plots/trends_plots.py"
      issue: "Line 528: plot_aop_lifetime() missing return type annotation -> tuple[str, str, str]"
  missing:
    - "API handler should check for empty/falsy HTML and return success:false with error message"
    - "API handler should detect fallback sentinel ('Data Unavailable') and return success:false"
    - "plot_aop_lifetime needs return type annotation for correct fallback tuple generation"
  debug_session: ""
