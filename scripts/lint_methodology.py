#!/usr/bin/env python3
"""Methodology JSON lint for the AOP-Wiki RDF dashboard.

Validates every entry in `static/data/methodology_notes.json` against the live
SPARQL endpoint and against simple static rules.

LINT codes
----------
LINT-01  SPARQL syntax/semantic error (endpoint returns HTTP 4xx for the
         entry's query after wrap-then-classify).
LINT-02  Empty-result drift against the latest snapshot (endpoint returns 200
         with zero bindings, OR a single binding whose numeric aggregate
         columns are all zero, AND the entry has not opted out via
         ``may_be_empty: true``).
LINT-03  ``latest_*`` entry lacks a latest-snapshot constraint: it has neither
         the ``__GRAPH_URI__`` substitution marker nor the combination of
         ``ORDER BY DESC(?graph)``, ``LIMIT 1`` and ``STRSTARTS(STR(?graph)``.
LINT-04  (soft warn this phase, see D2 in 11-CONTEXT.md) Plot function issues
         more SPARQL calls than the methodology entry discloses. Detected
         purely via AST scan of ``plots/*.py`` — the file is never imported.
LINT-05  Endpoint-call budget guard: refuse to start when the methodology JSON
         has more entries than the configured budget (default 50).
LINT-06  Deterministic PR-comment / JSON-report formatting (no severity; this
         is a presentation contract, not a check that emits findings).

Exit codes
----------
0  No findings (or only ``warn``-severity findings).
1  At least one ``fail``-severity finding.
2  Infrastructure error (DNS failure, connection refused, timeout, 5xx).
3  Pre-flight refusal (LINT-05 budget guard tripped).

Design constraints
------------------
- Pure stdlib + ``requests``. Do NOT import from ``plots.*`` (Pitfall P2).
- Do NOT use rdflib for LINT-01 (Pitfall P1: rdflib does not catch SP031).
- One endpoint round-trip per entry (LINT-05 budget).
- TLS validation enabled (default ``verify=True``); 30s per-call timeout.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from typing import Optional

import requests

DEFAULT_ENDPOINT = "https://aopwiki-multirdf.vhp4safety.nl/sparql"
DEFAULT_TIMEOUT = 30
INFRA_EXIT = 2
LINT_FAIL_EXIT = 1
PASS_EXIT = 0
BUDGET_EXIT = 3

# LINT-04: AST-scan helper-name set. The two canonical SPARQL helpers exported
# by ``plots/shared.py``. Centralised so future renames are a one-line change.
SPARQL_HELPER_NAMES = frozenset({"run_sparql_query", "run_sparql_query_with_retry"})

# LINT-04 is a SOFT warn this phase (per D2 in 11-CONTEXT.md). When the
# multi-query disclosure schema migration (issue #40) lands, flip this to
# "fail" — that one-line change deliberately breaks the
# ``test_lint04_severity_constant`` test so contributors notice.
LINT_04_SEVERITY = "warn"

# LINT-05: methodology JSON must not exceed this many entries per run.
# Today the real file has 46 entries; the ceiling is "at most ~50 endpoint
# requests per run" per the LINT-05 requirement.
DEFAULT_CALL_BUDGET = 50


class InfraError(RuntimeError):
    """Raised when the endpoint is unreachable or returns 5xx.

    Distinguished from lint failures so the CI runner can exit 2 (infra) vs.
    1 (lint) and avoid masking transient network issues as code defects.
    """


class BudgetExceeded(RuntimeError):
    """Raised by ``enforce_call_budget`` when entry count exceeds the budget.

    Caught in ``main`` and surfaced as exit code 3 (distinct from 0/1/2) so
    callers can distinguish ``lint cannot run`` from ``lint ran and infra
    broke``. See LINT-05 in 11-CONTEXT.md.
    """


# ---------------------------------------------------------------------------
# Static checks (no network)
# ---------------------------------------------------------------------------


def check_lint_03(name: str, entry: dict) -> Optional[dict]:
    """Static check: a ``latest_*`` entry must constrain to the latest graph.

    Returns ``None`` on pass, or a finding dict on fail. A ``latest_*`` entry
    passes if EITHER the ``__GRAPH_URI__`` substitution marker is present in
    the SPARQL string OR all three of ``ORDER BY DESC(?graph)``, ``LIMIT 1``
    and ``STRSTARTS(STR(?graph)`` are present (Pitfall P4).
    """
    if not name.startswith("latest_"):
        return None
    query = entry.get("sparql", "")
    if "__GRAPH_URI__" in query:
        return None
    has_order = "ORDER BY DESC(?graph)" in query
    has_limit1 = "LIMIT 1" in query
    has_strstarts = "STRSTARTS(STR(?graph)" in query
    if has_order and has_limit1 and has_strstarts:
        return None
    return {
        "name": name,
        "check": "LINT-03",
        "severity": "fail",
        "message": (
            "latest_* entry lacks a latest-snapshot constraint: needs either "
            "the __GRAPH_URI__ substitution marker or the ORDER BY "
            "DESC(?graph) LIMIT 1 + STRSTARTS(STR(?graph) trio."
        ),
    }


# ---------------------------------------------------------------------------
# Live endpoint checks (LINT-01 + LINT-02 combined)
# ---------------------------------------------------------------------------


_NUMERIC_TYPES = {
    "http://www.w3.org/2001/XMLSchema#integer",
    "http://www.w3.org/2001/XMLSchema#int",
    "http://www.w3.org/2001/XMLSchema#long",
    "http://www.w3.org/2001/XMLSchema#decimal",
    "http://www.w3.org/2001/XMLSchema#double",
    "http://www.w3.org/2001/XMLSchema#float",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
    "http://www.w3.org/2001/XMLSchema#positiveInteger",
}


def _bindings_drift_empty(bindings: list) -> bool:
    """True when a result set is empty or a single all-zero aggregate row.

    A GROUP BY query over OPTIONAL patterns (the shape of the
    ``latest_ke_components`` anchor) always returns at least one row with
    aggregate columns even when the inner predicates are bogus — in that
    case the COUNT/SUM values are all 0. Treat that as drift so LINT-02
    catches the H1-class anchor verbatim from the methodology JSON.

    Heuristic: examine the single row; ignore non-numeric columns
    (group keys like ``?graph`` URIs); if at least one numeric column is
    present and EVERY numeric column has a literal value of 0, treat as
    drift.
    """
    if not bindings:
        return True
    if len(bindings) != 1:
        return False
    row = bindings[0]
    if not row:
        return True
    numeric_values_seen = False
    for binding in row.values():
        if not isinstance(binding, dict):
            continue
        dtype = binding.get("datatype")
        if dtype not in _NUMERIC_TYPES:
            continue  # group-key or non-numeric column; doesn't count
        numeric_values_seen = True
        value = binding.get("value", "")
        try:
            if float(value) != 0.0:
                return False
        except (TypeError, ValueError):
            return False
    return numeric_values_seen


def check_lint_01_02(
    name: str,
    entry: dict,
    latest_graph_uri: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[dict]:
    """Wrap-then-classify endpoint round-trip covering LINT-01 + LINT-02.

    Substitutes ``__GRAPH_URI__`` with the resolved latest graph URI in angle
    brackets, wraps the inner query as ``SELECT * WHERE { { <inner> } } LIMIT
    1`` to preserve projection-list semantics so SP031-class errors still
    fire, POSTs once, and classifies the response.
    """
    query = entry.get("sparql", "")
    substituted = query.replace("__GRAPH_URI__", f"<{latest_graph_uri}>")
    wrapped = "SELECT * WHERE { { " + substituted + " } } LIMIT 1"

    http = session or requests
    try:
        response = http.post(
            endpoint,
            data={"query": wrapped},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise InfraError(f"{name}: endpoint unreachable: {exc}") from exc

    if response.status_code >= 500:
        raise InfraError(
            f"{name}: endpoint returned {response.status_code}: "
            f"{(response.text or '').splitlines()[:1]}"
        )

    if response.status_code >= 400:
        first_line = ""
        if response.text:
            first_line = response.text.splitlines()[0]
        return {
            "name": name,
            "check": "LINT-01",
            "severity": "fail",
            "message": f"{response.status_code}: {first_line}",
        }

    try:
        payload = response.json()
    except ValueError as exc:
        raise InfraError(f"{name}: non-JSON 200 response: {exc}") from exc

    bindings = payload.get("results", {}).get("bindings", [])
    if _bindings_drift_empty(bindings):
        if entry.get("may_be_empty") is True:
            return None
        return {
            "name": name,
            "check": "LINT-02",
            "severity": "fail",
            "message": "0 rows against latest graph",
        }
    return None


# ---------------------------------------------------------------------------
# Latest-graph URI discovery (one call per run)
# ---------------------------------------------------------------------------


def resolve_latest_graph_uri(
    endpoint: str = DEFAULT_ENDPOINT,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 15,
) -> str:
    """Resolve the latest ``http://aopwiki.org/graph/YYYY-MM-DD`` URI.

    Mirrors ``plots.shared.get_latest_version`` behaviourally without
    importing it (Pitfall P2). Raises ``InfraError`` so the orchestrator can
    exit 2 cleanly on transient endpoint failures.
    """
    query = "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } } LIMIT 100"
    http = session or requests
    try:
        response = http.post(
            endpoint,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise InfraError(f"latest-graph resolution failed: {exc}") from exc
    if response.status_code >= 400:
        raise InfraError(
            f"latest-graph resolution returned {response.status_code}: "
            f"{(response.text or '').splitlines()[:1]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise InfraError(f"latest-graph: non-JSON response: {exc}") from exc
    graphs = [
        b["g"]["value"]
        for b in payload.get("results", {}).get("bindings", [])
        if "aopwiki.org/graph" in b.get("g", {}).get("value", "")
    ]
    if not graphs:
        raise InfraError("latest-graph resolution returned no aopwiki graphs")
    return sorted(graphs, reverse=True)[0]


# ---------------------------------------------------------------------------
# LINT-04: AST scan of plots/*.py (no execution, no import — Pitfall P2)
# ---------------------------------------------------------------------------


def count_sparql_calls_per_function(path: pathlib.Path) -> dict[str, int]:
    """Count SPARQL helper calls per function in one ``.py`` file.

    Parses the file with ``ast.parse`` (Pitfall P2: NEVER import it) and walks
    each ``FunctionDef`` / ``AsyncFunctionDef``. Counts ``Call`` nodes whose
    ``func.attr`` (for ``obj.helper()``) or ``func.id`` (for ``helper()``)
    appears in ``SPARQL_HELPER_NAMES``.

    Returns ``{function_name: count}`` only for functions with at least one
    helper call. Functions with zero matches are omitted to keep the merged
    map small.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return {}

    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        n = 0
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            name: Optional[str] = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name and name in SPARQL_HELPER_NAMES:
                n += 1
        if n > 0:
            counts[node.name] = n
    return counts


def scan_plots_dir(plots_dir: pathlib.Path) -> dict[str, int]:
    """Walk ``plots_dir`` for ``*.py`` files and merge per-function call counts.

    One level deep — the dashboard's ``plots/`` package is flat. Skips
    ``__pycache__`` and any hidden directory. Defensive merge: if two files
    define the same function name, sum the counts.
    """
    merged: dict[str, int] = {}
    if not plots_dir.exists() or not plots_dir.is_dir():
        return merged
    for py in sorted(plots_dir.glob("*.py")):
        if py.name.startswith("."):
            continue
        per_file = count_sparql_calls_per_function(py)
        for fn, n in per_file.items():
            merged[fn] = merged.get(fn, 0) + n
    return merged


def check_lint_04(
    name: str,
    entry: dict,
    plot_call_counts: dict[str, int],
) -> Optional[dict]:
    """Soft-warn when a plot function issues more calls than the entry discloses.

    Maps methodology key ``foo`` to expected function ``plot_foo`` (e.g.
    ``latest_ke_components`` -> ``plot_latest_ke_components``). When the
    function is found in ``plot_call_counts`` and:

    - ``entry["sparql"]`` is a string (current schema): warn if ``count > 1``.
    - ``entry["sparql"]`` is a list (future multi-query schema; not present in
      Phase 11): warn if ``count > len(entry["sparql"])``.

    Returns ``None`` when no warning applies. Per D2 in 11-CONTEXT.md the
    severity is ``LINT_04_SEVERITY`` (``"warn"`` this phase). The future
    schema-migration phase flips that constant to ``"fail"``.
    """
    func_name = f"plot_{name}"
    count = plot_call_counts.get(func_name)
    if not count:
        return None
    disclosed = entry.get("sparql")
    if isinstance(disclosed, str):
        disclosed_count = 1
    elif isinstance(disclosed, list):
        disclosed_count = len(disclosed)
    else:
        return None
    if count <= disclosed_count:
        return None
    return {
        "name": name,
        "check": "LINT-04",
        "severity": LINT_04_SEVERITY,
        "message": (
            f"Plot function {func_name} issues {count} SPARQL calls but "
            f"methodology entry discloses only {disclosed_count} query "
            "string(s). After the multi-query schema migration "
            "(issue #40), this will become a hard failure."
        ),
    }


# ---------------------------------------------------------------------------
# LINT-05: endpoint-call-budget guard
# ---------------------------------------------------------------------------


def enforce_call_budget(
    entries: dict,
    *,
    budget: int = DEFAULT_CALL_BUDGET,
    no_network: bool = False,
) -> None:
    """Refuse to start when the entry count exceeds the endpoint-call budget.

    Each entry costs one wrapped round-trip for LINT-01/02, plus one shared
    latest-graph resolution call per run. With ``budget=50`` and 46 real
    entries today, a 30s per-call timeout keeps total run-time well under
    the LINT-05 "<2 minutes" target.

    The guard is a no-op when ``no_network=True`` because the static-only
    path issues zero endpoint calls. Raises ``BudgetExceeded`` otherwise.
    """
    if no_network:
        return
    count = len(entries)
    if count > budget:
        raise BudgetExceeded(
            f"methodology JSON has {count} entries but the endpoint-call "
            f"budget is {budget} (LINT-05). Either batch the per-entry "
            "round-trip, raise the budget with --call-budget, or split the "
            "methodology file."
        )


# ---------------------------------------------------------------------------
# LINT-06: deterministic PR-comment + JSON-report formatters
# ---------------------------------------------------------------------------


# Per-check actionable fix hints surfaced in the markdown PR comment.
FIX_HINTS = {
    "LINT-01": (
        "Test the wrapped query against the live endpoint; the SP031-class "
        "error names a specific projection/aggregate mismatch - usually a "
        "variable in the SELECT that is missing from GROUP BY."
    ),
    "LINT-02": (
        "Verify the predicate URIs exist in the latest graph. The May 2026 "
        "audit found 12 entries using the bogus `aopo:has_biological_process` "
        "form instead of `aopo:hasBiologicalEvent -> hasProcess`. Run the "
        "query directly at the SPARQL endpoint to inspect the result shape."
    ),
    "LINT-03": (
        "Add `__GRAPH_URI__` to the query OR add both "
        "`ORDER BY DESC(?graph) LIMIT 1` and "
        "`FILTER(STRSTARTS(STR(?graph), \"http://aopwiki.org/graph/\"))` so "
        "the lint can confirm a latest-snapshot constraint."
    ),
}

# Stable bot-comment marker so future idempotency work can edit-in-place.
PR_COMMENT_MARKER = "<!-- methodology-lint -->"


def _sorted_findings(findings: list[dict]) -> list[dict]:
    """Deterministic sort by ``(name, check)`` so output is byte-stable."""
    return sorted(
        findings,
        key=lambda f: (f.get("name", ""), f.get("check", "")),
    )


def format_pr_comment(
    findings: list[dict],
    *,
    methodology_path: str,
    total_entries: int,
    budget: int = DEFAULT_CALL_BUDGET,
    no_network: bool = False,
) -> str:
    """Render a deterministic markdown PR comment for the lint findings.

    Determinism: findings are sorted by ``(name, check)`` before rendering,
    so the same finding list always produces the same byte sequence. The
    header carries a ``<!-- methodology-lint -->`` HTML-comment marker so
    future idempotency work can locate and replace the prior bot comment.
    """
    findings = _sorted_findings(findings)
    fails = [f for f in findings if f.get("severity") == "fail"]
    warns = [f for f in findings if f.get("severity") == "warn"]

    lines = [
        PR_COMMENT_MARKER,
        (
            f"## Methodology lint: {total_entries} entries checked, "
            f"{len(fails)} failures, {len(warns)} warnings"
        ),
        "",
        f"Source: `{methodology_path}`",
        "",
    ]

    if fails:
        lines.append("### Failures")
        lines.append("")
        for f in fails:
            msg = f.get("message", "").replace("\n", " ")
            lines.append(
                f"- **{f['name']}** - `{f['check']}` - {msg}"
            )
            hint = FIX_HINTS.get(f["check"])
            if hint:
                lines.append(f"  > Fix hint: {hint}")
        lines.append("")

    if warns:
        lines.append("### Warnings (soft)")
        lines.append("")
        for f in warns:
            msg = f.get("message", "").replace("\n", " ")
            lines.append(
                f"- **{f['name']}** - `{f['check']}` - {msg}"
            )
        lines.append("")

    if not fails and not warns:
        lines.append("All checks passed.")
        lines.append("")

    lines.append(
        f"_Lint config: budget={budget}, no_network={str(bool(no_network))}_"
    )
    return "\n".join(lines) + "\n"


def format_json_report(
    findings: list[dict],
    *,
    methodology_path: str,
    total_entries: int,
) -> str:
    """Render a deterministic, machine-readable JSON report.

    Findings are sorted by ``(name, check)`` so byte-for-byte output is
    stable. ``indent=2`` for human-readable artifact diffs.
    """
    findings = _sorted_findings(findings)
    fails = sum(1 for f in findings if f.get("severity") == "fail")
    warns = sum(1 for f in findings if f.get("severity") == "warn")
    payload = {
        "methodology_path": methodology_path,
        "total_entries": total_entries,
        "findings": findings,
        "summary": {"fail": fails, "warn": warns},
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_lint(
    json_path: pathlib.Path,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    plots_dir: Optional[pathlib.Path] = None,
    no_network: bool = False,
    budget: int = DEFAULT_CALL_BUDGET,
) -> list[dict]:
    """Run all enabled checks and return the merged findings list.

    Per D2 in 11-CONTEXT.md, LINT-04 emits ``severity="warn"`` findings that
    do NOT change the lint's overall exit code; the future-phase schema
    migration (issue #40) flips that to ``"fail"`` via the
    ``LINT_04_SEVERITY`` constant.

    Order of work:
      1. Load JSON.
      2. LINT-05 budget guard - refuse early if oversize.
      3. AST scan of ``plots_dir`` (amortised across all entries).
      4. Per-entry checks (LINT-03 static, LINT-01/02 endpoint, LINT-04 AST).
    """
    with open(json_path) as fh:
        notes = json.load(fh)

    enforce_call_budget(notes, budget=budget, no_network=no_network)

    plot_call_counts: dict[str, int] = {}
    if plots_dir is not None:
        plot_call_counts = scan_plots_dir(plots_dir)

    findings: list[dict] = []
    session: Optional[requests.Session] = None
    latest_graph_uri: Optional[str] = None
    if not no_network:
        session = requests.Session()
        latest_graph_uri = resolve_latest_graph_uri(endpoint, session=session)

    for name, entry in notes.items():
        if not isinstance(entry, dict):
            continue
        lint03 = check_lint_03(name, entry)
        if lint03 is not None:
            findings.append(lint03)
        lint04 = check_lint_04(name, entry, plot_call_counts)
        if lint04 is not None:
            findings.append(lint04)
        if no_network:
            continue
        # Only skip the endpoint round-trip when LINT-03 fired AND the entry
        # truly has no graph constraint at all - otherwise a partial-constraint
        # latest_* entry (e.g. ORDER BY + LIMIT 1 but no STRSTARTS) would
        # silently bypass LINT-01/02 even though its query still binds.
        query = entry.get("sparql", "")
        truly_unconstrained = (
            lint03 is not None
            and "__GRAPH_URI__" not in query
            and "ORDER BY DESC(?graph)" not in query
        )
        if truly_unconstrained:
            continue
        assert latest_graph_uri is not None  # for type-checkers
        finding = check_lint_01_02(
            name,
            entry,
            latest_graph_uri,
            endpoint=endpoint,
            session=session,
        )
        if finding is not None:
            findings.append(finding)
    return findings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", required=True, type=pathlib.Path,
                        help="Path to methodology_notes.json to lint")
    parser.add_argument("--plots-dir", default=pathlib.Path("plots"),
                        type=pathlib.Path,
                        help="Directory to AST-scan for LINT-04 (default: plots/)")
    parser.add_argument("--report", default=pathlib.Path("lint-report.md"),
                        type=pathlib.Path,
                        help="Markdown report destination")
    parser.add_argument("--json-report", default=pathlib.Path("lint-report.json"),
                        type=pathlib.Path,
                        help="Machine-readable JSON report destination")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="SPARQL endpoint URL")
    parser.add_argument("--no-network", action="store_true",
                        help="Skip endpoint calls; run static + AST checks only")
    parser.add_argument("--call-budget", type=int, default=DEFAULT_CALL_BUDGET,
                        help=(
                            "LINT-05 entry-count ceiling. The lint refuses to "
                            "start when the methodology JSON has more than "
                            "this many entries (default: 50)."
                        ))
    args = parser.parse_args(argv)

    try:
        findings = run_lint(
            args.json,
            endpoint=args.endpoint,
            plots_dir=args.plots_dir,
            no_network=args.no_network,
            budget=args.call_budget,
        )
    except BudgetExceeded as exc:
        sys.stderr.write(f"call-budget refusal: {exc}\n")
        return BUDGET_EXIT
    except InfraError as exc:
        sys.stderr.write(f"infrastructure error: {exc}\n")
        return INFRA_EXIT

    # Total entry count for the report header - re-read for accuracy.
    try:
        with open(args.json) as fh:
            total_entries = sum(1 for v in json.load(fh).values()
                                if isinstance(v, dict))
    except (OSError, ValueError):
        total_entries = len(findings)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        format_pr_comment(
            findings,
            methodology_path=str(args.json),
            total_entries=total_entries,
            budget=args.call_budget,
            no_network=args.no_network,
        )
    )
    args.json_report.write_text(
        format_json_report(
            findings,
            methodology_path=str(args.json),
            total_entries=total_entries,
        )
    )

    has_fail = any(f.get("severity") == "fail" for f in findings)
    if findings:
        sys.stderr.write(
            f"methodology-lint: {len(findings)} finding(s); see "
            f"{args.report} and {args.json_report}\n"
        )
    return LINT_FAIL_EXIT if has_fail else PASS_EXIT


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
