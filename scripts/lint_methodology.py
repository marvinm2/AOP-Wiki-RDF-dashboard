#!/usr/bin/env python3
"""Methodology JSON lint for the AOP-Wiki RDF dashboard.

Validates every entry in `static/data/methodology_notes.json` against the live
SPARQL endpoint and against simple static rules. This file ships the first
half of the lint contract (LINT-01/02/03); LINT-04 (AST scan), LINT-05 (call
budget) and LINT-06 (PR comment formatting) land in a later plan.

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

Exit codes
----------
0  No findings.
1  At least one lint finding.
2  Infrastructure error (DNS failure, connection refused, timeout, 5xx).

Design constraints
------------------
- Pure stdlib + ``requests``. Do NOT import from ``plots.*`` (Pitfall P2).
- Do NOT use rdflib for LINT-01 (Pitfall P1: rdflib does not catch SP031).
- One endpoint round-trip per entry (LINT-05 budget).
- TLS validation enabled (default ``verify=True``); 30s per-call timeout.
"""

from __future__ import annotations

import argparse
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


class InfraError(RuntimeError):
    """Raised when the endpoint is unreachable or returns 5xx.

    Distinguished from lint failures so the CI runner can exit 2 (infra) vs.
    1 (lint) and avoid masking transient network issues as code defects.
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
# Orchestrator
# ---------------------------------------------------------------------------


def run_lint(
    json_path: pathlib.Path,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    plots_dir: Optional[pathlib.Path] = None,  # accepted for CLI stability
    no_network: bool = False,
) -> list[dict]:
    """Load the methodology JSON and run LINT-03 (+ LINT-01/02 when on-net).

    LINT-04/05/06 belong to a follow-up plan. ``plots_dir`` is accepted but
    unused at this stage so the CLI surface stays stable.
    """
    del plots_dir  # TODO(plan-02): AST scan for LINT-04
    with open(json_path) as fh:
        notes = json.load(fh)

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
        if no_network:
            continue
        # Only skip the endpoint round-trip when LINT-03 fired AND the entry
        # truly has no graph constraint at all — otherwise a partial-constraint
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


def _format_markdown(findings: list[dict], json_path: pathlib.Path) -> str:
    if not findings:
        return f"# methodology-lint\n\nAll entries in `{json_path}` pass.\n"
    lines = [
        "# methodology-lint",
        "",
        f"`{json_path}` produced {len(findings)} finding(s).",
        "",
        "| Entry | Check | Message |",
        "| --- | --- | --- |",
    ]
    for f in findings:
        msg = f.get("message", "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{f['name']}` | {f['check']} | {msg} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", required=True, type=pathlib.Path,
                        help="Path to methodology_notes.json to lint")
    parser.add_argument("--plots-dir", default=pathlib.Path("plots"),
                        type=pathlib.Path,
                        help="(Accepted for CLI stability; unused in this plan)")
    parser.add_argument("--report", default=pathlib.Path("lint-report.md"),
                        type=pathlib.Path,
                        help="Markdown report destination")
    parser.add_argument("--json-report", default=pathlib.Path("lint-report.json"),
                        type=pathlib.Path,
                        help="Machine-readable JSON report destination")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="SPARQL endpoint URL")
    parser.add_argument("--no-network", action="store_true",
                        help="Skip endpoint calls; run LINT-03 static check only")
    args = parser.parse_args(argv)

    try:
        findings = run_lint(
            args.json,
            endpoint=args.endpoint,
            plots_dir=args.plots_dir,
            no_network=args.no_network,
        )
    except InfraError as exc:
        sys.stderr.write(f"infrastructure error: {exc}\n")
        return INFRA_EXIT

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_format_markdown(findings, args.json))
    args.json_report.write_text(json.dumps({"findings": findings}, indent=2) + "\n")

    if findings:
        sys.stderr.write(
            f"methodology-lint: {len(findings)} finding(s); see "
            f"{args.report} and {args.json_report}\n"
        )
        return LINT_FAIL_EXIT
    return PASS_EXIT


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
