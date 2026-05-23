"""Tests for ``scripts/lint_methodology.py``.

Two tiers:

- Default (no marker): pure unit tests with no network. Uses the ``responses``
  library to stub ``requests.post`` for LINT-01/02 classifier coverage.
- ``@pytest.mark.slow``: integration tests against the live multi-version
  endpoint at ``https://aopwiki-multirdf.vhp4safety.nl/sparql``. Verify
  assumption A4 (wrapped SP031 still returns HTTP 400) and exercise the
  bogus-predicate H1 anchor.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import requests
import responses

_HERE = pathlib.Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "lint_methodology.py"

_spec = importlib.util.spec_from_file_location("lint_methodology", _SCRIPT)
lint_methodology = importlib.util.module_from_spec(_spec)
sys.modules["lint_methodology"] = lint_methodology
assert _spec.loader is not None
_spec.loader.exec_module(lint_methodology)


def _only_entry(fixture_path: pathlib.Path) -> tuple[str, dict]:
    with open(fixture_path) as fh:
        data = json.load(fh)
    name, entry = next(iter(data.items()))
    return name, entry


# ---------------------------------------------------------------------------
# LINT-03 static checks (no network)
# ---------------------------------------------------------------------------


def test_lint03_no_constraint_fails(fixtures_dir):
    name, entry = _only_entry(fixtures_dir / "known_broken_no_constraint.json")
    findings = lint_methodology.check_lint_03(name, entry)
    assert findings
    assert findings[0]["check"] == "LINT-03"
    assert findings[0]["severity"] == "fail"


def test_lint03_known_good_passes(fixtures_dir):
    name, entry = _only_entry(fixtures_dir / "known_good.json")
    assert lint_methodology.check_lint_03(name, entry) == []


def test_lint03_orderby_strstarts_passes():
    # Inline entry: latest_* key, no __GRAPH_URI__ but the alternate
    # ORDER BY DESC(?graph) LIMIT 1 + STRSTARTS pair is present.
    name = "latest_orderby_strstarts_alt"
    entry = {
        "sparql": (
            "SELECT ?s WHERE {\n"
            "  GRAPH ?graph { ?s a aopo:AdverseOutcomePathway }\n"
            '  FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))\n'
            "} ORDER BY DESC(?graph) LIMIT 1"
        ),
    }
    assert lint_methodology.check_lint_03(name, entry) == []


def test_lint03_non_latest_key_skipped():
    # Keys that don't start with "latest_" are out of scope for LINT-03.
    name = "main_ke_components"
    entry = {"sparql": "SELECT ?s WHERE { ?s a aopo:KeyEvent } LIMIT 5"}
    assert lint_methodology.check_lint_03(name, entry) == []


def test_latest_graph_constraint_check(fixtures_dir):
    """Roll-up coverage of LINT-03 across the four shipped fixtures."""
    cases = {
        "known_broken_no_constraint.json": "LINT-03",
        "known_good.json": None,
        "known_broken_sp031.json": None,  # has ORDER BY + STRSTARTS + LIMIT 1
        "known_broken_empty.json": "LINT-03",  # uses neither pattern (verbatim copy)
    }
    for fname, expected in cases.items():
        name, entry = _only_entry(fixtures_dir / fname)
        results = lint_methodology.check_lint_03(name, entry)
        if expected is None:
            assert results == [], f"{fname}: expected pass, got {result}"
        else:
            assert results, f"{fname}: expected {expected} fail"
            assert results[0]["check"] == expected, (
                f"{fname}: expected {expected}, got {results[0]['check']}"
            )


# ---------------------------------------------------------------------------
# LINT-01/02 mocked-HTTP unit tests (no network)
# ---------------------------------------------------------------------------


@responses.activate
def test_lint01_classifier_on_mocked_400(endpoint_url):
    responses.add(
        responses.POST,
        endpoint_url,
        body=(
            "Virtuoso 37000 Error SP031: SPARQL compiler: Variable ?graph "
            "is used in the result set outside aggregate and not mentioned "
            "in GROUP BY clause"
        ),
        status=400,
    )
    findings = lint_methodology.check_lint_01_02(
        "latest_synthetic",
        {"sparql": "SELECT ?graph WHERE { GRAPH ?graph { ?s ?p ?o } } LIMIT 1"},
        "http://aopwiki.org/graph/2026-04-01",
        endpoint=endpoint_url,
    )
    assert findings
    assert findings[0]["check"] == "LINT-01"
    assert "SP031" in findings[0]["message"]


@responses.activate
def test_lint02_classifier_on_empty_bindings(endpoint_url):
    responses.add(
        responses.POST,
        endpoint_url,
        json={"results": {"bindings": []}},
        status=200,
    )
    findings = lint_methodology.check_lint_01_02(
        "latest_synthetic",
        {"sparql": "SELECT ?s WHERE { ?s :nonexistentPredicate ?o }"},
        "http://aopwiki.org/graph/2026-04-01",
        endpoint=endpoint_url,
    )
    assert findings
    assert findings[0]["check"] == "LINT-02"


@responses.activate
def test_lint02_classifier_on_all_zero_aggregate_row(endpoint_url):
    # The latest_ke_components anchor returns one row with all-zero COUNT()
    # values because the inner pattern uses OPTIONAL + GROUP BY ?graph.
    # The classifier must treat that as drift (LINT-02), not as a pass.
    responses.add(
        responses.POST,
        endpoint_url,
        json={
            "results": {
                "bindings": [
                    {
                        "graph": {
                            "type": "uri",
                            "value": "http://aopwiki.org/graph/2026-04-01",
                        },
                        "processes": {
                            "type": "literal",
                            "datatype": "http://www.w3.org/2001/XMLSchema#integer",
                            "value": "0",
                        },
                        "objects": {
                            "type": "literal",
                            "datatype": "http://www.w3.org/2001/XMLSchema#integer",
                            "value": "0",
                        },
                        "actions": {
                            "type": "literal",
                            "datatype": "http://www.w3.org/2001/XMLSchema#integer",
                            "value": "0",
                        },
                    }
                ]
            }
        },
        status=200,
    )
    findings = lint_methodology.check_lint_01_02(
        "latest_ke_components",
        {"sparql": "SELECT ?graph (COUNT(?p) AS ?processes) WHERE { ... }"},
        "http://aopwiki.org/graph/2026-04-01",
        endpoint=endpoint_url,
    )
    assert findings
    assert findings[0]["check"] == "LINT-02"


@responses.activate
def test_lint01_02_passes_when_results_nonempty(endpoint_url):
    responses.add(
        responses.POST,
        endpoint_url,
        json={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "42"}}
                ]
            }
        },
        status=200,
    )
    findings = lint_methodology.check_lint_01_02(
        "latest_known_good_entity_counts",
        {"sparql": "SELECT (COUNT(?aop) AS ?count) WHERE { ... }"},
        "http://aopwiki.org/graph/2026-04-01",
        endpoint=endpoint_url,
    )
    assert findings == []


# ---------------------------------------------------------------------------
# Orchestrator unit test (no network)
# ---------------------------------------------------------------------------


def test_run_lint_orchestrator_no_network(fixtures_dir, tmp_path):
    findings = lint_methodology.run_lint(
        fixtures_dir / "known_broken_no_constraint.json",
        no_network=True,
    )
    assert len(findings) == 1
    assert findings[0]["check"] == "LINT-03"


# ---------------------------------------------------------------------------
# Integration tests against the live endpoint (marker: slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sp031_anchor_fails(fixtures_dir, endpoint_url):
    """Verifies Assumption A4: wrapped SP031 anchor still returns HTTP 400."""
    name, entry = _only_entry(fixtures_dir / "known_broken_sp031.json")
    latest = lint_methodology.resolve_latest_graph_uri(endpoint_url)
    findings = lint_methodology.check_lint_01_02(
        name, entry, latest, endpoint=endpoint_url
    )
    assert findings
    assert findings[0]["check"] == "LINT-01"
    assert "SP031" in findings[0]["message"]


@pytest.mark.slow
def test_empty_result_fails(fixtures_dir, endpoint_url):
    name, entry = _only_entry(fixtures_dir / "known_broken_empty.json")
    latest = lint_methodology.resolve_latest_graph_uri(endpoint_url)
    findings = lint_methodology.check_lint_01_02(
        name, entry, latest, endpoint=endpoint_url
    )
    assert findings
    assert findings[0]["check"] == "LINT-02"


@pytest.mark.slow
def test_may_be_empty_overrides_lint02(fixtures_dir, endpoint_url):
    name, entry = _only_entry(fixtures_dir / "known_broken_empty.json")
    entry = {**entry, "may_be_empty": True}
    latest = lint_methodology.resolve_latest_graph_uri(endpoint_url)
    findings = lint_methodology.check_lint_01_02(
        name, entry, latest, endpoint=endpoint_url
    )
    assert findings == []


@pytest.mark.slow
def test_known_good_passes(fixtures_dir, endpoint_url):
    name, entry = _only_entry(fixtures_dir / "known_good.json")
    assert lint_methodology.check_lint_03(name, entry) == []
    latest = lint_methodology.resolve_latest_graph_uri(endpoint_url)
    findings = lint_methodology.check_lint_01_02(
        name, entry, latest, endpoint=endpoint_url
    )
    assert findings == []


# ---------------------------------------------------------------------------
# LINT-04 AST-scan unit tests (no network)
# ---------------------------------------------------------------------------


def _synthetic_plots_dir(fixtures_dir: pathlib.Path) -> pathlib.Path:
    return fixtures_dir / "synthetic_plots"


def test_lint04_over_disclosed_warns(fixtures_dir):
    plots_dir = _synthetic_plots_dir(fixtures_dir)
    counts = lint_methodology.scan_plots_dir(plots_dir)
    assert counts.get("plot_latest_over_disclosed") == 3
    finding = lint_methodology.check_lint_04(
        "latest_over_disclosed",
        {"sparql": "SELECT 1"},
        counts,
    )
    assert finding is not None
    assert finding["check"] == "LINT-04"
    assert finding["severity"] == "warn"
    assert "3" in finding["message"]


def test_lint04_balanced_passes(fixtures_dir):
    plots_dir = _synthetic_plots_dir(fixtures_dir)
    counts = lint_methodology.scan_plots_dir(plots_dir)
    assert counts.get("plot_latest_balanced") == 1
    finding = lint_methodology.check_lint_04(
        "latest_balanced",
        {"sparql": "SELECT 1"},
        counts,
    )
    assert finding is None


def test_lint04_under_disclosed_passes(fixtures_dir):
    plots_dir = _synthetic_plots_dir(fixtures_dir)
    counts = lint_methodology.scan_plots_dir(plots_dir)
    # The under_disclosed.py fixture has zero SPARQL helper calls; the
    # function name should therefore not appear in the counts dict at all.
    assert "plot_latest_under_disclosed" not in counts
    finding = lint_methodology.check_lint_04(
        "latest_under_disclosed",
        {"sparql": "SELECT 1"},
        counts,
    )
    assert finding is None


def test_lint04_severity_constant():
    # This test deliberately locks the soft-warn semantics for Phase 11. The
    # future schema-migration phase (issue #40) flips this to "fail" — that
    # flip breaks one test, which is the intended forcing-function so the
    # contributor acknowledges the change.
    assert lint_methodology.LINT_04_SEVERITY == "warn"


def test_lint04_warn_does_not_change_exit_code(fixtures_dir, tmp_path):
    synthetic_dir = _synthetic_plots_dir(fixtures_dir)
    synthetic_json = synthetic_dir / "methodology.json"
    report_md = tmp_path / "r.md"
    report_json = tmp_path / "r.json"
    rc = lint_methodology.main([
        "--json", str(synthetic_json),
        "--plots-dir", str(synthetic_dir),
        "--no-network",
        "--report", str(report_md),
        "--json-report", str(report_json),
    ])
    assert rc == 0, f"expected exit code 0 with only LINT-04 warns, got {rc}"
    payload = json.loads(report_json.read_text())
    assert payload["summary"]["warn"] >= 1
    assert payload["summary"]["fail"] == 0
    lint04_findings = [f for f in payload["findings"] if f["check"] == "LINT-04"]
    assert lint04_findings, "expected at least one LINT-04 finding"
    assert all(f["severity"] == "warn" for f in lint04_findings)


# ---------------------------------------------------------------------------
# LINT-05 budget-guard unit tests (no network)
# ---------------------------------------------------------------------------


def _fake_entries(n: int) -> dict:
    return {f"entry_{i}": {"sparql": "SELECT 1"} for i in range(n)}


def test_lint05_call_budget_guard():
    entries = _fake_entries(51)
    with pytest.raises(lint_methodology.BudgetExceeded) as excinfo:
        lint_methodology.enforce_call_budget(entries, budget=50, no_network=False)
    assert "51" in str(excinfo.value)
    assert "50" in str(excinfo.value)
    # no-network bypass:
    lint_methodology.enforce_call_budget(entries, budget=50, no_network=True)
    # Higher budget:
    lint_methodology.enforce_call_budget(entries, budget=100, no_network=False)


def test_lint05_real_methodology_under_budget():
    """Regression guard: when someone adds a 51st entry, this test fails."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    real_json = repo_root / "static" / "data" / "methodology_notes.json"
    with open(real_json) as fh:
        notes = json.load(fh)
    # Should NOT raise at default budget=50:
    lint_methodology.enforce_call_budget(notes, budget=50, no_network=False)


# ---------------------------------------------------------------------------
# LINT-06 PR-comment / JSON-report formatter unit tests
# ---------------------------------------------------------------------------


def _sample_findings() -> list[dict]:
    return [
        {"name": "latest_ontology_usage", "check": "LINT-01", "severity": "fail",
         "message": "400: Virtuoso 37000 Error SP031: ..."},
        {"name": "latest_ke_components", "check": "LINT-02", "severity": "fail",
         "message": "0 rows against latest graph"},
        {"name": "latest_aop_completeness", "check": "LINT-03", "severity": "fail",
         "message": "latest_* entry lacks a latest-snapshot constraint"},
        {"name": "latest_over_disclosed", "check": "LINT-04", "severity": "warn",
         "message": "Plot function plot_latest_over_disclosed issues 3 SPARQL calls but disclosed 1"},
    ]


def test_lint06_pr_comment_format():
    md = lint_methodology.format_pr_comment(
        _sample_findings(),
        methodology_path="x.json",
        total_entries=4,
    )
    assert "<!-- methodology-lint -->" in md
    assert "## Methodology lint" in md
    # Each finding's name appears at least once.
    for name in ("latest_ontology_usage", "latest_ke_components",
                 "latest_aop_completeness", "latest_over_disclosed"):
        assert name in md, f"missing entry name in PR comment: {name}"
    # Each fail check code is backtick-wrapped.
    for code in ("LINT-01", "LINT-02", "LINT-03"):
        assert f"`{code}`" in md, f"missing backtick-wrapped check code: {code}"
    # Fix hints appear at least three times (LINT-01/02/03).
    assert md.count("Fix hint:") >= 3
    # Warn finding is under "Warnings (soft)" section, not "Failures".
    warn_idx = md.find("latest_over_disclosed")
    warnings_idx = md.find("Warnings (soft)")
    failures_idx = md.find("Failures")
    assert warnings_idx != -1 and warn_idx > warnings_idx, (
        "LINT-04 warn finding must appear after the Warnings section header"
    )
    assert warn_idx > failures_idx  # warnings section is after failures


def test_lint06_deterministic_ordering():
    findings_a = _sample_findings()
    findings_b = list(reversed(findings_a))
    md_a = lint_methodology.format_pr_comment(
        findings_a, methodology_path="x.json", total_entries=4)
    md_b = lint_methodology.format_pr_comment(
        findings_b, methodology_path="x.json", total_entries=4)
    assert md_a == md_b, "PR comment must be byte-for-byte deterministic"
    json_a = lint_methodology.format_json_report(
        findings_a, methodology_path="x.json", total_entries=4)
    json_b = lint_methodology.format_json_report(
        findings_b, methodology_path="x.json", total_entries=4)
    assert json_a == json_b, "JSON report must be byte-for-byte deterministic"


def test_lint06_all_passed_message():
    md = lint_methodology.format_pr_comment(
        [], methodology_path="x.json", total_entries=46)
    assert "<!-- methodology-lint -->" in md
    assert "All checks passed" in md
    # The summary header reports zero failures and zero warnings.
    assert "0 failures" in md
    assert "0 warnings" in md
