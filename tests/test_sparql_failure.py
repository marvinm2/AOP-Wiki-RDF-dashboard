"""A failed SPARQL query is an error, not an empty answer (#167).

The runner used to return ``[]`` after its retries, so a failed query looked
like "no data": per-version loops dropped the failed version and cached the
rest as a complete plot, and the bulk entity fetch cached empty sets for the
process lifetime (every entity then showed as removed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots import latest_plots, shared, trends_plots  # noqa: E402
from plots.shared import SparqlUnavailable  # noqa: E402
from plots.versions import VersionRegistry  # noqa: E402


def _wrapper(convert):
    class FakeWrapper:
        def __init__(self, endpoint): pass
        def setTimeout(self, t): pass
        def setReturnFormat(self, f): pass
        def setMethod(self, m): pass
        def setQuery(self, q): pass
        def query(self): return self
        def convert(self): return convert()
    return FakeWrapper


def _down():
    raise shared.requests.exceptions.ConnectionError("connection refused")


def test_runner_raises_after_retries(monkeypatch):
    monkeypatch.setattr(shared, "SPARQLWrapper", _wrapper(_down))
    monkeypatch.setattr(shared.time, "sleep", lambda s: None)
    with pytest.raises(SparqlUnavailable, match="connection refused"):
        shared.run_sparql_query_with_retry("SELECT * WHERE { ?s ?p ?o }")


def test_runner_returns_empty_list_for_a_genuinely_empty_answer(monkeypatch):
    monkeypatch.setattr(shared, "SPARQLWrapper", _wrapper(lambda: {"results": {"bindings": []}}))
    assert shared.run_sparql_query_with_retry("SELECT * WHERE { ?s ?p ?o }") == []


def test_bulk_entity_fetch_does_not_cache_a_failure(monkeypatch):
    monkeypatch.setattr(shared, "_entity_uris_cache", {})
    answers = iter([SparqlUnavailable("down"),
                    [{"graph": {"value": "http://aopwiki.org/graph/2026-10-01"},
                      "e": {"value": "http://aopwiki.org/aops/1"}}]])

    def run(query, **kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(shared, "run_sparql_query_with_retry", run)
    fetch = shared.fetch_entity_uris_by_version
    with pytest.raises(SparqlUnavailable):
        fetch("AOPs", ["2026-10-01"])
    assert shared._entity_uris_cache == {}
    assert fetch("AOPs", ["2026-10-01"]) == {"2026-10-01": {"http://aopwiki.org/aops/1"}}


def test_trend_with_one_failed_version_is_a_fallback_not_a_partial_plot(monkeypatch):
    versions = ["2026-04-01", "2026-07-01", "2026-10-01"]
    rows = [{"g": {"value": f"http://aopwiki.org/graph/{v}"}} for v in versions]
    monkeypatch.setattr(shared, "_version_registry", VersionRegistry(lambda q: rows))

    def query_version(version_info, use_distinct=False):
        if version_info["version"] == "2026-07-01":
            raise SparqlUnavailable("timeout")
        return {"version": version_info["version"], "Process": 1, "Object": 1, "Action": 1}

    monkeypatch.setattr(trends_plots, "_query_ke_components_version", query_version)
    key = "ke_component_annotations_absolute"
    before = trends_plots._plot_data_cache.get(key)

    absolute_html, _delta_html = trends_plots.plot_ke_components()[:2]

    assert "Data Unavailable" in absolute_html
    assert trends_plots._plot_data_cache.get(key) is before  # nothing partial cached


def test_snapshot_plot_raises_and_caches_nothing(monkeypatch):
    def down(*args, **kwargs):
        raise SparqlUnavailable("down")

    monkeypatch.setattr(latest_plots, "run_sparql_query", down)
    with pytest.raises(SparqlUnavailable):
        latest_plots.plot_latest_entity_counts("2026-10-01")
    assert "latest_entity_counts_2026-10-01" not in latest_plots._plot_data_cache
