"""Guards against silent wrong answers from Virtuoso (#173).

* Canary (slow, live endpoint): the dashboard's trend queries rely on
  ``GROUP BY ?graph`` with a ``STRSTARTS`` filter returning one correctly
  labelled row per version. The May 2026 ``:latest`` Virtuoso image returned the
  right counts under the wrong graph labels (#58), which is why the image is
  pinned by digest. Run this against a candidate image before moving the pin.
* Row cap (offline): a result as long as the endpoint's ResultSetMaxRows was
  probably truncated without any error, so it must at least be logged.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import requests

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots import shared  # noqa: E402

AOP = "<http://aopkb.org/aop_ontology#AdverseOutcomePathway>"


def _select(endpoint: str, query: str) -> list[dict]:
    r = requests.post(endpoint, data={"query": query},
                      headers={"Accept": "application/sparql-results+json"}, timeout=120)
    r.raise_for_status()
    return [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]


@pytest.mark.slow
def test_group_by_graph_labels_match_per_graph_counts(endpoint_url):
    grouped = _select(endpoint_url, f"""
        SELECT ?graph (COUNT(DISTINCT ?aop) AS ?n)
        WHERE {{ GRAPH ?graph {{ ?aop a {AOP} }}
                 FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/")) }}
        GROUP BY ?graph""")
    labels = [row["graph"] for row in grouped]
    assert len(labels) >= 30, f"expected one row per version, got {len(labels)}"
    assert len(set(labels)) == len(labels), "GROUP BY ?graph returned repeated labels (#58)"

    by_graph = {row["graph"]: int(row["n"]) for row in grouped}
    # Oldest, newest, and one in between: a mislabelling shows up as a mismatch.
    ordered = sorted(by_graph)
    for graph in {ordered[0], ordered[len(ordered) // 2], ordered[-1]}:
        single = _select(endpoint_url, f"""
            SELECT (COUNT(DISTINCT ?aop) AS ?n)
            WHERE {{ GRAPH <{graph}> {{ ?aop a {AOP} }} }}""")
        assert by_graph[graph] == int(single[0]["n"]), graph


@pytest.mark.slow
def test_optional_count_is_not_folded_across_graphs(endpoint_url):
    # The #168 shape: per-(graph, KE) counts with an OPTIONAL, over all graphs.
    grouped = _select(endpoint_url, """
        PREFIX aopo: <http://aopkb.org/aop_ontology#>
        SELECT ?graph (SUM(IF(?n = 0, 1, 0)) AS ?zero)
        WHERE {
          { SELECT ?graph ?ke (SUM(IF(BOUND(?b), 1, 0)) AS ?n)
            WHERE { GRAPH ?graph { ?ke a aopo:KeyEvent .
                                   OPTIONAL { ?ke aopo:hasBiologicalEvent ?b } }
                    FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/")) }
            GROUP BY ?graph ?ke }
        }
        GROUP BY ?graph""")
    newest = max(row["graph"] for row in grouped)
    single = _select(endpoint_url, f"""
        PREFIX aopo: <http://aopkb.org/aop_ontology#>
        SELECT (COUNT(?ke) AS ?zero)
        WHERE {{ GRAPH <{newest}> {{ ?ke a aopo:KeyEvent .
                 FILTER NOT EXISTS {{ ?ke aopo:hasBiologicalEvent ?b }} }} }}""")
    zero = {row["graph"]: int(row["zero"]) for row in grouped}[newest]
    assert zero == int(single[0]["zero"]) > 0


def _wrapper_returning(n_rows):
    class FakeWrapper:
        def __init__(self, endpoint): pass
        def setTimeout(self, t): pass
        def setReturnFormat(self, f): pass
        def setMethod(self, m): pass
        def setQuery(self, q): pass
        def query(self): return self
        def convert(self): return {"results": {"bindings": [{}] * n_rows}}
    return FakeWrapper


def test_result_at_the_row_cap_is_logged_as_truncated(monkeypatch, caplog):
    monkeypatch.setattr(shared.Config, "SPARQL_RESULT_ROW_CAP", 5)
    monkeypatch.setattr(shared, "SPARQLWrapper", _wrapper_returning(5))
    with caplog.at_level(logging.ERROR, logger="plots.shared"):
        rows = shared.run_sparql_query_with_retry("SELECT * WHERE { ?s ?p ?o }")
    assert len(rows) == 5
    assert "probably truncated" in caplog.text


def test_result_below_the_row_cap_is_not_flagged(monkeypatch, caplog):
    monkeypatch.setattr(shared.Config, "SPARQL_RESULT_ROW_CAP", 5)
    monkeypatch.setattr(shared, "SPARQLWrapper", _wrapper_returning(4))
    with caplog.at_level(logging.ERROR, logger="plots.shared"):
        shared.run_sparql_query_with_retry("SELECT * WHERE { ?s ?p ?o }")
    assert "truncated" not in caplog.text
