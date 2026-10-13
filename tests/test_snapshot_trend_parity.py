"""Plots served from the query store equal plots rendered live (#179).

Each case renders a trend plot and a snapshot plot against a stub endpoint
while the store records, then renders them again with the store read-only and
the endpoint down. The output must be identical and the endpoint must not be
called. (Against the live endpoint this was checked for every /api/plot id and
download when the store was introduced.) Offline.
"""

import re

import pytest
from SPARQLWrapper import SPARQLExceptions

import plots
from plots import query_store, shared

GRAPH = "http://aopwiki.org/graph/"
VERSION = "2026-10-01"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TITLE = "http://purl.org/dc/elements/1.1/title"


def answer(query):
    """Plausible bindings for the queries the two plots below issue."""
    if "SELECT ?graph\n" in query or "SELECT ?graph WHERE" in query:
        return [{"graph": {"value": GRAPH + VERSION}}]
    if "?total" in query and "?status" in query:
        return [{"status": {"value": "WPHA/WNT Endorsed"}, "total": {"value": "8"}}]
    if "?count_with_property" in query:
        return [{"status": {"value": "WPHA/WNT Endorsed"}, "p": {"value": TITLE},
                 "count_with_property": {"value": "6"}}]
    if "?total" in query:
        return [{"graph": {"value": GRAPH + v}, "total": {"value": "10"}}
                for v in ("2024-01-01", "2025-01-01")]
    return [{"graph": {"value": GRAPH + v}, "p": {"value": TITLE}, "count": {"value": n}}
            for v, n in (("2024-01-01", "4"), ("2025-01-01", "7"))]


class Endpoint:
    """Stands in for SPARQLWrapper; ``up=False`` makes every query fail."""

    def __init__(self):
        self.up, self.calls = True, 0

    def __call__(self, *args):
        endpoint, state = self, {}

        class Wrapper:
            def setTimeout(self, t): pass
            def setReturnFormat(self, f): pass
            def setMethod(self, m): pass
            def setQuery(self, q): state["q"] = q

            def query(self):
                endpoint.calls += 1
                if not endpoint.up:
                    raise SPARQLExceptions.EndPointNotFound("down")
                rows = answer(state["q"])
                return type("R", (), {"convert": lambda s: {"results": {"bindings": rows}}})()
        return Wrapper()


CASES = {
    "trend": lambda: plots.plot_property_presence("ke"),
    "snapshot": lambda: plots.plot_latest_completeness_by_status(VERSION, entity="ker"),
}


@pytest.mark.parametrize("case", CASES)
def test_store_renders_what_the_endpoint_rendered(case, tmp_path, monkeypatch):
    endpoint = Endpoint()
    monkeypatch.setattr(shared, "SPARQLWrapper", endpoint)
    monkeypatch.chdir(shared.os.path.dirname(shared.os.path.dirname(shared.__file__)))
    try:
        writer = query_store.configure(str(tmp_path), writable=True)
        writer.use_fingerprints({GRAPH + v: "fp" for v in ("2024-01-01", "2025-01-01", VERSION)})
        live = CASES[case]()
        assert writer.writes and endpoint.calls == writer.writes
        writer.publish()

        endpoint.up, calls = False, endpoint.calls
        reader = query_store.configure(str(tmp_path), manifest_ttl=0)
        stored = CASES[case]()
        assert endpoint.calls == calls, "a stored query went to the endpoint"
        assert reader.misses == 0 and reader.hits == writer.writes
        assert UUID.sub("U", str(stored)) == UUID.sub("U", str(live))
        assert "Data Unavailable" not in str(stored)
    finally:
        query_store.configure(None)
