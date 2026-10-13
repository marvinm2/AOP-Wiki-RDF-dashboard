"""The persistent SPARQL result store (#179). Offline."""

import gzip
import json
import os
import time

import pytest

from plots import query_store as qs
from plots import shared

G = qs.GRAPH_PREFIX
V1, V2 = f"{G}2024-01-01", f"{G}2026-10-01"


# --- which graphs a query reads -------------------------------------------

@pytest.mark.parametrize("query,expected", [
    (f"SELECT * WHERE {{ GRAPH <{V1}> {{ ?s ?p ?o }} }}", {V1}),
    (f"SELECT * WHERE {{ GRAPH <{V1}> {{ ?a ?b ?c }} GRAPH <{V2}> {{ ?a ?d ?e }} }}", {V1, V2}),
    # GRAPH ?graph pinned by an equality filter (the _build_graph_filter form)
    (f"SELECT ?graph WHERE {{ GRAPH ?graph {{ ?s a ?t }} FILTER(?graph = <{V2}>) }}", {V2}),
    (f"SELECT ?g WHERE {{ graph ?g {{ ?s a ?t }} filter( ?g =  <{V1}> ) }}", {V1}),
])
def test_queries_pinned_to_graphs(query, expected):
    assert qs.graph_dependencies(query) == frozenset(expected)


@pytest.mark.parametrize("query", [
    # every version
    f'SELECT ?graph WHERE {{ GRAPH ?graph {{ ?s a ?t }} FILTER(STRSTARTS(STR(?graph), "{G}")) }}',
    # latest via ORDER BY ... LIMIT 1
    "SELECT ?graph WHERE { GRAPH ?graph { ?s a ?t } } ORDER BY DESC(?graph) LIMIT 1",
    # a pinned graph next to an unpinned one
    f"SELECT * WHERE {{ GRAPH <{V1}> {{ ?s ?p ?o }} GRAPH ?other {{ ?s ?q ?r }} }}",
    # a comparison is not a pin
    f"SELECT * WHERE {{ GRAPH ?g {{ ?s ?p ?o }} FILTER(?g <= <{V1}>) }}",
    # no GRAPH at all: the default graph is every graph
    "SELECT * WHERE { ?s ?p ?o } LIMIT 1",
    f"SELECT * FROM <{V1}> WHERE {{ ?s ?p ?o }}",
])
def test_unpinned_queries_depend_on_every_graph(query):
    assert qs.graph_dependencies(query) is None


def test_fingerprints_ignore_value_order():
    rows = [{"g": {"value": V1}, "triples": {"value": "10"},
             "created_on": {"value": "2026-06-17 2026-06-16"}, "declared": {"value": "7 3"}},
            {"g": {"value": "http://example.org/other"}, "triples": {"value": "1"}}]
    same = [{"g": {"value": V1}, "triples": {"value": "10"},
             "created_on": {"value": "2026-06-16 2026-06-17"}, "declared": {"value": "3 7"}}]
    assert qs.fingerprints_from_rows(rows) == qs.fingerprints_from_rows(same)
    assert list(qs.fingerprints_from_rows(rows)) == [V1]


# --- keys move with exactly the graphs a query reads ------------------------

def _store(tmp_path, fps):
    store = qs.QueryStore(str(tmp_path))
    store.use_fingerprints(fps)
    return store


PER_V1 = f"SELECT * WHERE {{ GRAPH <{V1}> {{ ?s ?p ?o }} }}"
ALL = "SELECT * WHERE { GRAPH ?g { ?s ?p ?o } }"


def test_key_moves_only_with_the_graphs_read(tmp_path):
    before = _store(tmp_path, {V1: "a", V2: "b"})
    k_per, k_all = before.key(PER_V1), before.key(ALL)
    after_v2 = _store(tmp_path, {V1: "a", V2: "changed"})
    assert after_v2.key(PER_V1) == k_per
    assert after_v2.key(ALL) != k_all
    after_v1 = _store(tmp_path, {V1: "changed", V2: "b"})
    assert after_v1.key(PER_V1) != k_per
    # a new graph moves only the all-graph queries
    added = _store(tmp_path, {V1: "a", V2: "b", f"{G}2027-01-01": "c"})
    assert added.key(PER_V1) == k_per and added.key(ALL) != k_all


def test_query_on_an_unknown_graph_has_no_key(tmp_path):
    store = _store(tmp_path, {V2: "b"})
    assert store.key(PER_V1) is None
    assert store.get(PER_V1) is None
    store.put(PER_V1, [{"x": 1}])  # silently not stored
    assert store.writes == 0


def test_round_trip_and_atomic_files(tmp_path):
    store = _store(tmp_path, {V1: "a"})
    assert store.get(PER_V1) is None
    rows = [{"s": {"type": "uri", "value": "x"}}]
    store.put(PER_V1, rows)
    assert store.get(PER_V1) == rows
    assert store.get(PER_V1 + " ") is None  # the text is the key
    files = [f for _, _, fs in os.walk(tmp_path) for f in fs]
    assert files == [f"{store.key(PER_V1)}.json.gz"]
    entry = json.loads(gzip.decompress((tmp_path / "results" / files[0][:2] / files[0]).read_bytes()))
    assert entry["graphs"] == [V1] and entry["query"] == PER_V1


def test_empty_answer_is_a_hit(tmp_path):
    store = _store(tmp_path, {V1: "a"})
    store.put(PER_V1, [])
    assert store.get(PER_V1) == []


# --- readers follow the published manifest ----------------------------------

def test_reader_waits_for_the_manifest(tmp_path):
    writer = _store(tmp_path, {V1: "a"})
    writer.put(PER_V1, [{"n": 1}])
    reader = qs.QueryStore(str(tmp_path), manifest_ttl=0)
    assert reader.get(PER_V1) is None  # no manifest yet: nothing is served
    writer.publish({"ran": 1})
    assert reader.get(PER_V1) == [{"n": 1}]

    # The next build writes new results first; the reader keeps the old ones
    # until the new manifest is published.
    nxt = _store(tmp_path, {V1: "reloaded"})
    nxt.put(PER_V1, [{"n": 2}])
    assert reader.get(PER_V1) == [{"n": 1}]
    time.sleep(0.01)
    nxt.publish()
    os.utime(reader.manifest_path)  # coarse mtime on some filesystems
    assert reader.get(PER_V1) == [{"n": 2}]


def test_unreadable_manifest_keeps_the_last_good_one(tmp_path):
    writer = _store(tmp_path, {V1: "a"})
    writer.put(PER_V1, [{"n": 1}])
    writer.publish()
    reader = qs.QueryStore(str(tmp_path), manifest_ttl=0)
    assert reader.get(PER_V1) == [{"n": 1}]
    (tmp_path / "manifest.json").write_text("{ truncated")
    assert reader.get(PER_V1) == [{"n": 1}]


def test_prune_keeps_referenced_and_recent_results(tmp_path):
    store = _store(tmp_path, {V1: "a"})
    store.put(PER_V1, [1])
    store.put(ALL, [2])
    keep = {store.key(PER_V1)}
    assert store.prune(keep, older_than=3600) == 0  # both recent
    assert store.prune(keep, older_than=-1) == 1
    assert store.get(PER_V1) == [1] and store.get(ALL) is None


# --- single writer -------------------------------------------------------------

def test_build_lock_is_exclusive_until_stale(tmp_path):
    first, second = qs.BuildLock(str(tmp_path)), qs.BuildLock(str(tmp_path))
    assert first.acquire()
    assert not second.acquire()
    old = time.time() - 7 * 3600
    os.utime(first.path, (old, old))
    assert second.acquire()  # a killed build's lock is taken over
    second.release()
    assert first.acquire()
    first.release()


# --- the SPARQL runner consults the store ------------------------------------

class _Endpoint:
    def __init__(self, rows):
        self.rows, self.calls = rows, 0

    def __call__(self, *args, **kwargs):
        endpoint = self

        class Wrapper:
            def setTimeout(self, t): pass
            def setReturnFormat(self, f): pass
            def setQuery(self, q): pass
            def setMethod(self, m): pass

            def query(self):
                endpoint.calls += 1
                return type("R", (), {"convert": lambda s: {"results": {"bindings": endpoint.rows}}})()
        return Wrapper()


@pytest.fixture
def endpoint(monkeypatch):
    ep = _Endpoint([{"n": {"value": "1"}}])
    monkeypatch.setattr(shared, "SPARQLWrapper", ep)
    yield ep
    qs.configure(None)


def test_runner_writes_then_serves_from_the_store(tmp_path, endpoint):
    store = qs.configure(str(tmp_path), writable=True)
    store.use_fingerprints({V1: "a"})
    assert shared.run_sparql_query_with_retry(PER_V1) == endpoint.rows
    assert shared.run_sparql_query_with_retry(PER_V1) == endpoint.rows
    assert endpoint.calls == 1 and store.writes == 1 and store.hits == 1


def test_read_only_runner_falls_back_without_writing(tmp_path, endpoint):
    writer = qs.configure(str(tmp_path), writable=True)
    writer.use_fingerprints({V1: "a"})
    shared.run_sparql_query_with_retry(PER_V1)
    writer.publish()
    reader = qs.configure(str(tmp_path), manifest_ttl=0)
    calls = endpoint.calls
    assert shared.run_sparql_query_with_retry(PER_V1) == endpoint.rows
    assert endpoint.calls == calls  # served from the store
    assert shared.run_sparql_query_with_retry(ALL) == endpoint.rows
    assert endpoint.calls == calls + 1 and reader.writes == 0
    assert len(list((tmp_path / "results").rglob("*.json.gz"))) == 1  # nothing new written


def test_failed_query_is_counted_and_not_stored(tmp_path, monkeypatch):
    from SPARQLWrapper import SPARQLExceptions

    class Broken:
        def __init__(self, *a): pass
        def setTimeout(self, t): pass
        def setReturnFormat(self, f): pass
        def setQuery(self, q): pass
        def query(self): raise SPARQLExceptions.EndPointNotFound("down")

    monkeypatch.setattr(shared, "SPARQLWrapper", Broken)
    store = qs.configure(str(tmp_path), writable=True)
    store.use_fingerprints({V1: "a"})
    try:
        with pytest.raises(shared.SparqlUnavailable):
            shared.run_sparql_query_with_retry(PER_V1)
        assert store.failures == 1 and store.writes == 0
    finally:
        qs.configure(None)
