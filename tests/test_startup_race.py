"""Regression tests for the startup race with Virtuoso (#157).

On 2026-10-04 the dashboard and its Virtuoso were redeployed together. The
startup plot computation ran in the same second Virtuoso came online, got
``SP030: Undefined namespace prefix 'aopo'`` errors and empty results, and the
resulting fallback HTML was served for ~15 hours: 24 of 48 trend plots broken,
and all 72 of their downloads 404'ing.

Three defences, each pinned here:

* startup waits for a query that touches real AOP-Wiki data, not just HTTP;
* queries declare the prefixes they use instead of relying on Virtuoso's
  server-side namespace table;
* a trend plot whose startup run failed is recomputed on demand, rate-limited.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots import shared  # noqa: E402
from plots.recovery import TrendPlotRecovery, is_usable_plot_html  # noqa: E402

FALLBACK = '<div>Main graph - Data Unavailable</div>'


# --- add_missing_prefixes ---------------------------------------------------

def test_declares_a_prefix_the_query_uses():
    query = 'SELECT ?aop WHERE { ?aop a aopo:AdverseOutcomePathway . }'
    result = shared.add_missing_prefixes(query)
    assert result.startswith('PREFIX aopo: <http://aopkb.org/aop_ontology#>\n')
    assert result.endswith(query)


def test_leaves_an_explicit_declaration_alone():
    query = 'prefix aopo: <http://example.org/other#>\nSELECT * WHERE { ?a a aopo:X }'
    assert shared.add_missing_prefixes(query) == query


def test_full_iris_do_not_count_as_prefix_use():
    query = 'SELECT * WHERE { ?s <http://purl.org/dc/elements/1.1/title> "rdf:type" }'
    # The IRI and the string literal mention dc/rdf but use neither prefix...
    result = shared.add_missing_prefixes(query)
    assert 'PREFIX dc:' not in result
    # ...string literals are not parsed, so rdf: inside one is a harmless false positive.
    assert result.endswith(query)


def test_rdf_is_not_mistaken_for_rdfs():
    result = shared.add_missing_prefixes('SELECT * WHERE { ?s rdfs:label ?l }')
    assert 'PREFIX rdfs:' in result
    assert 'PREFIX rdf:' not in result


def test_only_missing_prefixes_are_added():
    query = 'PREFIX dc: <http://purl.org/dc/elements/1.1/>\nSELECT * WHERE { ?k a aopo:KeyEvent ; dc:title ?t ; nci:C1 ?x }'
    result = shared.add_missing_prefixes(query)
    assert 'PREFIX aopo:' in result and 'PREFIX nci:' in result
    assert result.count('PREFIX dc:') == 1


def test_run_sparql_query_sends_declared_prefixes(monkeypatch):
    sent = []

    class FakeWrapper:
        def __init__(self, endpoint): pass
        def setTimeout(self, t): pass
        def setReturnFormat(self, f): pass
        def setMethod(self, m): pass
        def setQuery(self, q): sent.append(q)
        def query(self): return self
        def convert(self): return {'results': {'bindings': []}}

    monkeypatch.setattr(shared, 'SPARQLWrapper', FakeWrapper)
    shared.run_sparql_query_with_retry('SELECT * WHERE { ?a a aopo:AdverseOutcomePathway }')
    assert sent and sent[0].startswith('PREFIX aopo:')


# --- wait_for_sparql_ready --------------------------------------------------

def _scripted_wrapper(answers):
    """A SPARQLWrapper stand-in that replays `answers` (an exception or an ASK boolean)."""
    answers = iter(answers)

    class FakeWrapper:
        def __init__(self, endpoint): pass
        def setTimeout(self, t): pass
        def setReturnFormat(self, f): pass
        def setQuery(self, q): assert 'AdverseOutcomePathway' in q
        def query(self): return self
        def convert(self):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return {'boolean': answer}

    return FakeWrapper


def test_waits_through_errors_and_empty_answers_until_data_is_there(monkeypatch):
    monkeypatch.setattr(shared, 'SPARQLWrapper', _scripted_wrapper(
        [ConnectionError('refused'), False, True]))
    assert shared.wait_for_sparql_ready(timeout=5, interval=0) is True


def test_gives_up_after_the_timeout(monkeypatch):
    monkeypatch.setattr(shared, 'SPARQLWrapper', _scripted_wrapper([False] * 1000))
    monkeypatch.setattr(shared.time, 'sleep', lambda s: None)
    clock = iter(range(0, 10_000, 3))
    monkeypatch.setattr(shared.time, 'time', lambda: next(clock))
    assert shared.wait_for_sparql_ready(timeout=10, interval=3) is False


# --- TrendPlotRecovery ------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _recovery(results, clock=None, cooldown=60):
    """A recovery over one tuple task and one single-HTML task, whose runs replay `results`."""
    calls = []
    results = iter(results)

    def main_graph():
        return next(results)

    def network_density():
        return '<div>density</div>'

    def run(func):
        calls.append(func.__name__)
        return func()

    recovery = TrendPlotRecovery(
        {'main_graph': main_graph, 'network_density': network_density},
        {
            'aop_entity_counts_absolute': ('main_graph', 0),
            'aop_entity_counts_delta': ('main_graph', 1),
            'aop_network_density': ('network_density', None),
        },
        run=run, cooldown=cooldown, clock=clock or FakeClock(),
    )
    return recovery, calls


def test_usable_startup_html_is_served_without_recomputing():
    recovery, calls = _recovery([])
    assert recovery.html('aop_entity_counts_absolute', '<div>ok</div>') == '<div>ok</div>'
    assert calls == []


def test_failed_startup_plot_is_recomputed_and_its_sibling_recovered_too():
    recovery, calls = _recovery([('<div>abs</div>', '<div>delta</div>', None)])
    assert recovery.html('aop_entity_counts_absolute', FALLBACK) == '<div>abs</div>'
    # The delta plot came from the same run: no second recompute.
    assert recovery.html('aop_entity_counts_delta', FALLBACK) == '<div>delta</div>'
    assert calls == ['main_graph']


def test_single_html_task_is_recovered():
    recovery, _ = _recovery([])
    assert recovery.html('aop_network_density', '') == '<div>density</div>'


def test_still_failing_plot_is_rate_limited():
    clock = FakeClock()
    recovery, calls = _recovery([(FALLBACK, FALLBACK, None), ('<div>abs</div>', '<div>d</div>', None)], clock)

    assert recovery.html('aop_entity_counts_absolute', FALLBACK) == FALLBACK
    clock.now += 30
    assert recovery.html('aop_entity_counts_absolute', FALLBACK) == FALLBACK
    assert calls == ['main_graph']  # inside the cooldown: no second query

    clock.now += 31
    assert recovery.html('aop_entity_counts_absolute', FALLBACK) == '<div>abs</div>'
    assert calls == ['main_graph', 'main_graph']


def test_recover_reports_unknown_plots_and_crashing_runs_as_unavailable():
    recovery, _ = _recovery([])
    assert recovery.recover('not_a_trend_plot') is None
    assert 'not_a_trend_plot' not in recovery

    def boom(func):
        raise RuntimeError('endpoint gone')

    crashing = TrendPlotRecovery({'t': lambda: None}, {'p': ('t', None)}, run=boom, cooldown=0)
    assert crashing.recover('p') is None


def test_concurrent_requests_trigger_one_recompute():
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_main_graph():
        calls.append(1)
        started.set()
        release.wait(5)
        return ('<div>abs</div>', '<div>delta</div>', None)

    recovery = TrendPlotRecovery(
        {'main_graph': slow_main_graph},
        {'aop_entity_counts_absolute': ('main_graph', 0), 'aop_entity_counts_delta': ('main_graph', 1)},
        run=lambda f: f(), cooldown=60,
    )
    results = []
    threads = [threading.Thread(target=lambda n=n: results.append(recovery.html(n, FALLBACK)))
               for n in ('aop_entity_counts_absolute', 'aop_entity_counts_delta')]
    threads[0].start()
    started.wait(5)
    threads[1].start()
    time.sleep(0.05)
    release.set()
    for t in threads:
        t.join(5)

    assert len(calls) == 1
    assert sorted(results) == ['<div>abs</div>', '<div>delta</div>']


@pytest.mark.parametrize('html, usable', [
    ('<div>plot</div>', True),
    ('', False),
    ('   ', False),
    (None, False),
    (FALLBACK, False),
])
def test_is_usable_plot_html(html, usable):
    assert is_usable_plot_html(html) is usable
