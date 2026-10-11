"""/health no longer counts every triple in the store (#177)."""

from __future__ import annotations

import sys
from pathlib import Path

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots import shared  # noqa: E402


def _wrapper(answer, sent):
    class FakeWrapper:
        def __init__(self, endpoint): pass
        def setTimeout(self, t): pass
        def setReturnFormat(self, f): pass
        def setQuery(self, q): sent.append(q)
        def query(self): return self
        def convert(self): return {"boolean": answer}
    return FakeWrapper


def test_health_uses_the_readiness_ask_and_is_memoized(monkeypatch):
    sent = []
    monkeypatch.setattr(shared, "_endpoint_health_cache", {})
    monkeypatch.setattr(shared, "SPARQLWrapper", _wrapper(True, sent))
    assert shared.check_sparql_endpoint_health() is True
    assert shared.check_sparql_endpoint_health() is True
    assert sent == [shared.READINESS_QUERY]
    assert "COUNT(*)" not in sent[0]


def test_an_endpoint_without_aopwiki_data_is_not_healthy(monkeypatch):
    monkeypatch.setattr(shared, "_endpoint_health_cache", {})
    monkeypatch.setattr(shared, "SPARQLWrapper", _wrapper(False, []))
    assert shared.check_sparql_endpoint_health() is False
