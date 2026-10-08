"""Memoized version registry (#169)."""

from __future__ import annotations

import sys
from pathlib import Path

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots import shared  # noqa: E402
from plots.versions import VERSIONS_QUERY, VersionRegistry  # noqa: E402


def _rows(*versions):
    return [{"g": {"value": f"http://aopwiki.org/graph/{v}"}} for v in versions]


class FakeEndpoint:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, query):
        self.calls += 1
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_query_is_typed_and_unlimited():
    assert "AdverseOutcomePathway" in VERSIONS_QUERY
    assert "LIMIT" not in VERSIONS_QUERY.upper()
    assert "?s ?p ?o" not in VERSIONS_QUERY


def test_newest_first_and_memoized_within_ttl():
    endpoint = FakeEndpoint(_rows("2018-04-01", "2026-10-01", "2026-07-01"))
    clock = Clock()
    reg = VersionRegistry(endpoint, ttl=600, clock=clock)
    assert reg.versions() == ["2026-10-01", "2026-07-01", "2018-04-01"]
    clock.now = 599
    assert reg.latest() == "2026-10-01"
    assert endpoint.calls == 1


def test_refreshes_after_ttl_and_picks_up_a_new_quarter():
    endpoint = FakeEndpoint(_rows("2026-07-01"), _rows("2026-07-01", "2026-10-01"))
    clock = Clock()
    reg = VersionRegistry(endpoint, ttl=600, clock=clock)
    assert reg.latest() == "2026-07-01"
    clock.now = 601
    assert reg.latest() == "2026-10-01"
    assert endpoint.calls == 2


def test_failure_keeps_last_good_list():
    endpoint = FakeEndpoint(_rows("2026-07-01", "2026-10-01"), [], RuntimeError("down"))
    clock = Clock()
    reg = VersionRegistry(endpoint, ttl=10, clock=clock)
    assert reg.latest() == "2026-10-01"
    clock.now = 11
    assert reg.latest() == "2026-10-01"  # empty answer
    clock.now = 12
    assert reg.latest() == "2026-10-01"  # exception


def test_nothing_known_yet_retries_on_next_call():
    endpoint = FakeEndpoint([], _rows("2026-10-01"))
    reg = VersionRegistry(endpoint, ttl=600, clock=Clock())
    assert reg.latest() is None
    assert reg.latest() == "2026-10-01"
    assert endpoint.calls == 2


def test_ignores_non_version_graphs():
    rows = _rows("2026-10-01") + [{"g": {"value": "http://aopwiki-multirdf.vhp4safety.nl/metadata"}}]
    reg = VersionRegistry(FakeEndpoint(rows), clock=Clock())
    assert reg.versions() == ["2026-10-01"]


def test_shared_helpers_keep_their_contract(monkeypatch):
    reg = VersionRegistry(FakeEndpoint(_rows("2026-07-01", "2026-10-01")), clock=Clock())
    monkeypatch.setattr(shared, "_version_registry", reg)
    assert shared.get_latest_version() == "2026-10-01"
    assert shared.get_all_versions()[0] == {
        "version": "2026-10-01",
        "graph_uri": "http://aopwiki.org/graph/2026-10-01",
        "date": "2026-10-01",
    }
    monkeypatch.setattr(shared, "_version_registry",
                        VersionRegistry(FakeEndpoint([]), clock=Clock()))
    assert shared.get_latest_version() == "Unknown"
    assert shared.get_all_versions() == []
