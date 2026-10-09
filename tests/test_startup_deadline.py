"""Startup precompute deadline and the network-cache lock (#170)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots import network  # noqa: E402
from plots.recovery import run_tasks_with_deadline  # noqa: E402


def _boom():
    raise RuntimeError("query failed")


def test_deadline_records_stragglers_as_failed_and_returns_on_time():
    release = threading.Event()

    def slow():
        release.wait(5)
        return "late"

    started = time.monotonic()
    try:
        results = run_tasks_with_deadline(
            {"fast": lambda: "<div>plot</div>", "slow": slow, "broken": _boom},
            max_workers=3,
            timeout=0.3,
        )
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert results == {"fast": "<div>plot</div>", "slow": None, "broken": None}
    # The old as_completed() loop would have waited for `slow` (5 s).
    assert elapsed < 2


def test_queued_tasks_past_the_deadline_are_not_started():
    release = threading.Event()
    ran = []

    def blocker():
        release.wait(5)

    def queued():
        ran.append(True)

    try:
        results = run_tasks_with_deadline(
            {"blocker": blocker, "queued": queued}, max_workers=1, timeout=0.2)
    finally:
        release.set()
    time.sleep(0.1)
    assert results["queued"] is None
    assert ran == []


def test_concurrent_first_requests_build_the_network_once(monkeypatch):
    calls = {"latest": 0, "build": 0}

    def fake_latest():
        calls["latest"] += 1
        return "2026-10-01"

    def fake_build(version=None):
        calls["build"] += 1
        assert version == "2026-10-01"  # passed in, not looked up again
        time.sleep(0.2)
        g = network.nx.Graph()
        g.add_edge("ke1", "ke2", type="ker")
        return g

    monkeypatch.setattr(network, "_network_cache", {})
    monkeypatch.setattr(network, "get_latest_version", fake_latest)
    monkeypatch.setattr(network, "build_aop_network", fake_build)
    monkeypatch.setattr(network, "detect_ke_roles", lambda graph: {})
    monkeypatch.setattr(network, "graph_to_cytoscape_json", lambda *a, **k: [])

    out = []
    threads = [threading.Thread(target=lambda: out.append(network.get_or_compute_network()))
               for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == {"latest": 1, "build": 1}
    assert len(out) == 6 and all(o is out[0] for o in out)
