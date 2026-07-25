"""Regression tests for export cache rewarming across gunicorn workers (#148).

The plot data/figure caches are plain per-process dicts. Production runs
``workers = 2`` with ``preload_app = True``, so only entries created *before*
the fork — the startup plot computation — are shared via copy-on-write.

Two categories of entry are therefore missing from at least one worker:

* plots not in the startup set, first rendered lazily through
  ``/api/plot/<name>`` (e.g. ``latest_ke_reuse``, ``latest_top_ontology_terms``);
* **every** ``?version=YYYY-MM-DD`` render, since historical versions are never
  precomputed.

A download request is load-balanced independently of the render that preceded
it, so roughly half of them landed on the worker that had never computed the
plot and returned "No data available" (HTTP 404). Reproduced live before the
fix: 0/10 downloads succeeded cold, 10/10 after priming both workers.

The fix makes exports self-healing: on a cache miss the export re-runs the plot
function, which repopulates both caches as a side effect. These tests pin that
contract — including that a rewarm is attempted exactly once, is serialised per
key, and that a failing or unknown rewarm still degrades to the old behaviour
rather than raising.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots import shared  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_caches():
    """Isolate each test from module-level cache and hook state."""
    shared._plot_data_cache.clear()
    shared._plot_figure_cache.clear()
    previous_hook = shared._cache_rewarm_hook
    shared._rewarm_locks.clear()
    yield
    shared._plot_data_cache.clear()
    shared._plot_figure_cache.clear()
    shared._cache_rewarm_hook = previous_hook
    shared._rewarm_locks.clear()


def test_csv_export_recovers_from_a_cold_worker_cache():
    """The headline bug: a plot this worker never rendered still exports."""
    key = "latest_ke_reuse_2026-07-01"

    def rewarm(cache_key):
        # Stand-in for the real plot function, which populates both caches.
        shared._plot_data_cache[cache_key] = pd.DataFrame({"KE": ["KE 1"], "AOPs": [3]})
        return True

    shared.register_cache_rewarm(rewarm)

    csv_data = shared.get_csv_with_metadata(key, include_metadata=False)

    assert csv_data is not None, "export must recover instead of returning None"
    assert "KE 1" in csv_data


def test_export_without_a_registered_hook_still_returns_none():
    """No hook registered (e.g. plots used as a library) must not raise."""
    shared._cache_rewarm_hook = None

    assert shared.get_csv_with_metadata("latest_ke_reuse_2026-07-01") is None


def test_unknown_key_is_not_rewarmed_twice_and_returns_none():
    """A key the hook can't resolve degrades to the old 'not available' path."""
    calls = []

    def rewarm(cache_key):
        calls.append(cache_key)
        return False  # e.g. a trend plot, which has no rewarm path

    shared.register_cache_rewarm(rewarm)

    assert shared.get_csv_with_metadata("main_graph_absolute") is None
    assert calls == ["main_graph_absolute"], "hook should be attempted exactly once"


def test_rewarm_failure_does_not_propagate():
    """A broken plot function must not turn a 404 into a 500."""
    def rewarm(cache_key):
        raise RuntimeError("SPARQL endpoint down")

    shared.register_cache_rewarm(rewarm)

    assert shared.get_csv_with_metadata("latest_ke_reuse_2026-07-01") is None


def test_concurrent_exports_rewarm_once_per_key():
    """CSV+PNG+SVG clicked together must trigger one recompute, not three."""
    call_count = []
    barrier = threading.Barrier(3)

    def rewarm(cache_key):
        call_count.append(cache_key)
        shared._plot_data_cache[cache_key] = pd.DataFrame({"a": [1]})
        return True

    shared.register_cache_rewarm(rewarm)

    def worker():
        barrier.wait()  # maximise the overlap
        shared.get_csv_with_metadata("latest_taxonomic_groups_2026-07-01")

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(call_count) == 1, f"expected one recompute, got {len(call_count)}"


def test_cached_key_never_triggers_a_rewarm():
    """The hot path must not regress into recomputing on every export."""
    calls = []

    def rewarm(cache_key):
        calls.append(cache_key)
        return True

    shared.register_cache_rewarm(rewarm)
    shared._plot_data_cache["latest_entity_counts_latest"] = pd.DataFrame({"a": [1]})

    assert shared.get_csv_with_metadata("latest_entity_counts_latest") is not None
    assert calls == [], "a cache hit must not call the rewarm hook"


class TestCacheKeyResolution:
    """`_rewarm_plot_cache` must map a cache key back to (plot function, version)."""

    @staticmethod
    def _resolve(cache_key):
        """Parse a key the way app._rewarm_plot_cache does, without importing app.

        Importing app.py runs the full ~65s startup plot computation against a
        live SPARQL endpoint, so the parsing contract is asserted directly.
        """
        import re
        pattern = re.compile(r'^(?P<name>.+)_(?P<version>\d{4}-\d{2}-\d{2}|latest)$')
        match = pattern.match(cache_key)
        if not match:
            return cache_key, None
        version = match.group('version')
        return match.group('name'), (None if version == 'latest' else version)

    def test_dated_version_suffix(self):
        assert self._resolve("latest_ke_reuse_2026-07-01") == ("latest_ke_reuse", "2026-07-01")

    def test_latest_suffix_means_no_version_argument(self):
        assert self._resolve("latest_ke_reuse_latest") == ("latest_ke_reuse", None)

    def test_bare_key_has_no_version(self):
        assert self._resolve("latest_ke_reuse") == ("latest_ke_reuse", None)

    def test_plot_name_containing_digits_is_not_mistaken_for_a_version(self):
        assert self._resolve("latest_mie_ao_path_length") == ("latest_mie_ao_path_length", None)
