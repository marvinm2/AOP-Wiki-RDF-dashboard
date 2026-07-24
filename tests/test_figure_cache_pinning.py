"""Regression tests for VersionedPlotCache singleton-figure retention (#146).

The figure cache carries two kinds of entry:

* **Versioned historical snapshots** — keys ending in ``_YYYY-MM-DD`` (e.g.
  ``latest_taxonomic_groups_2020-01-01``). These are bounded by TTL + a
  max-versions cap so a user paging through 33 quarters can't grow the cache
  without limit. Only the pinned *current* version survives indefinitely.

* **Singleton figures** — everything else: the startup-computed trend figures
  (``main_graph_absolute``, ``ke_property_presence_absolute``, …) and the
  default no-version latest views (``latest_entity_counts``). These are
  computed once and never change during a process's lifetime.

The original ``_is_pinned`` only protected keys containing the pinned version
string, so singleton trend figures fell under the 1800s TTL and were evicted.
Trend plots serve pre-rendered HTML and never repopulate the figure cache on
view, so once evicted their PNG/SVG downloads 404'd permanently until the next
restart. These tests pin the retention contract so that regression can't return.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots.shared import VersionedPlotCache  # noqa: E402


# Representative singleton keys that must never expire — a spread of the trend
# figure suffixes plus a default (no-version) latest view.
SINGLETON_KEYS = [
    "main_graph_absolute",
    "ke_property_presence_percentage",
    "aop_completeness_boxplot_all",
    "oecd_status_distribution_percentage",
    "kes_by_kec_count_delta",
    "entity_birth_death",
    "latest_entity_counts",
    "latest_taxonomic_groups_latest",  # version_key='latest' when no date chosen
]


@pytest.fixture
def expired_cache():
    """A cache whose TTL has already elapsed for every non-pinned entry."""
    cache = VersionedPlotCache(max_versions=5, ttl_seconds=0)
    cache.pin_version("2026-07-01")
    return cache


@pytest.mark.parametrize("key", SINGLETON_KEYS)
def test_singleton_figures_survive_ttl(expired_cache, key):
    """Un-versioned figures are pinned and never expire, even past the TTL."""
    expired_cache[key] = "FIG"
    time.sleep(0.01)
    assert key in expired_cache
    assert expired_cache.get(key) == "FIG"
    # And they surface in keys() rather than being swept as expired.
    assert key in expired_cache.keys()


def test_current_version_snapshot_is_pinned(expired_cache):
    """A versioned entry matching the pinned prefix survives the TTL."""
    expired_cache["latest_taxonomic_groups_2026-07-01"] = "FIG"
    time.sleep(0.01)
    assert "latest_taxonomic_groups_2026-07-01" in expired_cache


def test_old_version_snapshot_expires(expired_cache):
    """A versioned entry NOT matching the pinned prefix follows the TTL."""
    expired_cache["latest_taxonomic_groups_2020-01-01"] = "FIG"
    time.sleep(0.01)
    assert "latest_taxonomic_groups_2020-01-01" not in expired_cache
    assert expired_cache.get("latest_taxonomic_groups_2020-01-01") is None


def test_version_cap_evicts_old_but_keeps_singletons_and_pinned():
    """The max-versions cap trims old snapshots without touching singletons."""
    cache = VersionedPlotCache(max_versions=2, ttl_seconds=3600)
    cache.pin_version("2026-07-01")

    cache["main_graph_absolute"] = "TREND"          # singleton
    cache["latest_ke_reuse_2026-07-01"] = "CURRENT"  # pinned current version
    # Three historical versions with a cap of 2 → the oldest must be evicted.
    cache["latest_ke_reuse_2019-01-01"] = "OLD1"
    cache["latest_ke_reuse_2020-01-01"] = "OLD2"
    cache["latest_ke_reuse_2021-01-01"] = "OLD3"

    # Singleton and pinned current version always retained.
    assert "main_graph_absolute" in cache
    assert "latest_ke_reuse_2026-07-01" in cache
    # The cap bounds the historical (non-pinned) versions.
    historical = [k for k in cache.keys() if k.startswith("latest_ke_reuse_20")
                  and "2026-07-01" not in k]
    assert len(historical) <= 2
