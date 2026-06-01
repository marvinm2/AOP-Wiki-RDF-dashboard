"""Unit tests for the privacy-respecting usage analytics store.

Fast and offline: no SPARQL endpoint, no Flask app startup. Exercises the SQLite
event store directly against a temporary database.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))


@pytest.fixture
def usage(tmp_path, monkeypatch):
    """Fresh usage_analytics module bound to a temp DB."""
    monkeypatch.setenv("USAGE_DB_PATH", str(tmp_path / "usage.sqlite"))
    monkeypatch.setenv("ENABLE_USAGE_ANALYTICS", "True")
    # Reimport config + module so the env vars take effect and module state resets.
    for mod in ("usage_analytics", "config"):
        sys.modules.pop(mod, None)
    return importlib.import_module("usage_analytics")


def test_records_and_aggregates(usage):
    usage.record_event("view", plot="latest_entity_counts", version="2026-04-01", route="/api/plot/latest_entity_counts")
    usage.record_event("download", plot="latest_entity_counts", version="2026-04-01", fmt="csv", route="/download/latest_entity_counts")
    usage.record_event("download", plot="latest_entity_counts", version="2026-04-01", fmt="png", route="/download/latest_entity_counts")
    usage.record_event("download", plot="aop_network_density", version=None, fmt="csv", route="/download/trend/aop_network_density")

    s = usage.get_summary()
    assert s["enabled"] is True
    assert s["total_events"] == 4
    assert s["by_event"] == {"view": 1, "download": 3}
    assert s["by_format"]["csv"] == 2
    assert s["by_format"]["png"] == 1
    top = {d["plot"]: d["count"] for d in s["top_downloads"]}
    assert top["latest_entity_counts"] == 2
    assert top["aop_network_density"] == 1


def test_invalid_event_ignored(usage):
    usage.record_event("hack", plot="x")
    assert usage.get_summary()["total_events"] == 0


def test_disabled_flag_blocks_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("USAGE_DB_PATH", str(tmp_path / "u.sqlite"))
    monkeypatch.setenv("ENABLE_USAGE_ANALYTICS", "False")
    for mod in ("usage_analytics", "config"):
        sys.modules.pop(mod, None)
    mod = importlib.import_module("usage_analytics")
    mod.record_event("view", plot="x")
    assert mod.get_summary() == {"enabled": False}


def test_write_never_raises(usage, monkeypatch):
    # Point at an unwritable path mid-flight; record_event must swallow the error.
    monkeypatch.setattr(usage.Config, "USAGE_DB_PATH", "/proc/cannot/write/here.sqlite")
    usage._initialized = False
    usage.record_event("view", plot="x")  # should not raise
