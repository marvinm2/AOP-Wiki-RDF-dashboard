"""Snapshot plots are cached per concrete version, never under a bare key (#166).

Before this, most ``latest_*`` plots wrote a bare key that every ``?version=``
render overwrote, and downloads preferred that bare key, so a CSV named for one
version could hold another version's numbers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots import shared  # noqa: E402
from plots.versions import VersionRegistry  # noqa: E402

LATEST = "2026-10-01"
OLD = "2018-04-01"


@pytest.fixture
def registry(monkeypatch):
    rows = [{"g": {"value": f"http://aopwiki.org/graph/{v}"}} for v in (OLD, LATEST)]
    monkeypatch.setattr(shared, "_version_registry", VersionRegistry(lambda q: rows))


@pytest.fixture
def fake_plot(monkeypatch, registry):
    """A snapshot plot that caches the version it was asked for, like the real ones."""
    calls = []

    @shared.resolves_version
    def plot_latest_fake(version=None):
        calls.append(version)
        df = pd.DataFrame({"Entity": ["AOPs"], "Count": [599 if version == LATEST else 219],
                           "Version": [version]})
        shared._plot_data_cache[f"latest_fake_{version or 'latest'}"] = df
        shared._plot_figure_cache[f"latest_fake_{version or 'latest'}"] = object()
        return "<div>plot</div>"

    monkeypatch.setattr(shared, "_cache_rewarm_hook", lambda key: plot_latest_fake(
        re.search(r"_(\d{4}-\d{2}-\d{2})$", key).group(1)))
    yield plot_latest_fake, calls
    for key in list(shared._plot_data_cache.keys()):
        if key.startswith("latest_fake"):
            shared._plot_data_cache._data.pop(key, None)
            shared._plot_figure_cache._data.pop(key, None)


def test_resolve_version(registry):
    assert shared.resolve_version(None) == LATEST
    assert shared.resolve_version(OLD) == OLD


def test_plot_cache_key(registry):
    assert shared.plot_cache_key("latest_entity_counts") == f"latest_entity_counts_{LATEST}"
    assert shared.plot_cache_key("latest_entity_counts", OLD) == f"latest_entity_counts_{OLD}"
    assert shared.plot_cache_key(f"latest_entity_counts_{OLD}") == f"latest_entity_counts_{OLD}"
    assert shared.plot_cache_key("aop_network_density", OLD) == "aop_network_density"


def test_default_render_uses_the_concrete_latest_version(fake_plot):
    plot, calls = fake_plot
    plot()
    assert calls == [LATEST]


def test_historical_render_does_not_change_the_latest_download(fake_plot):
    plot, _ = fake_plot
    plot()             # startup / default view
    plot(OLD)          # someone looks at 2018-04-01
    latest_csv = shared.get_csv_with_metadata("latest_fake", include_metadata=False)
    old_csv = shared.get_csv_with_metadata("latest_fake", include_metadata=False, version=OLD)
    assert f"599,{LATEST}" in latest_csv and OLD not in latest_csv
    assert f"219,{OLD}" in old_csv and LATEST not in old_csv


def test_download_of_an_unrendered_version_recomputes_that_version(fake_plot):
    plot, calls = fake_plot
    plot()
    csv = shared.get_csv_with_metadata("latest_fake", include_metadata=False, version=OLD)
    assert calls == [LATEST, OLD]
    assert f"219,{OLD}" in csv


def test_export_filename_names_the_exported_version(registry):
    assert f"_v{LATEST}." in shared.build_export_filename("latest_entity_counts", "csv")
    assert f"_v{OLD}." in shared.build_export_filename("latest_entity_counts", "csv", OLD)
    assert "_v" not in shared.build_export_filename("aop_network_density", "csv")


def test_no_snapshot_plot_writes_a_bare_cache_key():
    src = "\n".join(p.read_text() for p in (_DASHBOARD_ROOT / "plots").glob("*.py"))
    bare = re.findall(r"_plot_(?:data|figure)_cache\[\s*['\"](latest_[A-Za-z0-9_]+)['\"]\s*\]\s*=", src)
    assert bare == [], f"bare snapshot cache writes: {bare}"


def test_every_snapshot_plot_function_resolves_its_version():
    for module in ("latest_plots", "domain_plots"):
        src = (_DASHBOARD_ROOT / "plots" / f"{module}.py").read_text()
        undecorated = re.findall(r"(?<!@resolves_version\n)^def (plot_latest_\w+)", src, re.M)
        assert undecorated == [], f"{module}: {undecorated}"
