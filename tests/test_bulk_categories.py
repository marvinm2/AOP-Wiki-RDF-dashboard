"""Every plot id named in a bulk-download category must have a producer (#172).

``create_bulk_download`` looks each id up in the plot data/figure caches by name
and silently skips ids it can't find, so a stale id just yields a ZIP with files
missing. This test reads the category table from ``app.py`` (parsed, not
imported: importing ``app`` starts the live-endpoint precompute) and checks each
id against the cache keys that the plot modules write.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOT_SOURCES = sorted((ROOT / "plots").glob("*.py"))


def _bulk_categories() -> dict[str, list[str]]:
    tree = ast.parse((ROOT / "app.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "categories" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return ast.literal_eval(node.value)
    raise AssertionError("bulk-download `categories` dict not found in app.py")


def _cache_writers() -> tuple[set[str], set[str]]:
    """Return (exact keys, key prefixes) written to the plot caches."""
    src = "\n".join(p.read_text() for p in PLOT_SOURCES)
    cache = r"_plot_(?:data|figure)_cache\["
    exact = set(re.findall(cache + r"\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]\s*=", src))
    prefixes = set(re.findall(cache + r"\s*f['\"]([A-Za-z0-9_]+?)_\{", src))
    # domain_plots caches through _cache_plot("<stub>", version_key, df, fig)
    exact |= set(re.findall(r"_cache_plot\(\s*['\"]([A-Za-z0-9_]+)['\"]", src))
    return exact, prefixes


def test_every_bulk_id_has_a_producer():
    exact, prefixes = _cache_writers()
    missing = sorted({
        plot_id
        for ids in _bulk_categories().values()
        for plot_id in ids
        if plot_id not in exact and plot_id not in prefixes
    })
    assert not missing, f"bulk-download ids with no cache writer: {missing}"


def test_scanner_sees_known_writers():
    # Guard against the regexes silently matching nothing.
    exact, prefixes = _cache_writers()
    assert "aop_property_presence_absolute" in exact
    assert "latest_ke_by_bio_level" in prefixes
