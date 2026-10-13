"""The snapshot page's versioned plots come from one place (#171).

The version selector used to keep its own list of plot names, which drifted from
LATEST_PLOT_FUNCTIONS so some plots never followed a version switch. The page
now publishes the set from the server; these checks keep it that way.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _latest_plot_functions() -> set[str]:
    # LATEST_PLOT_FUNCTIONS in app.py is built from these specs (#178).
    from plots import registry
    return {spec.id for spec in registry.snapshot_specs()}


def test_every_snapshot_container_is_versionable():
    containers = set(re.findall(r'class="lazy-plot"[^>]*data-plot-name="([^"]+)"',
                                (ROOT / "templates" / "latest.html").read_text()))
    assert len(containers) >= 30
    assert containers - _latest_plot_functions() == set()


def test_selector_keeps_no_plot_list_of_its_own():
    js = (ROOT / "static" / "js" / "version-selector.js").read_text()
    assert "versionedPlots" not in js
    assert "'latest_entity_counts'" not in js


def test_page_publishes_the_versioned_set_and_one_url_builder_exists():
    assert "window.VERSIONED_PLOTS" in (ROOT / "templates" / "latest.html").read_text()
    loader = (ROOT / "static" / "js" / "lazy-loading.js").read_text()
    assert "static plotUrl(element)" in loader
    for script in ("version-selector.js", "coverage-toggle.js"):
        assert "fetch(`/api/plot" not in (ROOT / "static" / "js" / script).read_text(), script
