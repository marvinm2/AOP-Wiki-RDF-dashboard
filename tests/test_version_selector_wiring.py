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
    page = (ROOT / "templates" / "latest.html").read_text()
    assert "window.VERSIONED_PLOTS" in page and "window.PLOT_PARAMS" in page
    runtime = (ROOT / "static" / "js" / "runtime.js").read_text()
    assert "plotUrl(element)" in runtime


def test_only_the_runtime_fetches_and_injects_plots():
    """One fetch-and-inject path (#181): every other script changes the state."""
    for path in sorted((ROOT / "static" / "js").glob("*.js")):
        if path.name == "runtime.js":
            continue
        js = path.read_text()
        assert "/api/plot/" not in js, path.name
        assert "data.html" not in js, path.name


def test_snapshot_controls_cover_every_plot_setting():
    """Every setting a snapshot plot takes is a state key the controls can write."""
    from plots import registry
    state_js = (ROOT / "static" / "js" / "state.js").read_text()
    keys = set(re.search(r"var KEYS = \[([^\]]+)\]", state_js).group(1).replace("'", "").replace(" ", "").split(","))
    params = {p for s in registry.snapshot_specs() for p in s.params} - {"min_shared_kes"}
    assert params <= keys
