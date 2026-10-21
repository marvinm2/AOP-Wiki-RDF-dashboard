"""The plot registry agrees with the plot code, the templates and the notes (#178).

``plots/registry.py`` is the one place a plot is declared. Adding a plot means
adding its spec, its container in a page template, and its methodology note;
these tests fail when one of the three is missing or they disagree. Offline:
no endpoint is contacted and ``app`` is not imported (that would run the
startup precompute).
"""

import inspect
import json
import pathlib
import re

import pytest

import plots
from plots import registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTES = json.loads((ROOT / "static" / "data" / "methodology_notes.json").read_text())
TEMPLATES = {"snapshot": "latest.html", "trends": "trends.html", "diff": "diff.html"}


NOTE_RE = r"methodology_note\(methodology_notes, '([^']+)'\)"


def _cards(page):
    """Plot cards of a page template: one methodology note, its containers and download links.

    A card whose tab panels each carry their own note (the property heatmap,
    #187) counts as one card per panel; the card's download links may point at
    any of its panels' plots.
    """
    html = (ROOT / "templates" / TEMPLATES[page]).read_text()
    cards = []
    for chunk in re.split(r'<div class="plot-box', html)[1:]:
        downloads = re.findall(r'href="/download/([^"?]+)', chunk)
        card_plots = re.findall(r'data-plot-name="([^"]+)"', chunk)
        panels = re.split(r'role="tabpanel"', chunk)[1:]
        units = panels if panels and all(re.search(NOTE_RE, p) for p in panels) else [chunk]
        cards += [{"notes": re.findall(NOTE_RE, unit), "plots": re.findall(r'data-plot-name="([^"]+)"', unit),
                   "downloads": downloads, "card_plots": card_plots} for unit in units]
    return cards


CARDS = {page: _cards(page) for page in TEMPLATES}
SPECS = registry.SPECS


def test_ids_are_unique():
    ids = [s.id for s in SPECS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.id)
def test_producer_exists(spec):
    assert callable(getattr(plots, spec.producer, None)), spec.producer


@pytest.mark.parametrize("spec", registry.snapshot_specs(), ids=lambda s: s.id)
def test_params_match_signature(spec):
    """/api/plot passes exactly the tuning args the function accepts."""
    signature = inspect.signature(getattr(plots, spec.producer)).parameters
    accepted = tuple(p for p in signature if p in registry.tuning_params())
    assert spec.params == accepted


@pytest.mark.parametrize("spec", [s for s in SPECS if s.args], ids=lambda s: s.id)
def test_fixed_args_are_accepted_by_the_producer(spec):
    signature = inspect.signature(getattr(plots, spec.producer)).parameters
    assert set(spec.kwargs) <= set(signature), spec.kwargs
    assert not set(spec.kwargs) & set(spec.params)


def test_trend_outputs_are_distinct_per_task():
    seen = {}
    for s in registry.trend_specs():
        key = (s.task, s.output)
        assert key not in seen, f"{s.id} and {seen[key]} both read output {key}"
        seen[key] = s.id
    for s in registry.trend_specs():
        same_task = [t for t in registry.trend_specs() if t.task == s.task]
        assert {(t.producer, t.args) for t in same_task} == {(s.producer, s.args)}


@pytest.mark.parametrize("spec", [s for s in SPECS if s.note], ids=lambda s: s.id)
def test_note_exists(spec):
    assert spec.note in NOTES


def test_every_note_belongs_to_a_plot():
    assert set(NOTES) <= {s.note for s in SPECS} | registry.CONTROL_NOTES


@pytest.mark.parametrize("spec", [s for s in SPECS if s.page], ids=lambda s: s.id)
def test_container_is_on_its_page_with_its_note(spec):
    cards = [c for c in CARDS[spec.page] if spec.id in c["plots"]]
    assert len(cards) == 1, f"{spec.id}: expected one container on {TEMPLATES[spec.page]}"
    assert cards[0]["notes"] == [spec.note]


@pytest.mark.parametrize("page", TEMPLATES)
def test_every_container_has_a_spec_for_its_page(page):
    for card in CARDS[page]:
        for plot_id in card["plots"]:
            assert plot_id in registry.BY_ID, plot_id
            assert registry.BY_ID[plot_id].page == page, plot_id


def _download_target(path):
    """The plot a template download link resolves to, mirroring app.py's routes."""
    if path.startswith("latest/"):
        return "latest_" + path.split("/", 1)[1]
    if path.startswith("trend/"):
        return path.split("/", 1)[1]
    return registry.DOWNLOAD_ALIASES.get(path, path)


@pytest.mark.parametrize("page", TEMPLATES)
def test_download_links_point_at_the_cards_plots(page):
    for card in CARDS[page]:
        for path in card["downloads"]:
            assert _download_target(path) in card["card_plots"], path


def test_snapshot_ids_are_prefixed():
    """Export helpers key snapshot caches per version by the latest_ prefix."""
    for s in SPECS:
        assert s.id.startswith("latest_") == (s.kind == "snapshot"), s.id


def test_aliases_and_bulk_categories_resolve():
    assert set(registry.DOWNLOAD_ALIASES.values()) <= set(registry.BY_ID)
    for spec in SPECS:
        assert set(spec.bulk) <= set(registry.BULK_CATEGORIES), spec.id
    assert all(registry.bulk_categories().values())


def test_app_keeps_no_hand_written_plot_lists():
    """app.py derives its tables from the registry instead of listing plots."""
    source = (ROOT / "app.py").read_text()
    assert "plot_latest_" not in source
    assert not re.search(r'@app\.route\("/download/(latest_|trend/[a-z]|main_graph)', source)
    assert "'aop_entity_counts_absolute'" not in source


def test_delta_views_cover_every_delta_plot():
    """Each *_delta trend plot is drawn in the browser from its absolute plot (#181)."""
    deltas = {s.id for s in registry.trend_specs() if s.id.endswith("_delta")}
    assert set(registry.DELTA_VIEWS) == deltas
    for delta_id, view in registry.DELTA_VIEWS.items():
        source = registry.BY_ID[view["of"]]
        assert source.kind == "trend" and source.task == registry.BY_ID[delta_id].task, delta_id
        assert view["y_title"]
        assert set(view) <= {"of", "y_title", "hover_label", "bar", "name_replace"}, delta_id
