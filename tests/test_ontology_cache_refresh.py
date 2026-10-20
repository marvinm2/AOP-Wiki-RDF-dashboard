"""Ontology cache refresh (multi-endpoint #22): change detection, summary, /health and /about dates."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("refresh", ROOT / "scripts" / "refresh_ontology_caches.py")
refresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh)


def cache(generated, elapsed, terms):
    return {"generated_at": generated, "domains": {"x": [1, 2]},
            "stats": {"elapsed_s": elapsed, "used_terms_ever": terms}}


def test_a_rebuild_that_only_moves_the_timestamps_is_no_change():
    a, b = cache("2026-08-16", 41.6, 1133), cache("2026-10-20", 41.2, 1133)
    assert refresh.comparable(a) == refresh.comparable(b)
    assert refresh.stat_changes(a, b) == []


def test_new_terms_are_a_change_and_the_stats_say_how():
    a, b = cache("2026-08-16", 41.6, 1133), cache("2026-10-20", 41.2, 1138)
    assert refresh.comparable(a) != refresh.comparable(b)
    assert refresh.stat_changes(a, b) == ["used_terms_ever: 1133 -> 1138"]


def test_refresh_keeps_an_unchanged_cache_and_replaces_a_changed_one(tmp_path, monkeypatch):
    target = tmp_path / "static" / "data" / "x_cache.json"
    target.parent.mkdir(parents=True)
    builder = tmp_path / "build.py"
    builder.write_text(
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        f"json.dump({{'generated_at': 'new', 'stats': {{'elapsed_s': 2, 'n': N}}}}, open(out, 'w'))\n")
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    for n, expect_change in ((1, False), (2, True)):
        target.write_text(json.dumps({"generated_at": "old", "stats": {"elapsed_s": 1, "n": 1}}))
        builder.write_text(builder.read_text().replace("N", str(n)) if "N}" in builder.read_text()
                           else builder.read_text().replace(f"'n': {n - 1}", f"'n': {n}"))
        r = refresh.refresh("x", "static/data/x_cache.json", "build.py", tmp_path)
        assert r["changed"] is expect_change
        assert json.loads(target.read_text())["generated_at"] == ("new" if expect_change else "old")
    text = refresh.summary([r])
    assert text.startswith("1 of 1 ontology caches changed.") and "n: 1 -> 2" in text


def test_every_committed_cache_records_when_it_was_built():
    for _, (cache_rel, builder_rel) in refresh.CACHES.items():
        assert (ROOT / builder_rel).exists()
        generated = json.loads((ROOT / cache_rel).read_text()).get("generated_at")
        assert generated and generated[:4].isdigit(), cache_rel


@pytest.mark.slow  # importing app runs the startup precompute
def test_health_and_about_show_when_each_cache_was_built():
    import app as dashboard
    dates = dashboard._ontology_cache_dates()
    assert set(dates) == {"organ_systems", "domain_lens", "ontology_branches"}
    for key, name in dashboard.ONTOLOGY_CACHES.items():
        assert dates[key] == json.loads((ROOT / "static" / "data" / name).read_text())["generated_at"]
    page = dashboard.app.test_client().get("/about").get_data(as_text=True)
    assert f"organ systems {dates['organ_systems'][:10]}" in page
