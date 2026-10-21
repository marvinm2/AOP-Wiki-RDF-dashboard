"""The Version Diff Explorer (#188): change detection, kinds, stages, caching, CSV = /api/v2/data."""

from __future__ import annotations

import collections
import hashlib
import io
import json
import pathlib

import pandas as pd
import pytest

import plots
from plots import data_api, diff_plots as d, registry, shared
from plots.trends_plots import OECD_WORK_PLAN_FLAG

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTES = json.loads((ROOT / "static" / "data" / "methodology_notes.json").read_text())

AOPO = "http://aopkb.org/aop_ontology#"
DC = "http://purl.org/dc/elements/1.1/"
DCT = "http://purl.org/dc/terms/"
TITLE, DESC, MODIFIED, PART_OF = DC + "title", DC + "description", DCT + "modified", DCT + "isPartOf"


def token(p, o):
    """A triple hash as FINGERPRINT_QUERY builds it."""
    return d.predicate_hash(p) + hashlib.md5(f"{p} {o}".encode()).hexdigest()[:d.T_LEN]


def snapshot(entities, statuses=None):
    """entities: {iri: (type, {predicate: [values]})}."""
    hashes, types, predicates, titles = {}, {}, {}, {}
    for iri, (etype, triples) in entities.items():
        hashes[iri] = collections.Counter(token(p, o) for p, values in triples.items() for o in values)
        types[iri] = etype
        for p in triples:
            predicates[d.predicate_hash(p)] = p
        if TITLE in triples:
            titles[iri] = triples[TITLE][0]
    return d.Snapshot(hashes, types, predicates, titles, statuses or {})


AOP = "https://identifiers.org/aop/"
KE = "https://identifiers.org/aop.events/"


def pair():
    base = {
        AOP + "1": ("AOPs", {TITLE: ["Liver AOP"], DESC: ["old"], MODIFIED: ["2026-01-01"]}),
        AOP + "2": ("AOPs", {TITLE: ["Kidney AOP"], MODIFIED: ["2026-01-01"]}),
        AOP + "3": ("AOPs", {TITLE: ["Gone"]}),
        KE + "10": ("KEs", {TITLE: ["Steatosis"], PART_OF: [AOP + "1"]}),
        KE + "11": ("KEs", {TITLE: ["Unchanged"]}),
    }
    a = snapshot(base, {AOP + "1": f"{OECD_WORK_PLAN_FLAG}|Under Development", AOP + "2": "Under Development"})
    newer = dict(base)
    del newer[AOP + "3"]
    newer[AOP + "1"] = ("AOPs", {TITLE: ["Liver AOP"], DESC: ["new"], MODIFIED: ["2026-04-01"]})
    newer[AOP + "2"] = ("AOPs", {TITLE: ["Kidney AOP"], MODIFIED: ["2026-04-01"]})
    newer[KE + "10"] = ("KEs", {TITLE: ["Steatosis"], PART_OF: [AOP + "1", AOP + "4"]})
    newer[AOP + "4"] = ("AOPs", {TITLE: ["New AOP"]})
    b = snapshot(newer, {AOP + "1": "Under Review", AOP + "2": "Under Development"})
    return a, b


def test_kinds_of_change_and_fields():
    a, b = pair()
    rows = d.diff_rows(a, b).set_index("ID")
    assert rows.loc["1", "Change"] == "Edited"
    assert rows.loc["1", "Fields Changed"] == "Description; Modification Date"   # property_labels.csv
    assert rows.loc["2", "Change"] == "Metadata only"
    assert rows.loc["3", "Change"] == "Removed"
    assert rows.loc["4", "Change"] == "Added"
    assert rows.loc["10", "Change"] == "Links only" and rows.loc["10", "Fields Changed"] == "Part of AOP"
    assert "11" not in rows.index   # unchanged entities are left out
    assert rows.loc["1", "Link"] == AOP + "1"


def test_rows_are_ordered_by_type_then_kind():
    a, b = pair()
    rows = d.diff_rows(a, b)
    assert list(rows["ID"]) == ["4", "3", "1", "2", "10"]
    assert list(rows.columns) == d.WORKLIST_COLUMNS


def test_the_work_plan_flag_is_not_a_stage():
    a, b = pair()
    rows = d.diff_rows(a, b).set_index("ID")
    assert rows.loc["1", "OECD Stage A"] == "Under Development"
    assert rows.loc["1", "OECD Stage B"] == "Under Review"
    assert rows.loc["4", "OECD Stage A"] == "" and rows.loc["4", "OECD Stage B"] == "No Status"
    assert rows.loc["10", "OECD Stage A"] == ""   # not an AOP


def test_multivalued_fields_compare_as_sets():
    p = AOPO + "has_key_event"
    a = snapshot({AOP + "1": ("AOPs", {p: [KE + "1", KE + "2"]})})
    b = snapshot({AOP + "1": ("AOPs", {p: [KE + "2", KE + "1"]})})
    assert d.diff_rows(a, b).empty


def test_summary_counts_each_entity_once():
    a, b = pair()
    worklist = d.diff_rows(a, b)
    summary = d.summary_rows(worklist, a, b).set_index("Entity Type")
    assert summary.loc["AOPs", ["Added", "Removed", "Edited", "Links only", "Metadata only"]].tolist() == [1, 1, 1, 0, 1]
    assert summary.loc["KEs", "Links only"] == 1
    assert summary.loc["AOPs", "In A"] == 3 and summary.loc["AOPs", "In B"] == 3
    assert list(summary.index) == list(shared.ENTITY_TYPE_CLASSES)


def test_stage_links_include_new_and_removed_aops():
    a, b = pair()
    links, moved = d.stage_links(d.diff_rows(a, b), "2026-01-01", "2026-04-01")
    got = {(r["From Status"], r["To Status"]): r["AOPs"] for _, r in links.iterrows()}
    assert got == {
        ("Under Development", "Under Review"): 1,
        ("Not in 2026-01-01", "No Status"): 1,
        ("No Status", "Not in 2026-04-01"): 1,
    }
    assert moved == 3


@pytest.fixture
def versions(monkeypatch):
    vs = ["2026-01-01", "2026-04-01", "2026-07-01"]
    monkeypatch.setattr(d, "get_all_versions", lambda: [{"version": v} for v in reversed(vs)])
    return vs


def test_version_a_defaults_to_the_one_before_b(versions):
    assert d.resolve_against("2026-07-01", d.PREVIOUS) == "2026-04-01"
    assert d.resolve_against("2026-07-01", None) == "2026-04-01"
    assert d.resolve_against("2026-07-01", "2026-01-01") == "2026-01-01"
    assert d.resolve_against("2026-01-01", None) is None
    with pytest.raises(ValueError):
        d.resolve_against("2026-07-01", "2020-01-01")


@pytest.fixture
def fake_store(monkeypatch, versions):
    a, b = pair()
    fetched = []

    def fetch(version):
        fetched.append(version)
        return {"2026-01-01": a, "2026-04-01": a, "2026-07-01": b}[version]

    monkeypatch.setattr(d, "fetch_snapshot", fetch)
    monkeypatch.setattr(d, "_lru", collections.OrderedDict())
    data = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    figs = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    monkeypatch.setattr(d, "_plot_data_cache", data)
    monkeypatch.setattr(d, "_plot_figure_cache", figs)
    monkeypatch.setattr(shared, "_plot_data_cache", data)
    return fetched, data, figs


def test_one_comparison_serves_the_three_views(fake_store):
    fetched, data, figs = fake_store
    for spec in (s for s in registry.snapshot_specs() if s.page == "diff"):
        plots.bind_producer(spec)("2026-07-01", against="2026-04-01")
        assert f"{spec.id}_2026-07-01" in data and f"{spec.id}_2026-07-01" in figs
    assert fetched == ["2026-04-01", "2026-07-01"]   # the LRU served the second and third view
    assert len(data["latest_version_diff_worklist_2026-07-01"]) == 5
    assert list(data["latest_version_diff_worklist_2026-07-01"].columns[:2]) == ["Version A", "Version B"]
    assert "2026-04-01 → 2026-07-01" in figs["latest_version_diff_2026-07-01"].layout.title.text


def test_a_message_replaces_the_previous_pair(fake_store):
    """The plots take settings, so each request renders again: a request that
    cannot be compared must not leave the last pair's result in the cache."""
    _, data, figs = fake_store
    plots.plot_latest_version_diff("2026-07-01", against="2026-04-01", part="worklist")
    assert len(data["latest_version_diff_worklist_2026-07-01"]) == 5
    assert list(data["latest_version_diff_worklist_2026-07-01"].columns[:2]) == ["Version A", "Version B"]
    plots.plot_latest_version_diff("2026-07-01", against="2026-07-01", part="worklist")
    assert data["latest_version_diff_worklist_2026-07-01"].empty
    assert "must be older" in figs["latest_version_diff_worklist_2026-07-01"].layout.annotations[0].text
    plots.plot_latest_version_diff("2026-01-01", part="summary")
    assert "No version before" in figs["latest_version_diff_2026-01-01"].layout.annotations[0].text


def test_csv_download_equals_the_v2_data(fake_store):
    plots.plot_latest_version_diff("2026-07-01", against="2026-04-01", part="worklist")
    worklist = shared._plot_data_cache["latest_version_diff_worklist_2026-07-01"]
    csv = shared.get_csv_with_metadata("latest_version_diff_worklist", version="2026-07-01")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False, dtype=str)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(worklist))
    pd.testing.assert_frame_equal(from_csv, from_api.astype(str), check_dtype=False)


def test_the_notes_disclose_the_queries_the_code_runs():
    for note in ("latest_version_diff", "latest_version_diff_stages", "latest_version_diff_worklist"):
        assert [q["query"] for q in NOTES[note]["queries"]] == list(d.QUERIES), note


def test_every_query_names_one_graph_so_the_store_keys_it_per_version():
    from plots.query_store import graph_dependencies
    for template in d.QUERIES:
        query = template.replace("__GRAPH_URI__", "<http://aopwiki.org/graph/2026-07-01>")
        assert graph_dependencies(query) == frozenset({"http://aopwiki.org/graph/2026-07-01"})


def test_the_build_stores_every_version_against_the_previous():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_query_store", ROOT / "scripts" / "build_query_store.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert build.setting_variants(registry.BY_ID["latest_version_diff"]) == [{"against": "previous"}]
