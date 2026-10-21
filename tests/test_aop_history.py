"""AOP composition history (#190): roles per version, added/removed, the producer, CSV = /api/v2/data."""

from __future__ import annotations

import io
import json
import pathlib

import pandas as pd
import pytest

from plots import aop_history as ah, data_api, shared

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTES = json.loads((ROOT / "static" / "data" / "methodology_notes.json").read_text())

KE = "https://identifiers.org/aop.events/"
AOPO = "http://aopkb.org/aop_ontology#"
HAS_KE, HAS_MIE, HAS_AO = AOPO + "has_key_event", AOPO + "has_molecular_initiating_event", AOPO + "has_adverse_outcome"
VERSIONS = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]


def links():
    out = []
    for v in VERSIONS[1:]:                       # the AOP appears in the second version
        out += [(v, HAS_KE, KE + "10"), (v, HAS_MIE, KE + "10"), (v, HAS_KE, KE + "30"), (v, HAS_AO, KE + "30")]
    out += [(VERSIONS[1], HAS_KE, KE + "20"), (VERSIONS[3], HAS_KE, KE + "20")]   # left, came back
    out += [(VERSIONS[3], HAS_KE, KE + "5")]                                      # joined last
    return out


def aop_rows():
    return [(VERSIONS[1], "Liver fibrosis", ["Under Development"]),
            (VERSIONS[2], "Liver fibrosis", ["Under Development", "Included in OECD Work Plan"]),
            (VERSIONS[3], "Liver fibrosis", [])]


def frame():
    return ah.composition_rows(7, links(), aop_rows(), VERSIONS, {KE + "10": "Binding", KE + "30": "Fibrosis"})


def test_aop_numbers_are_parsed_strictly():
    assert ah.aop_number("3") == 3 and ah.aop_number("AOP 12") == 12
    assert ah.aop_number("https://identifiers.org/aop/45") == 45
    for bad in ("", None, "abc", "3; DROP", "1.5", "-2"):
        assert ah.aop_number(bad) is None


def test_the_queries_name_one_aop_in_a_values_block():
    q = ah._query(ah.KE_QUERY, 3)
    assert "VALUES ?aop { <https://identifiers.org/aop/3> }" in q and "__AOP__" not in q


def test_rows_are_the_aop_then_mie_ke_ao_with_roles_per_version():
    df = frame()
    assert list(df.columns) == ah.COLUMNS
    assert list(df["ID"]) == ["AOP 7", "KE 10", "KE 20", "KE 5", "KE 30"]
    rows = df.set_index("ID")
    assert rows.loc["AOP 7", "Roles"] == "-xxx"
    assert rows.loc["AOP 7", "Stages"] == "|Under Development|Under Development|No Status"
    assert rows.loc["KE 10", "Roles"] == "-MMM" and rows.loc["KE 10", "Role"] == "MIE"
    assert rows.loc["KE 30", "Roles"] == "-AAA" and rows.loc["KE 30", "Title"] == "Fibrosis"
    assert rows.loc["KE 20", "Roles"] == "-K-K"
    assert (rows.loc["KE 5", "First Seen"], rows.loc["KE 5", "Last Seen"]) == ("2024-10-01", "2024-10-01")


def test_only_the_versions_asked_for_count():
    df = ah.composition_rows(7, links(), aop_rows(), VERSIONS[:3], {})
    assert list(df["ID"]) == ["AOP 7", "KE 10", "KE 20", "KE 30"]
    assert df.set_index("ID").loc["KE 20", "Roles"] == "-K-"


def test_changes_count_joins_and_departures():
    assert ah.changes("-K-K") == (2, 1)
    assert ah.changes("MMMM") == (0, 0)


def test_role_prefers_mie_then_ao():
    assert ah.role_in({HAS_KE, HAS_AO, HAS_MIE}) == "MIE"
    assert ah.role_in({HAS_KE, HAS_AO}) == "AO"
    assert ah.role_in({HAS_KE}) == "KE"


def test_the_figure_marks_added_and_removed():
    fig = ah._render(frame(), VERSIONS)
    by_name = {t.name: t for t in fig.data}
    assert len(by_name["MIE"].x) == 3 and len(by_name["AO"].x) == 3
    # KE 20 joins again in 2024-10-01 and KE 5 joins there; KE 20 leaves in 2024-07-01.
    assert sorted(by_name["Added"].x) == ["2024-10-01", "2024-10-01"]
    assert list(by_name["Removed"].x) == ["2024-07-01"]
    assert "2024-01-01" not in fig.layout.xaxis.categoryarray   # before the AOP existed


@pytest.fixture
def stubbed(monkeypatch):
    data = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    figures = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    monkeypatch.setattr(ah, "_plot_data_cache", data)
    monkeypatch.setattr(ah, "_plot_figure_cache", figures)
    monkeypatch.setattr(shared, "_plot_data_cache", data)
    monkeypatch.setattr(ah, "get_all_versions", lambda: [{"version": v} for v in VERSIONS])
    monkeypatch.setattr(shared, "resolve_version", lambda v=None: v or VERSIONS[-1])
    monkeypatch.setattr(ah, "_lifeline_labels", lambda kind: {KE + "10": "Binding"})
    calls = []
    monkeypatch.setattr(ah, "fetch_aop", lambda n: calls.append(n) or (links(), aop_rows()))
    return data, figures, calls


def test_the_producer_caches_frame_and_figure_in_every_branch(stubbed):
    data, figures, calls = stubbed
    key = f"{ah.PLOT_ID}_{VERSIONS[-1]}"
    for aop in (None, "abc"):
        ah.plot_latest_aop_composition(VERSIONS[-1], aop=aop)
        assert data[key].empty and key in figures
    assert calls == []           # no query without a valid AOP
    ah.plot_latest_aop_composition(VERSIONS[-1], aop="7")
    assert list(data[key]["ID"])[:2] == ["AOP 7", "KE 10"] and calls == [7]
    ah.plot_latest_aop_composition(VERSIONS[0], aop="7")
    assert data[f"{ah.PLOT_ID}_{VERSIONS[0]}"].empty     # not in AOP-Wiki yet


def test_csv_download_equals_the_v2_data(stubbed):
    data, _, _ = stubbed
    ah.plot_latest_aop_composition(VERSIONS[-1], aop="7")
    key = f"{ah.PLOT_ID}_{VERSIONS[-1]}"
    df = data[key]
    lines = shared.get_csv_with_metadata(key).splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False, dtype=str)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df)).astype(str)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)


def test_the_store_build_runs_no_aop():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_query_store", ROOT / "scripts" / "build_query_store.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    from plots import registry
    assert build.setting_variants(registry.BY_ID["latest_aop_composition"]) == []


def test_the_note_discloses_the_queries():
    queries = [q["query"] for q in NOTES["aop_composition"]["queries"]]
    assert queries[0] == ah._query(ah.KE_QUERY, 3)
    assert queries[1] == ah._query(ah.AOP_QUERY, 3)
    assert any("dc:title ?title" in q and "aopo:KeyEvent" in q for q in queries)
