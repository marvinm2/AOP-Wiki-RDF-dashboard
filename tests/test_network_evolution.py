"""Network evolution (#189): union graph, fixed layout, presence rows, figure, CSV = /api/v2/data."""

from __future__ import annotations

import io
import json
import pathlib

import pandas as pd

from plots import data_api, network_evolution as ne, shared

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTES = json.loads((ROOT / "static" / "data" / "methodology_notes.json").read_text())

KE = "https://identifiers.org/aop.events/"
KER = "https://identifiers.org/aop.relationships/"
VERSIONS = ["2025-01-01", "2025-04-01", "2025-07-01"]


def data():
    ke_by_version = {
        "2025-01-01": {KE + "1", KE + "2"},
        "2025-04-01": {KE + "1", KE + "2", KE + "3"},
        "2025-07-01": {KE + "1", KE + "3", KE + "10"},
    }
    ker_by_version = {
        "2025-01-01": {KER + "5"},
        "2025-04-01": {KER + "5", KER + "6"},
        "2025-07-01": {KER + "6", KER + "7"},
    }
    ends = {KER + "5": (KE + "1", KE + "2"), KER + "6": (KE + "1", KE + "3")}   # KER 7 has none
    labels = {KE + "1": "Binding", KE + "2": "Inflammation", KE + "3": "Fibrosis"}
    return ke_by_version, ker_by_version, ends, labels


def rows():
    ke_by_version, ker_by_version, ends, labels = data()
    positions = ne.union_layout(ne.union_graph(ke_by_version, ends))
    return ne.evolution_rows(ke_by_version, ker_by_version, ends, labels, VERSIONS, positions)


def test_the_union_holds_every_ke_and_ker_ever_seen():
    ke_by_version, _, ends, _ = data()
    G = ne.union_graph(ke_by_version, ends)
    assert set(G.nodes) == {KE + "1", KE + "2", KE + "3", KE + "10"}
    assert G.number_of_edges() == 2


def test_the_layout_is_the_same_every_time():
    ke_by_version, _, ends, _ = data()
    a = ne.union_layout(ne.union_graph(ke_by_version, ends))
    b = ne.union_layout(ne.union_graph(ke_by_version, ends))
    assert a == b and set(a) == {KE + "1", KE + "2", KE + "3", KE + "10"}


def test_rows_carry_presence_and_first_and_last_seen():
    df = rows().set_index("ID")
    assert df.loc["KE 2", "Presence"] == "xx-"
    assert (df.loc["KE 2", "First Seen"], df.loc["KE 2", "Last Seen"]) == ("2025-01-01", "2025-04-01")
    assert df.loc["KE 10", "Presence"] == "--x" and df.loc["KE 10", "Label"] == ""
    assert (df.loc["KER 6", "Source"], df.loc["KER 6", "Target"]) == ("KE 1", "KE 3")
    assert df.loc["KER 6", "Presence"] == "-xx"
    assert "KER 7" not in df.index   # no endpoints in any snapshot: cannot be drawn
    assert pd.isna(df.loc["KER 5", "X"])


def test_rows_are_kes_then_kers_in_number_order():
    assert list(rows()["ID"]) == ["KE 1", "KE 2", "KE 3", "KE 10", "KER 5", "KER 6"]
    assert list(rows().columns) == ne.COLUMNS


def test_the_figure_shows_the_latest_snapshot():
    fig = ne._render(rows(), VERSIONS)
    edges, nodes = fig.data
    assert len(nodes.x) == 3                # KE 1, 3, 10
    assert len(edges.x) == 3                # one KER (6): two points and a gap
    assert "2025-07-01" in fig.layout.title.text


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = rows()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["network_evolution"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)
    csv = shared.get_csv_with_metadata("network_evolution")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))
    for col in ("X", "Y"):   # empty for edges: "" in the CSV, None in the API
        from_csv[col] = pd.to_numeric(from_csv[col].replace("", None))
        from_api[col] = pd.to_numeric(from_api[col])
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)


def test_the_note_discloses_the_shared_queries():
    queries = [q["query"] for q in NOTES["network_evolution"]["queries"]]
    assert len(queries) == 4
    assert any("aopo:KeyEvent ." in q for q in queries)
    assert any("aopo:KeyEventRelationship ." in q for q in queries)
    assert any("has_upstream_key_event" in q for q in queries)
    assert any("dc:title" in q for q in queries)
