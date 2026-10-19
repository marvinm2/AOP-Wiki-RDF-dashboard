"""The ontology coverage map (#187): layout, statuses, figure, CSV = /api/v2/data.

Fixture branch data only: no endpoint, no Ubergraph, no cache file.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from plots import data_api, shared
from plots.latest_plots import (
    COVERAGE_MAP_COLUMNS,
    _render_ontology_coverage_map,
    find_ontology_coverage_holes,
    ontology_coverage_map_rows,
)

# heart (annotated) > cardiac chamber (annotated below) > cardiac ventricle
# (annotated); heart > cardiac valve (hole) > mitral valve (inside); a vessel
# hangs off the root and also off the heart, so it is drawn under the heart;
# lymph vessel sits under two depth-1 parents and goes to the larger one.
BRANCH = {
    "roots": ["UBERON_0004535"],
    "root_labels": ["cardiovascular system"],
    "branch_size": 200,
    "candidates": {
        "UBERON_0000948": {"label": "heart", "subtree_size": 100, "depth": 1, "parents": []},
        "UBERON_0001981": {"label": "blood vessel", "subtree_size": 60, "depth": 1, "parents": []},
        "UBERON_0004151": {"label": "cardiac chamber", "subtree_size": 40, "depth": 2, "parents": ["UBERON_0000948"]},
        "UBERON_0000946": {"label": "cardiac valve", "subtree_size": 30, "depth": 2, "parents": ["UBERON_0000948"]},
        "UBERON_0002082": {"label": "cardiac ventricle", "subtree_size": 10, "depth": 3, "parents": ["UBERON_0004151"]},
        "UBERON_0002135": {"label": "mitral valve", "subtree_size": 5, "depth": 3, "parents": ["UBERON_0000946"]},
        "UBERON_0001473": {"label": "lymph vessel", "subtree_size": 20, "depth": 2,
                           "parents": ["UBERON_0001981", "UBERON_0000948"]},
    },
    "used_term_ancestors": {
        "UBERON_0002082": ["UBERON_0000948", "UBERON_0004151", "UBERON_0002082"],
    },
    "used_terms_in_branch": ["UBERON_0002082"],
}
USED = {"UBERON_0002082"}


def frame():
    df = pd.DataFrame(ontology_coverage_map_rows(BRANCH, USED, "Cardiovascular"))
    df["Branch"] = "Cardiovascular"
    df["Version"] = "2026-10-01"
    return df[COVERAGE_MAP_COLUMNS]


def by_id():
    return frame().set_index("Ontology ID")


def test_one_row_per_term_plus_the_branch_root():
    df = frame()
    assert len(df) == len(BRANCH["candidates"]) + 1
    root = df.iloc[0]
    assert (root["Level"], root["Start"], root["Width"]) == (0, 0.0, 1.0)
    assert root["Term"] == "cardiovascular system"
    assert root["Status"] == "Contains annotated terms"


def test_statuses_follow_the_holes_rule():
    df = by_id()
    assert df.loc["UBERON:0002082", "Status"] == "Annotated"
    assert df.loc["UBERON:0000948", "Status"] == "Contains annotated terms"
    assert df.loc["UBERON:0004151", "Status"] == "Contains annotated terms"
    assert df.loc["UBERON:0000946", "Status"] == "Coverage hole"
    assert df.loc["UBERON:0002135", "Status"] == "Inside a hole"
    holes = {h["Ontology ID"] for h in find_ontology_coverage_holes(BRANCH, USED)}
    assert holes == set(df.index[df["Status"] == "Coverage hole"])


def test_a_multi_parent_term_is_drawn_once_under_its_shallowest_larger_parent():
    df = by_id()
    lymph = df.loc["UBERON:0001473"]
    # Both parents are depth 1; the heart has the larger subtree.
    assert lymph["Drawn Under"] == "UBERON:0000948"
    assert lymph["Other Parents"] == "UBERON:0001981"
    assert lymph["Level"] == 2
    assert df.loc["UBERON:0001981", "Drawn Under"] == ""


def test_children_fit_inside_their_parent_in_subtree_order():
    df = by_id()
    for oid, row in df.iterrows():
        if not row["Drawn Under"]:
            continue
        parent = df.loc[row["Drawn Under"]]
        assert row["Start"] >= parent["Start"] - 1e-9
        assert row["Start"] + row["Width"] <= parent["Start"] + parent["Width"] + 1e-9
    # heart (100) and blood vessel (60) share the root's 200 in proportion.
    assert df.loc["UBERON:0000948", "Width"] == pytest.approx(0.5)
    assert df.loc["UBERON:0001981", "Width"] == pytest.approx(0.3)
    assert df.loc["UBERON:0001981", "Start"] == pytest.approx(0.5)
    # Under the heart: chamber 40, valve 30, lymph 20 of 100, largest first.
    kids = df[df["Drawn Under"] == "UBERON:0000948"].sort_values("Start")
    assert list(kids["Term"]) == ["cardiac chamber", "cardiac valve", "lymph vessel"]
    assert kids["Width"].sum() == pytest.approx(0.5 * 90 / 100)


def test_overlapping_siblings_are_scaled_to_fit():
    branch = {**BRANCH, "branch_size": 100}
    rows = pd.DataFrame(ontology_coverage_map_rows(branch, USED, "Cardiovascular"))
    top = rows[(rows["Level"] == 1)]
    # 100 + 60 > 100: the two share the full width in proportion.
    assert top["Width"].sum() == pytest.approx(1.0)


def test_figure_is_one_bar_trace_per_status_on_level_rows():
    df = frame()
    fig = _render_ontology_coverage_map(df, "t")
    names = [t.name for t in fig.data]
    assert names == ["Annotated (1)", "Contains annotated terms (3)", "Coverage hole (3)", "Inside a hole (1)"]
    assert all(t.type == "bar" and t.orientation == "h" for t in fig.data)
    assert list(fig.layout.yaxis.categoryarray) == ["Branch", "Level 1", "Level 2", "Level 3"]
    assert list(fig.layout.xaxis.range) == [0, 100]
    hole = fig.data[2]
    assert set(hole.text) == {"cardiac valve", "blood vessel", "lymph vessel"}
    # customdata carries what static/js/viz/coverage-map.js navigates by.
    cd = dict(zip(hole.text, hole.customdata))
    assert list(cd["cardiac valve"][6:]) == ["UBERON:0000948", 2]


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = frame()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["latest_ontology_coverage_map_2026-10-01"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("latest_ontology_coverage_map_2026-10-01")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
