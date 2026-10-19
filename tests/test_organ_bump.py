"""The organ-system bump chart (#187): ranks, ties, gaps, figure, CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd

from plots import data_api, shared
from plots.organ_systems import NO_ANNOTATION_BUCKET
from plots.trends_plots import (
    ORGAN_BUMP_COLUMNS,
    ORGAN_BUMP_SYSTEMS,
    _render_organ_bump,
    organ_bump_rows,
)

VERSIONS = ["2019-01-01", "2020-01-01", "2021-01-01"]


def wide():
    # Nervous leads, then falls behind Hepatobiliary; Cardiovascular ties
    # Hepatobiliary in 2020 (the 2019 order wins); Sensory has no AOPs in 2020;
    # Other/Unclassified and No annotation are not ranked.
    counts = {
        "Nervous": [10, 12, 12],
        "Hepatobiliary": [5, 8, 20],
        "Cardiovascular": [3, 8, 9],
        "Sensory": [1, 0, 2],
        "Other/Unclassified": [50, 60, 70],
        NO_ANNOTATION_BUCKET: [99, 99, 99],
    }
    return pd.DataFrame({"version": VERSIONS, **counts})


def rows():
    return organ_bump_rows(wide())


def at(df, version, system):
    hit = df[(df["Version"] == version) & (df["Organ System"] == system)]
    return hit.iloc[0] if len(hit) else None


def test_ranks_per_snapshot_leave_out_non_systems():
    df = rows()
    assert list(df.columns) == ORGAN_BUMP_COLUMNS
    assert "Other/Unclassified" not in ORGAN_BUMP_SYSTEMS
    assert set(df["Organ System"]) == {"Nervous", "Hepatobiliary", "Cardiovascular", "Sensory"}
    first = df[df["Version"] == VERSIONS[0]].sort_values("Rank")
    assert list(first["Organ System"]) == ["Nervous", "Hepatobiliary", "Cardiovascular", "Sensory"]
    assert set(first["Ranked Systems"]) == {4}
    assert set(first["Rank Change"]) == {""}


def test_ties_keep_the_previous_order_and_changes_are_signed():
    df = rows()
    # 2020: Hepatobiliary and Cardiovascular both have 8; Hepatobiliary was ahead.
    assert at(df, VERSIONS[1], "Hepatobiliary")["Rank"] == 2
    assert at(df, VERSIONS[1], "Cardiovascular")["Rank"] == 3
    assert at(df, VERSIONS[1], "Cardiovascular")["Rank Change"] == "0"
    # 2021: Hepatobiliary overtakes Nervous.
    assert at(df, VERSIONS[2], "Hepatobiliary")["Rank Change"] == "+1"
    assert at(df, VERSIONS[2], "Nervous")["Rank Change"] == "-1"


def test_a_system_without_aops_is_not_ranked_and_comes_back_new():
    df = rows()
    assert at(df, VERSIONS[1], "Sensory") is None
    assert set(df[df["Version"] == VERSIONS[1]]["Ranked Systems"]) == {3}
    back = at(df, VERSIONS[2], "Sensory")
    assert (back["Rank"], back["Rank Change"]) == (4, "new")


def test_figure_is_one_line_per_system_in_latest_rank_order():
    fig = _render_organ_bump(rows())
    assert [t.name for t in fig.data] == ["Hepatobiliary", "Nervous", "Cardiovascular", "Sensory"]
    sensory = fig.data[3]
    assert list(sensory.y) == [4, None, 4]
    assert sensory.connectgaps is False
    assert sensory.customdata[2][1] == "newly ranked"
    assert list(fig.data[0].y) == [2, 2, 1]
    # Rank 1 on top, with a fixed range that the trends range clamp leaves alone.
    assert fig.layout.yaxis.autorange is False
    assert list(fig.layout.yaxis.range) == [len(ORGAN_BUMP_SYSTEMS) + 0.5, 0.5]
    assert shared._figure_x_categories(fig) == VERSIONS


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = rows()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["organ_coverage_bump"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("organ_coverage_bump")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False,
                           dtype={"Rank Change": str})
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
