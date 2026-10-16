"""The organ-system UpSet (#184): rows, exclusive intersections, set sizes, CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd

from plots import data_api, latest_plots, shared
from plots.latest_plots import (
    UPSET_SCOPE_COLUMNS,
    _render_organ_upset,
    organ_intersections,
    organ_set_sizes,
    organ_upset_rows,
)

AOP = "https://identifiers.org/aop/"
UNIVERSE = {AOP + "10": "Ten", AOP + "2": "Two  spaced\ttitle", AOP + "3": "Three", AOP + "4": "Four"}
SETS = {
    "all": {
        AOP + "2": ("Hepatobiliary", "Immune/Haematopoietic"),
        AOP + "3": ("Hepatobiliary", "Immune/Haematopoietic"),
        AOP + "4": ("Nervous",),
        AOP + "10": ("Hepatobiliary",),
    },
    "apical": {AOP + "2": ("Hepatobiliary",), AOP + "4": ("Nervous",)},
    "ao": {AOP + "4": ("Nervous",)},
}


def frame():
    return organ_upset_rows(SETS, UNIVERSE, "2026-10-01")


def test_one_row_per_aop_in_number_order_with_every_scope():
    df = frame()
    assert df["aop_id"].tolist() == ["AOP 2", "AOP 3", "AOP 4", "AOP 10"]
    assert list(df.columns) == ["aop", "aop_id", "title", *UPSET_SCOPE_COLUMNS.values(),
                                "n_systems_all_kes", "version"]
    two = df.iloc[0]
    assert two["title"] == "Two spaced title"
    assert two["systems_all_kes"] == "Hepatobiliary; Immune/Haematopoietic"
    assert two["systems_apical_kes"] == "Hepatobiliary"
    assert two["systems_adverse_outcome"] == ""      # no signal under this scope
    assert two["n_systems_all_kes"] == 2
    assert (df["version"] == "2026-10-01").all()


def test_intersections_are_exclusive_and_ordered():
    df = frame()
    inter = organ_intersections(df, "all")
    assert inter == [
        (("Hepatobiliary", "Immune/Haematopoietic"), 2),
        (("Hepatobiliary",), 1),     # ties: fewer buckets first, then by name
        (("Nervous",), 1),
    ]
    # Every classified AOP is in exactly one column.
    assert sum(n for _, n in inter) == (df["systems_all_kes"] != "").sum()
    assert organ_intersections(df, "ao") == [(("Nervous",), 1)]


def test_set_sizes_count_each_bucket_across_combinations():
    assert organ_set_sizes(frame(), "all") == [
        ("Hepatobiliary", 3), ("Immune/Haematopoietic", 2), ("Nervous", 1)]


def test_server_figure_has_bars_matrix_and_totals():
    fig = _render_organ_upset(frame(), "all", "2026-10-01")
    bars, background = fig.data[0], fig.data[1]
    assert list(bars.y) == [2, 1, 1]
    assert len(background.x) == 3 * 3                 # every column x every bucket
    totals = fig.data[-1]
    assert list(totals.y) == ["Hepatobiliary", "Immune/Haematopoietic", "Nervous"]
    assert list(totals.x) == [3, 2, 1]
    assert "3 distinct combinations" in fig.layout.title.text


def test_data_does_not_depend_on_the_scope(monkeypatch):
    granular = pd.DataFrame()
    monkeypatch.setattr(latest_plots, "_get_coverage_dataframe",
                        lambda version: (granular, UNIVERSE, "2026-10-01"))
    monkeypatch.setattr(latest_plots, "_organ_sets", lambda g, u, scope: SETS[scope])
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    monkeypatch.setattr(latest_plots, "_plot_data_cache", cache)
    monkeypatch.setattr(latest_plots, "_plot_figure_cache",
                        shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800))
    key = "latest_organ_upset_2026-10-01"
    latest_plots.plot_latest_organ_upset.__wrapped__("2026-10-01", scope="all")
    first = cache[key].copy()
    latest_plots.plot_latest_organ_upset.__wrapped__("2026-10-01", scope="ao")
    pd.testing.assert_frame_equal(cache[key], first)


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = frame()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["latest_organ_upset_2026-10-01"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("latest_organ_upset", version="2026-10-01")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
