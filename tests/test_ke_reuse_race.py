"""KE reuse rank race (#187): ranks, ties, racers, figure, CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd

from plots import data_api, shared
from plots.trends_plots import KE_RACE_COLUMNS, _render_ke_reuse_race, ke_race_rows

KE = "https://identifiers.org/aop.events/{}".format


def counts():
    # KE 2 overtakes KE 1; KE 3 ties KE 4 in 2020 and keeps its 2019 place;
    # KE 5 only enters the top 2 in 2021; KE 9 never does.
    return {
        "2019-01-01": {KE(1): 9, KE(2): 5, KE(3): 4, KE(4): 3, KE(9): 1},
        "2020-01-01": {KE(1): 9, KE(2): 8, KE(3): 6, KE(4): 6, KE(9): 1},
        "2021-01-01": {KE(1): 9, KE(2): 12, KE(5): 10, KE(9): 2},
    }


def rows(top=2):
    return ke_race_rows(counts(), {KE(1): "First", KE(2): "Second"}, top=top)


def at(df, version, n):
    hit = df[(df["Version"] == version) & (df["KE ID"] == f"KE {n}")]
    return hit.iloc[0]


def test_racers_are_the_kes_ever_in_the_top_with_a_row_per_snapshot():
    df = rows()
    assert list(df.columns) == KE_RACE_COLUMNS
    assert sorted(set(df["KE ID"])) == ["KE 1", "KE 2", "KE 5"]
    assert len(df) == 3 * 3
    assert at(df, "2021-01-01", 2)["Rank"] == 1
    assert at(df, "2021-01-01", 1)["Rank"] == 3
    assert at(df, "2021-01-01", 2)["Title"] == "Second"


def test_ties_keep_the_previous_order_and_absent_kes_have_no_rank():
    df = rows(top=4)
    # 2020: KE 3 and KE 4 both have 6; KE 3 was ahead in 2019.
    assert at(df, "2020-01-01", 3)["Rank"] == 3
    assert at(df, "2020-01-01", 4)["Rank"] == 4
    # KE 5 is in no AOP before 2021, KE 3 in none in 2021.
    assert (at(df, "2019-01-01", 5)["AOPs"], pd.isna(at(df, "2019-01-01", 5)["Rank"])) == (0, True)
    assert pd.isna(at(df, "2021-01-01", 3)["Rank"])


def test_figure_shows_the_top_of_one_snapshot_on_a_fixed_axis():
    df = rows()
    fig = _render_ke_reuse_race(df, "2021-01-01")
    bar = fig.data[0]
    # The figure always shows the top 15, here every ranked racer.
    assert list(bar.y) == [1, 2, 3]
    assert list(bar.x) == [12, 10, 9]
    assert fig.layout.yaxis.ticktext[0] == "KE 2 · Second"
    assert fig.layout.xaxis.range[1] == 12 * 1.12
    assert fig.layout.yaxis.autorange is False


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = rows()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["ke_reuse_race"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("ke_reuse_race")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False,
                           na_values={"Rank": [""]}, dtype={"Rank": "Int64"})
    payload = data_api.frame_to_columnar(df)
    # No rank is null in the JSON (not NaN), so the page can test it.
    assert None in payload["data"]["Rank"]
    from_api = data_api.columnar_to_frame(payload).astype({"Rank": "Int64"})

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
