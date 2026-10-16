"""The AOP-AOP overlap matrix (#185): cluster order, rows = coloured cells, CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd

from plots import data_api, shared
from plots.latest_plots import (
    OVERLAP_COLUMNS,
    _render_overlap_matrix,
    overlap_cluster_colours,
    overlap_clusters,
    overlap_order,
    overlap_rows,
    score_ke_set_similarity,
)

AOP = "https://identifiers.org/aop/"
KE = "https://identifiers.org/aop.events/"


def kes(*numbers):
    return {KE + str(n) for n in numbers}


# A three-AOP family (10, 2, 7), where 2 and 7 are identical and 10 chains in
# through 2; a separate pair (30, 31); and a loner (40) that reaches no one.
KE_SETS = {
    AOP + "10": kes(1, 2, 3, 4, 5),
    AOP + "2": kes(1, 2, 3, 4),
    AOP + "7": kes(1, 2, 3, 4),
    AOP + "30": kes(20, 21, 22),
    AOP + "31": kes(20, 21, 23),
    AOP + "40": kes(1, 99),
}
TITLES = {AOP + "2": "Two", AOP + "7": "Seven", AOP + "10": "Ten", AOP + "30": "Thirty", AOP + "31": "Thirty-one"}


def frame():
    scored = score_ke_set_similarity(KE_SETS, 0.34, 2)
    return overlap_rows(scored, KE_SETS, TITLES, "2026-10-01")


def test_clusters_largest_first_in_number_order():
    scored = score_ke_set_similarity(KE_SETS, 0.34, 2)
    assert overlap_clusters(scored) == [[AOP + "2", AOP + "7", AOP + "10"], [AOP + "30", AOP + "31"]]


def test_order_puts_the_most_similar_aops_next_to_each_other():
    sets = {AOP + "1": kes(1, 2, 3), AOP + "2": kes(7, 8, 9), AOP + "3": kes(1, 2, 3, 7),
            AOP + "4": kes(7, 8, 9, 3)}
    order = overlap_order([[AOP + "1", AOP + "2", AOP + "3", AOP + "4"]], sets)
    # 1~3 and 2~4 are the close pairs; whichever way round, they end up adjacent.
    pos = {a: i for i, a in enumerate(order)}
    assert abs(pos[AOP + "1"] - pos[AOP + "3"]) == 1
    assert abs(pos[AOP + "2"] - pos[AOP + "4"]) == 1


def test_rows_are_the_coloured_cells_in_matrix_order():
    df = frame()
    assert list(df.columns) == OVERLAP_COLUMNS
    assert (df["matrix_row"] < df["matrix_col"]).all()
    assert df[["matrix_row", "matrix_col"]].values.tolist() == sorted(df[["matrix_row", "matrix_col"]].values.tolist())
    # The family takes positions 0-2, the pair 3-4; the loner is not drawn.
    assert set(df.loc[df["cluster_id"] == 1, "matrix_row"]) | set(df.loc[df["cluster_id"] == 1, "matrix_col"]) == {0, 1, 2}
    assert set(df.loc[df["cluster_id"] == 2, "aop1_id"]) | set(df.loc[df["cluster_id"] == 2, "aop2_id"]) == {"AOP 30", "AOP 31"}
    assert "AOP 40" not in set(df["aop1_id"]) | set(df["aop2_id"])
    same = df[df["identical_ke_set"]]
    assert {same.iloc[0]["aop1_id"], same.iloc[0]["aop2_id"]} == {"AOP 2", "AOP 7"}
    assert same.iloc[0]["jaccard"] == 1.0 and same.iloc[0]["shared_kes"] == 4


def test_matrix_is_symmetric_and_drawn_from_the_rows():
    df = frame()
    fig = _render_overlap_matrix(df, 0.34, "2026-10-01")
    heat = fig.data[0]
    z = [list(r) for r in heat.z]
    n = len(z)
    assert n == 5 and list(heat.x) == list(heat.y)
    assert all(z[i][j] == z[j][i] for i in range(n) for j in range(n))
    assert sum(v is not None for row in z for v in row) == 2 * len(df)
    # One outline (the 3-AOP family), and the identical pair marked both ways.
    assert len(fig.layout.shapes) == 1
    assert (fig.layout.shapes[0].x0, fig.layout.shapes[0].x1) == (-0.5, 2.5)
    assert len(fig.data[1].x) == 2
    assert "5 AOPs in 2 clusters" in fig.layout.title.text


def test_cluster_colours_skip_magenta_and_grey_out_pairs():
    colours = overlap_cluster_colours(frame())
    assert colours[1] == shared.BRAND_COLORS["primary"]
    assert colours[2] == "#B8B8B8"
    assert shared.BRAND_COLORS["magenta"] not in colours.values()


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = frame()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["latest_aop_aop_overlap_2026-10-01"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("latest_aop_aop_overlap", version="2026-10-01")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
