"""The AOP completeness beeswarm (#186): rows keyed on the AOP, no overlapping dots, CSV = /api/v2/data."""

from __future__ import annotations

import io
import random

import pandas as pd

from plots import data_api, shared
from plots.trends_plots import (
    BEESWARM_COLUMNS,
    BEESWARM_RADIUS,
    _render_beeswarm,
    beeswarm_rows,
    swarm_offsets,
)

AOP = "https://identifiers.org/aop/"


def cohort(rows):
    return pd.DataFrame([{"version": v, "aop_uri": AOP + str(a), "completeness": c, "entity_count": n}
                         for v, a, c, n in rows])


# AOP 10 matures between the snapshots; AOP 2 is new in the second; AOP 7 is dropped from it.
OPTIONAL = cohort([("2025-01-01", 10, 20.0, 5), ("2025-01-01", 7, 50.0, 3),
                   ("2025-04-01", 10, 61.234, 7), ("2025-04-01", 2, 40.0, 2)])
ALL = cohort([("2025-01-01", 10, 55.0, 5), ("2025-01-01", 7, 70.0, 3),
              ("2025-04-01", 10, 80.0, 7), ("2025-04-01", 2, 65.0, 2)])
TITLES = {AOP + "10": "Ten", AOP + "2": "Two"}


def frame():
    return beeswarm_rows(OPTIONAL, ALL, TITLES)


def test_rows_pair_the_cohorts_per_aop_in_number_order():
    df = frame()
    assert list(df.columns) == BEESWARM_COLUMNS
    assert df[["version", "aop_id"]].values.tolist() == [
        ["2025-01-01", "AOP 7"], ["2025-01-01", "AOP 10"], ["2025-04-01", "AOP 2"], ["2025-04-01", "AOP 10"]]
    ten = df[df["aop_id"] == "AOP 10"]
    assert ten["completeness_optional"].tolist() == [20.0, 61.23]
    assert ten["completeness_all"].tolist() == [55.0, 80.0]
    assert ten["entity_count"].tolist() == [5, 7]
    assert ten["aop"].unique().tolist() == [AOP + "10"]
    assert df.loc[df["aop_id"] == "AOP 7", "aop_title"].tolist() == [""]


def test_swarm_offsets_never_overlap_and_hug_the_axis():
    rng = random.Random(4)
    xs = [rng.uniform(0, 300) for _ in range(400)] + [150.0] * 30
    ys = swarm_offsets(xs, 4)
    d = 8
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            assert (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 >= d * d - 1e-3
    # A lone dot sits on the axis; well-spread dots stay on it.
    assert swarm_offsets([5.0], 4) == [0.0]
    assert swarm_offsets([0.0, 20.0, 40.0], 4) == [0.0, 0.0, 0.0]
    # Dots at one value stack symmetrically, upper side first.
    assert swarm_offsets([1.0, 1.0, 1.0], 4) == [0.0, 8.0, -8.0]


def test_server_figure_is_the_newest_snapshot_keyed_on_the_aop():
    df = frame()
    fig = _render_beeswarm(df, "2025-04-01", "completeness_optional", "x")
    dots = fig.data[0]
    assert list(dots.ids) == [AOP + "2", AOP + "10"]
    assert list(dots.x) == [40.0, 61.23]
    assert dots.marker.size == 2 * BEESWARM_RADIUS
    assert dots.marker.color == shared.BRAND_COLORS["primary"]
    assert "2 AOPs in 2025-04-01, median 50.6%" in fig.layout.title.text


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = frame()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["aop_completeness_beeswarm"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("aop_completeness_beeswarm")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
