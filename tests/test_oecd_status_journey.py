"""The OECD status journey (#187): anchors, stages, flows, figure, CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd

from plots import data_api, shared
from plots.trends_plots import (
    _JOURNEY_BOTTOM,
    _JOURNEY_HEIGHT,
    _JOURNEY_PAD,
    _JOURNEY_TOP,
    OECD_JOURNEY_COLUMNS,
    OECD_WORK_PLAN_FLAG,
    _oecd_stage,
    _render_oecd_status_journey,
    oecd_journey_anchors,
    oecd_status_journey_rows,
)

ANCHORS = ["2023-01-01", "2024-01-01", "2025-01-01"]


def statuses():
    rows = []

    def add(aop, *per_anchor):
        for version, status in zip(ANCHORS, per_anchor):
            for s in status if isinstance(status, tuple) else (status,):
                if s is not None:
                    rows.append({"version": version, "aop": aop, "status": s})

    # a1: work-plan flag beside Under Development, then wiped to No Status.
    add("a1", (OECD_WORK_PLAN_FLAG, "Under Development"), (OECD_WORK_PLAN_FLAG, "Under Development"), "No Status")
    add("a2", "Under Development", "Under Development", "No Status")
    # a3 never changes, so it is left out.
    add("a3", "WPHA/WNT Endorsed", "WPHA/WNT Endorsed", "WPHA/WNT Endorsed")
    # a4 only exists from 2024 and moves on.
    add("a4", None, "Under Development", "Under Review")
    # a5: the flag on its own is kept as a stage.
    add("a5", OECD_WORK_PLAN_FLAG, "Under Development", "Under Development")
    # Snapshots that are not anchors are ignored.
    rows.append({"version": "2024-04-01", "aop": "a3", "status": "No Status"})
    return pd.DataFrame(rows)


def journey():
    return oecd_status_journey_rows(statuses(), ANCHORS)


def test_one_anchor_per_year_and_the_latest_for_the_latest_year():
    versions = ["2019-01-01", "2019-04-01", "2020-01-01", "2021-01-03", "2022-01-01", "2022-07-01",
                "2023-04-01", "2026-01-01", "2026-10-01"]
    # 2021's January snapshot is on the 3rd; 2023 has none.
    assert oecd_journey_anchors(versions) == ["2019-01-01", "2020-01-01", "2021-01-03", "2022-01-01", "2026-10-01"]
    assert oecd_journey_anchors(["2025-01-01", "2026-01-01"]) == ["2025-01-01", "2026-01-01"]
    assert oecd_journey_anchors([]) == []


def test_the_work_plan_flag_is_not_a_stage_unless_alone():
    assert _oecd_stage([OECD_WORK_PLAN_FLAG, "Under Development"]) == "Under Development"
    assert _oecd_stage([OECD_WORK_PLAN_FLAG]) == OECD_WORK_PLAN_FLAG
    assert _oecd_stage(["No Status"]) == "No Status"


def test_flows_keep_only_aops_that_changed():
    links, changed, total = journey()
    assert list(links.columns) == OECD_JOURNEY_COLUMNS
    assert (changed, total) == (4, 5)
    flows = {tuple(r[:4]): r[4] for r in links.itertuples(index=False)}
    assert flows == {
        ("2023-01-01", "2024-01-01", "Under Development", "Under Development"): 2,
        ("2023-01-01", "2024-01-01", OECD_WORK_PLAN_FLAG, "Under Development"): 1,
        ("2024-01-01", "2025-01-01", "Under Development", "No Status"): 2,
        ("2024-01-01", "2025-01-01", "Under Development", "Under Review"): 1,
        ("2024-01-01", "2025-01-01", "Under Development", "Under Development"): 1,
    }
    # Snapshot order, then the stages in workflow order.
    assert list(links["From Snapshot"]) == sorted(links["From Snapshot"])


def test_figure_places_one_column_per_anchor_without_overlap():
    links, changed, total = journey()
    fig = _render_oecd_status_journey(links, changed, total)
    sankey = fig.data[0]
    assert sankey.type == "sankey"
    assert sum(sankey.link.value) == links["AOPs"].sum()
    snapshots = [cd[0] for cd in sankey.node.customdata]
    xs = dict(zip(snapshots, sankey.node.x))
    assert [round(xs[a], 3) for a in ANCHORS] == [0.001, 0.5, 0.999]
    # a4 joins in 2024 without an incoming band, but is still drawn in the 2024 column.
    assert ("2024-01-01", "Under Development") in set(zip(snapshots, sankey.node.label))

    # Stacked nodes in a column leave at least the pad between them, at Plotly's scale.
    plot_h = _JOURNEY_HEIGHT - _JOURNEY_TOP - _JOURNEY_BOTTOM
    sizes = [cd[1] for cd in sankey.node.customdata]
    columns = {a: [i for i, s in enumerate(snapshots) if s == a] for a in ANCHORS}
    ky = min((plot_h - (len(c) - 1) * _JOURNEY_PAD) / sum(sizes[i] for i in c) for c in columns.values())
    for nodes in columns.values():
        for upper, lower in zip(nodes, nodes[1:]):
            gap = (sankey.node.y[lower] - sankey.node.y[upper]) * plot_h
            assert gap >= (sizes[upper] + sizes[lower]) * ky / 2 + _JOURNEY_PAD - 1e-6

    # The 2024-04-01 wipe falls between the 2024 and 2025 columns.
    assert [round(s.x0, 3) for s in fig.layout.shapes] == [0.75]
    assert any("status wipe" in a.text for a in fig.layout.annotations)
    assert "4 of 5 AOPs changed status" in fig.layout.title.text


def test_csv_download_equals_the_v2_data(monkeypatch):
    links, _, _ = journey()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["oecd_status_journey"] = links
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("oecd_status_journey")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(links))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
