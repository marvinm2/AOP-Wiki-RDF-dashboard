"""Cohort survival (#187): cohorts, Kaplan-Meier steps, cut-off, figure, CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd

from plots import data_api, shared
from plots.trends_plots import (
    SURVIVAL_COLUMNS,
    SURVIVAL_MIN_AT_RISK,
    _render_entity_survival,
    survival_cohort,
    survival_rows,
)

VERSIONS = ["2018-04-01", "2018-07-01", "2019-01-01", "2019-04-01", "2020-01-01", "2020-04-01"]


def presence(spec: dict) -> dict:
    """{uri: "x-x..."} -> {version: set of uris}."""
    return {v: {u for u, p in spec.items() if p[i] == "x"} for i, v in enumerate(VERSIONS)}


def test_cohorts_are_two_years_wide_and_the_first_snapshot_is_its_own():
    assert survival_cohort("2018-04-01", VERSIONS) == "Already in 2018-04-01"
    assert survival_cohort("2018-07-01", VERSIONS) == "First seen 2018–2019"
    assert survival_cohort("2019-04-01", VERSIONS) == "First seen 2018–2019"
    # The last cohort is cut at the latest year.
    assert survival_cohort("2020-01-01", VERSIONS) == "First seen 2020"


def test_kaplan_meier_counts_the_first_gap_and_censors_at_the_latest():
    n = SURVIVAL_MIN_AT_RISK
    spec = {"old": "xxxxxx"}  # the first snapshot is not empty
    spec.update({f"s{i}": "-xxxxx" for i in range(n)})  # 10 that stay
    spec.update({f"g{i}": "-x-xxx" for i in range(n)})  # 10 gone after one step, back later
    spec.update({f"late{i}": "---xxx" for i in range(n)})  # 10 that stay, followed 2 steps
    rows = pd.DataFrame(survival_rows("KEs", presence(spec), VERSIONS), columns=SURVIVAL_COLUMNS)
    cohort = rows[rows["Cohort"] == "First seen 2018–2019"]
    assert list(cohort["Cohort Size"].unique()) == [30]
    assert list(cohort["Years Since First Seen"]) == [0, 0.25, 0.5, 0.75, 1.0]
    # late* drop out of the count after 2 steps (censored, not removed).
    assert list(cohort["At Risk"]) == [30, 30, 20, 10, 10]
    assert list(cohort["Removed"]) == [0, 10, 0, 0, 0]
    # 1 - 10/30 after one step; nothing removed after that.
    assert list(cohort["Surviving (%)"]) == [100.0, 66.67, 66.67, 66.67, 66.67]


def test_a_line_stops_below_the_minimum_at_risk_and_empty_versions_are_skipped():
    n = SURVIVAL_MIN_AT_RISK
    spec = {f"a{i}": "xxxxxx" for i in range(n - 1)}
    spec.update({f"b{i}": "xxx---" for i in range(n)})
    by_version = presence(spec)
    by_version["2019-04-01"] = set()  # a failed query, not a mass removal
    rows = pd.DataFrame(survival_rows("KERs", by_version, VERSIONS), columns=SURVIVAL_COLUMNS)
    assert set(rows["Cohort"]) == {"Already in 2018-04-01"}
    # 19 followed; b* go at the 4th kept snapshot (index 3), leaving 9 < 10.
    assert list(rows["Years Since First Seen"]) == [0, 0.25, 0.5, 0.75]
    assert list(rows["At Risk"]) == [19, 19, 19, 19]
    assert rows["Surviving (%)"].iloc[-1] == round(100 * 9 / 19, 2)


def test_figure_has_a_panel_per_entity_and_a_legend_entry_per_cohort():
    spec = {f"k{i}": "xxxxxx" for i in range(12)}
    spec.update({f"n{i}": "-xxxxx" for i in range(12)})
    rows = []
    for entity in ("AOPs", "KEs", "KERs"):
        rows.extend(survival_rows(entity, presence(spec), VERSIONS))
    fig = _render_entity_survival(pd.DataFrame(rows, columns=SURVIVAL_COLUMNS),
                                  {"AOPs": 24, "KEs": 24, "KERs": 24})
    assert len(fig.data) == 6
    assert [t.name for t in fig.data if t.showlegend] == ["Already in 2018-04-01", "First seen 2018–2019"]
    assert {t.line.shape for t in fig.data} == {"hv"}
    assert [a.text for a in fig.layout.annotations][:3] == [f"{e} (24 ever seen)" for e in ("AOPs", "KEs", "KERs")]
    # Numeric x: the server-side range clamp leaves it alone.
    assert shared._figure_x_categories(fig) == []
    clamped = shared._clamp_figure_to_range(fig, "2019-01-01", "2020-01-01")
    assert clamped.layout.xaxis.range is None


def test_csv_download_equals_the_v2_data(monkeypatch):
    spec = {f"k{i}": "xx-xxx" for i in range(12)}
    df = pd.DataFrame(survival_rows("KEs", presence(spec), VERSIONS), columns=SURVIVAL_COLUMNS)
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["entity_survival"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("entity_survival")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
