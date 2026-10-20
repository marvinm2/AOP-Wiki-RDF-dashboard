"""KER curation calendar (#187): days, bulk imports, calendar cells, CSV = /api/v2/data."""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from plots import data_api, shared
from plots.trends_plots import (
    KER_CALENDAR_BULK_MIN,
    KER_CALENDAR_CAP,
    KER_CALENDAR_COLUMNS,
    _calendar_week,
    _render_ker_calendar,
    ker_calendar_days,
)


def created():
    bulk = ["2016-11-29T18:41:33"] * (KER_CALENDAR_BULK_MIN + 1)
    return bulk + [
        "2023-10-17T08:00:00", "2023-10-17T23:59:59",   # a Tuesday, two KERs
        "2024-01-01T00:00:01",                          # a Monday, 1 January
        "2024-12-31T12:00:00",                          # week 53 of a leap year
        *(["2025-03-05T10:00:00"] * 25),                 # above the colour cap
        "",                                             # no value: skipped
    ]


def test_days_are_counted_and_the_bulk_import_is_flagged():
    df = ker_calendar_days(created())
    assert list(df.columns) == KER_CALENDAR_COLUMNS
    rows = {r.Date: r for r in df.itertuples(index=False)}
    assert rows["2016-11-29"][4] == KER_CALENDAR_BULK_MIN + 1
    assert rows["2016-11-29"][5] is True
    assert (rows["2023-10-17"][1], rows["2023-10-17"][4], rows["2023-10-17"][5]) == ("Tuesday", 2, False)
    assert df["KERs Created"].sum() == len(created()) - 1


def test_weeks_start_on_monday_with_1_january_in_week_0():
    assert _calendar_week(dt.date(2024, 1, 1)) == 0      # a Monday
    assert _calendar_week(dt.date(2024, 1, 7)) == 0      # the Sunday after
    assert _calendar_week(dt.date(2024, 1, 8)) == 1
    assert _calendar_week(dt.date(2025, 1, 1)) == 0      # a Wednesday
    assert _calendar_week(dt.date(2025, 1, 6)) == 1
    assert _calendar_week(dt.date(2024, 12, 31)) == 52
    assert max(_calendar_week(dt.date(y, 12, 31)) for y in range(2016, 2030)) <= 53


def test_one_strip_per_curated_year_with_blank_days_after_the_newest_snapshot():
    fig = _render_ker_calendar(ker_calendar_days(created()), until="2025-06-30")
    # 2016 only has the bulk import, so it gets no strip.
    assert [t.name for t in fig.data] == ["2023", "2024", "2025"]
    by_year = {t.name: t for t in fig.data}
    tue = by_year["2023"]
    w = _calendar_week(dt.date(2023, 10, 17))
    assert tue.z[1][w] == 2
    assert tue.customdata[1][w] == "Tue 2023-10-17: 2"
    # Capped colour, real count in the hover text.
    w = _calendar_week(dt.date(2025, 3, 5))
    assert by_year["2025"].z[2][w] == KER_CALENDAR_CAP
    assert by_year["2025"].customdata[2][w].endswith(": 25")
    # Days after the newest snapshot, and days of other years, are blank.
    w = _calendar_week(dt.date(2025, 7, 1))
    assert by_year["2025"].z[1][w] is None and by_year["2025"].customdata[1][w] == ""
    assert by_year["2024"].z[0][0] == 1 and by_year["2025"].z[0][0] is None  # 2024-12-30, not 2025
    assert "101 KERs of a bulk import on 2016-11-29 left out" in fig.layout.title.text
    # Numeric x: the server-side range clamp leaves it alone.
    assert shared._figure_x_categories(fig) == []


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = ker_calendar_days(created())
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["ker_curation_calendar"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("ker_curation_calendar")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
