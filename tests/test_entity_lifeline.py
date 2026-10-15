"""The entity lifeline's rows (#183): status, ordering, links, and CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from plots import data_api, shared
from plots.trends_plots import aopwiki_page, lifeline_rows, lifeline_status

V = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]
KE = "https://identifiers.org/aop.events/"


@pytest.mark.parametrize("presence,expected", [
    ("xxxx", ("always present", 0)),
    ("--xx", ("added", 0)),
    ("xx--", ("removed", 0)),
    ("x-xx", ("returned", 1)),
    ("x-x-x", ("returned", 2)),
    ("x-x-", ("removed", 1)),   # gone now: removed wins over returned
    ("-x--", ("removed", 0)),
])
def test_status(presence, expected):
    assert lifeline_status(presence) == expected


def test_rows_order_by_arrival_then_lifetime_then_number():
    uris = {
        V[0]: {KE + "9", KE + "10"},
        V[1]: {KE + "9", KE + "10", KE + "2"},
        V[2]: {KE + "10", KE + "2"},
        V[3]: {KE + "10", KE + "2", KE + "9"},
    }
    rows = lifeline_rows("KEs", uris, V, {KE + "2": "Increase, X"})
    assert [r["id"] for r in rows] == ["KE 9", "KE 10", "KE 2"]  # 9 and 10 arrive first, live to the end; 9 < 10
    nine = rows[0]
    assert nine["presence"] == "xx-x" and nine["status"] == "returned" and nine["returns"] == 1
    assert nine["first_seen"] == V[0] and nine["last_seen"] == V[3] and nine["snapshots_present"] == 3
    assert rows[2]["label"] == "Increase, X" and rows[2]["status"] == "added"
    assert nine["aopwiki_url"] == "https://aopwiki.org/events/9"


def test_aopwiki_pages():
    assert aopwiki_page("https://identifiers.org/aop/12") == "https://aopwiki.org/aops/12"
    assert aopwiki_page("https://identifiers.org/aop.relationships/5") == "https://aopwiki.org/relationships/5"
    assert aopwiki_page("https://identifiers.org/aop.stressor/7") == "https://aopwiki.org/stressors/7"
    assert aopwiki_page("http://example.org/x") == ""


def test_csv_download_equals_the_v2_data(monkeypatch):
    uris = {V[0]: {KE + "1"}, V[1]: set(), V[2]: {KE + "1", KE + "3"}, V[3]: {KE + "3"}}
    df = pd.DataFrame(lifeline_rows("KEs", uris, V, {KE + "1": "A, label # with a hash"}))
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)  # no __delitem__, so swap the whole cache
    cache["entity_lifeline"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("entity_lifeline")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
    assert from_csv["presence"].tolist() == ["x-x-", "--xx"]  # text, leading "-" kept
