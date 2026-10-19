"""JSON forms of plot data and figures for the v2 data API (#180). Offline."""

import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from plots import data_api


def _frame():
    return pd.DataFrame({
        "version": ["2024-01-01", "2025-01-01", "2026-01-01"],
        "count": np.array([1, 2, 3], dtype="int64"),
        "share": [0.5, float("nan"), 1 / 3],
        "created": pd.to_datetime(["2016-11-29T18:41:15", None, "2020-01-02"], format="mixed"),
        "flag": [True, False, True],
        "label": ["a", None, "c"],
    })


def test_columnar_round_trip_keeps_rows_and_order():
    df = _frame()
    payload = data_api.frame_to_columnar(df)
    assert [c["name"] for c in payload["columns"]] == list(df.columns)
    assert {c["name"]: c["type"] for c in payload["columns"]} == {
        "version": "string", "count": "integer", "share": "number",
        "created": "datetime", "flag": "boolean", "label": "string"}
    assert payload["data"]["count"] == [1, 2, 3]
    assert payload["data"]["share"][1] is None and math.isclose(payload["data"]["share"][2], 1 / 3)
    assert payload["data"]["created"] == ["2016-11-29T18:41:15", None, "2020-01-02T00:00:00"]
    assert payload["data"]["label"] == ["a", None, "c"]
    back = data_api.columnar_to_frame(payload)
    assert list(back.columns) == list(df.columns) and len(back) == 3


def test_body_is_strict_json_and_stable():
    payload = {"meta": {"id": "x"}, "data": data_api.frame_to_columnar(_frame())["data"]}
    body = data_api.encode(payload)
    json.loads(body)  # no NaN/Infinity tokens
    assert data_api.encode(payload) == body
    assert data_api.etag(body) == data_api.etag(data_api.encode(json.loads(body)))


def test_infinity_and_numpy_scalars():
    assert data_api._json_value(np.float64("inf")) is None
    assert data_api._json_value(np.int32(7)) == 7 and isinstance(data_api._json_value(np.int32(7)), int)
    assert data_api._json_value({"a": np.float32(0.5)}) == {"a": 0.5}
    assert data_api._json_value(pd.NaT) is None


@pytest.mark.parametrize("header,expected", [
    (None, False),
    ('"abc"', True),
    ('W/"abc"', True),
    ('"zzz", "abc"', True),
    ('"zzz"', False),
    ("*", True),
])
def test_if_none_match(header, expected):
    assert data_api.etag_matches(header, '"abc"') is expected


def test_figure_json_drops_the_template():
    fig = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))
    fig.update_layout(template="plotly_white", title="t")
    spec = data_api.figure_to_json(fig)
    assert "template" not in spec["layout"]
    assert spec["layout"]["title"]["text"] == "t"
    assert spec["data"][0]["y"] == [1, 2]


def test_disclosed_queries_fill_the_graph():
    # Notes write the marker bare, as static/data/methodology_notes.json does.
    note = {"queries": [{"caption": "c", "query": "SELECT * { GRAPH __GRAPH_URI__ {} }"}]}
    out = data_api.disclosed_queries(note, "http://aopwiki.org/graph/2024-01-01")
    assert out == [{"caption": "c", "query": "SELECT * { GRAPH <http://aopwiki.org/graph/2024-01-01> {} }"}]
    assert data_api.disclosed_queries({"sparql": "Q"}, None) == [{"caption": "", "query": "Q"}]
    assert data_api.disclosed_queries(None, None) == []


def test_disclosed_queries_parse_for_every_note():
    """Every query /api/v2 discloses is valid SPARQL once the graph is filled in (#244)."""
    import json
    from pathlib import Path
    from rdflib.plugins.sparql.parser import parseQuery  # syntax only: notes lean on the endpoint's prefixes

    notes = json.loads((Path(__file__).resolve().parent.parent / "static/data/methodology_notes.json").read_text())
    checked = 0
    for note in notes.values():
        if not isinstance(note, dict) or "__GRAPH_URI__" not in json.dumps(note):
            continue
        for item in data_api.disclosed_queries(note, "http://aopwiki.org/graph/2024-01-01"):
            parseQuery(item["query"])
            checked += 1
    assert checked > 50
