"""JSON forms of a plot's data and figure for the v2 data API (#180).

``/api/v2/data/<plot>`` serves the same DataFrame the CSV download writes, in
columnar form, and ``/api/v2/figure/<plot>`` the same figure the image export
draws, as Plotly JSON without its template. Both carry an ETag over the body,
so a client that sends it back gets a 304 while nothing changed. The body is
deterministic (#220), so a content hash is exact and never goes stale.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    """One cell as a JSON value: NaN/NaT become null, numpy scalars plain numbers."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, str) else str(value)


def _column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def frame_to_columnar(df: pd.DataFrame) -> Dict[str, Any]:
    """``{"columns": [{"name", "type"}], "data": {name: [values]}}``, rows in frame order."""
    columns: List[Dict[str, str]] = []
    data: Dict[str, list] = {}
    for position, name in enumerate(df.columns):
        series = df.iloc[:, position]
        key = str(name)
        columns.append({"name": key, "type": _column_type(series)})
        data[key] = [_json_value(v) for v in series.tolist()]
    return {"columns": columns, "data": data}


def columnar_to_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    """The inverse of :func:`frame_to_columnar`, for clients and tests."""
    names = [c["name"] for c in payload["columns"]]
    return pd.DataFrame({n: payload["data"][n] for n in names}, columns=names)


def figure_to_json(fig) -> Dict[str, Any]:
    """Plotly figure JSON without ``layout.template``.

    The template is the same ~7 KB for every figure; a client applies its own
    (Plotly's default, or the dashboard's) with ``Plotly.react``.
    """
    spec = json.loads(fig.to_json())
    spec.get("layout", {}).pop("template", None)
    return spec


def encode(payload: Dict[str, Any]) -> bytes:
    """Compact, key-sorted JSON, so equal payloads give equal bytes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def etag(body: bytes) -> str:
    return '"' + hashlib.sha1(body).hexdigest() + '"'


def etag_matches(if_none_match: Optional[str], tag: str) -> bool:
    """RFC 7232 weak comparison against an If-None-Match header."""
    if not if_none_match:
        return False
    if if_none_match.strip() == "*":
        return True
    wanted = tag.strip('"')
    return any(t.strip().removeprefix("W/").strip('"') == wanted
               for t in if_none_match.split(","))


def disclosed_queries(note: Optional[dict], graph_uri: Optional[str]) -> List[Dict[str, str]]:
    """The SPARQL a methodology note discloses, with its graph marker filled in."""
    if not note:
        return []
    if isinstance(note.get("queries"), list):
        items = [{"caption": q.get("caption", ""), "query": q.get("query", "")} for q in note["queries"]]
    elif isinstance(note.get("sparql"), str):
        items = [{"caption": "", "query": note["sparql"]}]
    elif isinstance(note.get("sparql"), list):
        items = [{"caption": "", "query": q} for q in note["sparql"]]
    else:
        items = []
    if graph_uri:
        for item in items:
            item["query"] = item["query"].replace("__GRAPH_URI__", graph_uri)
    return items
