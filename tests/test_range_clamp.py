"""Regression tests for the date-axis vs ordinal-index range-unit mismatch (#61).

Both bugs fixed during the #44 test cycle (commits e521b63 client-side,
be0c7b9 server-side) shared one root cause: passing **ordinal indices** to a
Plotly x-axis that auto-detects as ``type='date'``. Plotly then reinterprets
the indices as milliseconds-since-epoch and the rendered window collapses to
1970.

These tests pin the server-side clamp (`plots.shared._clamp_figure_to_range`,
the helper behind `export_figure_as_image`) so a future change that feeds
ordinal indices to a date-string axis fails fast. They use tiny in-memory
figures rather than `compute_plots_parallel()`, so the suite stays fast and
needs no SPARQL endpoint.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Make the dashboard package importable when pytest is invoked from the repo root.
_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

import plotly.graph_objects as go  # noqa: E402

from plots.shared import _clamp_figure_to_range  # noqa: E402

# Quarterly-style snapshot dates as they appear on a trend plot's x-axis.
SNAPSHOT_DATES = [
    "2018-04-01",
    "2020-01-01",
    "2022-01-01",
    "2023-01-01",
    "2024-01-01",
    "2026-01-01",
]


def _date_axis_figure() -> go.Figure:
    """A trend-style figure whose x-axis is YYYY-MM-DD strings (Plotly → date)."""
    return go.Figure(
        go.Scatter(x=list(SNAPSHOT_DATES), y=list(range(len(SNAPSHOT_DATES))))
    )


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def test_date_bounds_produce_date_string_range():
    """The core regression guard: date bounds must yield ISO-date range units.

    A regression to ordinal indices would put small numbers (e.g. -0.5, 3.5)
    here instead of ISO strings, collapsing the rendered window to 1970.
    """
    fig = _clamp_figure_to_range(_date_axis_figure(), "2022-01-01", "2024-01-01")
    rng = fig.layout.xaxis.range

    assert rng is not None, "clamp did not set an x-axis range"
    assert len(rng) == 2

    # Must be date strings — NOT ordinal indices. This is the bug signature.
    for bound in rng:
        assert not _is_number(bound), (
            f"x-axis range bound {bound!r} is numeric — ordinal indices were "
            "passed to a date axis (regression of #44/#61)"
        )
        # Parseable as an ISO datetime (the ±12h padded boundary).
        datetime.fromisoformat(str(bound))

    # The window must bracket the requested range (with the ±12h padding).
    lo, hi = datetime.fromisoformat(str(rng[0])), datetime.fromisoformat(str(rng[1]))
    assert lo < datetime(2022, 1, 1, 12)
    assert hi > datetime(2024, 1, 1, 0)
    assert lo < hi


def test_open_ended_bounds_use_axis_extremes():
    """A one-sided range still clamps in date units, defaulting the open end."""
    fig = _clamp_figure_to_range(_date_axis_figure(), "2022-01-01", None)
    rng = fig.layout.xaxis.range
    assert rng is not None
    for bound in rng:
        assert not _is_number(bound)
        datetime.fromisoformat(str(bound))


def test_clamp_does_not_mutate_input_figure():
    """`_clamp_figure_to_range` returns a deep copy; the cached figure is safe."""
    fig = _date_axis_figure()
    _clamp_figure_to_range(fig, "2022-01-01", "2024-01-01")
    assert fig.layout.xaxis.range is None, "input figure was mutated by the clamp"


def test_non_date_bounds_fall_back_to_ordinal_indices():
    """When bounds aren't YYYY-MM-DD, the ordinal fallback path is used.

    This locks the *other* branch so the date/ordinal split stays explicit.
    """
    fig = _clamp_figure_to_range(_date_axis_figure(), "2018", "2024")
    rng = fig.layout.xaxis.range
    assert rng is not None
    assert all(_is_number(b) for b in rng), (
        "non-date bounds should fall back to numeric ordinal positions"
    )


def test_no_x_categories_is_a_noop():
    """A figure with no string x-values is returned unclamped (no crash)."""
    fig = go.Figure(go.Bar(x=[1, 2, 3], y=[4, 5, 6]))
    out = _clamp_figure_to_range(fig, "2022-01-01", "2024-01-01")
    assert out.layout.xaxis.range is None


def test_clamped_date_figure_renders_to_decodable_png():
    """End-to-end smoke: a clamped date-axis figure exports to a real PNG.

    Guards against the rendered-frame collapse directly. Skips cleanly if the
    Kaleido image backend isn't usable in this environment (e.g. no headless
    browser) — the unit tests above remain the always-on regression guard.
    """
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    import plotly.io as pio

    fig = _clamp_figure_to_range(_date_axis_figure(), "2022-01-01", "2024-01-01")
    try:
        png = pio.to_image(fig, format="png", width=800, height=500, engine="kaleido")
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"Kaleido PNG export unavailable: {e}")

    assert png and len(png) > 1000, "PNG is suspiciously small / empty"
    import io

    img = Image.open(io.BytesIO(png))
    img.verify()  # raises if the bytes are not a valid image
