"""Server image export on Kaleido 1.x (#267): one Chromium per worker, exports never swapped."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import plotly.graph_objects as go
import pytest

from plots import shared


def figure(title):
    return go.Figure(go.Bar(x=["a", "b"], y=[1, 2]), layout={"title": {"text": title}})


@pytest.fixture(scope="module")
def export():
    try:
        shared._to_image(figure("warm-up"), format="svg")
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"Kaleido export unavailable: {e}")
    return shared._to_image


def test_concurrent_exports_each_get_their_own_figure(export):
    titles = [f"figure-{i:02d}" for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        svgs = list(pool.map(lambda t: export(figure(t), format="svg").decode(), titles))
    for title, svg in zip(titles, svgs):
        assert [t for t in titles if t in svg] == [title]


def test_a_failed_export_restarts_the_server(export, monkeypatch):
    calls = []
    real = shared.pio.to_image

    def flaky(fig, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("browser gone")
        return real(fig, **kwargs)

    monkeypatch.setattr(shared.pio, "to_image", flaky)
    with pytest.raises(RuntimeError):
        export(figure("x"), format="svg")
    assert shared._kaleido_pid is None
    assert b"after" in export(figure("after"), format="svg")


def test_the_browser_is_restarted_after_a_number_of_exports(export, monkeypatch):
    monkeypatch.setattr(shared, "KALEIDO_RECYCLE_AFTER", 2)
    shared._stop_kaleido_when_idle()            # start from a fresh browser
    export(figure("one"), format="svg")
    assert shared._kaleido_pid is not None and shared._kaleido_exports == 1
    export(figure("two"), format="svg")
    assert shared._kaleido_pid is None and shared._kaleido_exports == 0
    assert b"three" in export(figure("three"), format="svg")


def test_an_idle_browser_is_stopped(export, monkeypatch):
    import time
    monkeypatch.setattr(shared, "KALEIDO_IDLE_SECONDS", 0.2)
    export(figure("idle"), format="svg")
    assert shared._kaleido_pid is not None
    deadline = time.time() + 10
    while shared._kaleido_pid is not None and time.time() < deadline:
        time.sleep(0.1)
    assert shared._kaleido_pid is None


def test_png_has_the_requested_size(export):
    png = export(figure("size"), format="png", width=640, height=360)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")) == (640, 360)


def numeric_figure():
    import numpy as np
    return go.Figure([go.Scatter(x=np.array([1, 2, 3]), y=np.array([0.5, np.nan, 2.5])),
                      go.Heatmap(z=np.array([[1, 2], [3, 4]]), x=["2024-01-01", "2024-04-01"])])


def test_figures_for_the_browser_carry_plain_lists():
    from plots import data_api
    spec = data_api.figure_to_json(numeric_figure())
    scatter, heatmap = spec["data"]
    assert scatter["x"] == [1, 2, 3] and scatter["y"] == [0.5, None, 2.5]
    assert heatmap["z"] == [[1, 2], [3, 4]] and heatmap["x"] == ["2024-01-01", "2024-04-01"]
    html = shared.render_plot_html(numeric_figure())
    assert "bdata" not in html and "[[1,2],[3,4]]" in html.replace(" ", "")
