"""Bar | Pie switch on five snapshot plots (#255): settings, defaults, the pie view."""

from __future__ import annotations

import inspect

import pandas as pd

import plots
from plots import registry
from plots.latest_plots import _share_pie

SWITCHES = {
    "latest_ontology_usage": "ontology_chart",
    "latest_process_usage": "process_chart",
    "latest_object_usage": "object_chart",
    "latest_ke_annotation_depth": "depth_chart",
    "latest_organ_coverage_pie": "organ_chart",
}


def test_each_plot_takes_its_own_switch_with_bars_by_default():
    for plot_id, key in SWITCHES.items():
        spec = registry.BY_ID[plot_id]
        assert key in spec.params, plot_id
        func = inspect.unwrap(getattr(plots, spec.producer))
        assert inspect.signature(func).parameters[key].default == "bar", plot_id


def test_the_pie_keeps_the_given_order_and_shows_shares():
    df = pd.DataFrame({"Depth": ["No annotations", "1 components", "2 components"], "KE Count": [5, 30, 10]})
    fig = _share_pie(df, "Depth", "KE Count", "Key Events", order=df["Depth"].tolist())
    pie = fig.data[0]
    assert pie.type == "pie"
    assert list(pie.labels) == ["No annotations", "1 components", "2 components"]
    assert pie.sort is False and pie.direction == "clockwise"
    assert pie.textinfo == "percent+label"
    assert "Key Events: %{value}" in pie.hovertemplate


def test_a_colour_map_colours_the_slices_by_name():
    df = pd.DataFrame({"Organ System": ["Nervous", "Unclassified"], "AOPs": [40, 7]})
    fig = _share_pie(df, "Organ System", "AOPs", "AOPs", color_map={"Nervous": "#111111", "Unclassified": "#cccccc"})
    assert list(fig.data[0].marker.colors) == ["#111111", "#cccccc"]
