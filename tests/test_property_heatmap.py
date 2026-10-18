"""The property x snapshot heatmap (#187): rows, figure, CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd

from plots import data_api, shared
from plots.trends_plots import (
    PROPERTY_HEATMAP_COLUMNS,
    PROPERTY_PRESENCE_ENTITIES,
    RDF_TYPE,
    _render_property_heatmap,
    property_heatmap_rows,
)

DC = "http://purl.org/dc/elements/1.1/"
AOPO = "http://aopkb.org/aop_ontology#"
VERSIONS = ["2019-01-01", "2020-01-01", "2021-01-01"]

LABELS = pd.DataFrame([
    {"uri": DC + "title", "label": "Title", "type": "Essential", "applies_to": "AOP"},
    {"uri": DC + "creator", "label": "Creator", "type": "Metadata", "applies_to": "AOP"},
    {"uri": AOPO + "has_key_event", "label": "Has Key Event", "type": "Content", "applies_to": "AOP"},
])


def frames():
    # Title is always there; Has Key Event arrives in 2020; Creator is gone in
    # 2021; an unlabelled predicate shows up once; rdf:type is everywhere.
    props = [
        *[(v, DC + "title", t) for v, t in zip(VERSIONS, [10, 20, 40])],
        ("2020-01-01", AOPO + "has_key_event", 5), ("2021-01-01", AOPO + "has_key_event", 30),
        ("2019-01-01", DC + "creator", 10), ("2020-01-01", DC + "creator", 4),
        ("2021-01-01", "http://example.org/vocab#some_note", 2),
        *[(v, RDF_TYPE, t) for v, t in zip(VERSIONS, [10, 20, 40])],
    ]
    df_props = pd.DataFrame(props, columns=["version", "property", "count"])
    df_total = pd.DataFrame({"version": VERSIONS, "total_aops": [10, 20, 40]})
    return df_props, df_total


def rows():
    return property_heatmap_rows(*frames(), LABELS)


def test_rows_keep_every_property_but_rdf_type_in_type_order():
    df = rows()
    assert list(df.columns) == PROPERTY_HEATMAP_COLUMNS
    assert RDF_TYPE not in set(df["property"])
    # Essential, Content, Metadata, then unlabelled; each over every snapshot.
    assert list(dict.fromkeys(df["label"])) == ["Title", "Has Key Event", "Creator", "Some Note"]
    assert len(df) == 4 * len(VERSIONS)
    assert df[df["label"] == "Some Note"]["type"].unique().tolist() == ["Other"]


def test_absent_snapshots_are_zero_and_first_filled_is_the_first_nonzero():
    df = rows().set_index(["label", "version"])
    assert df.loc[("Has Key Event", "2019-01-01"), "filled"] == 0
    assert df.loc[("Has Key Event", "2021-01-01"), "percentage"] == 75.0
    assert df.loc[("Creator", "2021-01-01"), "percentage"] == 0.0
    assert set(df.xs("Has Key Event")["first_filled"]) == {"2020-01-01"}
    assert set(df.xs("Title")["first_filled"]) == {"2019-01-01"}


def test_figure_is_one_heatmap_cell_per_row():
    df = rows()
    fig = _render_property_heatmap(df, PROPERTY_PRESENCE_ENTITIES["aop"])
    heat = fig.data[0]
    assert heat.type == "heatmap"
    assert list(heat.x) == VERSIONS
    assert list(heat.y) == ["Title", "Has Key Event", "Creator", "Some Note"]
    assert [list(r) for r in heat.z][1] == [0.0, 25.0, 75.0]
    # A separator between each change of property type.
    assert len(fig.layout.shapes) == 3
    assert "4 AOPs properties across 3 snapshots" in fig.layout.title.text
    # The x values stay strings, so the /trends range clamp keeps working.
    assert shared._figure_x_categories(fig) == VERSIONS


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = rows()
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["aop_property_heatmap"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)

    csv = shared.get_csv_with_metadata("aop_property_heatmap")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))

    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
