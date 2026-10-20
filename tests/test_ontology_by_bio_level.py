"""Ontologies by level of biological organisation: rows, shares, order, figure, CSV = /api/v2/data."""

from __future__ import annotations

import io

import pandas as pd

from plots import data_api, shared
from plots.latest_plots import (
    ONTOLOGY_BY_LEVEL_COLUMNS,
    _render_ontology_by_level,
    ontology_by_level_rows,
)


def row(level, role, ontology, n, kes):
    v = lambda x: {"value": str(x)}
    return {"level": v(level), "role": v(role), "ontology": v(ontology), "annotations": v(n), "level_kes": v(kes)}


def results():
    return [
        row("Population", "Process", "PCO", 3, 4), row("Population", "Process", "GO", 1, 4),
        row("Molecular", "Process", "GO", 18, 20), row("Molecular", "Process", "MP", 2, 20),
        row("Molecular", "Object", "PR", 9, 20), row("Molecular", "Object", "CHEBI", 1, 20),
        row("Not annotated", "Object", "MESH", 5, 7),
    ]


def test_shares_are_per_level_and_role_and_levels_run_small_to_large():
    df = ontology_by_level_rows(results(), "2026-10-01")
    assert list(df.columns) == ONTOLOGY_BY_LEVEL_COLUMNS
    assert list(dict.fromkeys(df["Biological Level"])) == ["Molecular", "Population", "Not annotated"]
    go = df[(df["Biological Level"] == "Molecular") & (df["Ontology"] == "GO")].iloc[0]
    assert (go["Share (%)"], go["Level Annotations"], go["Level KEs"]) == (90.0, 20, 20)
    assert df[df["Biological Level"] == "Population"]["Share (%)"].tolist() == [75.0, 25.0]
    # Process before object within a level, biggest first.
    assert df[df["Biological Level"] == "Molecular"]["Ontology"].tolist() == ["GO", "MP", "PR", "CHEBI"]
    assert ontology_by_level_rows([], "2026-10-01").empty


def test_figure_has_a_panel_per_role_one_legend_entry_and_colour_per_ontology():
    fig = _render_ontology_by_level(ontology_by_level_rows(results(), "2026-10-01"))
    assert [a.text for a in fig.layout.annotations] == ["Process terms", "Object terms"]
    legend = [t.name for t in fig.data if t.showlegend]
    assert sorted(legend) == sorted(set(t.name for t in fig.data))
    colours = {}
    for t in fig.data:
        assert colours.setdefault(t.name, t.marker.color) == t.marker.color
    # The first level is drawn on top.
    assert list(fig.layout.yaxis.categoryarray)[-1] == "Molecular (20 KEs)"
    pco = next(t for t in fig.data if t.name == "PCO")
    assert list(pco.x) == [75.0]
    assert fig.layout.barmode == "stack"


def test_small_ontologies_share_other_unless_they_matter_somewhere():
    many = [row("Cellular", "Object", f"O{i}", 100 - i, 50) for i in range(14)]
    many.append(row("Population", "Object", "PCO", 2, 3))  # 2 annotations, but 100% of its level
    fig = _render_ontology_by_level(ontology_by_level_rows(many, "2026-10-01"))
    names = {t.name for t in fig.data}
    assert "PCO" in names and "Other" in names
    assert "O13" not in names


def test_csv_download_equals_the_v2_data(monkeypatch):
    df = ontology_by_level_rows(results(), "2026-10-01")
    cache = shared.VersionedPlotCache(max_versions=5, ttl_seconds=1800)
    cache["latest_ontology_by_bio_level_2026-10-01"] = df
    monkeypatch.setattr(shared, "_plot_data_cache", cache)
    csv = shared.get_csv_with_metadata("latest_ontology_by_bio_level_2026-10-01")
    lines = csv.splitlines()
    header = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    from_csv = pd.read_csv(io.StringIO("\n".join(lines[header:])), keep_default_na=False)
    from_api = data_api.columnar_to_frame(data_api.frame_to_columnar(df))
    assert list(from_csv.columns) == list(from_api.columns)
    pd.testing.assert_frame_equal(from_csv, from_api, check_dtype=False)
