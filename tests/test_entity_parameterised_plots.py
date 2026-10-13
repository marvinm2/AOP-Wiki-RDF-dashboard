"""One function per plot family, parameterised by entity type (#178).

``plot_property_presence`` replaced four per-entity trend functions and
``plot_latest_completeness_by_status`` three per-entity snapshot functions.
Their output was checked byte for byte against the old functions on the live
endpoint when they were merged; these tests pin what differs per entity, so a
later edit cannot quietly make one entity render like another. Offline: SPARQL
is answered by a stub.
"""

import pathlib

import pytest

import plots
from plots import latest_plots, registry, trends_plots
from plots.shared import _plot_data_cache, _plot_figure_cache

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = "http://aopwiki.org/graph/"
TITLE_URI = "http://purl.org/dc/elements/1.1/title"


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    """property_labels.csv is read relative to the working directory."""
    monkeypatch.chdir(ROOT)


def _stub(monkeypatch, module, answer):
    queries = []

    def run(query, *args, **kwargs):
        queries.append(query)
        return answer(query)

    monkeypatch.setattr(module, "run_sparql_query", run)
    return queries


# --- Property presence (trend) ------------------------------------------------

PRESENCE = {
    # entity: (class in the query, total column, title noun, y-axis noun)
    "aop": ("aopo:AdverseOutcomePathway", "total_aops", "AOPs", "AOPs"),
    "ke": ("aopo:KeyEvent", "total_kes", "Key Events", "KEs"),
    "ker": ("aopo:KeyEventRelationship", "total_kers", "Key Event Relationships", "KERs"),
    "stressor": ("nci:C54571", "total_stressors", "Stressors", "Stressors"),
}


def _presence_answer(query):
    if "?total" in query:
        return [{"graph": {"value": GRAPH + v}, "total": {"value": "10"}}
                for v in ("2024-01-01", "2025-01-01")]
    return [{"graph": {"value": GRAPH + v}, "p": {"value": TITLE_URI}, "count": {"value": n}}
            for v, n in (("2024-01-01", "4"), ("2025-01-01", "7"))]


@pytest.mark.parametrize("entity", PRESENCE)
def test_property_presence_per_entity(monkeypatch, entity):
    cls, total_col, _, noun = PRESENCE[entity]
    queries = _stub(monkeypatch, trends_plots, _presence_answer)

    absolute, percentage = plots.plot_property_presence(entity)

    assert len(queries) == 2 and all(f"?s a {cls}" in q for q in queries)
    df = _plot_data_cache[f"{entity}_property_presence_absolute"]
    assert total_col + "_y" in df.columns
    assert list(df["count"]) == [4, 7]
    fig = _plot_figure_cache[f"{entity}_property_presence_absolute"]
    assert fig.layout.yaxis.title.text == f"Number of {noun}"
    assert "plotly" in absolute and "plotly" in percentage


@pytest.mark.parametrize("entity", PRESENCE)
def test_property_presence_fallback_names_the_entity(monkeypatch, entity):
    _stub(monkeypatch, trends_plots, lambda q: [])
    absolute, percentage = plots.plot_property_presence(entity)
    title = f"Property Presence in {PRESENCE[entity][2]} Over Time"
    assert f"{title} (Count)" in absolute
    assert f"{title} (Percentage)" in percentage


# --- Completeness by OECD status (snapshot) ----------------------------------

STATUS = {
    # entity: (link predicate or None, class, x-axis title, small labels)
    "aop": (None, "aopo:AdverseOutcomePathway", "OECD Status", False),
    "ke": ("aop_ontology#has_key_event>", "aopo:KeyEvent", "OECD Status of Parent AOPs", False),
    "ker": ("aop_ontology#has_key_event_relationship>", "aopo:KeyEventRelationship",
            "OECD Status of Parent AOPs", True),
}
VERSION = "2026-10-01"
ENDORSED = "WPHA/WNT Endorsed"


def _status_answer(query):
    if "SELECT ?graph" in query:
        return [{"graph": {"value": GRAPH + VERSION}}]
    if "?total" in query:
        return [{"status": {"value": ENDORSED}, "total": {"value": "8"}}]
    return [{"status": {"value": ENDORSED}, "p": {"value": TITLE_URI},
             "count_with_property": {"value": "6"}},
            {"p": {"value": TITLE_URI}, "count_with_property": {"value": "1"}}]


@pytest.mark.parametrize("entity", STATUS)
def test_completeness_by_status_per_entity(monkeypatch, entity):
    link, cls, x_title, small_labels = STATUS[entity]
    queries = _stub(monkeypatch, latest_plots, _status_answer)

    plots.plot_latest_completeness_by_status(VERSION, entity=entity)

    _, aggregated, totals = queries
    if link:
        assert link in aggregated and link in totals
        assert f"?entity a {cls}" in aggregated
        assert "COUNT(DISTINCT ?entity)" in aggregated and "COUNT(DISTINCT ?entity)" in totals
    else:
        assert "?entity" not in aggregated and "?entity" not in totals
        assert "COUNT(DISTINCT ?aop)" in aggregated
    key = f"latest_{entity}_completeness_by_status_{VERSION}"
    df = _plot_data_cache[key]
    row = df[df["OECD Status"] == ENDORSED].iloc[0]
    assert (row["Count"], row["Total"]) == (6, 8)
    # A row without a status that has no total is kept, at 0% (as before).
    assert df[df["OECD Status"] == "No Status"]["Completeness"].tolist() == [0]
    fig = _plot_figure_cache[key]
    assert fig.layout.xaxis.title.text == x_title
    assert (fig.data[0].textfont.size == 9) == small_labels


@pytest.mark.parametrize("entity", STATUS)
def test_completeness_by_status_fallback_names_the_entity(monkeypatch, entity):
    _stub(monkeypatch, latest_plots, lambda q: [])
    html = plots.plot_latest_completeness_by_status(VERSION, entity=entity)
    assert f"{entity.upper()} Completeness by OECD Status" in html


# --- Registry binding ---------------------------------------------------------

@pytest.mark.parametrize("spec", [s for s in registry.SPECS if s.args], ids=lambda s: s.id)
def test_bound_producer_carries_the_spec_args(spec):
    bound = plots.bind_producer(spec)
    assert bound.keywords == spec.kwargs
    assert spec.id.startswith(f"latest_{spec.kwargs['entity']}_") or \
        spec.id.startswith(f"{spec.kwargs['entity']}_")
    assert spec.producer in bound.__name__ and repr(spec.kwargs["entity"]) in bound.__name__


def test_unparameterised_producer_is_the_function_itself():
    spec = registry.BY_ID["latest_entity_counts"]
    assert plots.bind_producer(spec) is plots.plot_latest_entity_counts
