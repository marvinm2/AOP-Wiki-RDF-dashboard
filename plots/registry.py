"""The plot registry: one definition per plot, from which every list is generated (#178).

Before this module, adding or renaming a plot meant editing about ten lists by
hand (the startup set, the snapshot functions, the trend outputs, /api/plot's
map, the bulk categories, the recovery map, ...), and they drifted (#171, #172).
Each ``PlotSpec`` below is now the only place a plot is declared; ``app.py`` and
``plots/__init__.py`` derive their tables from ``SPECS``.

This module is pure data and imports nothing from the dashboard, so tools that
must not import the plotting code (``scripts/lint_methodology.py``) can load it
by path. Producers are therefore named by the function's name in ``plots``, not
held as references; ``app.py`` resolves them.

Adding a plot: add a ``snapshot(...)`` or ``trend(...)`` entry here, its
container in ``templates/latest.html`` or ``templates/trends.html``, and its
methodology note. ``tests/test_registry_consistency.py`` checks the three agree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotSpec:
    """One plot served by ``/api/plot/<id>``.

    Attributes:
        id: Plot name in URLs, containers and cache keys.
        kind: ``"snapshot"`` (rendered per version on request) or ``"trend"``
            (one HTML output of a startup task covering every version).
        producer: Name of the function exported by ``plots`` that renders it.
        task: Startup/recovery task that produces it. A snapshot plot is its
            own task.
        output: Position of this plot's HTML in the task's result tuple, or
            None when the task returns the HTML itself.
        note: Key of its entry in ``static/data/methodology_notes.json``.
        page: ``"snapshot"`` or ``"trends"`` for the page holding its
            container, or None for a plot served by the API only.
        precompute: Rendered at startup. Always true for trend plots.
        params: Query-string tuning arguments ``/api/plot`` passes through.
        bulk: Bulk-download categories (``/download/bulk?category=``).
        args: Fixed keyword arguments the producer is always called with, as
            ``(name, value)`` pairs; one function serves several plots this
            way (e.g. ``entity="ke"``).
    """

    id: str
    kind: str
    producer: str
    task: str
    output: int | None
    note: str | None
    page: str | None
    precompute: bool
    params: tuple[str, ...] = ()
    bulk: tuple[str, ...] = ()
    args: tuple[tuple[str, str], ...] = ()

    @property
    def kwargs(self) -> dict[str, str]:
        return dict(self.args)


def snapshot(id, producer, *, note="", page="snapshot", precompute=False,
             params=(), bulk=(), args=None) -> PlotSpec:
    """A version-selectable snapshot plot. ``note`` defaults to the plot id."""
    return PlotSpec(id=id, kind="snapshot", producer=producer, task=id,
                    output=None, note=id if note == "" else note, page=page,
                    precompute=precompute, params=tuple(params), bulk=tuple(bulk),
                    args=tuple((args or {}).items()))


def trend(task, producer, *, note, outputs, args=None) -> tuple[PlotSpec, ...]:
    """The plots one trend task produces: ``outputs`` maps id -> (index, bulk)."""
    return tuple(
        PlotSpec(id=id, kind="trend", producer=producer, task=task, output=index,
                 note=note, page="trends", precompute=True, bulk=tuple(bulk),
                 args=tuple((args or {}).items()))
        for id, (index, bulk) in outputs.items()
    )


SPECS: tuple[PlotSpec, ...] = (
    # --- Trend tasks (computed once at startup over every version) -----------
    *trend('main_graph', 'plot_main_graph', note='aop_entity_counts', outputs={
        'aop_entity_counts_absolute': (0, ('trends-all', 'trends-main', 'all')),
        'aop_entity_counts_delta': (1, ('trends-main',)),
    }),
    *trend('entity_birth_death', 'plot_entity_birth_death', note='entity_birth_death', outputs={
        'entity_birth_death': (0, ('trends-all', 'trends-main', 'all')),
    }),
    *trend('entity_cumulative_removed', 'plot_entity_cumulative_removed', note='entity_cumulative_removed', outputs={
        'entity_cumulative_removed': (0, ('trends-all', 'trends-main', 'all')),
    }),
    *trend('oecd_status_distribution', 'plot_oecd_status_distribution', note='oecd_status_distribution', outputs={
        'oecd_status_distribution_absolute': (0, ()),
        'oecd_status_distribution_percentage': (1, ()),
    }),
    *trend('stressor_coverage_growth', 'plot_stressor_coverage_growth', note='stressor_coverage_growth', outputs={
        'stressor_coverage_growth_absolute': (0, ()),
        'stressor_coverage_growth_delta': (1, ()),
    }),
    *trend('aops_per_stressor_distribution', 'plot_aops_per_stressor_distribution', note='aops_per_stressor_distribution', outputs={
        'aops_per_stressor_distribution_absolute': (0, ()),
        'aops_per_stressor_distribution_percentage': (1, ()),
    }),
    *trend('ke_mmo_coverage', 'plot_ke_mmo_coverage_trends', note='ke_mmo_coverage', outputs={
        'ke_mmo_coverage_absolute': (0, ()),
        'ke_mmo_coverage_percentage': (1, ()),
    }),
    *trend('ke_migration_map', 'plot_ke_migration_map', note='ke_migration_map', outputs={
        'ke_migration_map': (0, ()),
    }),
    *trend('avg_per_aop', 'plot_avg_per_aop', note='average_components_per_aop', outputs={
        'average_components_per_aop_absolute': (0, ('trends-all', 'trends-network', 'all')),
        'average_components_per_aop_delta': (1, ('trends-network',)),
    }),
    *trend('network_density', 'plot_network_density', note='aop_network_density', outputs={
        'aop_network_density': (None, ('trends-all', 'trends-network', 'all')),
    }),
    *trend('ke_components', 'plot_ke_components', note='ke_component_annotations', outputs={
        'ke_component_annotations_absolute': (0, ('trends-all', 'trends-components', 'all')),
        'ke_component_annotations_delta': (1, ('trends-components',)),
    }),
    *trend('unique_ke_components', 'plot_unique_ke_components', note='unique_ke_components', outputs={
        'unique_ke_components_absolute': (0, ('trends-all', 'trends-components', 'all')),
        'unique_ke_components_delta': (1, ('trends-components',)),
    }),
    *trend('ke_components_percentage', 'plot_ke_components_percentage', note='ke_components_percentage', outputs={
        'ke_components_percentage_absolute': (0, ('trends-all', 'trends-components', 'all')),
        'ke_components_percentage_delta': (1, ('trends-components',)),
    }),
    *trend('bio_processes', 'plot_bio_processes', note='bio_annotations', outputs={
        'biological_process_annotations_absolute': (0, ('trends-all', 'trends-ontology', 'all')),
        'biological_process_annotations_delta': (1, ('trends-ontology',)),
    }),
    *trend('bio_objects', 'plot_bio_objects', note='bio_annotations', outputs={
        'biological_object_annotations_absolute': (0, ('trends-all', 'trends-ontology', 'all')),
        'biological_object_annotations_delta': (1, ('trends-ontology',)),
    }),
    *trend('author_counts', 'plot_author_counts', note='aop_authors', outputs={
        'aop_authors_absolute': (0, ('trends-all', 'trends-authors', 'all')),
        'aop_authors_delta': (1, ('trends-authors',)),
    }),
    *trend('aop_lifetime', 'plot_aop_lifetime', note='aop_lifetime', outputs={
        'aops_created_over_time': (0, ('trends-all', 'trends-authors', 'all')),
        'aop_creation_vs_modification_timeline': (1, ('trends-all', 'trends-authors', 'all')),
    }),
    *trend('aop_property_presence', 'plot_property_presence', note='aop_property_presence', args={'entity': 'aop'}, outputs={
        'aop_property_presence_absolute': (0, ('trends-all', 'trends-properties', 'all')),
        'aop_property_presence_percentage': (1, ('trends-properties',)),
    }),
    *trend('ke_property_presence', 'plot_property_presence', note='ke_property_presence', args={'entity': 'ke'}, outputs={
        'ke_property_presence_absolute': (0, ()),
        'ke_property_presence_percentage': (1, ()),
    }),
    *trend('ker_property_presence', 'plot_property_presence', note='ker_property_presence', args={'entity': 'ker'}, outputs={
        'ker_property_presence_absolute': (0, ()),
        'ker_property_presence_percentage': (1, ()),
    }),
    *trend('stressor_property_presence', 'plot_property_presence', note='stressor_property_presence', args={'entity': 'stressor'}, outputs={
        'stressor_property_presence_absolute': (0, ()),
        'stressor_property_presence_percentage': (1, ()),
    }),
    *trend('entity_completeness_trends', 'plot_entity_completeness_trends', note='entity_completeness_trends', outputs={
        'entity_completeness_trends': (None, ()),
    }),
    *trend('aop_completeness_boxplot', 'plot_aop_completeness_boxplot', note='aop_completeness_boxplot', outputs={
        'aop_completeness_boxplot': (0, ()),
        'aop_completeness_boxplot_all': (1, ()),
    }),
    *trend('oecd_completeness_trend', 'plot_oecd_completeness_trend', note='oecd_completeness_trend', outputs={
        'oecd_completeness_trend': (None, ()),
    }),
    *trend('kes_by_kec_count', 'plot_kes_by_kec_count', note='kes_by_kec_count', outputs={
        'kes_by_kec_count_absolute': (0, ('trends-all', 'all')),
        'kes_by_kec_count_delta': (1, ()),
    }),
    *trend('ontology_term_growth', 'plot_ontology_term_growth', note='ontology_term_growth', outputs={
        'ontology_term_growth_absolute': (0, ()),
        'ontology_term_growth_delta': (1, ()),
    }),
    *trend('organ_coverage', 'plot_organ_coverage_trends', note='organ_coverage', outputs={
        'organ_coverage_absolute': (0, ()),
        'organ_coverage_percentage': (1, ()),
    }),

    # --- Snapshot plots (rendered per version on request) ---------------------
    snapshot('latest_entity_counts', 'plot_latest_entity_counts', precompute=True, bulk=('latest-all', 'database-state', 'all')),
    snapshot('latest_ke_components', 'plot_latest_ke_components', precompute=True, bulk=('latest-all', 'ke-analysis', 'all')),
    snapshot('latest_aop_connectivity', 'plot_latest_aop_connectivity', precompute=True, bulk=('latest-all', 'network-analysis', 'all')),
    snapshot('latest_avg_per_aop', 'plot_latest_avg_per_aop', precompute=True, bulk=('latest-all', 'network-analysis', 'all')),
    snapshot('latest_process_usage', 'plot_latest_process_usage', precompute=True, bulk=('latest-all', 'data-quality', 'all')),
    snapshot('latest_object_usage', 'plot_latest_object_usage', precompute=True, bulk=('latest-all', 'data-quality', 'all')),
    snapshot('latest_aop_completeness', 'plot_latest_aop_completeness', precompute=True, bulk=('latest-all', 'data-quality', 'all')),
    snapshot('latest_aop_completeness_by_status', 'plot_latest_completeness_by_status', precompute=True, args={'entity': 'aop'}),
    snapshot('latest_ke_completeness_by_status', 'plot_latest_completeness_by_status', precompute=True, args={'entity': 'ke'}),
    snapshot('latest_ker_completeness_by_status', 'plot_latest_completeness_by_status', precompute=True, args={'entity': 'ker'}),
    snapshot('latest_ontology_usage', 'plot_latest_ontology_usage', precompute=True),
    snapshot('latest_ke_annotation_depth', 'plot_latest_ke_annotation_depth', precompute=True, bulk=('latest-all', 'ke-analysis', 'all')),
    snapshot('latest_ke_by_bio_level', 'plot_latest_ke_by_bio_level', bulk=('latest-all',)),
    snapshot('latest_taxonomic_groups', 'plot_latest_taxonomic_groups', bulk=('latest-all',)),
    snapshot('latest_entity_by_oecd_status', 'plot_latest_entity_by_oecd_status', bulk=('latest-all',)),
    snapshot('latest_ke_reuse', 'plot_latest_ke_reuse', bulk=('latest-all',)),
    snapshot('latest_ke_reuse_distribution', 'plot_latest_ke_reuse_distribution', bulk=('latest-all',)),
    snapshot('latest_mie_ao_path_length', 'plot_latest_mie_ao_path_length', bulk=('latest-all',)),
    snapshot('latest_stressor_mie_coverage', 'plot_latest_stressor_mie_coverage', bulk=('latest-all',)),
    snapshot('latest_ker_directionality', 'plot_latest_ker_directionality', bulk=('latest-all',)),
    snapshot('latest_top_ontology_terms', 'plot_latest_top_ontology_terms', bulk=('latest-all',)),
    snapshot('latest_completeness_correlation', 'plot_latest_completeness_correlation', bulk=('latest-all',)),
    snapshot('latest_ontology_diversity', 'plot_latest_ontology_diversity'),
    snapshot('latest_organ_coverage_unified', 'plot_latest_organ_coverage_unified', params=('scope', 'view')),
    snapshot('latest_organ_coverage_pie', 'plot_latest_organ_coverage_pie', note='latest_organ_coverage_unified', params=('scope',)),
    snapshot('latest_multi_organ_aops', 'plot_latest_multi_organ_aops'),
    snapshot('latest_life_stage', 'plot_latest_life_stage'),
    snapshot('latest_ke_mmo_coverage', 'plot_latest_ke_mmo_coverage', precompute=True),
    snapshot('latest_aop_aop_overlap', 'plot_latest_aop_aop_overlap', precompute=True, params=('min_jaccard', 'min_shared_kes')),
    snapshot('latest_ontology_coverage_holes', 'plot_latest_ontology_coverage_holes', params=('branch',), bulk=('latest-all', 'data-quality')),
    snapshot('latest_domain_coverage', 'plot_latest_domain_coverage', params=('domain',), bulk=('latest-all',)),
    snapshot('latest_domain_completeness', 'plot_latest_domain_completeness', params=('domain',), bulk=('latest-all',)),
    snapshot('latest_domain_status', 'plot_latest_domain_status', params=('domain',), bulk=('latest-all',)),
)

BY_ID: dict[str, PlotSpec] = {s.id: s for s in SPECS}

# The order bulk categories are listed in; their members follow SPECS order.
BULK_CATEGORIES: tuple[str, ...] = (
    "latest-all", "database-state", "network-analysis", "ke-analysis",
    "data-quality", "trends-all", "trends-main", "trends-network",
    "trends-authors", "trends-components", "trends-ontology",
    "trends-properties", "all",
)

# The delta (quarter-over-quarter) trend views, drawn in the browser from their
# absolute plot (#181): y becomes y[i] - y[i-1], 0 for the first snapshot. The
# server still renders the *_delta figures for downloads; these entries carry
# the labels so the browser draws the same thing. Checked against the server
# figures with scripts/check_delta_views.js (run against the live API).
DELTA_VIEWS: dict[str, dict] = {
    "aop_entity_counts_delta": {"of": "aop_entity_counts_absolute", "y_title": "Count"},
    "stressor_coverage_growth_delta": {"of": "stressor_coverage_growth_absolute", "y_title": "Quarter-over-quarter Δ", "bar": True},
    "average_components_per_aop_delta": {"of": "average_components_per_aop_absolute", "y_title": "Δ Average", "name_replace": ["Average ", "Δ "]},
    "ke_component_annotations_delta": {"of": "ke_component_annotations_absolute", "y_title": "Change"},
    "unique_ke_components_delta": {"of": "unique_ke_components_absolute", "y_title": "Change"},
    "ke_components_percentage_delta": {"of": "ke_components_percentage_absolute", "y_title": "Percentage Change (%)", "hover_label": "Percentage Change"},
    "biological_process_annotations_delta": {"of": "biological_process_annotations_absolute", "y_title": "Change in distinct terms"},
    "biological_object_annotations_delta": {"of": "biological_object_annotations_absolute", "y_title": "Change in distinct terms"},
    "aop_authors_delta": {"of": "aop_authors_absolute", "y_title": "author_count_Δ"},
    "kes_by_kec_count_delta": {"of": "kes_by_kec_count_absolute", "y_title": "Change in KEs", "bar": True},
    "ontology_term_growth_delta": {"of": "ontology_term_growth_absolute", "y_title": "New Terms Added", "bar": True},
}

# Legacy download URLs whose path names a different plot (/download/<alias>).
DOWNLOAD_ALIASES: dict[str, str] = {
    "latest_network_density": "latest_aop_connectivity",
}


def snapshot_specs() -> list[PlotSpec]:
    return [s for s in SPECS if s.kind == "snapshot"]


def trend_specs() -> list[PlotSpec]:
    return [s for s in SPECS if s.kind == "trend"]


def startup_tasks() -> dict[str, PlotSpec]:
    """Startup task name -> the (first) spec it renders, trend tasks first."""
    tasks: dict[str, PlotSpec] = {}
    for s in SPECS:
        if s.precompute:
            tasks.setdefault(s.task, s)
    return tasks


def trend_outputs() -> dict[str, tuple[str, int | None]]:
    """Trend plot id -> (task, output index), as plots/recovery.py expects."""
    return {s.id: (s.task, s.output) for s in trend_specs()}


def bulk_categories() -> dict[str, list[str]]:
    return {c: [s.id for s in SPECS if c in s.bulk] for c in BULK_CATEGORIES}


def tuning_params() -> tuple[str, ...]:
    """Every tuning argument any plot accepts, in first-declared order."""
    return tuple(dict.fromkeys(p for s in SPECS for p in s.params))


def manifest() -> list[dict]:
    """JSON-ready description of every plot, for /api/v2/manifest."""
    return [
        {"id": s.id, "kind": s.kind, "task": s.task, "note": s.note,
         "page": s.page, "versioned": s.kind == "snapshot",
         "params": list(s.params), "bulk": list(s.bulk), "args": s.kwargs}
        for s in SPECS
    ]
