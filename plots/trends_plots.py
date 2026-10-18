"""AOP-Wiki RDF Dashboard - Historical Trends Plot Functions.

This module contains all historical trend and time-series visualization functions
for the AOP-Wiki RDF Dashboard. These functions analyze data evolution over time
by querying multiple RDF graph versions and creating comparative visualizations.

Key Features:
    - Time-series analysis of entity counts and relationships
    - Delta change calculations between versions
    - Historical pattern identification and trend analysis
    - Parallel SPARQL query execution for optimal performance
    - Comprehensive data caching for CSV export functionality

Historical Plot Functions:
    Core Evolution Analysis:
        - plot_main_graph(): Entity count evolution (AOPs, KEs, KERs, Stressors)
        - plot_avg_per_aop(): Average components per AOP trends
        - plot_network_density(): Graph connectivity evolution
        - plot_author_counts(): Author contribution patterns

    Component Analysis:
        - plot_ke_components(): Component annotation trends
        - plot_ke_components_percentage(): Percentage-based component trends
        - plot_unique_ke_components(): Unique component evolution
        - plot_kes_by_kec_count(): KE component count distributions

    Ontology Usage:
        - plot_bio_processes(): Biological process annotation trends
        - plot_bio_objects(): Biological object annotation trends

    Property Analysis:
        - plot_property_presence(entity): Property presence evolution with marker shapes

    Temporal Analysis:
        - plot_aop_lifetime(): AOP creation and modification patterns

Visualization Characteristics:
    - Consistent VHP4Safety branding and color palette
    - Interactive hover modes with unified tooltips
    - Responsive design for desktop and mobile
    - Professional margins and typography
    - Delta change analysis for trend identification

Data Processing:
    - Robust SPARQL query execution with retry logic
    - Version-based data sorting and chronological ordering
    - Data quality validation and error handling
    - Global caching for CSV export functionality

Author:
    Marvin Martens
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import re
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed
from .shared import (
    SparqlUnavailable,
    BRAND_COLORS, config, _plot_data_cache, _plot_figure_cache,
    run_sparql_query, run_sparql_query_with_retry, extract_counts,
    safe_read_csv, create_fallback_plot, get_properties_for_entity,
    get_all_versions, render_plot_html, apply_snapshot_xaxis,
    build_property_presence_figure, mask_property_prefix_zeros,
    humanize_predicate_uri,
    ENTITY_TYPE_CLASSES, diff_entities_between_versions,
    fetch_entity_uris_by_version,
    ontology_bucket_sparql,
)
from .organ_systems import (
    ORGAN_SYSTEM_BUCKETS,
    NO_ANNOTATION_BUCKET,
    best_signal,
    classify_anatomy,
    classify_process,
    classify_text,
)

logger = logging.getLogger(__name__)

# Distinct marker glyphs to pair with colour when a trend has more series than
# the palette has distinct hues, so repeated colours stay tellable apart
# (#plot-review). Mirrors plots.shared._MARKER_SYMBOLS.
_TREND_MARKERS = [
    'circle', 'square', 'diamond', 'triangle-up', 'cross', 'x', 'star',
    'pentagon', 'hexagon', 'triangle-down', 'triangle-left', 'triangle-right',
    'octagon', 'star-diamond',
]


def _apply_marker_symbols(fig, size: int = 6, line_width: float = 1.6):
    """Give each trace a distinct marker symbol (cycling _TREND_MARKERS) plus a
    thinner line, so a high-cardinality line chart stays separable even where the
    palette colours repeat (#plot-review)."""
    for i, tr in enumerate(fig.data):
        tr.update(
            marker=dict(symbol=_TREND_MARKERS[i % len(_TREND_MARKERS)], size=size),
            line=dict(width=line_width),
        )
    return fig


def plot_main_graph() -> tuple[str, str, pd.DataFrame]:
    """Generate the main AOP entity evolution visualization with absolute and delta views.

    Creates comprehensive time-series visualizations showing the evolution of core
    AOP-Wiki entities (AOPs, Key Events, Key Event Relationships, and Stressors)
    across different RDF graph versions. Provides both absolute counts and
    delta changes between versions with global data caching for CSV exports.

    The function executes multiple SPARQL queries to gather entity counts across
    all available graph versions, processes the data into time-series format,
    and generates interactive Plotly visualizations with VHP4Safety branding.

    SPARQL Queries:
        - AOPs: Counts AdverseOutcomePathway instances per graph
        - KEs: Counts KeyEvent instances per graph
        - KERs: Counts KeyEventRelationship instances per graph
        - Stressors: Counts Stressor (nci:C54571) instances per graph

    Returns:
        tuple[str, str, pd.DataFrame]: A 3-tuple containing:
            - str: HTML for absolute counts line chart
            - str: HTML for delta changes line chart
            - pd.DataFrame: Processed data with all entity counts by version

    Data Processing:
        1. Executes 4 parallel SPARQL queries for entity types
        2. Extracts and validates count data using extract_counts()
        3. Merges results into unified DataFrame with outer join
        4. Sorts by datetime-converted version strings
        5. Computes delta changes between consecutive versions
        6. Caches data globally for CSV export functionality

    Visualizations:
        Absolute Plot:
            - Multi-line chart showing entity counts over time
            - Uses VHP4Safety color palette for consistency
            - Interactive hover with unified mode
            - Responsive design with proper margins

        Delta Plot:
            - Shows changes between consecutive versions
            - Highlights growth/decline patterns
            - Same styling and interactivity as absolute plot
            - Useful for identifying update patterns

    Global Caching:
        Stores processed data in _plot_data_cache for CSV downloads:
        - 'main_graph_absolute': Melted absolute data
        - 'main_graph_delta': Melted delta data

    Example Data Structure:
        >>> abs_html, delta_html, df = plot_main_graph()
        >>> print(df.head())
           version  AOPs  KEs  KERs  Stressors
        0  2023-01-15   120  800   650        45
        1  2023-06-01   135  850   720        48
        2  2024-01-15   142  890   780        52

    Error Handling:
        - Empty query results: Logs warnings, continues with available data
        - Invalid data formats: Uses extract_counts() error handling
        - Merge failures: Returns empty DataFrame and fallback plots
        - Visualization errors: Handled by safe_plot_execution wrapper

    Performance:
        - Parallel query execution for optimal speed
        - Efficient DataFrame operations with pandas
        - Cached results prevent recomputation
        - Optimized for time-series data processing

    Usage:
        Called during application startup for dashboard generation:
        >>> # In app.py startup
        >>> abs_plot, delta_plot, data = plot_main_graph()
        >>> # Data automatically cached for CSV downloads

    Note:
        This is the primary historical analysis function and forms the
        foundation for understanding AOP-Wiki data evolution patterns.
        The function assumes RDF graphs follow the naming pattern:
        "http://aopwiki.org/graph/{version}"
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        sparql_queries = {
            "AOPs": """
                SELECT ?graph (COUNT(?aop) AS ?count)
                WHERE {
                    GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
                    FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                }
                GROUP BY ?graph
            """,
            "KEs": """
                SELECT ?graph (COUNT(?ke) AS ?count)
                WHERE {
                    GRAPH ?graph {?ke a aopo:KeyEvent .
                    }
                    FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                }
                GROUP BY ?graph
            """,
            "KERs": """
                SELECT ?graph (COUNT(?ker) AS ?count)
                WHERE {
                    GRAPH ?graph {
                        ?ker a aopo:KeyEventRelationship .
                    }
                    FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                }
                GROUP BY ?graph
            """,
            "Stressors": """
                SELECT ?graph (COUNT(?s) AS ?count)
                WHERE {
                    GRAPH ?graph {
                        ?s a nci:C54571 .
                    }
                    FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
                }
                GROUP BY ?graph
            """
        }

        # --- Query and merge results ---
        df_list = []
        for label, query in sparql_queries.items():
            results = run_sparql_query(query)
            df = extract_counts(results)
            df.rename(columns={"count": label}, inplace=True)
            df_list.append(df)

        df_all = reduce(lambda left, right: pd.merge(left, right, on="version", how="outer"), df_list)

        if df_all.empty:
            logger.warning("Main graph query returned no results")
            return (
                create_fallback_plot("Entity Evolution Over Time", "No data available"),
                create_fallback_plot("Entity Change Between Versions", "No data available"),
                pd.DataFrame()
            )

        # Convert to datetime for correct sorting
        df_all["version_dt"] = pd.to_datetime(df_all["version"], errors="coerce")
        df_all = df_all.sort_values("version_dt").drop(columns="version_dt").reset_index(drop=True)

        # --- Absolute plot ---
        df_abs_melted = df_all.melt(id_vars="version", var_name="Entity", value_name="Count")
        # Clean data: fill NaN with 0 and ensure numeric type
        df_abs_melted["Count"] = pd.to_numeric(df_abs_melted["Count"], errors="coerce").fillna(0).astype(int)
        fig_abs = px.line(
            df_abs_melted, x="version", y="Count", color="Entity", markers=True,

            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_abs.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        apply_snapshot_xaxis(fig_abs)

        # --- Delta plot ---
        df_delta = df_all.copy()
        for col in sparql_queries.keys():
            df_delta[f"{col}_Δ"] = df_all[col].diff().fillna(0).astype(int)

        df_delta_melted = df_delta.melt(
            id_vars="version",
            value_vars=[f"{k}_Δ" for k in sparql_queries.keys()],
            var_name="Entity",
            value_name="Count"
        )
        df_delta_melted["Entity"] = df_delta_melted["Entity"].str.replace("_Δ", "")

        fig_delta = px.line(
            df_delta_melted, x="version", y="Count", color="Entity", markers=True,

            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_delta.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        apply_snapshot_xaxis(fig_delta)

        # Store absolute and delta data in cache for CSV download.
        # Legacy `main_graph_*` keys kept for backwards-compat with old
        # /download/main_graph_absolute and /download/main_graph_delta routes;
        # `aop_entity_counts_*` aliases match the template's current download
        # URLs (#34).
        _plot_data_cache['main_graph_absolute'] = df_abs_melted
        _plot_data_cache['main_graph_delta'] = df_delta_melted
        _plot_data_cache['aop_entity_counts_absolute'] = df_abs_melted
        _plot_data_cache['aop_entity_counts_delta'] = df_delta_melted

        # Cache figures for image export. Mirror the data-cache aliases above so
        # PNG/SVG export works on both the legacy main_graph_* routes and the
        # current aop_entity_counts_* download URLs.
        _plot_figure_cache['main_graph_absolute'] = fig_abs
        _plot_figure_cache['main_graph_delta'] = fig_delta
        _plot_figure_cache['aop_entity_counts_absolute'] = fig_abs
        _plot_figure_cache['aop_entity_counts_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta),
            df_all
        )

    except Exception as e:
        logger.error(f"Failed to generate main graph plots: {str(e)}")
        return (
            create_fallback_plot("Entity Evolution Over Time", str(e)),
            create_fallback_plot("Entity Change Between Versions", str(e)),
            pd.DataFrame()
        )


def plot_entity_birth_death() -> tuple[str, pd.DataFrame]:
    """Gross additions vs removals per snapshot transition (birth/death curve).

    Net entity counts hide churn: a delta of zero can mask "5 new KEs added,
    5 existing KEs removed/merged/deprecated." This plot exposes both flows
    by diffing the entity URI set of each pair of adjacent named graphs.

    The result is a faceted diverging bar chart — additions above the zero
    line, removals below — one panel per entity type. The cached DataFrame
    also stores the actual added/removed URIs per quarter, so curators can
    inspect whether removals were intentional.

    See `diff_entities_between_versions` (plots/shared.py) for the
    underlying set-difference helper; that helper is reused by issues #66,
    #70, and #72.

    Returns:
        tuple[str, pd.DataFrame]:
            - HTML for the faceted diverging-bar plot.
            - DataFrame with one row per (version, entity_type) transition,
              columns: version, prev_version, entity_type, added_count,
              removed_count, stable_count, added_uris, removed_uris.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = [v['version'] for v in get_all_versions()]
        versions = sorted(set(versions))
        if len(versions) < 2:
            logger.warning("Not enough versions for birth/death curve")
            return (
                create_fallback_plot(
                    "Entity Birth/Death Curve",
                    "Need at least two snapshot versions to compute a diff."
                ),
                pd.DataFrame(),
            )

        # Run the four entity types in parallel — each spawns its own per-version
        # fetch pool internally. Cap workers at 4 since there are 4 types.
        per_type_dfs: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(diff_entities_between_versions, entity_type, versions): entity_type
                for entity_type in ENTITY_TYPE_CLASSES
            }
            for fut in as_completed(futures):
                entity_type = futures[fut]
                try:
                    per_type_dfs.append(fut.result())
                except Exception as e:
                    logger.error(f"Birth/death diff failed for {entity_type}: {e}")
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        if not per_type_dfs:
            return (
                create_fallback_plot("Entity Birth/Death Curve", "No diff data available"),
                pd.DataFrame(),
            )

        df = pd.concat(per_type_dfs, ignore_index=True)
        # Stable chronological + entity ordering for downstream UX.
        entity_order = list(ENTITY_TYPE_CLASSES.keys())
        df['entity_type'] = pd.Categorical(df['entity_type'], categories=entity_order, ordered=True)
        df = df.sort_values(['entity_type', 'version']).reset_index(drop=True)

        # Single-panel diverging stacked bars: additions positive, removals negative,
        # one colour per entity type. Avoids the multi-subplot rendering quirks of
        # facet_col_wrap which caused the chart to render blank on /trends.
        versions = sorted(df['version'].unique())
        # Four maximally-separated, colourblind-safe hues (blue/magenta/green/
        # orange) instead of two blue-family colours, so the entity types stay
        # distinct at each stack (#plot-review).
        entity_colors = dict(zip(entity_order, BRAND_COLORS['palette'][:4]))
        fig = go.Figure()
        for et in entity_order:
            sub = df[df['entity_type'] == et].set_index('version').reindex(versions)
            color = entity_colors[et]
            fig.add_bar(
                x=versions, y=sub['added_count'].fillna(0).astype(int).tolist(),
                name=f'{et} added', marker_color=color, legendgroup=et,
                hovertemplate=f'<b>{et}</b><br>%{{x}}<br>added: %{{y}}<extra></extra>',
            )
            fig.add_bar(
                x=versions, y=(-sub['removed_count'].fillna(0).astype(int)).tolist(),
                name=f'{et} removed', marker_color=color, legendgroup=et,
                opacity=0.55, showlegend=False,
                hovertemplate=f'<b>{et}</b><br>%{{x}}<br>removed: %{{customdata}}<extra></extra>',
                customdata=sub['removed_count'].fillna(0).astype(int).tolist(),
            )
        fig.update_xaxes(tickmode='array', tickvals=versions, ticktext=versions, tickangle=-45)
        fig.update_yaxes(title='Entities (added above / removed below)',
                         zeroline=True, zerolinecolor='#888', zerolinewidth=1)
        fig.update_layout(
            barmode='relative',
            margin=dict(l=60, r=30, t=60, b=80),
            height=480,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        )
        # The below-zero bars reuse each entity's colour at reduced opacity but
        # carry no legend entry; spell out the encoding so the faded bars aren't
        # a mystery family of colours (#plot-review).
        fig.add_annotation(
            text="Solid bars above 0 = entities added · faded bars below 0 = entities removed",
            xref='paper', yref='paper', x=0.0, y=1.10, showarrow=False,
            font=dict(size=11, color='#666'), align='left',
        )

        _plot_data_cache['entity_birth_death'] = df
        _plot_figure_cache['entity_birth_death'] = fig

        return render_plot_html(fig), df

    except Exception as e:
        logger.error(f"Failed to generate entity birth/death plot: {str(e)}")
        return (
            create_fallback_plot("Entity Birth/Death Curve", str(e)),
            pd.DataFrame(),
        )


def plot_entity_cumulative_removed() -> tuple[str, pd.DataFrame]:
    """Running graveyard view — entities that once existed and are now absent.

    Companion to `plot_entity_birth_death` (#71): #71 shows the *flow*
    (per-quarter births and deaths); this plot shows the *stock* (total
    ever-removed up to each snapshot).

    For each snapshot N and entity type, the count is
    ``len(union(URIs in graphs ≤ N) − URIs in N)`` — entities seen at any
    earlier point but absent right now. An entity that disappears in v_N
    and reappears in v_{N+2} is counted in N and N+1 but not in N+2; the
    line is monotonic modulo re-additions.

    The downloadable CSV preserves the actual absent URIs per snapshot so
    curators can audit a concrete "deprecated entities" list.

    Returns:
        tuple[str, pd.DataFrame]:
            - HTML for a 4-trace line chart (one trace per entity type).
            - DataFrame with columns: version, entity_type,
              cumulative_removed_count, removed_uris.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = sorted({v['version'] for v in get_all_versions()})
        if len(versions) < 2:
            logger.warning("Not enough versions for cumulative-removed curve")
            return (
                create_fallback_plot(
                    "Cumulative Removed Entities",
                    "Need at least two snapshot versions to compute the running set."
                ),
                pd.DataFrame(),
            )

        # Each entity type runs its own per-version fetch pool (5 workers each).
        # Cap the outer pool at 4 since there are 4 entity types.
        per_type_uris: dict[str, dict[str, set]] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(fetch_entity_uris_by_version, entity_type, versions): entity_type
                for entity_type in ENTITY_TYPE_CLASSES
            }
            for fut in as_completed(futures):
                entity_type = futures[fut]
                try:
                    per_type_uris[entity_type] = fut.result()
                except Exception as e:
                    logger.error(f"Cumulative-removed fetch failed for {entity_type}: {e}")
                    per_type_uris[entity_type] = {v: set() for v in versions}
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        rows = []
        for entity_type in ENTITY_TYPE_CLASSES:
            uris_by_version = per_type_uris.get(entity_type, {})
            ever_seen: set[str] = set()
            for v in versions:
                current = uris_by_version.get(v, set())
                ever_seen |= current
                # Entities seen at any earlier point AND absent in the current snapshot.
                absent_now = ever_seen - current
                rows.append({
                    'version': v,
                    'entity_type': entity_type,
                    'cumulative_removed_count': len(absent_now),
                    'removed_uris': ';'.join(sorted(absent_now)),
                })

        df = pd.DataFrame(rows)
        entity_order = list(ENTITY_TYPE_CLASSES.keys())
        df['entity_type'] = pd.Categorical(df['entity_type'], categories=entity_order, ordered=True)
        df = df.sort_values(['entity_type', 'version']).reset_index(drop=True)

        fig = px.line(
            df,
            x='version',
            y='cumulative_removed_count',
            color='entity_type',
            markers=True,
            category_orders={'entity_type': entity_order},
            color_discrete_sequence=BRAND_COLORS['palette'],
            labels={
                'version': 'Snapshot',
                'cumulative_removed_count': 'Entities ever-removed and currently absent',
                'entity_type': 'Entity type',
            },
        )
        # Let Plotly auto-thin the date ticks (one per year) instead of printing
        # all ~33 rotated snapshot labels, which collapse into a smear (#plot-review).
        apply_snapshot_xaxis(fig)
        fig.update_layout(margin=dict(l=60, r=30, t=40, b=80), height=440)

        _plot_data_cache['entity_cumulative_removed'] = df
        _plot_figure_cache['entity_cumulative_removed'] = fig

        return render_plot_html(fig), df

    except Exception as e:
        logger.error(f"Failed to generate cumulative-removed plot: {str(e)}")
        return (
            create_fallback_plot("Cumulative Removed Entities", str(e)),
            pd.DataFrame(),
        )


# --- Entity lifeline (#183) -------------------------------------------------

# Page on aopwiki.org for each entity namespace (identifiers.org IRIs).
_LIFELINE_SHORT = {"AOPs": "AOP", "KEs": "KE", "KERs": "KER", "Stressors": "Stressor"}
LIFELINE_STATUSES = ("always present", "added", "returned", "removed")


# Presence strings use "x" (present) and "-" (absent), oldest version first:
# text a spreadsheet keeps as it is, unlike a 0/1 string that loses its
# leading zeros.
PRESENT, ABSENT = "x", "-"


def lifeline_status(presence: str) -> tuple[str, int]:
    """(status, returns) for a presence string over the versions, oldest first.

    ``returns`` counts the gaps between the first and last appearance. The
    status is "removed" when the entity is missing from the latest version,
    else "returned" when it had a gap, else "added" when it was missing from
    the first version, else "always present".
    """
    first, last = presence.index(PRESENT), presence.rindex(PRESENT)
    returns = len([run for run in presence[first:last + 1].split(PRESENT) if run])
    if last < len(presence) - 1:
        return "removed", returns
    if returns:
        return "returned", returns
    return ("added" if first > 0 else "always present"), returns


def lifeline_rows(entity_type: str, uris_by_version: dict, versions: list,
                  labels: dict) -> list[dict]:
    """One row per entity ever seen: when it appears, vanishes and returns.

    Rows are ordered by first appearance, then longest-lived first, then
    number, so the lifeline reads as a staircase of arrivals.
    """
    ever = set().union(*uris_by_version.values()) if uris_by_version else set()
    short = _LIFELINE_SHORT[entity_type]
    rows = []
    for uri in ever:
        presence = "".join(PRESENT if uri in uris_by_version.get(v, ()) else ABSENT for v in versions)
        status, returns = lifeline_status(presence)
        number = uri.rsplit("/", 1)[-1]
        first, last = presence.index(PRESENT), presence.rindex(PRESENT)
        rows.append({
            "entity_type": entity_type,
            "id": f"{short} {number}",
            "label": labels.get(uri, ""),
            "first_seen": versions[first],
            "last_seen": versions[last],
            "snapshots_present": presence.count(PRESENT),
            "returns": returns,
            "status": status,
            "presence": presence,
            "uri": uri,
            "_order": (first, -last, int(number) if number.isdigit() else 0),
        })
    rows.sort(key=lambda r: r["_order"])
    for r in rows:
        del r["_order"]
    return rows


def _lifeline_labels(entity_type: str) -> dict:
    """IRI -> label: the newest dc:title (KERs: their upstream and downstream KE)."""
    cls = ENTITY_TYPE_CLASSES[entity_type]
    if entity_type == "KERs":
        query = f"""
        SELECT ?e ?up ?down (MAX(STR(?graph)) AS ?last)
        WHERE {{
            GRAPH ?graph {{
                ?e a {cls} ;
                   aopo:has_upstream_key_event ?up ;
                   aopo:has_downstream_key_event ?down .
            }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?e ?up ?down
        """
        value = lambda r: (f"KE {r['up']['value'].rsplit('/', 1)[-1]} → "
                           f"KE {r['down']['value'].rsplit('/', 1)[-1]}")
    else:
        query = f"""
        SELECT ?e ?title (MAX(STR(?graph)) AS ?last)
        WHERE {{
            GRAPH ?graph {{ ?e a {cls} ; dc:title ?title . }}
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?e ?title
        """
        value = lambda r: " ".join(r['title']['value'].split())
    newest: dict = {}
    for r in run_sparql_query_with_retry(query):
        uri, last = r['e']['value'], r['last']['value']
        if uri not in newest or last > newest[uri][0]:
            newest[uri] = (last, value(r))
    return {uri: label for uri, (_, label) in newest.items()}


def plot_entity_lifeline() -> tuple[str, pd.DataFrame]:
    """When each AOP, KE, KER and stressor appears, vanishes and comes back (#183).

    The page draws this with a canvas view (static/js/viz/lifeline.js) from
    /api/v2/data; the Plotly figure here is the same picture as a heatmap, for
    the PNG/SVG downloads and /api/v2/figure. Presence comes from the bulk
    per-graph entity query shared with the birth/death plots.

    Returns:
        tuple[str, pd.DataFrame]: the heatmap HTML, and one row per entity
        ever seen (see ``lifeline_rows``).
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = sorted({v['version'] for v in get_all_versions()})
        if not versions:
            return create_fallback_plot("Entity Lifeline", "No versions"), pd.DataFrame()
        rows = []
        for entity_type in ENTITY_TYPE_CLASSES:
            uris = fetch_entity_uris_by_version(entity_type, versions)
            rows.extend(lifeline_rows(entity_type, uris, versions, _lifeline_labels(entity_type)))
        df = pd.DataFrame(rows)

        colors = dict(zip(ENTITY_TYPE_CLASSES, BRAND_COLORS['palette'][:4]))
        fig = go.Figure()
        offset = 0
        for entity_type in ENTITY_TYPE_CLASSES:
            sub = df[df['entity_type'] == entity_type]
            if sub.empty:
                continue
            z = np.array([[c == PRESENT for c in p] for p in sub['presence']], dtype=np.int8)
            fig.add_heatmap(
                z=z, x=versions, y=np.arange(offset, offset + len(sub)),
                colorscale=[[0, '#f0f0f0'], [1, colors[entity_type]]], zmin=0, zmax=1,
                showscale=False, name=entity_type,
                hovertemplate=f'{entity_type}<br>%{{x}}<extra></extra>',
            )
            fig.add_annotation(x=0, xref='paper', xanchor='right', y=offset + len(sub) / 2,
                               text=f"{entity_type} ({len(sub)})", showarrow=False)
            offset += len(sub) + 40  # a gap between the entity blocks
        fig.update_yaxes(autorange='reversed', showticklabels=False, showgrid=False, zeroline=False)
        fig.update_xaxes(showgrid=False)
        apply_snapshot_xaxis(fig)
        fig.update_layout(height=640, margin=dict(l=130, r=20, t=40, b=60),
                          hovermode='closest')

        _plot_data_cache['entity_lifeline'] = df
        _plot_figure_cache['entity_lifeline'] = fig
        return render_plot_html(fig), df

    except SparqlUnavailable:
        raise
    except Exception as e:
        logger.error(f"Failed to generate entity lifeline: {e}")
        return create_fallback_plot("Entity Lifeline", str(e)), pd.DataFrame()


def plot_oecd_status_distribution() -> tuple[str, str, pd.DataFrame]:
    """Per-version count of AOPs in each OECD status (absolute + percentage).

    Companion to `plot_oecd_completeness_trend` (which measures *completeness*
    grouped by status). This view answers a different question: how is the
    status *mix* of AOP-Wiki evolving? Are we accumulating drafts or
    converting them to endorsed AOPs?

    Returns:
        tuple[str, str, pd.DataFrame]:
            - absolute stacked-area HTML
            - percentage (100%-stacked) HTML
            - long-form DataFrame (version, status, n_aops, percentage).
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        query = """
        SELECT ?graph ?status_str (COUNT(DISTINCT ?aop) AS ?n_aops)
        WHERE {
            GRAPH ?graph {
                ?aop a aopo:AdverseOutcomePathway .
                OPTIONAL {
                    ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?s .
                }
                BIND(COALESCE(STR(?s), "No Status") AS ?status_str)
            }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ?status_str
        """
        results = run_sparql_query_with_retry(query)
        if not results:
            return (
                create_fallback_plot("OECD Status Distribution", "No data available"),
                create_fallback_plot("OECD Status Distribution (%)", "No data available"),
                pd.DataFrame(),
            )

        rows = []
        for r in results:
            graph_uri = r.get('graph', {}).get('value', '')
            version = graph_uri.rsplit('/', 1)[-1] if graph_uri else None
            rows.append({
                'version': version,
                'status': r.get('status_str', {}).get('value', 'No Status'),
                'n_aops': int(r.get('n_aops', {}).get('value', 0)),
            })
        df = pd.DataFrame(rows).dropna(subset=['version'])
        df['_dt'] = pd.to_datetime(df['version'], errors='coerce')
        df = df.sort_values(['_dt', 'status']).drop(columns='_dt').reset_index(drop=True)

        totals = df.groupby('version', as_index=False)['n_aops'].sum().rename(
            columns={'n_aops': 'total'}
        )
        df = df.merge(totals, on='version')
        df['percentage'] = (df['n_aops'] / df['total'] * 100).round(2)

        # Use brand OECD palette where status names match; fall back to palette.
        oecd_colors = BRAND_COLORS.get('oecd_status', {})
        status_order = sorted(df['status'].unique(), key=lambda s: (s == 'No Status', s))
        color_map = {
            s: oecd_colors.get(s, BRAND_COLORS['palette'][i % len(BRAND_COLORS['palette'])])
            for i, s in enumerate(status_order)
        }

        fig_abs = px.area(
            df, x='version', y='n_aops', color='status',
            category_orders={'status': status_order},
            color_discrete_map=color_map,
            labels={'version': 'Snapshot', 'n_aops': 'AOPs', 'status': 'OECD status'},
        )
        fig_abs.update_layout(margin=dict(l=60, r=30, t=50, b=80))
        fig_abs.update_xaxes(tickangle=-45)

        fig_pct = px.area(
            df, x='version', y='percentage', color='status',
            category_orders={'status': status_order},
            color_discrete_map=color_map,
            groupnorm='percent',
            labels={'version': 'Snapshot', 'percentage': '% of AOPs', 'status': 'OECD status'},
        )
        fig_pct.update_layout(margin=dict(l=60, r=30, t=50, b=80))
        fig_pct.update_xaxes(tickangle=-45)
        fig_pct.update_yaxes(range=[0, 100])

        _plot_data_cache['oecd_status_distribution_absolute'] = df
        _plot_data_cache['oecd_status_distribution_percentage'] = df
        _plot_figure_cache['oecd_status_distribution_absolute'] = fig_abs
        _plot_figure_cache['oecd_status_distribution_percentage'] = fig_pct

        return render_plot_html(fig_abs), render_plot_html(fig_pct), df

    except Exception as e:
        logger.error(f"Failed to generate OECD status distribution: {str(e)}")
        return (
            create_fallback_plot("OECD Status Distribution", str(e)),
            create_fallback_plot("OECD Status Distribution (%)", str(e)),
            pd.DataFrame(),
        )


def plot_stressor_coverage_growth() -> tuple[str, str, pd.DataFrame]:
    """Stressor coverage growth (absolute + delta) over snapshots.

    Same numbers as the Stressors trace of `plot_main_graph` but isolated
    so it can be downloaded and methodology-cited on its own (#75). Used
    in conjunction with the AOPs-per-stressor distribution view to answer
    "is AOP-Wiki broadening (more stressors) or deepening (same stressors,
    more AOPs)?"

    Returns:
        tuple[str, str, pd.DataFrame]: absolute_html, delta_html, df.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        query = """
        SELECT ?graph (COUNT(DISTINCT ?s) AS ?n_stressors)
        WHERE {
            GRAPH ?graph { ?s a <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C54571> . }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        """
        results = run_sparql_query_with_retry(query)
        if not results:
            return (
                create_fallback_plot("Stressor Coverage Growth", "No data"),
                create_fallback_plot("Stressor Coverage Growth (Δ)", "No data"),
                pd.DataFrame(),
            )
        rows = []
        for r in results:
            uri = r.get('graph', {}).get('value', '')
            rows.append({
                'version': uri.rsplit('/', 1)[-1] if uri else None,
                'Stressors': int(r.get('n_stressors', {}).get('value', 0)),
            })
        df = pd.DataFrame(rows).dropna(subset=['version'])
        df['_dt'] = pd.to_datetime(df['version'], errors='coerce')
        df = df.sort_values('_dt').drop(columns='_dt').reset_index(drop=True)
        df['Δ'] = df['Stressors'].diff().fillna(0).astype(int)

        fig_abs = px.line(
            df, x='version', y='Stressors', markers=True,
            color_discrete_sequence=[BRAND_COLORS['orange']],
            labels={'version': 'Snapshot', 'Stressors': 'Distinct stressors'},
        )
        fig_abs.update_xaxes(tickangle=-45)
        fig_abs.update_layout(margin=dict(l=60, r=30, t=50, b=80))

        fig_delta = px.bar(
            df, x='version', y='Δ',
            color_discrete_sequence=[BRAND_COLORS['orange']],
            labels={'version': 'Snapshot', 'Δ': 'Quarter-over-quarter Δ'},
        )
        fig_delta.update_xaxes(tickangle=-45)
        fig_delta.update_layout(margin=dict(l=60, r=30, t=50, b=80))

        _plot_data_cache['stressor_coverage_growth_absolute'] = df[['version', 'Stressors']]
        _plot_data_cache['stressor_coverage_growth_delta'] = df[['version', 'Δ']].rename(columns={'Δ': 'delta'})
        _plot_figure_cache['stressor_coverage_growth_absolute'] = fig_abs
        _plot_figure_cache['stressor_coverage_growth_delta'] = fig_delta

        return render_plot_html(fig_abs), render_plot_html(fig_delta), df

    except Exception as e:
        logger.error(f"Failed to generate stressor coverage growth: {str(e)}")
        return (
            create_fallback_plot("Stressor Coverage Growth", str(e)),
            create_fallback_plot("Stressor Coverage Growth (Δ)", str(e)),
            pd.DataFrame(),
        )


def plot_aops_per_stressor_distribution() -> tuple[str, str, pd.DataFrame]:
    """Stressors bucketed by #AOPs that reference them, over snapshots (#75).

    Answers whether AOP-Wiki is *broadening* (more stressors used in one AOP
    each) or *deepening* (same set of "popular" stressors reused across many
    AOPs). Bucket boundaries: {1, 2, 3-5, 6-10, 11+}.

    Returns:
        tuple[str, str, pd.DataFrame]: absolute_html, percentage_html, df.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        query = """
        SELECT ?graph ?stressor (COUNT(DISTINCT ?aop) AS ?aop_count)
        WHERE {
            GRAPH ?graph {
                ?aop a aopo:AdverseOutcomePathway ;
                     <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C54571> ?stressor .
                ?stressor a <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C54571> .
            }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ?stressor
        """
        results = run_sparql_query_with_retry(query)
        if not results:
            return (
                create_fallback_plot("AOPs per Stressor Distribution", "No data"),
                create_fallback_plot("AOPs per Stressor Distribution (%)", "No data"),
                pd.DataFrame(),
            )

        rows = []
        for r in results:
            uri = r.get('graph', {}).get('value', '')
            rows.append({
                'version': uri.rsplit('/', 1)[-1] if uri else None,
                'stressor': r.get('stressor', {}).get('value', ''),
                'aop_count': int(r.get('aop_count', {}).get('value', 0)),
            })
        raw = pd.DataFrame(rows).dropna(subset=['version'])

        bucket_order = ['1 AOP', '2 AOPs', '3-5 AOPs', '6-10 AOPs', '11+ AOPs']

        def _bucket(n: int) -> str:
            if n <= 1:
                return '1 AOP'
            if n == 2:
                return '2 AOPs'
            if n <= 5:
                return '3-5 AOPs'
            if n <= 10:
                return '6-10 AOPs'
            return '11+ AOPs'

        raw['bucket'] = raw['aop_count'].map(_bucket)
        df = raw.groupby(['version', 'bucket'], as_index=False).size().rename(columns={'size': 'n_stressors'})

        # Ensure every (version, bucket) combination is present (even zero).
        all_versions = sorted(df['version'].unique(), key=lambda v: pd.to_datetime(v, errors='coerce'))
        full_idx = pd.MultiIndex.from_product([all_versions, bucket_order], names=['version', 'bucket'])
        df = df.set_index(['version', 'bucket']).reindex(full_idx, fill_value=0).reset_index()

        # Percentages computed against snapshot total.
        totals = df.groupby('version', as_index=False)['n_stressors'].sum().rename(columns={'n_stressors': 'total'})
        df = df.merge(totals, on='version')
        df['percentage'] = (df['n_stressors'] / df['total'] * 100).fillna(0).round(2)

        # Brand colour ramp — light → dark for low → high concentration.
        ramp = [BRAND_COLORS['sky_blue'], BRAND_COLORS['light_blue'], BRAND_COLORS['blue'],
                BRAND_COLORS['violet'], BRAND_COLORS['deep_magenta']]
        color_map = dict(zip(bucket_order, ramp))

        fig_abs = px.area(
            df, x='version', y='n_stressors', color='bucket',
            category_orders={'bucket': bucket_order},
            color_discrete_map=color_map,
            labels={'version': 'Snapshot', 'n_stressors': 'Stressors', 'bucket': 'AOPs referencing this stressor'},
        )
        fig_abs.update_xaxes(tickangle=-45)
        fig_abs.update_layout(margin=dict(l=60, r=30, t=50, b=80))

        # `percentage` already sums to 100 per snapshot (buckets are exhaustive
        # and mutually exclusive), so stack it directly — an extra groupnorm here
        # would be a redundant re-normalisation.
        fig_pct = px.area(
            df, x='version', y='percentage', color='bucket',
            category_orders={'bucket': bucket_order},
            color_discrete_map=color_map,
            labels={'version': 'Snapshot', 'percentage': '% of stressors', 'bucket': 'AOPs referencing this stressor'},
        )
        fig_pct.update_xaxes(tickangle=-45)
        fig_pct.update_yaxes(range=[0, 100])
        fig_pct.update_layout(margin=dict(l=60, r=30, t=50, b=80))
        # The 3-5 / 6-10 / 11+ buckets compress into a thin sliver at the top of
        # the 100% stack; a thin white border between bands delineates them so the
        # minority categories are separable (#plot-review). The fill colour must
        # be pinned explicitly first — plotly otherwise derives the stacked fill
        # from the line colour, so a white line would blank the whole band.
        for _tr in fig_pct.data:
            _tr.update(fillcolor=color_map.get(_tr.name), line=dict(width=0.6, color='white'))

        _plot_data_cache['aops_per_stressor_distribution_absolute'] = df
        _plot_data_cache['aops_per_stressor_distribution_percentage'] = df
        _plot_figure_cache['aops_per_stressor_distribution_absolute'] = fig_abs
        _plot_figure_cache['aops_per_stressor_distribution_percentage'] = fig_pct

        return render_plot_html(fig_abs), render_plot_html(fig_pct), df

    except Exception as e:
        logger.error(f"Failed to generate AOPs-per-stressor distribution: {str(e)}")
        return (
            create_fallback_plot("AOPs per Stressor Distribution", str(e)),
            create_fallback_plot("AOPs per Stressor Distribution (%)", str(e)),
            pd.DataFrame(),
        )


def plot_ke_mmo_coverage_trends() -> tuple[str, str, pd.DataFrame]:
    """Historical trend of KE measurement-method (obo:MMO_0000000) coverage (#68).

    Per-version share of KEs that carry an `obo:MMO_0000000` description.
    Companion to the latest-snapshot donut on /snapshot. Shows whether the
    NAM-readiness signal is improving over time.

    Returns:
        tuple[str, str, pd.DataFrame]:
            - absolute_html (count of KEs with vs without MMO)
            - percentage_html (% of KEs with MMO over time)
            - long DataFrame (version, with_mmo, without_mmo, total, percentage)
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        query = """
        SELECT ?graph
               (SUM(IF(BOUND(?mmo), 1, 0)) AS ?with_mmo)
               (COUNT(?ke) AS ?total)
        WHERE {
            GRAPH ?graph {
                ?ke a aopo:KeyEvent .
                OPTIONAL { ?ke <http://purl.obolibrary.org/obo/MMO_0000000> ?mmo }
            }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        """
        results = run_sparql_query_with_retry(query)
        if not results:
            return (
                create_fallback_plot("KE MMO Coverage (absolute)", "No data"),
                create_fallback_plot("KE MMO Coverage (%)", "No data"),
                pd.DataFrame(),
            )

        rows = []
        for r in results:
            uri = r.get('graph', {}).get('value', '')
            with_mmo = int(r.get('with_mmo', {}).get('value', 0))
            total = int(r.get('total', {}).get('value', 0))
            rows.append({
                'version': uri.rsplit('/', 1)[-1] if uri else None,
                'with_mmo': with_mmo,
                'without_mmo': max(0, total - with_mmo),
                'total': total,
                'percentage': round(with_mmo / total * 100, 2) if total else 0.0,
            })
        df = pd.DataFrame(rows).dropna(subset=['version'])
        df['_dt'] = pd.to_datetime(df['version'], errors='coerce')
        df = df.sort_values('_dt').drop(columns='_dt').reset_index(drop=True)

        # Absolute: stacked bar (with + without).
        long_abs = df.melt(id_vars='version', value_vars=['with_mmo', 'without_mmo'],
                           var_name='Status', value_name='KEs')
        long_abs['Status'] = long_abs['Status'].map({
            'with_mmo': 'With MMO assay',
            'without_mmo': 'Without MMO assay',
        })
        fig_abs = px.bar(
            long_abs, x='version', y='KEs', color='Status',
            color_discrete_map={
                'With MMO assay': BRAND_COLORS['magenta'],
                'Without MMO assay': BRAND_COLORS['primary'],
            },
            labels={'version': 'Snapshot', 'KEs': 'Key Events'},
        )
        fig_abs.update_xaxes(tickangle=-45)
        fig_abs.update_layout(barmode='stack', margin=dict(l=60, r=30, t=50, b=80))

        # Percentage: single line.
        fig_pct = px.line(
            df, x='version', y='percentage', markers=True,
            color_discrete_sequence=[BRAND_COLORS['magenta']],
            labels={'version': 'Snapshot', 'percentage': '% of KEs with MMO assay'},
        )
        fig_pct.update_xaxes(tickangle=-45)
        fig_pct.update_yaxes(range=[0, 100])
        fig_pct.update_layout(margin=dict(l=60, r=30, t=50, b=80))

        _plot_data_cache['ke_mmo_coverage_absolute'] = df
        _plot_data_cache['ke_mmo_coverage_percentage'] = df
        _plot_figure_cache['ke_mmo_coverage_absolute'] = fig_abs
        _plot_figure_cache['ke_mmo_coverage_percentage'] = fig_pct

        return render_plot_html(fig_abs), render_plot_html(fig_pct), df

    except Exception as e:
        logger.error(f"Failed to generate KE MMO coverage trend: {e}")
        return (
            create_fallback_plot("KE MMO Coverage (absolute)", str(e)),
            create_fallback_plot("KE MMO Coverage (%)", str(e)),
            pd.DataFrame(),
        )


def plot_ke_migration_map(top_n: int = 50) -> tuple[str, pd.DataFrame]:
    """KE AOP-membership migration across snapshot history (#66).

    The issue asks for a Sankey of KE→AOP membership over versions. With
    ~1.5K KEs and ~500 AOPs across 32 snapshots that visualisation is
    unreadable; v1 ships the same underlying data as a heatmap (Y = top-N
    most-mobile KEs, X = snapshot date, fill = #AOPs the KE is in that
    quarter). The heatmap reveals exactly what the issue wants exposed:
    KEs first introduced, removed, merged, or jumping between AOPs.

    Mobility is scored per KE as `distinct_aops_ever × snapshots_present`
    so KEs that touch many AOPs and persist across many quarters rank
    highest. Tunable via the methodology note.
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        q = """
        SELECT ?graph ?ke (COUNT(DISTINCT ?aop) AS ?n_aops)
        WHERE {
            GRAPH ?graph {
                ?aop a aopo:AdverseOutcomePathway ;
                     aopo:has_key_event ?ke .
            }
            FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ?ke
        """
        rows = run_sparql_query_with_retry(q)
        if not rows:
            return (
                create_fallback_plot("KE Migration Map", "No data"),
                pd.DataFrame(),
            )

        df = pd.DataFrame([{
            'version': r['graph']['value'].rsplit('/', 1)[-1],
            'ke_uri': r['ke']['value'],
            'n_aops': int(r['n_aops']['value']),
        } for r in rows])
        df['ke_id'] = df['ke_uri'].str.rsplit('/', n=1).str[-1]

        # Mobility = distinct snapshots present × peak #AOPs.
        mobility = df.groupby('ke_id').agg(
            snapshots_present=('version', 'nunique'),
            peak_aops=('n_aops', 'max'),
        )
        mobility['score'] = mobility['snapshots_present'] * mobility['peak_aops']
        top_kes = mobility.sort_values('score', ascending=False).head(top_n).index.tolist()
        top_df = df[df['ke_id'].isin(top_kes)].copy()

        # Pivot to heatmap matrix.
        versions = sorted(top_df['version'].unique())
        pivot = (top_df.pivot_table(index='ke_id', columns='version',
                                    values='n_aops', aggfunc='max')
                 .reindex(top_kes).reindex(columns=versions))

        # Fetch KE titles from the latest snapshot so the y-axis is readable.
        latest_graph = f"http://aopwiki.org/graph/{versions[-1]}"
        values_block = ' '.join(
            f'<https://identifiers.org/aop.events/{kid}>' for kid in top_kes
        )
        title_q = f"""
        SELECT ?ke ?title WHERE {{
            GRAPH <{latest_graph}> {{
                VALUES ?ke {{ {values_block} }}
                OPTIONAL {{ ?ke <http://purl.org/dc/elements/1.1/title> ?title }}
            }}
        }}
        """
        titles: dict[str, str] = {}
        try:
            for r in run_sparql_query_with_retry(title_q) or []:
                uri = r.get('ke', {}).get('value', '')
                kid = uri.rsplit('/', 1)[-1] if uri else ''
                if kid:
                    titles[kid] = r.get('title', {}).get('value', '') or ''
        except Exception as exc:
            logger.warning(f"KE-title fetch failed for migration map: {exc}")

        def _label(kid: str) -> str:
            t = titles.get(kid, '').strip()
            if not t:
                return f"KE {kid}"
            # Truncate long titles so the y-axis stays readable.
            short = t if len(t) <= 60 else t[:57] + '…'
            return f"KE {kid} — {short}"

        y_labels = [_label(kid) for kid in pivot.index]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=versions,
            y=y_labels,
            # Single-hue blues ramp, monotonic in lightness (light→dark), so equal
            # value steps read as equal colour steps and mid-range differences in
            # the crowded 10-30 band separate cleanly — unlike the old
            # light-blue→blue→magenta scale that was dominated by look-alike blues
            # then jumped hue (#plot-review).
            colorscale=[
                [0.0, '#f2f2f7'],
                [0.25, BRAND_COLORS['sky_blue']],
                [0.5, BRAND_COLORS['light_blue']],
                [0.75, BRAND_COLORS['blue']],
                [1.0, BRAND_COLORS['primary']],
            ],
            zmin=0,
            colorbar=dict(title='#AOPs<br>referencing<br>this KE'),
            hovertemplate='<b>%{y}</b><br>Snapshot %{x}<br>In %{z} AOPs<extra></extra>',
        ))
        fig.update_layout(
            title={"text": f"KE migration map — top {len(top_kes)} most-mobile Key Events<br><sub>Cell colour = #AOPs the KE is in at that snapshot; empty cells = KE absent</sub>"},
            xaxis=dict(title='Snapshot date', tickangle=-45),
            yaxis=dict(autorange='reversed'),
            margin=dict(l=80, r=40, t=80, b=80),
            height=900,
        )

        # Export DataFrame: rows = top KEs × snapshots, columns = #AOPs, csv-friendly.
        export = top_df.sort_values(['ke_id', 'version'])[['version', 'ke_id', 'ke_uri', 'n_aops']]

        _plot_data_cache['ke_migration_map'] = export
        _plot_figure_cache['ke_migration_map'] = fig

        return render_plot_html(fig), export

    except Exception as e:
        logger.error(f"Failed to generate KE migration map: {e}")
        return (
            create_fallback_plot("KE Migration Map", str(e)),
            pd.DataFrame(),
        )


def plot_avg_per_aop() -> tuple[str, str]:
    """Generate average components per AOP visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        query_aops = """
            SELECT ?graph (COUNT(?aop) AS ?count)
            WHERE {
                GRAPH ?graph { ?aop a aopo:AdverseOutcomePathway . }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
        """
        # Count has_key_event / has_key_event_relationship MEMBERSHIPS (one row per
        # AOP-KE / AOP-KER link), not distinct KE/KER entities. Averaging distinct
        # entities graph-wide understates the metric because Key Events are reused
        # across AOPs (e.g. 1583 distinct KEs / 588 AOPs = 2.7, whereas the average
        # AOP actually contains ~6 KEs).
        query_kes = """
            SELECT ?graph (COUNT(?ke) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?aop a aopo:AdverseOutcomePathway ;
                         aopo:has_key_event ?ke .
                }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
        """
        query_kers = """
            SELECT ?graph (COUNT(?ker) AS ?count)
            WHERE {
                GRAPH ?graph {
                    ?aop a aopo:AdverseOutcomePathway ;
                         aopo:has_key_event_relationship ?ker .
                }
                FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph
        """

        df_aops = extract_counts(run_sparql_query(query_aops))
        df_aops.rename(columns={"count": "aop_count"}, inplace=True)
        df_aops["version_dt"] = pd.to_datetime(df_aops["version"], errors="coerce")

        df_kes = extract_counts(run_sparql_query(query_kes))
        df_kes.rename(columns={"count": "ke_count"}, inplace=True)
        df_kes["version_dt"] = pd.to_datetime(df_kes["version"], errors="coerce")

        df_kers = extract_counts(run_sparql_query(query_kers))
        df_kers.rename(columns={"count": "ker_count"}, inplace=True)
        df_kers["version_dt"] = pd.to_datetime(df_kers["version"], errors="coerce")

        df_all = df_aops.merge(df_kes, on="version").merge(df_kers, on="version")
        df_all = df_all.drop_duplicates("version").copy()
        df_all["version_dt"] = pd.to_datetime(df_all["version"], errors="coerce")
        df_all = df_all.sort_values("version_dt").drop(columns="version_dt")

        if df_all.empty:
            logger.warning("Average per AOP query returned no results")
            return (
                create_fallback_plot("Average Components per AOP Over Time", "No data available"),
                create_fallback_plot("Change in Average Components per AOP", "No data available")
            )

        # Guard against division by zero
        df_all["avg_KEs_per_AOP"] = df_all.apply(
            lambda row: row["ke_count"] / row["aop_count"] if row["aop_count"] > 0 else 0, axis=1
        )
        df_all["avg_KERs_per_AOP"] = df_all.apply(
            lambda row: row["ker_count"] / row["aop_count"] if row["aop_count"] > 0 else 0, axis=1
        )

        # Absolute
        df_melted = df_all.melt(
            id_vars="version",
            value_vars=["avg_KEs_per_AOP", "avg_KERs_per_AOP"],
            var_name="Metric",
            value_name="Average"
        ).sort_values("version")

        df_melted["Metric"] = df_melted["Metric"].replace({
            "avg_KEs_per_AOP": "Average KEs per AOP",
            "avg_KERs_per_AOP": "Average KERs per AOP"
        })

        fig_abs = px.line(df_melted, x="version", y="Average", color="Metric", markers=True,
                          color_discrete_sequence=BRAND_COLORS['palette'])
        fig_abs.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        # Auto-thin the date ticks (one per year) instead of all ~32 rotated
        # snapshot labels crowding the axis (#plot-review).
        apply_snapshot_xaxis(fig_abs)

        # Delta
        df_all["avg_KEs_per_AOP_Δ"] = df_all["avg_KEs_per_AOP"].diff().fillna(0)
        df_all["avg_KERs_per_AOP_Δ"] = df_all["avg_KERs_per_AOP"].diff().fillna(0)
        df_delta_melted = df_all.melt(
            id_vars="version",
            value_vars=["avg_KEs_per_AOP_Δ", "avg_KERs_per_AOP_Δ"],
            var_name="Metric",
            value_name="Δ Average"
        ).sort_values("version")

        df_delta_melted["Metric"] = df_delta_melted["Metric"].replace({
            "avg_KEs_per_AOP_Δ": "Δ KEs per AOP",
            "avg_KERs_per_AOP_Δ": "Δ KERs per AOP"
        })
        fig_delta = px.line(df_delta_melted, x="version", y="Δ Average", color="Metric", markers=True,
                            color_discrete_sequence=BRAND_COLORS['palette'])
        fig_delta.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        apply_snapshot_xaxis(fig_delta)

        # Cache data for CSV export
        _plot_data_cache['average_components_per_aop_absolute'] = df_melted.copy()
        _plot_data_cache['average_components_per_aop_delta'] = df_delta_melted.copy()

        # Cache figures for image export
        _plot_figure_cache['average_components_per_aop_absolute'] = fig_abs
        _plot_figure_cache['average_components_per_aop_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta)
        )

    except Exception as e:
        logger.error(f"Failed to generate average per AOP plots: {str(e)}")
        return (
            create_fallback_plot("Average Components per AOP Over Time", str(e)),
            create_fallback_plot("Change in Average Components per AOP", str(e))
        )


def plot_network_density() -> str:
    """Generate network density evolution visualization."""
    global _plot_data_cache, _plot_figure_cache

    try:
        # Per-AOP graph density: each AOP's own KE->KER graph, averaged across
        # AOPs. Grouping by ?aop with COUNT(DISTINCT ...) yields correct per-AOP
        # node/edge counts despite the OPTIONAL cross-product. The previous
        # graph-wide query used non-distinct COUNT(?ker) under a KE x KER join,
        # which inflated edges ~12x and treated all AOPs as one global graph.
        query_density = """
        SELECT ?graph ?aop (COUNT(DISTINCT ?ke) AS ?nodes) (COUNT(DISTINCT ?ker) AS ?edges)
        WHERE {
          GRAPH ?graph {
            ?aop a aopo:AdverseOutcomePathway .
            OPTIONAL { ?aop aopo:has_key_event ?ke . }
            OPTIONAL { ?aop aopo:has_key_event_relationship ?ker . }
          }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph ?aop
        """

        results = run_sparql_query(query_density)

        if not results:
            logger.warning("Network density query returned no results")
            return create_fallback_plot("Network Density Evolution Over Time", "No data available")

        df_aop = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "nodes": int(r["nodes"]["value"]),
            "edges": int(r["edges"]["value"])
        } for r in results])

        # Density is undefined for AOPs with fewer than 2 KEs — exclude them.
        df_aop = df_aop[df_aop["nodes"] >= 2].copy()

        if df_aop.empty:
            logger.warning("Network density DataFrame is empty")
            return create_fallback_plot("Network Density Evolution Over Time", "No data available")

        df_aop["density"] = (
            2 * df_aop["edges"] / (df_aop["nodes"] * (df_aop["nodes"] - 1))
        )

        # Mean per-AOP density per version (with the count of contributing AOPs).
        df_density = df_aop.groupby("version", as_index=False).agg(
            density=("density", "mean"), aops=("density", "size")
        )

        # Sort and ensure tick labels
        df_density["version_dt"] = pd.to_datetime(df_density["version"], errors="coerce")
        df_density = df_density.sort_values("version_dt").drop(columns="version_dt")

        fig = px.line(
            df_density,
            x="version",
            y="density",
            markers=True,
            labels={"density": "Mean per-AOP graph density"},
            color_discrete_sequence=[BRAND_COLORS['accent']]
        )
        fig.update_layout(
            margin=dict(l=50, r=20, t=50, b=50),
            yaxis=dict(title="Mean per-AOP graph density")
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=df_density["version"],
            ticktext=df_density["version"],
            tickangle=-45
        )

        # Cache data for CSV export
        _plot_data_cache['aop_network_density'] = df_density.copy()

        # Cache figure for image export
        _plot_figure_cache['aop_network_density'] = fig

        return render_plot_html(fig)

    except Exception as e:
        logger.error(f"Failed to generate network density plot: {str(e)}")
        return create_fallback_plot("Network Density Evolution Over Time", str(e))


def plot_author_counts() -> tuple[str, str]:
    """Generate author contribution visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        query = """
        SELECT ?graph (COUNT(DISTINCT ?c) AS ?author_count)
        WHERE {
          GRAPH ?graph {
            ?aop a aopo:AdverseOutcomePathway ;
                 dc:creator ?c .
          }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        GROUP BY ?graph
        ORDER BY ?graph
        """

        results = run_sparql_query(query)

        if not results:
            logger.warning("Author counts query returned no results")
            return (
                create_fallback_plot("Author Contributions Over Time", "No data available"),
                create_fallback_plot("Change in Author Contributions", "No data available")
            )

        df_authors = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "author_count": int(r["author_count"]["value"])
        } for r in results])

        if df_authors.empty:
            logger.warning("Author counts DataFrame is empty")
            return (
                create_fallback_plot("Author Contributions Over Time", "No data available"),
                create_fallback_plot("Change in Author Contributions", "No data available")
            )

        df_authors["version_dt"] = pd.to_datetime(df_authors["version"], errors="coerce")
        df_authors = df_authors.sort_values("version_dt").drop(columns="version_dt")

        # Absolute
        fig_abs = px.line(df_authors, x="version", y="author_count", markers=True,
                          color_discrete_sequence=[BRAND_COLORS['blue']])
        fig_abs.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        # Auto-thin the date ticks (one per year) rather than printing all ~32
        # rotated snapshot labels, which crowd into a wall at -45° (#plot-review).
        apply_snapshot_xaxis(fig_abs)

        # Delta
        df_authors["author_count_Δ"] = df_authors["author_count"].diff().fillna(0)
        fig_delta = px.line(df_authors, x="version", y="author_count_Δ", markers=True,
                            color_discrete_sequence=[BRAND_COLORS['blue']])
        fig_delta.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        apply_snapshot_xaxis(fig_delta)

        # Cache data for CSV export
        _plot_data_cache['aop_authors_absolute'] = df_authors.copy()
        _plot_data_cache['aop_authors_delta'] = df_authors[['version', 'author_count_Δ']].copy()

        # Cache figures for image export
        _plot_figure_cache['aop_authors_absolute'] = fig_abs
        _plot_figure_cache['aop_authors_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta)
        )

    except Exception as e:
        logger.error(f"Failed to generate author count plots: {str(e)}")
        return (
            create_fallback_plot("Author Contributions Over Time", str(e)),
            create_fallback_plot("Change in Author Contributions", str(e))
        )


def plot_aop_lifetime() -> tuple[str, str]:
    """Generate AOP lifetime analysis visualizations.

    Returns (creation-year histogram, creation-vs-modification scatter).
    The per-year modification histogram was dropped — its information is
    captured by the y-axis of the scatter, which subsumes it.
    """
    global _plot_data_cache, _plot_figure_cache

    fallback_created = create_fallback_plot("AOP Creation Timeline", "No data available")
    fallback_scatter = create_fallback_plot("AOP Creation vs. Modification Timeline", "No data available")

    try:
        query_lifetime = """
        SELECT ?graph ?aop ?created ?modified
        WHERE {
          GRAPH ?graph {
            ?aop a aopo:AdverseOutcomePathway ;
                 dcterms:created ?created ;
                 dcterms:modified ?modified .
          }
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }
        """
        results_lifetime = run_sparql_query(query_lifetime)

        if not results_lifetime:
            logger.warning("AOP lifetime query returned no results")
            return fallback_created, fallback_scatter

        df_lifetime = pd.DataFrame([{
            "aop": r["aop"]["value"],
            "version": r["graph"]["value"].split("/")[-1],
            "created": r["created"]["value"],
            "modified": r["modified"]["value"],
        } for r in results_lifetime])
        # One vectorised parse; parsing ~28k values one call at a time made
        # pandas guess the format each time and took ~10 s (#179). "mixed"
        # parses each value on its own, as the per-value calls did.
        for col in ("created", "modified"):
            df_lifetime[col] = pd.to_datetime(df_lifetime[col], errors="coerce", format="mixed")

        if df_lifetime.empty:
            logger.warning("AOP lifetime DataFrame is empty after parsing")
            return fallback_created, fallback_scatter

        # Drop rows where date parsing failed
        df_lifetime = df_lifetime.dropna(subset=["created", "modified"])
        if df_lifetime.empty:
            logger.warning("AOP lifetime DataFrame is empty after dropping invalid dates")
            return fallback_created, fallback_scatter

        df_lifetime["lifetime_days"] = (df_lifetime["modified"] - df_lifetime["created"]).dt.days
        df_lifetime["year_created"] = df_lifetime["created"].dt.year

        # Deduplicate to one row per AOP (earliest creation wins for the histogram).
        df_created = df_lifetime.sort_values("created").drop_duplicates("aop", keep="first")

        # Cache data for CSV export
        _plot_data_cache['aops_created_over_time'] = df_created.copy()
        _plot_data_cache['aop_creation_vs_modification_timeline'] = df_lifetime.copy()

        # --- Plot 1: AOPs Created ---
        html1 = fallback_created
        try:
            fig1 = px.histogram(df_created, x="year_created",
                                labels={"year_created": "Year", "count": "AOP Count"},
                                color_discrete_sequence=[BRAND_COLORS['blue']])
            # Anchor one bin per calendar year, centred on the integer year, so
            # each bar maps cleanly to a single year instead of straddling the
            # year tick (#plot-review).
            yrs = df_created["year_created"].dropna()
            if not yrs.empty:
                fig1.update_traces(xbins=dict(start=int(yrs.min()) - 0.5,
                                              end=int(yrs.max()) + 0.5, size=1))
                fig1.update_xaxes(dtick=1, tick0=int(yrs.min()))
            fig1.update_layout(height=400, bargap=0.05)
            html1 = render_plot_html(fig1)
            _plot_figure_cache['aops_created_over_time'] = fig1
        except Exception as e:
            logger.error(f"Failed to generate AOP creation timeline plot: {str(e)}")

        # --- Plot 2: Created vs. Modified scatter ---
        html2 = fallback_scatter
        try:
            # One point per distinct (AOP, created, modified): the query returns
            # every AOP once per version, so an AOP unchanged across 30 snapshots
            # was drawn 30 times on one spot (13,930 points -> 3,969 at 35
            # versions), which also made long-lived AOPs look denser (#176).
            df_points = df_lifetime.drop_duplicates(["aop", "created", "modified"])
            fig2 = px.scatter(df_points, x="created", y="modified", hover_name="aop",
                              labels={"created": "Created", "modified": "Modified"},
                              color_discrete_sequence=[BRAND_COLORS['blue']],
                              render_mode='svg')
            # Semi-transparent, slightly smaller markers so the dense batch-
            # modification bands reveal point density instead of saturating to
            # solid blobs (#plot-review).
            fig2.update_traces(marker=dict(opacity=0.45, size=5))
            fig2.update_layout(height=500)
            html2 = render_plot_html(fig2)
            _plot_figure_cache['aop_creation_vs_modification_timeline'] = fig2
        except Exception as e:
            logger.error(f"Failed to generate AOP creation vs modification scatter plot: {str(e)}")

        return html1, html2

    except Exception as e:
        logger.error(f"Failed to generate AOP lifetime plots: {str(e)}")
        return fallback_created, fallback_scatter


def _query_ke_components_version(version_info, use_distinct=False):
    """Query KE component counts for a single version graph.

    Args:
        version_info: Dict with 'version' and 'graph_uri' keys
        use_distinct: If True, count DISTINCT values (for unique component queries)

    Returns:
        Dict with version, Process, Object, Action counts, or None on error
    """
    graph_uri = version_info["graph_uri"]
    version_str = version_info["version"]
    distinct = "DISTINCT " if use_distinct else ""

    query = f"""
    SELECT (COUNT({distinct}?process) AS ?process_count)
           (COUNT({distinct}?object) AS ?object_count)
           (COUNT({distinct}?action) AS ?action_count)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL {{ ?bioevent aopo:hasProcess ?process . }}
        OPTIONAL {{ ?bioevent aopo:hasObject ?object . }}
        OPTIONAL {{ ?bioevent aopo:hasAction ?action . }}
      }}
    }}
    """

    try:
        results = run_sparql_query_with_retry(query, max_retries=2)
        if results:
            r = results[0]
            return {
                "version": version_str,
                "Process": int(r["process_count"]["value"]),
                "Object": int(r["object_count"]["value"]),
                "Action": int(r["action_count"]["value"])
            }
    except Exception as e:
        logger.warning(f"KE components query failed for version {version_str}: {e}")
    return None


def plot_ke_components() -> tuple[str, str]:
    """Generate KE component annotation visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = get_all_versions()
        if not versions:
            logger.warning("No versions available for KE components query")
            return (
                create_fallback_plot("KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Annotations", "No data available")
            )

        logger.info(f"KE components: querying {len(versions)} versions in parallel")

        # Query each version in parallel to avoid Virtuoso cross-graph timeout
        data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_query_ke_components_version, v): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        data.append(result)
                except Exception as e:
                    logger.warning(f"KE components query timed out for version {version_str}: {e}")
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        if not data:
            logger.warning("KE components query returned no results")
            return (
                create_fallback_plot("KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Annotations", "No data available")
            )

        df_components = pd.DataFrame(data)

        if df_components.empty:
            logger.warning("KE components DataFrame is empty")
            return (
                create_fallback_plot("KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Annotations", "No data available")
            )

        # Sort by datetime
        df_components["version_dt"] = pd.to_datetime(df_components["version"], errors="coerce")
        df_components = df_components.sort_values("version_dt").drop(columns="version_dt")

        # --- Absolute line plot ---
        df_melted = df_components.melt(
            id_vars=["version"],
            value_vars=["Process", "Object", "Action"],
            var_name="Component", value_name="Count"
        )

        fig_abs = px.line(
            df_melted,
            x="version",
            y="Count",
            color="Component",
            markers=True,
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_abs.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        fig_abs.update_xaxes(
            tickmode='array',
            tickvals=df_components["version"],
            ticktext=df_components["version"],
            tickangle=-45
        )

        # --- Delta line plot ---
        df_diff = df_components.copy()
        df_diff[["Process", "Object", "Action"]] = \
            df_diff[["Process", "Object", "Action"]].diff().fillna(0)

        df_melted_diff = df_diff.melt(
            id_vars=["version"],
            value_vars=["Process", "Object", "Action"],
            var_name="Component", value_name="Change"
        )

        fig_delta = px.line(
            df_melted_diff,
            x="version",
            y="Change",
            color="Component",
            markers=True,
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_delta.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        fig_delta.update_xaxes(
            tickmode='array',
            tickvals=df_components["version"],
            ticktext=df_components["version"],
            tickangle=-45
        )

        # Cache data for CSV export
        _plot_data_cache['ke_component_annotations_absolute'] = df_melted.copy()
        _plot_data_cache['ke_component_annotations_delta'] = df_melted_diff.copy()

        # Cache figures for image export
        _plot_figure_cache['ke_component_annotations_absolute'] = fig_abs
        _plot_figure_cache['ke_component_annotations_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta)
        )

    except Exception as e:
        logger.error(f"Failed to generate KE component annotations plots: {str(e)}")
        return (
            create_fallback_plot("KE Component Annotations Over Time", str(e)),
            create_fallback_plot("Change in KE Component Annotations", str(e))
        )


def _query_ke_components_percentage_version(version_info):
    """Query KE component counts and total KEs for a single version graph.

    Returns:
        Dict with version, Process, Object, Action, TotalKEs counts, or None on error
    """
    graph_uri = version_info["graph_uri"]
    version_str = version_info["version"]

    # Query components
    query_comp = f"""
    SELECT (COUNT(?process) AS ?process_count)
           (COUNT(?object) AS ?object_count)
           (COUNT(?action) AS ?action_count)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?ke a aopo:KeyEvent ;
            aopo:hasBiologicalEvent ?bioevent .
        OPTIONAL {{ ?bioevent aopo:hasProcess ?process . }}
        OPTIONAL {{ ?bioevent aopo:hasObject ?object . }}
        OPTIONAL {{ ?bioevent aopo:hasAction ?action . }}
      }}
    }}
    """

    # Query total KEs
    query_total = f"""
    SELECT (COUNT(?ke) AS ?total_kes)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?ke a aopo:KeyEvent .
      }}
    }}
    """

    try:
        results_comp = run_sparql_query_with_retry(query_comp, max_retries=2)
        results_total = run_sparql_query_with_retry(query_total, max_retries=2)
        if results_comp and results_total:
            rc = results_comp[0]
            rt = results_total[0]
            return {
                "version": version_str,
                "Process": int(rc["process_count"]["value"]),
                "Object": int(rc["object_count"]["value"]),
                "Action": int(rc["action_count"]["value"]),
                "TotalKEs": int(rt["total_kes"]["value"])
            }
    except Exception as e:
        logger.warning(f"KE components percentage query failed for version {version_str}: {e}")
    return None


def plot_ke_components_percentage() -> tuple[str, str]:
    """Generate KE component percentage visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = get_all_versions()
        if not versions:
            logger.warning("No versions available for KE components percentage query")
            return (
                create_fallback_plot("KE Component Annotations as Percentage Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Percentage", "No data available")
            )

        logger.info(f"KE components percentage: querying {len(versions)} versions in parallel")

        # Query each version in parallel
        data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_query_ke_components_percentage_version, v): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        data.append(result)
                except Exception as e:
                    logger.warning(f"KE components percentage query timed out for version {version_str}: {e}")
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        if not data:
            logger.warning("KE components percentage query returned no results")
            return (
                create_fallback_plot("KE Component Annotations as Percentage Over Time", "No data available"),
                create_fallback_plot("Change in KE Component Percentage", "No data available")
            )

        df_merged = pd.DataFrame(data)

        # Filter out versions with zero total KEs to avoid division by zero
        df_merged = df_merged[df_merged["TotalKEs"] > 0]

        if df_merged.empty:
            logger.warning("KE components percentage: no versions with non-zero KE counts")
            return (
                create_fallback_plot("KE Component Annotations as Percentage Over Time", "No versions with KE data"),
                create_fallback_plot("Change in KE Component Percentage", "No versions with KE data")
            )

        # Calculate percentages
        for col in ["Process", "Object", "Action"]:
            df_merged[col] = (df_merged[col] / df_merged["TotalKEs"]) * 100

        # Sort by version
        df_merged["version_dt"] = pd.to_datetime(df_merged["version"], errors="coerce")
        df_merged = df_merged.sort_values("version_dt").drop(columns="version_dt")

        # --- Absolute percentage plot ---
        df_melted = df_merged.melt(
            id_vars=["version"],
            value_vars=["Process", "Object", "Action"],
            var_name="Component", value_name="Percentage"
        )

        fig_abs = px.line(
            df_melted,
            x="version",
            y="Percentage",
            color="Component",
            markers=True,
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_abs.update_layout(
            yaxis=dict(title="Percentage (%)"),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        fig_abs.update_xaxes(
            tickmode='array',
            tickvals=df_merged["version"],
            ticktext=df_merged["version"],
            tickangle=-45
        )

        # --- Delta percentage plot ---
        df_delta = df_merged.copy()
        df_delta[["Process", "Object", "Action"]] = df_delta[["Process", "Object", "Action"]].diff().fillna(0)

        df_melted_delta = df_delta.melt(
            id_vars=["version"],
            value_vars=["Process", "Object", "Action"],
            var_name="Component", value_name="Percentage Change"
        )

        fig_delta = px.line(
            df_melted_delta,
            x="version",
            y="Percentage Change",
            color="Component",
            markers=True,
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_delta.update_layout(
            yaxis=dict(title="Percentage Change (%)"),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        fig_delta.update_xaxes(
            tickmode='array',
            tickvals=df_merged["version"],
            ticktext=df_merged["version"],
            tickangle=-45
        )

        # Cache data for CSV export
        _plot_data_cache['ke_components_percentage_absolute'] = df_melted.copy()
        _plot_data_cache['ke_components_percentage_delta'] = df_melted_delta.copy()

        # Cache figures for image export
        _plot_figure_cache['ke_components_percentage_absolute'] = fig_abs
        _plot_figure_cache['ke_components_percentage_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta)
        )

    except Exception as e:
        logger.error(f"Failed to generate KE components percentage plots: {str(e)}")
        return (
            create_fallback_plot("KE Component Annotations as Percentage Over Time", str(e)),
            create_fallback_plot("Change in KE Component Percentage", str(e))
        )


def plot_unique_ke_components() -> tuple[str, str]:
    """Generate unique KE component visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = get_all_versions()
        if not versions:
            logger.warning("No versions available for unique KE components query")
            return (
                create_fallback_plot("Unique KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in Unique KE Component Annotations", "No data available")
            )

        logger.info(f"Unique KE components: querying {len(versions)} versions in parallel")

        # Query each version in parallel using shared helper with use_distinct=True
        data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_query_ke_components_version, v, True): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        data.append(result)
                except Exception as e:
                    logger.warning(f"Unique KE components query timed out for version {version_str}: {e}")
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        if not data:
            logger.warning("Unique KE components query returned no results")
            return (
                create_fallback_plot("Unique KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in Unique KE Component Annotations", "No data available")
            )

        df_unique = pd.DataFrame(data)

        if df_unique.empty:
            logger.warning("Unique KE components DataFrame is empty")
            return (
                create_fallback_plot("Unique KE Component Annotations Over Time", "No data available"),
                create_fallback_plot("Change in Unique KE Component Annotations", "No data available")
            )

        # Sort by datetime
        df_unique["version_dt"] = pd.to_datetime(df_unique["version"], errors="coerce")
        df_unique = df_unique.sort_values("version_dt").drop(columns="version_dt")

        # --- Absolute plot ---
        df_melted = df_unique.melt(
            id_vars=["version"],
            value_vars=["Process", "Object", "Action"],
            var_name="Component", value_name="Unique Count"
        )

        fig_abs = px.line(
            df_melted,
            x="version",
            y="Unique Count",
            color="Component",
            markers=True,
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_abs.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        fig_abs.update_xaxes(
            tickmode='array',
            tickvals=df_unique["version"],
            ticktext=df_unique["version"],
            tickangle=-45
        )

        # --- Delta plot ---
        df_diff = df_unique.copy()
        df_diff[["Process", "Object", "Action"]] = \
            df_diff[["Process", "Object", "Action"]].diff().fillna(0)

        df_melted_diff = df_diff.melt(
            id_vars=["version"],
            value_vars=["Process", "Object", "Action"],
            var_name="Component", value_name="Change"
        )

        fig_delta = px.line(
            df_melted_diff,
            x="version",
            y="Change",
            color="Component",
            markers=True,
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_delta.update_layout(
            margin=dict(l=50, r=20, t=50, b=50)
        )
        fig_delta.update_xaxes(
            tickmode='array',
            tickvals=df_unique["version"],
            ticktext=df_unique["version"],
            tickangle=-45
        )

        # Cache data for CSV export
        _plot_data_cache['unique_ke_components_absolute'] = df_melted.copy()
        _plot_data_cache['unique_ke_components_delta'] = df_melted_diff.copy()

        # Cache figures for image export
        _plot_figure_cache['unique_ke_components_absolute'] = fig_abs
        _plot_figure_cache['unique_ke_components_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta)
        )

    except Exception as e:
        logger.error(f"Failed to generate unique KE component plots: {str(e)}")
        return (
            create_fallback_plot("Unique KE Component Annotations Over Time", str(e)),
            create_fallback_plot("Change in Unique KE Component Annotations", str(e))
        )


def plot_bio_processes() -> tuple[str, str]:
    """Generate biological process ontology usage visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        query_ontologies = f"""
        SELECT ?graph ?ontology (COUNT(DISTINCT ?process) AS ?count)
        WHERE {{
          GRAPH ?graph {{
            ?ke a aopo:KeyEvent ;
                aopo:hasBiologicalEvent ?bioevent .
            ?bioevent aopo:hasProcess ?process .
            FILTER(isIRI(?process))
            BIND({ontology_bucket_sparql("process")} AS ?ontology)
          }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph ?ontology
        ORDER BY ?graph ?ontology
        """

        results_ont = run_sparql_query(query_ontologies)

        if not results_ont:
            logger.warning("Biological process annotations query returned no results")
            return (
                create_fallback_plot("Biological Process Annotations by Ontology Over Time", "No data available"),
                create_fallback_plot("Change in Biological Process Annotations by Ontology", "No data available")
            )

        df_ont = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "ontology": r["ontology"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_ont if "ontology" in r])

        if df_ont.empty:
            logger.warning("Biological process annotations DataFrame is empty")
            return (
                create_fallback_plot("Biological Process Annotations by Ontology Over Time", "No data available"),
                create_fallback_plot("Change in Biological Process Annotations by Ontology", "No data available")
            )

        # Sort by datetime
        df_ont["version_dt"] = pd.to_datetime(df_ont["version"], errors="coerce")
        df_ont = df_ont.sort_values("version_dt").drop(columns="version_dt")

        # --- Absolute line chart ---
        # An ordered time axis with 8 ontology series reads as lines, not a forest
        # of ~256 thin grouped bars where the small-count ontologies vanished
        # (#plot-review). Marker shapes back up colour so the low-count series stay
        # tellable apart.
        fig_abs = px.line(
            df_ont,
            x="version",
            y="count",
            color="ontology",
            markers=True,
            labels={"count": "Distinct terms", "ontology": "Ontology"},
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        _apply_marker_symbols(fig_abs)
        apply_snapshot_xaxis(fig_abs)

        # --- Delta calculation ---
        df_delta = df_ont.copy()
        df_delta["count"] = df_delta.groupby("ontology")["count"].diff().fillna(0)

        fig_delta = px.line(
            df_delta,
            x="version",
            y="count",
            color="ontology",
            markers=True,
            labels={"count": "Change in distinct terms", "ontology": "Ontology"},
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        _apply_marker_symbols(fig_delta)
        apply_snapshot_xaxis(fig_delta)

        # Cache data for CSV export
        _plot_data_cache['biological_process_annotations_absolute'] = df_ont.copy()
        _plot_data_cache['biological_process_annotations_delta'] = df_delta.copy()

        # Cache figures for image export
        _plot_figure_cache['biological_process_annotations_absolute'] = fig_abs
        _plot_figure_cache['biological_process_annotations_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta)
        )

    except Exception as e:
        logger.error(f"Failed to generate biological process annotation plots: {str(e)}")
        return (
            create_fallback_plot("Biological Process Annotations by Ontology Over Time", str(e)),
            create_fallback_plot("Change in Biological Process Annotations by Ontology", str(e))
        )


def plot_bio_objects() -> tuple[str, str]:
    """Generate biological object ontology usage visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        query_objects = f"""
        SELECT ?graph ?ontology (COUNT(DISTINCT ?object) AS ?count)
        WHERE {{
          GRAPH ?graph {{
            ?ke a aopo:KeyEvent ;
                aopo:hasBiologicalEvent ?bioevent .
            ?bioevent aopo:hasObject ?object .
            FILTER(isIRI(?object))
            BIND({ontology_bucket_sparql("object")} AS ?ontology)
          }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph ?ontology
        ORDER BY ?graph ?ontology
        """

        results_obj = run_sparql_query(query_objects)

        if not results_obj:
            logger.warning("Biological object annotations query returned no results")
            return (
                create_fallback_plot("Biological Object Annotations by Ontology Over Time", "No data available"),
                create_fallback_plot("Change in Biological Object Annotations by Ontology", "No data available")
            )

        df_obj = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "ontology": r["ontology"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_obj if "ontology" in r])

        if df_obj.empty:
            logger.warning("Biological object annotations DataFrame is empty")
            return (
                create_fallback_plot("Biological Object Annotations by Ontology Over Time", "No data available"),
                create_fallback_plot("Change in Biological Object Annotations by Ontology", "No data available")
            )

        df_obj["version_dt"] = pd.to_datetime(df_obj["version"], errors="coerce")
        df_obj = df_obj.sort_values("version_dt").drop(columns="version_dt")

        # --- Absolute line chart ---
        # One line per ontology over the ordered time axis, replacing the dense
        # grouped-bar forest where CHEBI/GO's near-identical blues collided and
        # the small ontologies collapsed to slivers (#plot-review). Marker shapes
        # reinforce colour.
        fig_abs = px.line(
            df_obj,
            x="version",
            y="count",
            color="ontology",
            markers=True,
            labels={"count": "Distinct terms", "ontology": "Ontology"},
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        _apply_marker_symbols(fig_abs)
        apply_snapshot_xaxis(fig_abs)

        # --- Delta calculation ---
        df_delta = df_obj.copy()
        df_delta["count"] = df_delta.groupby("ontology")["count"].diff().fillna(0)

        fig_delta = px.line(
            df_delta,
            x="version",
            y="count",
            color="ontology",
            markers=True,
            labels={"count": "Change in distinct terms", "ontology": "Ontology"},
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        _apply_marker_symbols(fig_delta)
        apply_snapshot_xaxis(fig_delta)

        # Cache data for CSV export
        _plot_data_cache['biological_object_annotations_absolute'] = df_obj.copy()
        _plot_data_cache['biological_object_annotations_delta'] = df_delta.copy()

        # Cache figures for image export
        _plot_figure_cache['biological_object_annotations_absolute'] = fig_abs
        _plot_figure_cache['biological_object_annotations_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta)
        )

    except Exception as e:
        logger.error(f"Failed to generate biological object annotation plots: {str(e)}")
        return (
            create_fallback_plot("Biological Object Annotations by Ontology Over Time", str(e)),
            create_fallback_plot("Change in Biological Object Annotations by Ontology", str(e))
        )


# Entity types the property-presence trend covers: RDF class, the per-version
# total column in the exported table, and the wording of titles and axes.
PROPERTY_PRESENCE_ENTITIES = {
    "aop": {"cls": "aopo:AdverseOutcomePathway", "total": "total_aops",
            "plural": "AOPs", "short": "AOPs", "label": "AOP"},
    "ke": {"cls": "aopo:KeyEvent", "total": "total_kes",
           "plural": "Key Events", "short": "KEs", "label": "KE"},
    "ker": {"cls": "aopo:KeyEventRelationship", "total": "total_kers",
            "plural": "Key Event Relationships", "short": "KERs", "label": "KER"},
    "stressor": {"cls": "nci:C54571", "total": "total_stressors",
                 "plural": "Stressors", "short": "Stressors", "label": "Stressor"},
}


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
PROPERTY_HEATMAP_COLUMNS = ["version", "property", "label", "type", "filled", "total",
                            "percentage", "first_filled"]
# Absent stays white; anything above 0% starts at the lightest brand blue, so a
# rarely used field never looks the same as an unused one.
PROPERTY_HEATMAP_SCALE = [[0.0, "#FFFFFF"], [1e-6, BRAND_COLORS["sky_blue"]],
                          [0.5, BRAND_COLORS["blue"]], [1.0, BRAND_COLORS["primary"]]]


def property_heatmap_rows(df_props: pd.DataFrame, df_total: pd.DataFrame,
                          df_labels: pd.DataFrame) -> pd.DataFrame:
    """One row per (snapshot, property): how many entities fill the property in.

    Unlike the presence lines, properties at 100% are kept, since "always there"
    answers the heatmap's question too; only rdf:type is left out. A property
    missing from a snapshot is a 0 row, and ``first_filled`` is the first
    snapshot where any entity has it. Rows are in display order: by property
    type (``PROPERTY_TYPE_ORDER``, unlabelled last), then label, then snapshot.
    """
    from .shared import PROPERTY_TYPE_ORDER

    props = sorted(set(df_props["property"]) - {RDF_TYPE})
    versions = sorted(df_total["version"])
    if not props or not versions:
        return pd.DataFrame(columns=PROPERTY_HEATMAP_COLUMNS)
    counts = df_props.set_index(["version", "property"])["count"]
    totals = df_total.set_index("version").iloc[:, 0]
    labels = {} if df_labels.empty else df_labels.set_index("uri")[["label", "type"]].to_dict("index")

    rows = []
    for prop in props:
        meta = labels.get(prop, {})
        label = meta.get("label") if isinstance(meta.get("label"), str) else humanize_predicate_uri(prop)
        kind = meta.get("type") if isinstance(meta.get("type"), str) else "Other"
        first = ""
        for version in versions:
            filled = int(counts.get((version, prop), 0))
            if filled and not first:
                first = version
            total = int(totals.get(version, 0))
            rows.append({"version": version, "property": prop, "label": label, "type": kind,
                         "filled": filled, "total": total,
                         "percentage": round(filled / total * 100, 1) if total else 0.0})
        for row in rows[-len(versions):]:
            row["first_filled"] = first
    df = pd.DataFrame(rows, columns=PROPERTY_HEATMAP_COLUMNS)
    rank = {t: i for i, t in enumerate(PROPERTY_TYPE_ORDER)}
    df["_k"] = df["type"].map(lambda t: rank.get(t, len(rank)))
    return df.sort_values(["_k", "label", "version"], kind="stable").drop(columns="_k").reset_index(drop=True)


def _render_property_heatmap(df: pd.DataFrame, cfg: dict) -> go.Figure:
    """Property x snapshot heatmap: rows grouped by property type, colour = % filled in."""
    order = list(dict.fromkeys(df["label"]))
    versions = sorted(df["version"].unique())
    cell = df.set_index(["label", "version"])
    z = [[cell.loc[(lab, v), "percentage"] for v in versions] for lab in order]
    text = [[f"{cell.loc[(lab, v), 'filled']} of {cell.loc[(lab, v), 'total']}" for v in versions] for lab in order]
    meta = df.drop_duplicates("label").set_index("label")
    custom = [[[meta.loc[lab, "type"], meta.loc[lab, "first_filled"] or "never"] for _ in versions] for lab in order]

    fig = go.Figure(go.Heatmap(
        x=versions, y=order, z=z, text=text, customdata=custom,
        zmin=0, zmax=100, xgap=1, ygap=1, colorscale=PROPERTY_HEATMAP_SCALE,
        colorbar=dict(title=dict(text="% filled in", side="right"), ticksuffix="%", thickness=12, len=0.8),
        hovertemplate=("<b>%{y}</b> (%{customdata[0]})<br>%{x}: %{z:.1f}% of "
                       + cfg["short"] + " (%{text})<br>First filled in: %{customdata[1]}<extra></extra>"),
    ))
    # A line between property types, so the groups read without a legend.
    kinds = list(meta.loc[order, "type"])
    shapes = [dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=i - 0.5, y1=i - 0.5,
                   line=dict(color=BRAND_COLORS["primary"], width=1.5))
              for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1]]
    apply_snapshot_xaxis(fig)
    fig.update_yaxes(autorange="reversed", title_text=None, ticks="", automargin=True)
    fig.update_layout(
        title=dict(text=f"{len(order)} {cfg['short']} properties across {len(versions)} snapshots"),
        height=max(260, 22 * len(order) + 150), shapes=shapes,
        margin=dict(l=10, r=10, t=60, b=60), plot_bgcolor="white",
    )
    return fig


def plot_property_presence(entity: str, label_file="property_labels.csv") -> tuple[str, str, str]:
    """Property presence over time for one entity type: count and percentage lines,
    plus a property x snapshot heatmap (#187).

    One function for AOPs, KEs, KERs and Stressors (#178); ``entity`` is a key
    of ``PROPERTY_PRESENCE_ENTITIES``. Uses marker shapes to differentiate
    properties when colors repeat, ensuring visual distinction across all
    properties.
    """
    global _plot_data_cache, _plot_figure_cache

    cfg = PROPERTY_PRESENCE_ENTITIES[entity]
    title = f"Property Presence in {cfg['plural']} Over Time"
    total_col = cfg["total"]

    def fallback(message):
        return (
            create_fallback_plot(f"{title} (Count)", message),
            create_fallback_plot(f"{title} (Percentage)", message),
            create_fallback_plot(f"{cfg['label']} Properties by Snapshot", message),
        )

    try:
        results_props = run_sparql_query(f"""
        SELECT ?graph ?p (COUNT(DISTINCT ?s) AS ?count)
        WHERE {{
          GRAPH ?graph {{ ?s a {cfg['cls']} ; ?p ?o . }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph ?p ORDER BY ?graph ?p
        """)
        results_total = run_sparql_query(f"""
        SELECT ?graph (COUNT(DISTINCT ?s) AS ?total)
        WHERE {{
          GRAPH ?graph {{ ?s a {cfg['cls']} . }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph ORDER BY ?graph
        """)

        if not results_props or not results_total:
            logger.warning(f"{cfg['label']} property presence query returned no results")
            return fallback("No data available")

        df_props = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            "property": r["p"]["value"],
            "count": int(r["count"]["value"])
        } for r in results_props])
        df_total = pd.DataFrame([{
            "version": r["graph"]["value"].split("/")[-1],
            total_col: int(r["total"]["value"])
        } for r in results_total])

        if df_props.empty or df_total.empty:
            return fallback("No data available")

        # The heatmap keeps every property, so it is built before the 100% filter.
        df_heat = property_heatmap_rows(df_props, df_total, safe_read_csv(label_file, []))

        df = df_props.merge(df_total, on="version", how="left")
        # Guard against division by zero in percentage calculation
        df["percentage"] = df.apply(
            lambda row: (row["count"] / row[total_col]) * 100 if row[total_col] > 0 else 0, axis=1
        )

        # Remove properties that are 100% in all versions
        props_to_keep = df.groupby("property")["percentage"].max().loc[lambda x: x < 100].index
        df = df[df["property"].isin(props_to_keep)]

        # Ensure complete data: fill missing property-version combinations with 0
        if not df.empty:
            all_versions = sorted(df['version'].unique())
            all_props = sorted(df['property'].unique())
            complete_index = pd.MultiIndex.from_product([all_versions, all_props], names=['version', 'property'])
            df_complete = df.set_index(['version', 'property']).reindex(complete_index, fill_value=0).reset_index()
            # The re-merge suffixes the total column (_x holds the reindex
            # zeros, _y the real totals); the export keeps both, as it always has.
            df_complete = df_complete.merge(df_total, on="version", how="left")
            merged_total = [col for col in df_complete.columns if col.startswith('total_')][0]
            df_complete["percentage"] = df_complete.apply(
                lambda row: (row["count"] / row[merged_total]) * 100 if row[merged_total] > 0 else 0, axis=1
            )
            df = df_complete

        # Label mapping with safe file reading
        default_labels = [
            {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential"},
            {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata"}
        ]
        df_labels = safe_read_csv(label_file, default_labels)
        if not df_labels.empty:
            df = df.merge(df_labels, how="left", left_on="property", right_on="uri")
            # Curated label where property_labels.csv has one; otherwise a
            # humanized last-resort label so an unmapped predicate never renders
            # as a raw URI (#130).
            df["display_label"] = df["label"].fillna(
                df["property"].map(humanize_predicate_uri)
            )
        else:
            df["display_label"] = df["property"].map(humanize_predicate_uri)

        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")

        # Raw 0s intact for the CSV/raw-data export.
        _plot_data_cache[f'{entity}_property_presence_absolute'] = df.copy()
        _plot_data_cache[f'{entity}_property_presence_percentage'] = df.copy()

        # Small multiples, one panel per property (#130). Mask leading zeros so a
        # property's line starts when it first appears, not from a flat zero run.
        plot_df = mask_property_prefix_zeros(df)
        fig_abs = build_property_presence_figure(
            plot_df, "count", f"Number of {cfg['short']}", cfg["label"])
        fig_pct = build_property_presence_figure(
            plot_df, "percentage", "Percentage (%)", cfg["label"], percentage=True)

        # Cache figures for image export
        _plot_figure_cache[f'{entity}_property_presence_absolute'] = fig_abs
        _plot_figure_cache[f'{entity}_property_presence_percentage'] = fig_pct

        fig_heat = _render_property_heatmap(df_heat, cfg)
        _plot_data_cache[f'{entity}_property_heatmap'] = df_heat
        _plot_figure_cache[f'{entity}_property_heatmap'] = fig_heat

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_pct),
            render_plot_html(fig_heat),
        )

    except SparqlUnavailable:
        raise
    except Exception as e:
        logger.error(f"Failed to generate {cfg['label']} property presence plots: {str(e)}")
        return fallback(str(e))


def plot_kes_by_kec_count() -> tuple[str, str]:
    """Generate KE distribution by KEC count visualization with absolute and delta views."""
    global _plot_data_cache, _plot_figure_cache

    try:
        # Count per (graph, KE) first and bucket afterwards. The previous form
        # nested an ungrouped sub-SELECT inside GRAPH ?graph, so each KE's count
        # was taken across every version at once and leaked between graphs
        # (#168). SUM(IF(BOUND(...))) rather than COUNT(DISTINCT ?bioevent):
        # over several graphs Virtuoso counts an unmatched OPTIONAL as 1, which
        # folds the 0 bucket into 1. Checked against per-graph COUNTs for all 35
        # graphs.
        query_kec_count = """
        PREFIX aopo: <http://aopkb.org/aop_ontology#>

        SELECT ?graph ?bioevent_count_group (COUNT(?ke) AS ?total_kes)
        WHERE {
          {
            SELECT ?graph ?ke (SUM(IF(BOUND(?bioevent), 1, 0)) AS ?bioevent_count)
            WHERE {
              GRAPH ?graph {
                ?ke a aopo:KeyEvent .
                OPTIONAL { ?ke aopo:hasBiologicalEvent ?bioevent . }
              }
              FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }
            GROUP BY ?graph ?ke
          }
          BIND(IF(?bioevent_count >= 6, "6+", STR(?bioevent_count)) AS ?bioevent_count_group)
        }
        GROUP BY ?graph ?bioevent_count_group
        ORDER BY ?graph ?bioevent_count_group
        """

        results = run_sparql_query(query_kec_count)

        data = []
        for r in results:
            if "graph" in r and "bioevent_count_group" in r and "total_kes" in r:
                data.append({
                    "version": r["graph"]["value"].split("/")[-1],
                    "bioevent_count_group": r["bioevent_count_group"]["value"],
                    "total_kes": int(r["total_kes"]["value"])
                })

        if not data:
            logger.warning("No data for KE by KEC count plot")
            return (
                create_fallback_plot("KE Distribution by Component Count Over Time", "No data available"),
                create_fallback_plot("Change in KE Distribution by Component Count", "No data available")
            )

        df = pd.DataFrame(data)

        all_versions = sorted(df["version"].unique())
        all_groups = sorted(df["bioevent_count_group"].unique())

        idx = pd.MultiIndex.from_product([all_versions, all_groups],
                                         names=["version", "bioevent_count_group"])
        df_full = df.set_index(["version", "bioevent_count_group"]).reindex(idx, fill_value=0).reset_index()

        df_full["version_dt"] = pd.to_datetime(df_full["version"], errors="coerce")
        df_full = df_full.sort_values(["version_dt", "bioevent_count_group"]).drop(columns="version_dt")

        fig_abs = px.area(
            df_full, x="version", y="total_kes", color="bioevent_count_group",
            labels={"total_kes": "Number of KEs", "bioevent_count_group": "Number of Components"},
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_abs.update_layout(margin=dict(l=50, r=20, t=50, b=50),
                              xaxis=dict(tickmode='array', tickvals=all_versions, ticktext=all_versions, tickangle=-45))

        df_delta = df_full.copy()
        df_delta["total_kes_delta"] = df_delta.groupby("bioevent_count_group")["total_kes"].diff().fillna(0)

        # The delta is signed (several snapshots dip below zero), which breaks the
        # additive, non-negative assumption of a stacked area. Relative bars stack
        # positive changes up and negative changes down from the baseline, so the
        # composition reads honestly (#plot-review).
        fig_delta = px.bar(
            df_delta, x="version", y="total_kes_delta", color="bioevent_count_group",
            barmode="relative",
            labels={"total_kes_delta": "Change in KEs", "bioevent_count_group": "Number of Components"},
            color_discrete_sequence=BRAND_COLORS['palette']
        )
        fig_delta.update_layout(margin=dict(l=50, r=20, t=50, b=50),
                                xaxis=dict(tickmode='array', tickvals=all_versions, ticktext=all_versions, tickangle=-45))

        _plot_data_cache['kes_by_kec_count_absolute'] = df_full.copy()
        _plot_data_cache['kes_by_kec_count_delta'] = df_delta.copy()
        _plot_figure_cache['kes_by_kec_count_absolute'] = fig_abs
        _plot_figure_cache['kes_by_kec_count_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta)
        )

    except Exception as e:
        logger.error(f"Failed to generate KE by KEC count plots: {str(e)}")
        return (
            create_fallback_plot("KE Distribution by Component Count Over Time", str(e)),
            create_fallback_plot("Change in KE Distribution by Component Count", str(e))
        )


def plot_entity_completeness_trends(label_file="property_labels.csv") -> str:
    """Generate entity completeness trend visualization showing average completeness percentage over time.

    Calculates average completeness for AOPs, KEs, KERs, and Stressors across all versions.
    Excludes properties that are 100% present in any version (e.g., rdf:type, mandatory fields)
    to focus on optional/variable properties that reflect true data richness.

    Completeness is calculated per entity as (properties present / non-ubiquitous properties) × 100,
    then averaged across all entities of that type in each version.

    Args:
        label_file: Path to CSV file containing property labels and types

    Returns:
        HTML string containing the Plotly visualization
    """
    global _plot_data_cache, _plot_figure_cache

    # Load property labels
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential", "applies_to": "AOP|KE|KER|Stressor"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata", "applies_to": "AOP|KE|KER"}
    ]
    df_labels = safe_read_csv(label_file, default_properties)

    if df_labels.empty:
        return create_fallback_plot("Entity Completeness Trends", "No property labels available")

    # Entity type configurations
    entity_configs = [
        {
            "name": "AOP",
            "rdf_type": "aopo:AdverseOutcomePathway",
            "variable": "?aop",
            "label": "AOPs"
        },
        {
            "name": "KE",
            "rdf_type": "aopo:KeyEvent",
            "variable": "?ke",
            "label": "Key Events"
        },
        {
            "name": "KER",
            "rdf_type": "aopo:KeyEventRelationship",
            "variable": "?ker",
            "label": "Key Event Relationships"
        },
        {
            "name": "Stressor",
            "rdf_type": "nci:C54571",
            "variable": "?stressor",
            "label": "Stressors"
        }
    ]

    all_completeness_data = []

    for config in entity_configs:
        entity_type = config["name"]
        rdf_type = config["rdf_type"]
        variable = config["variable"]
        display_label = config["label"]

        # Filter properties applicable to this entity type
        applicable_props = df_labels[
            df_labels["applies_to"].fillna("").str.contains(entity_type, case=False, na=False)
        ]

        if applicable_props.empty:
            continue

        applicable_uris = set(applicable_props["uri"].tolist())

        if not applicable_uris:
            continue

        # Query to identify 100% present properties (to exclude them)
        # First get total entity counts per version
        uri_filter_presence = ">, <".join(sorted(applicable_uris))
        query_total = f"""
        SELECT ?graph (COUNT(DISTINCT {variable}) AS ?total)
        WHERE {{
          GRAPH ?graph {{
            {variable} a {rdf_type} .
          }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph
        ORDER BY ?graph
        """

        query_presence = f"""
        SELECT ?graph ?p (COUNT(DISTINCT {variable}) AS ?count)
        WHERE {{
          GRAPH ?graph {{
            {variable} a {rdf_type} ;
                 ?p ?o .
            FILTER(?p IN (<{uri_filter_presence}>))
          }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph ?p
        ORDER BY ?graph ?p
        """

        # True entity totals per version — used as the completeness denominator so
        # entities with none of the counted properties count as 0% (not dropped).
        totals_map = {}
        try:
            results_total = run_sparql_query(query_total)
            results_presence = run_sparql_query(query_presence)

            # Build totals map
            for r in results_total:
                version = r["graph"]["value"].split("/")[-1]
                totals_map[version] = int(r["total"]["value"])

            # Calculate property presence percentages
            prop_presence = {}
            for r in results_presence:
                version = r["graph"]["value"].split("/")[-1]
                prop_uri = r["p"]["value"]
                count = int(r["count"]["value"])
                total = totals_map.get(version, 1)
                percentage = (count / total) * 100

                if prop_uri not in prop_presence:
                    prop_presence[prop_uri] = []
                prop_presence[prop_uri].append(percentage)

            # Filter out properties that reach 100% in any version
            props_to_exclude = {
                prop_uri for prop_uri, percentages in prop_presence.items()
                if max(percentages) >= 100
            }

            # Update applicable URIs to exclude 100% properties
            applicable_uris = applicable_uris - props_to_exclude
            total_properties = len(applicable_uris)

            if total_properties == 0:
                logger.info(f"All properties for {entity_type} are 100% present, skipping completeness calculation")
                continue

        except SparqlUnavailable:
            raise
        except Exception as e:
            logger.warning(f"Error filtering 100% properties for {entity_type}: {str(e)}, proceeding with all properties")
            total_properties = len(applicable_uris)

        # Query to get property counts per entity per version (using filtered properties)
        uri_filter = ">, <".join(sorted(applicable_uris))
        query = f"""
        SELECT ?graph {variable} (COUNT(DISTINCT ?p) AS ?prop_count)
        WHERE {{
          GRAPH ?graph {{
            {variable} a {rdf_type} ;
                 ?p ?o .
            FILTER(?p IN (<{uri_filter}>))
          }}
          FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
        }}
        GROUP BY ?graph {variable}
        ORDER BY ?graph {variable}
        """

        try:
            results = run_sparql_query(query)

            if not results:
                continue

            # Process results to calculate average completeness per version
            version_data = {}
            for result in results:
                version = result["graph"]["value"].split("/")[-1]
                prop_count = int(result["prop_count"]["value"])
                completeness = (prop_count / total_properties) * 100

                if version not in version_data:
                    version_data[version] = []
                version_data[version].append(completeness)

            # Average completeness per version over ALL entities of the type:
            # entities absent from `completeness_list` have none of the counted
            # properties and must count as 0%, so divide by the true entity total
            # (falls back to the observed count if the total query returned nothing).
            for version, completeness_list in version_data.items():
                total_entities = totals_map.get(version) or len(completeness_list)
                avg_completeness = sum(completeness_list) / total_entities
                all_completeness_data.append({
                    "version": version,
                    "entity_type": display_label,
                    "avg_completeness": avg_completeness,
                    "entity_count": total_entities
                })

        except Exception as e:
            logger.error(f"Error querying completeness for {entity_type}: {str(e)}")
            # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
            raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

    if not all_completeness_data:
        return create_fallback_plot("Entity Completeness Trends", "No completeness data available")

    # Create DataFrame
    df = pd.DataFrame(all_completeness_data)

    # Sort by version
    df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
    df = df.sort_values("version_dt")

    # Cache data for CSV export
    _plot_data_cache['entity_completeness_trends'] = df.copy()

    # Create line plot
    fig = px.line(
        df,
        x="version",
        y="avg_completeness",
        color="entity_type",
        markers=True,
        labels={
            "avg_completeness": "Average Completeness (%)",
            "entity_type": "Entity Type",
            "version": "Version"
        },
        color_discrete_sequence=BRAND_COLORS['palette']
    )

    # Apply marker shapes for visual distinction
    marker_symbols = ['circle', 'square', 'diamond', 'triangle-up']
    for i, trace in enumerate(fig.data):
        trace.update(
            marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=10),
            line=dict(width=3)
        )

    fig.update_layout(
        margin=dict(l=50, r=20, t=50, b=50),
        yaxis=dict(title="Average Completeness (%)", range=[0, 105]),
        xaxis=dict(title="Version", tickangle=-45),
    )

    # Cache figure for image export
    _plot_figure_cache['entity_completeness_trends'] = fig

    return render_plot_html(fig)


def _query_boxplot_version(version_info, aop_props, ke_props, ker_props, all_props):
    """Query boxplot data for a single version.

    Runs three scoped queries per version:
    1. Entity totals (AOP, KE, KER counts)
    2. Property presence per entity type
    3. AOP network structure (KEs/KERs per AOP)

    Returns:
        Dict with 'version', 'totals', 'presence', 'network' data, or None on error
    """
    graph_uri = version_info["graph_uri"]
    version_str = version_info["version"]

    all_props_filter = ">, <".join(sorted(all_props))

    # Query 1: Entity totals for this version
    query_totals = f"""
    SELECT ?entity_type (COUNT(DISTINCT ?entity) AS ?total)
    WHERE {{
      GRAPH <{graph_uri}> {{
        {{
          ?entity a aopo:AdverseOutcomePathway .
          BIND("AOP" AS ?entity_type)
        }}
        UNION
        {{
          ?entity a aopo:KeyEvent .
          BIND("KE" AS ?entity_type)
        }}
        UNION
        {{
          ?entity a aopo:KeyEventRelationship .
          BIND("KER" AS ?entity_type)
        }}
      }}
    }}
    GROUP BY ?entity_type
    """

    # Query 2: Property presence per entity type
    query_presence = f"""
    SELECT ?entity_type ?p (COUNT(DISTINCT ?entity) AS ?count)
    WHERE {{
      GRAPH <{graph_uri}> {{
        {{
          ?entity a aopo:AdverseOutcomePathway ;
                  ?p ?o .
          BIND("AOP" AS ?entity_type)
        }}
        UNION
        {{
          ?entity a aopo:KeyEvent ;
                  ?p ?o .
          BIND("KE" AS ?entity_type)
        }}
        UNION
        {{
          ?entity a aopo:KeyEventRelationship ;
                  ?p ?o .
          BIND("KER" AS ?entity_type)
        }}
        FILTER(?p IN (<{all_props_filter}>))
      }}
    }}
    GROUP BY ?entity_type ?p
    """

    # Query 3: AOP network structure
    query_network = f"""
    SELECT ?aop
           (GROUP_CONCAT(DISTINCT ?ke; separator="|") AS ?kes)
           (GROUP_CONCAT(DISTINCT ?ker; separator="|") AS ?kers)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {{ ?aop aopo:has_key_event ?ke }}
        OPTIONAL {{ ?aop aopo:has_key_event_relationship ?ker }}
      }}
    }}
    GROUP BY ?aop
    """

    try:
        results_totals = run_sparql_query_with_retry(query_totals, max_retries=2)
        results_presence = run_sparql_query_with_retry(query_presence, max_retries=2)
        results_network = run_sparql_query_with_retry(query_network, max_retries=2)

        return {
            "version": version_str,
            "totals": results_totals or [],
            "presence": results_presence or [],
            "network": results_network or []
        }
    except Exception as e:
        logger.warning(f"Boxplot query failed for version {version_str}: {e}")
        # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
        raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e


def _query_boxplot_entity_props(version_info, aop_props, ke_props, ker_props,
                                aop_prop_count, ke_prop_count, ker_prop_count):
    """Query per-entity property counts for a single version for composite score calculation.

    Returns:
        Dict with 'version', 'entity_props' (entity completeness data), 'network', or None on error
    """
    graph_uri = version_info["graph_uri"]
    version_str = version_info["version"]

    # Build UNION parts for entity property counts
    union_parts = []

    if aop_prop_count > 0:
        aop_uri_filter = ">, <".join(sorted(aop_props))
        union_parts.append(f"""
          {{
            ?entity a aopo:AdverseOutcomePathway ;
                 ?p ?o .
            FILTER(?p IN (<{aop_uri_filter}>))
            BIND("AOP" AS ?entity_type)
          }}
        """)

    if ke_prop_count > 0:
        ke_uri_filter = ">, <".join(sorted(ke_props))
        union_parts.append(f"""
          {{
            ?entity a aopo:KeyEvent ;
                ?p ?o .
            FILTER(?p IN (<{ke_uri_filter}>))
            BIND("KE" AS ?entity_type)
          }}
        """)

    if ker_prop_count > 0:
        ker_uri_filter = ">, <".join(sorted(ker_props))
        union_parts.append(f"""
          {{
            ?entity a aopo:KeyEventRelationship ;
                 ?p ?o .
            FILTER(?p IN (<{ker_uri_filter}>))
            BIND("KER" AS ?entity_type)
          }}
        """)

    if not union_parts:
        return {"version": version_str, "entity_props": [], "network": []}

    union_clause = "\nUNION\n".join(union_parts)

    # Query per-entity property counts
    query_props = f"""
    SELECT ?entity (COUNT(DISTINCT ?p) AS ?prop_count) ?entity_type
    WHERE {{
      GRAPH <{graph_uri}> {{
        {union_clause}
      }}
    }}
    GROUP BY ?entity ?entity_type
    """

    # Query AOP network structure
    query_network = f"""
    SELECT ?aop
           (GROUP_CONCAT(DISTINCT ?ke; separator="|") AS ?kes)
           (GROUP_CONCAT(DISTINCT ?ker; separator="|") AS ?kers)
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {{ ?aop aopo:has_key_event ?ke }}
        OPTIONAL {{ ?aop aopo:has_key_event_relationship ?ker }}
      }}
    }}
    GROUP BY ?aop
    """

    try:
        results_props = run_sparql_query_with_retry(query_props, max_retries=2)
        results_network = run_sparql_query_with_retry(query_network, max_retries=2)

        return {
            "version": version_str,
            "entity_props": results_props or [],
            "network": results_network or []
        }
    except Exception as e:
        logger.warning(f"Boxplot entity props query failed for version {version_str}: {e}")
        # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
        raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e


# --- AOP completeness beeswarm (#186) ------------------------------------------

BEESWARM_COLUMNS = ["version", "aop", "aop_id", "aop_title",
                    "completeness_optional", "completeness_all", "entity_count"]
BEESWARM_RADIUS = 4.5    # dot radius in px (static/js/viz/beeswarm.js uses the same)
BEESWARM_WIDTH = 1000    # plot-area width in px the server figure is laid out for


def _query_aop_titles(version_info) -> dict[str, str]:
    """AOP IRI -> title in one snapshot (the newest), for the beeswarm's hover."""
    query = f"""
    SELECT ?aop ?title
    WHERE {{
      GRAPH <{version_info["graph_uri"]}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             dc:title ?title .
      }}
    }}
    """
    rows = run_sparql_query_with_retry(query, max_retries=2)
    return {r["aop"]["value"]: r["title"]["value"] for r in rows}


def _aop_sort_number(iri: str) -> int:
    m = re.search(r"(\d+)$", iri)
    return int(m.group(1)) if m else 10**9


def beeswarm_rows(df_optional: pd.DataFrame, df_all: pd.DataFrame,
                  titles: dict[str, str]) -> pd.DataFrame:
    """One row per AOP per snapshot with both cohorts' scores (#186).

    The beeswarm keys its dots on ``aop``, so the same AOP is the same dot in
    every snapshot. Rows are in snapshot order, then AOP number.
    """
    key = ["version", "aop_uri"]
    opt = df_optional[key + ["completeness"]].rename(columns={"completeness": "completeness_optional"})
    full = df_all[key + ["completeness", "entity_count"]].rename(columns={"completeness": "completeness_all"})
    df = opt.merge(full, on=key, how="outer").rename(columns={"aop_uri": "aop"})
    df["aop_id"] = df["aop"].map(lambda iri: f"AOP {_aop_sort_number(iri)}")
    df["aop_title"] = df["aop"].map(lambda iri: titles.get(iri, ""))
    for col in ("completeness_optional", "completeness_all"):
        df[col] = df[col].round(2)
    df["entity_count"] = df["entity_count"].fillna(0).astype(int)
    df["_n"] = df["aop"].map(_aop_sort_number)
    df = df.sort_values(["version", "_n", "aop"], kind="mergesort").drop(columns="_n")
    return df[BEESWARM_COLUMNS].reset_index(drop=True)


def swarm_offsets(xs, radius: float) -> list[float]:
    """Vertical offsets for dots at pixel positions ``xs`` so that none overlap.

    Each dot, taken left to right (ties in input order), goes as close to the
    axis as it can without touching a dot already placed: the classic
    "dodge" beeswarm. static/js/viz/beeswarm.js has the same function, so the
    server figure and the page agree.
    """
    order = sorted(range(len(xs)), key=lambda i: (xs[i], i))
    placed: list[tuple[float, float]] = []   # (x, y), in placing order (x ascending)
    out = [0.0] * len(xs)
    d = 2 * radius
    start = 0
    for i in order:
        x = xs[i]
        while start < len(placed) and placed[start][0] < x - d:
            start += 1
        near = placed[start:]
        candidates = [0.0]
        for px, py in near:
            dy = (d * d - (x - px) ** 2) ** 0.5
            candidates += [py + dy, py - dy]
        candidates.sort(key=lambda y: (abs(y), -y))
        for y in candidates:
            if all((x - px) ** 2 + (y - py) ** 2 >= d * d - 1e-6 for px, py in near):
                break
        out[i] = y
        placed.append((x, y))
    return out


def _render_beeswarm(df: pd.DataFrame, version: str, column: str, y_title: str) -> go.Figure:
    """The beeswarm of one snapshot, for the PNG/SVG downloads and /api/v2/figure."""
    rows = df[df["version"] == version].dropna(subset=[column])
    xs = [v / 100 * BEESWARM_WIDTH for v in rows[column]]
    ys = swarm_offsets(xs, BEESWARM_RADIUS)
    reach = max([abs(y) for y in ys] + [0]) + 2 * BEESWARM_RADIUS
    fig = go.Figure(go.Scatter(
        x=rows[column], y=ys, mode="markers", ids=rows["aop"],
        customdata=rows[["aop", "aop_id", "aop_title"]].values,
        marker=dict(size=2 * BEESWARM_RADIUS, color=BRAND_COLORS["primary"], opacity=0.75,
                    line=dict(width=0.5, color="white")),
        hovertemplate="<b>%{customdata[1]}</b> %{customdata[2]}<br>%{x:.1f}%<extra></extra>",
    ))
    median = rows[column].median() if len(rows) else 0
    fig.update_layout(
        title=dict(text=f"{len(rows)} AOPs in {version}, median {median:.1f}%", x=0.01),
        # One y unit is one pixel, so the dots stay round and apart.
        height=max(240, int(2 * reach) + 110), margin=dict(l=20, r=20, t=50, b=60), showlegend=False,
        xaxis=dict(title=y_title, range=[-2, 102], zeroline=False),
        yaxis=dict(visible=False, range=[-reach, reach], fixedrange=True),
        hovermode="closest",
    )
    return fig


def plot_aop_completeness_boxplot(label_file="property_labels.csv") -> tuple[str, str, str]:
    """Render two side-by-side composite-completeness boxplots (optional-only and
    all-properties), and the AOP-by-AOP beeswarm built from the same scores (#186).

    The composite score per AOP is the flat average of the AOP entity's own
    property completeness with the completeness of every KE and KER it links
    to, computed against a per-entity-type property denominator (see
    ``property_labels.csv``).

    Two cohorts are returned so the reader can see the discriminating signal
    AND the headline number:

    - **Optional-only** (left): excludes properties that reach 100% presence
      in *any* snapshot. Removes mandatory wiki-form fields like ``dc:title``
      that don't discriminate between curated and abandoned AOPs.
    - **All properties** (right): full denominator. Values are uniformly
      higher because mandatory fields are always in the numerator.

    The data and figures are cached under two keys
    (``aop_completeness_boxplot`` and ``aop_completeness_boxplot_all``) so
    the standard ``/download/trend/<plot>`` route works for both.

    Args:
        label_file: Path to CSV file containing property labels and types.

    The beeswarm (``aop_completeness_beeswarm``) has one row per AOP per
    snapshot with both cohorts' scores; the page draws it from /api/v2/data
    (static/js/viz/beeswarm.js), the figure here is the newest snapshot.

    Returns:
        Tuple of (html_optional, html_all, html_beeswarm).
    """
    global _plot_data_cache, _plot_figure_cache

    # Load property labels from CSV
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential", "applies_to": "AOP|KE|KER|Stressor"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata", "applies_to": "AOP|KE|KER"}
    ]
    df_labels = safe_read_csv(label_file, default_properties)

    if df_labels.empty:
        fallback = create_fallback_plot("Composite AOP Completeness Distribution", "No property labels available")
        return fallback, fallback, fallback

    # Load properties for each entity type from CSV
    aop_properties_df = df_labels[df_labels['applies_to'].str.contains('AOP', na=False)]
    ke_properties_df = df_labels[df_labels['applies_to'].str.contains('KE', na=False)]
    ker_properties_df = df_labels[df_labels['applies_to'].str.contains('KER', na=False)]

    # Get initial property sets
    aop_props_all = set(aop_properties_df['uri'].tolist())
    ke_props_all = set(ke_properties_df['uri'].tolist())
    ker_props_all = set(ker_properties_df['uri'].tolist())

    # Combine all properties for querying
    all_props = aop_props_all | ke_props_all | ker_props_all

    if not all_props:
        fallback = create_fallback_plot("Composite AOP Completeness Distribution", "No properties found")
        return fallback, fallback, fallback

    # Get all versions for per-version parallel queries
    versions = get_all_versions()
    if not versions:
        fallback = create_fallback_plot("Composite AOP Completeness Distribution", "No versions available")
        return fallback, fallback, fallback

    logger.info(f"Boxplot: querying {len(versions)} versions in parallel for property filtering")

    # Step 1: Query presence and totals per version to filter 100% properties
    try:
        version_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_query_boxplot_version, v, aop_props_all, ke_props_all, ker_props_all, all_props): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        version_data.append(result)
                except Exception as e:
                    logger.warning(f"Boxplot presence query timed out for version {version_str}: {e}")
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        # Build totals map from per-version results: {version: {entity_type: total}}
        totals_map = {}
        for vd in version_data:
            version = vd["version"]
            totals_map[version] = {}
            for r in vd["totals"]:
                entity_type = r["entity_type"]["value"]
                total = int(r["total"]["value"])
                totals_map[version][entity_type] = total

        # Calculate property presence percentages by entity type
        # {entity_type: {prop_uri: [percentages]}}
        prop_presence = {"AOP": {}, "KE": {}, "KER": {}}

        for vd in version_data:
            version = vd["version"]
            for r in vd["presence"]:
                entity_type = r["entity_type"]["value"]
                prop_uri = r["p"]["value"]
                count = int(r["count"]["value"])
                total = totals_map.get(version, {}).get(entity_type, 1)
                percentage = (count / total) * 100

                if prop_uri not in prop_presence[entity_type]:
                    prop_presence[entity_type][prop_uri] = []
                prop_presence[entity_type][prop_uri].append(percentage)

        # Filter out properties that reach 100% in any version for each entity type
        aop_props_to_exclude = {
            prop_uri for prop_uri, percentages in prop_presence["AOP"].items()
            if max(percentages) >= 100
        }
        ke_props_to_exclude = {
            prop_uri for prop_uri, percentages in prop_presence["KE"].items()
            if max(percentages) >= 100
        }
        ker_props_to_exclude = {
            prop_uri for prop_uri, percentages in prop_presence["KER"].items()
            if max(percentages) >= 100
        }

        # Get filtered property sets
        aop_props = aop_props_all - aop_props_to_exclude
        ke_props = ke_props_all - ke_props_to_exclude
        ker_props = ker_props_all - ker_props_to_exclude

        aop_prop_count = len(aop_props)
        ke_prop_count = len(ke_props)
        ker_prop_count = len(ker_props)

    except SparqlUnavailable:
        raise
    except Exception as e:
        logger.warning(f"Error filtering 100% properties: {str(e)}")
        # Fall back to using all properties
        aop_props = aop_props_all
        ke_props = ke_props_all
        ker_props = ker_props_all
        aop_prop_count = len(aop_props)
        ke_prop_count = len(ke_props)
        ker_prop_count = len(ker_props)

    if aop_prop_count == 0 and ke_prop_count == 0 and ker_prop_count == 0:
        fallback = create_fallback_plot("Composite AOP Completeness Distribution", "All properties are 100% present")
        return fallback, fallback, fallback

    # Property counts for the unfiltered "all-properties" cohort (parallel pass below).
    aop_prop_count_all = len(aop_props_all)
    ke_prop_count_all = len(ke_props_all)
    ker_prop_count_all = len(ker_props_all)

    logger.info(
        f"Boxplot: querying {len(versions)} versions × 2 cohorts for per-entity property counts "
        f"(optional AOP:{aop_prop_count}, KE:{ke_prop_count}, KER:{ker_prop_count}; "
        f"all AOP:{aop_prop_count_all}, KE:{ke_prop_count_all}, KER:{ker_prop_count_all})"
    )

    # Step 2: For each cohort (optional-only, all-properties) run the per-version
    # entity-props query and build a composite-score DataFrame. The network query
    # runs inside each per-version helper invocation but the result is identical
    # across cohorts; the trade-off is a small extra SPARQL load (~32 redundant
    # network queries) in exchange for not rewriting the helper signature.
    def _build_cohort_df(label, props_aop, props_ke, props_ker, n_aop, n_ke, n_ker):
        cohort_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    _query_boxplot_entity_props, v,
                    props_aop, props_ke, props_ker,
                    n_aop, n_ke, n_ker,
                ): v["version"]
                for v in versions
            }
            for future in as_completed(futures):
                version_str = futures[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        cohort_data.append(result)
                except Exception as e:
                    logger.warning(f"Boxplot entity query timed out for version {version_str} ({label}): {e}")
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        if not cohort_data:
            return None
        # as_completed() yields in finishing order; fix it (#220).
        cohort_data.sort(key=lambda vd: vd["version"])

        network_map: dict[str, dict] = {}
        aop_completeness: dict[str, dict] = {}
        ke_completeness: dict[str, dict] = {}
        ker_completeness: dict[str, dict] = {}

        for vd in cohort_data:
            version = vd["version"]

            network_map[version] = {}
            for r in vd["network"]:
                aop_uri = r["aop"]["value"]
                network_map[version][aop_uri] = {"kes": set(), "kers": set()}
                if "kes" in r and r["kes"] and r["kes"]["value"]:
                    network_map[version][aop_uri]["kes"] = set(r["kes"]["value"].split("|"))
                if "kers" in r and r["kers"] and r["kers"]["value"]:
                    network_map[version][aop_uri]["kers"] = set(r["kers"]["value"].split("|"))

            for r in vd["entity_props"]:
                entity_uri = r["entity"]["value"]
                entity_type = r["entity_type"]["value"]
                prop_count = int(r["prop_count"]["value"])
                if entity_type == "AOP" and n_aop > 0:
                    aop_completeness.setdefault(version, {})[entity_uri] = (prop_count / n_aop) * 100
                elif entity_type == "KE" and n_ke > 0:
                    ke_completeness.setdefault(version, {})[entity_uri] = (prop_count / n_ke) * 100
                elif entity_type == "KER" and n_ker > 0:
                    ker_completeness.setdefault(version, {})[entity_uri] = (prop_count / n_ker) * 100

        rows = []
        for version, aops in network_map.items():
            for aop_uri, network in aops.items():
                entity_scores = []
                if version in aop_completeness and aop_uri in aop_completeness[version]:
                    entity_scores.append(aop_completeness[version][aop_uri])
                elif n_aop > 0:
                    entity_scores.append(0)
                # Sorted sets: the summation order moves the last float digits (#220).
                for ke_uri in sorted(network["kes"]):
                    if version in ke_completeness and ke_uri in ke_completeness[version]:
                        entity_scores.append(ke_completeness[version][ke_uri])
                    elif n_ke > 0:
                        entity_scores.append(0)
                for ker_uri in sorted(network["kers"]):
                    if version in ker_completeness and ker_uri in ker_completeness[version]:
                        entity_scores.append(ker_completeness[version][ker_uri])
                    elif n_ker > 0:
                        entity_scores.append(0)
                if entity_scores:
                    rows.append({
                        "version": version,
                        "aop_uri": aop_uri,
                        "completeness": sum(entity_scores) / len(entity_scores),
                        "entity_count": len(entity_scores),
                    })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        # Stable sort with a tie-break, so AOPs keep one order within a version.
        return df.sort_values(["version_dt", "aop_uri"], kind="mergesort")

    def _render_boxplot(df, y_title):
        fig = px.box(
            df, x="version", y="completeness",
            labels={"completeness": y_title, "version": "Version"},
        )
        fig.update_traces(
            marker=dict(opacity=0.6, color=BRAND_COLORS["primary"]),
            line=dict(width=2),
        )
        fig.update_layout(
            margin=dict(l=50, r=20, t=50, b=100),
            yaxis=dict(title=y_title, range=[0, 105]),
            xaxis=dict(title="Version", tickangle=-45),
            showlegend=False,
        )
        return fig

    try:
        df_optional = _build_cohort_df(
            "optional", aop_props, ke_props, ker_props,
            aop_prop_count, ke_prop_count, ker_prop_count,
        )
        df_all = _build_cohort_df(
            "all", aop_props_all, ke_props_all, ker_props_all,
            aop_prop_count_all, ke_prop_count_all, ker_prop_count_all,
        )

        if df_optional is None or df_all is None:
            fallback = create_fallback_plot("Composite AOP Completeness Distribution", "No completeness data available")
            return fallback, fallback, fallback

        _plot_data_cache['aop_completeness_boxplot'] = df_optional.copy()
        _plot_data_cache['aop_completeness_boxplot_all'] = df_all.copy()

        fig_optional = _render_boxplot(df_optional, "Optional-property composite completeness (%)")
        fig_all = _render_boxplot(df_all, "All-property composite completeness (%)")

        _plot_figure_cache['aop_completeness_boxplot'] = fig_optional
        _plot_figure_cache['aop_completeness_boxplot_all'] = fig_all

        newest = max(versions, key=lambda v: v["version"])
        df_swarm = beeswarm_rows(df_optional, df_all, _query_aop_titles(newest))
        fig_swarm = _render_beeswarm(df_swarm, df_swarm["version"].max(), "completeness_optional",
                                     "Optional-property composite completeness (%)")
        _plot_data_cache['aop_completeness_beeswarm'] = df_swarm
        _plot_figure_cache['aop_completeness_beeswarm'] = fig_swarm

        return render_plot_html(fig_optional), render_plot_html(fig_all), render_plot_html(fig_swarm)

    except SparqlUnavailable:
        raise
    except Exception as e:
        logger.error(f"Error creating composite AOP completeness boxplot: {str(e)}")
        fallback = create_fallback_plot("Composite AOP Completeness Distribution", f"Error: {str(e)}")
        return fallback, fallback, fallback


def plot_oecd_completeness_trend(label_file="property_labels.csv") -> str:
    """Generate OECD completeness trend visualization showing mean completeness per OECD status over time.

    Replaces the removed plot_aop_completeness_boxplot_by_status() which hit Virtuoso limits.
    This version queries aggregated means per status per version (~5 rows per version),
    producing ~75 total data points instead of 240K raw rows.

    Strategy:
        - For each version, query per-AOP property counts and OECD status
        - Compute mean completeness per OECD status in Python
        - Plot as a line chart with one line per status

    Args:
        label_file: Path to CSV file containing property labels and types

    Returns:
        HTML string containing the Plotly line chart visualization
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        # Load property labels
        default_properties = [
            {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential", "applies_to": "AOP|KE|KER|Stressor"},
            {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata", "applies_to": "AOP|KE|KER"}
        ]
        df_labels = safe_read_csv(label_file, default_properties)

        if df_labels.empty:
            return create_fallback_plot("OECD Completeness Trend", "No property labels available")

        # Filter properties applicable to AOPs
        aop_props_df = df_labels[
            df_labels["applies_to"].fillna("").str.contains("AOP", case=False, na=False)
        ]
        if aop_props_df.empty:
            return create_fallback_plot("OECD Completeness Trend", "No AOP properties configured")

        aop_property_uris = set(aop_props_df["uri"].tolist())
        total_properties = len(aop_property_uris)

        if total_properties == 0:
            return create_fallback_plot("OECD Completeness Trend", "No AOP properties configured")

        # Build property filter for SPARQL
        prop_filter_values = ", ".join([f"<{uri}>" for uri in sorted(aop_property_uris)])

        # Get all versions
        versions = get_all_versions()
        if not versions:
            return create_fallback_plot("OECD Completeness Trend", "No versions available")

        logger.info(f"OECD completeness trend: querying {len(versions)} versions with {total_properties} AOP properties")

        # Query function for a single version
        def query_version(version_info):
            """Query per-AOP property count and OECD status for a single version."""
            graph_uri = version_info["graph_uri"]
            version_str = version_info["version"]

            query = f"""
            SELECT ?aop (STR(?status_obj) AS ?status) (COUNT(DISTINCT ?p) AS ?prop_count)
            WHERE {{
                GRAPH <{graph_uri}> {{
                    ?aop a aopo:AdverseOutcomePathway .
                    OPTIONAL {{
                        ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj .
                    }}
                    ?aop ?p ?o .
                    FILTER(?p IN ({prop_filter_values}))
                }}
            }}
            GROUP BY ?aop ?status_obj
            """

            try:
                results = run_sparql_query_with_retry(query, max_retries=2)
                if not results:
                    return []

                # Collect per-AOP completeness with status
                version_data = []
                for r in results:
                    status = r.get("status", {}).get("value", "No Status") if "status" in r else "No Status"
                    prop_count = int(r["prop_count"]["value"])
                    completeness = (prop_count / total_properties) * 100

                    version_data.append({
                        "version": version_str,
                        "status": status,
                        "completeness": completeness
                    })

                return version_data
            except Exception as e:
                logger.warning(f"OECD completeness query failed for version {version_str}: {e}")
                # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        # Query versions in parallel (4 workers to avoid overloading endpoint)
        all_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_version = {
                executor.submit(query_version, v): v["version"]
                for v in versions
            }
            for future in as_completed(future_to_version):
                version_str = future_to_version[future]
                try:
                    result = future.result(timeout=60)
                    all_data.extend(result)
                except Exception as e:
                    logger.warning(f"OECD completeness query timed out for version {version_str}: {e}")
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        if not all_data:
            return create_fallback_plot("OECD Completeness Trend", "No completeness data available")

        # Create DataFrame and compute mean completeness per status per version
        df_raw = pd.DataFrame(all_data)

        # Aggregate: mean completeness per (version, status)
        df = df_raw.groupby(["version", "status"], as_index=False).agg(
            mean_completeness=("completeness", "mean"),
            aop_count=("completeness", "count")
        )

        # Sort by version date
        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")

        # Cache data for CSV export
        _plot_data_cache['oecd_completeness_trend'] = df.copy()

        logger.info(f"OECD completeness trend: {len(df)} data points across {df['version'].nunique()} versions, {df['status'].nunique()} statuses")

        # Use shared OECD status brand colors for consistency
        oecd_colors = BRAND_COLORS['oecd_status']
        unique_statuses = sorted(df['status'].unique())
        # Map each status to its brand color, fall back to palette cycling
        palette = BRAND_COLORS['palette']
        fallback_idx = 0
        status_color_map = {}
        for status in unique_statuses:
            # Try exact match first, then substring match for URI-based status labels
            if status in oecd_colors:
                status_color_map[status] = oecd_colors[status]
            else:
                # Check if any key is a substring of the status (handles URI-derived labels)
                matched = False
                for key, color in oecd_colors.items():
                    if key.lower() in status.lower() or status.lower() in key.lower():
                        status_color_map[status] = color
                        matched = True
                        break
                if not matched:
                    status_color_map[status] = palette[fallback_idx % len(palette)]
                    fallback_idx += 1

        # Create line chart
        fig = px.line(
            df,
            x="version",
            y="mean_completeness",
            color="status",
            markers=True,
            labels={
                "mean_completeness": "Mean Completeness (%)",
                "status": "OECD Status",
                "version": "Version"
            },
            color_discrete_map=status_color_map
        )

        # Apply marker shapes for visual distinction
        marker_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross',
                          'star', 'hexagon', 'pentagon', 'triangle-down', 'x']
        for i, trace in enumerate(fig.data):
            trace.update(
                marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=9),
                line=dict(width=2.5)
            )

        fig.update_layout(
            margin=dict(l=50, r=150, t=50, b=50),
            yaxis=dict(title="Mean Completeness (%)", range=[0, 105]),
            xaxis=dict(title="Version", tickangle=-45),
            legend=dict(title="OECD Status")
        )

        # Cache figure for image export
        _plot_figure_cache['oecd_completeness_trend'] = fig

        return render_plot_html(fig)

    except Exception as e:
        logger.error(f"Error creating OECD completeness trend: {str(e)}")
        return create_fallback_plot("OECD Completeness Trend", f"Error: {str(e)}")


def plot_aop_completeness_boxplot_by_status():
    """
    Plot composite AOP completeness distribution grouped by OECD status across all versions.
    
    Similar to plot_aop_completeness_boxplot() but adds color-grouped visualization
    by OECD approval status (Approved, Under Review, Under Development, No Status).
    
    Uses fallback strategy: queries status from latest version and applies across
    all historical versions to avoid timeout issues.
    
    Returns:
        str: HTML string of the Plotly boxplot visualization
    """
    import pandas as pd
    import plotly.express as px

    # Load properties from CSV for each entity type
    default_properties = [
        {"uri": "http://purl.org/dc/elements/1.1/title", "label": "Title", "type": "Essential"},
        {"uri": "http://purl.org/dc/terms/abstract", "label": "Abstract", "type": "Essential"},
        {"uri": "http://purl.org/dc/elements/1.1/creator", "label": "Creator", "type": "Metadata"},
    ]
    properties_df = safe_read_csv("property_labels.csv", default_properties)

    # Filter properties for each entity type
    aop_properties_df = properties_df[properties_df['applies_to'].str.contains('AOP', na=False)]
    ke_properties_df = properties_df[properties_df['applies_to'].str.contains('KE', na=False)]
    ker_properties_df = properties_df[properties_df['applies_to'].str.contains('KER', na=False)]

    aop_props = set([p["uri"] for _, p in aop_properties_df.iterrows()])
    ke_props = set([p["uri"] for _, p in ke_properties_df.iterrows()])
    ker_props = set([p["uri"] for _, p in ker_properties_df.iterrows()])

    aop_prop_count = len(aop_props)
    ke_prop_count = len(ke_props)
    ker_prop_count = len(ker_props)

    if aop_prop_count == 0 and ke_prop_count == 0 and ker_prop_count == 0:
        return create_fallback_plot("Composite AOP Completeness by OECD Status", "No properties configured")
    
    # Phase 1: Optimized network structure query - combine network + status in single query
    # Get individual AOP-KE and AOP-KER relationships with status (but no entity IN filters later)
    query_network = """
    SELECT ?graph ?aop (STR(?status_obj) AS ?status) ?ke ?ker
    WHERE {
      GRAPH ?graph {
        ?aop a aopo:AdverseOutcomePathway .
        OPTIONAL {
            ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status_obj
        }
        OPTIONAL { ?aop aopo:has_key_event ?ke }
        OPTIONAL { ?aop aopo:has_key_event_relationship ?ker }
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    ORDER BY ?graph ?aop
    """

    try:
        results_network = run_sparql_query(query_network)

        if not results_network:
            return create_fallback_plot("Composite AOP Completeness by OECD Status", "No AOP network data available")

        # Build network map from individual rows
        network_map = {}  # {version: {aop_uri: {"kes": set(), "kers": set(), "status": str}}}
        all_aops = set()
        all_kes = set()
        all_kers = set()

        for r in results_network:
            version = r["graph"]["value"].split("/")[-1]
            aop_uri = r["aop"]["value"]
            status = r.get("status", {}).get("value", "No Status") if "status" in r else "No Status"

            # Track all unique entities
            all_aops.add(aop_uri)

            if version not in network_map:
                network_map[version] = {}
            if aop_uri not in network_map[version]:
                network_map[version][aop_uri] = {"kes": set(), "kers": set(), "status": status}

            # Add KE if present
            if "ke" in r and r["ke"]:
                ke_uri = r["ke"]["value"]
                network_map[version][aop_uri]["kes"].add(ke_uri)
                all_kes.add(ke_uri)

            # Add KER if present
            if "ker" in r and r["ker"]:
                ker_uri = r["ker"]["value"]
                network_map[version][aop_uri]["kers"].add(ker_uri)
                all_kers.add(ker_uri)

        logger.info(f"Phase 1 complete: Found {len(all_aops)} unique AOPs, {len(all_kes)} unique KEs, {len(all_kers)} unique KERs across all versions")

        # Phase 2: Query property counts with optimized filtering
        # Use property URI filtering but avoid large entity IN filters

        # Build property filter strings
        if aop_prop_count > 0:
            aop_prop_filter_values = ", ".join([f"<{uri}>" for uri in sorted(aop_props)])
            aop_prop_filter = f"FILTER(?p IN ({aop_prop_filter_values}))"
        else:
            aop_prop_filter = ""

        if ke_prop_count > 0:
            ke_prop_filter_values = ", ".join([f"<{uri}>" for uri in sorted(ke_props)])
            ke_prop_filter = f"FILTER(?p IN ({ke_prop_filter_values}))"
        else:
            ke_prop_filter = ""

        if ker_prop_count > 0:
            ker_prop_filter_values = ", ".join([f"<{uri}>" for uri in sorted(ker_props)])
            ker_prop_filter = f"FILTER(?p IN ({ker_prop_filter_values}))"
        else:
            ker_prop_filter = ""

        # Query AOP property counts
        if aop_prop_count > 0:
            query_aop = f"""
            SELECT ?graph ?aop (COUNT(DISTINCT ?p) AS ?prop_count)
            WHERE {{
              GRAPH ?graph {{
                ?aop a aopo:AdverseOutcomePathway ;
                     ?p ?o .
                {aop_prop_filter}
              }}
              FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }}
            GROUP BY ?graph ?aop
            ORDER BY ?graph ?aop
            """
            results_aop = run_sparql_query(query_aop)
        else:
            results_aop = []

        # Query KE property counts
        if ke_prop_count > 0:
            query_ke = f"""
            SELECT ?graph ?ke (COUNT(DISTINCT ?p) AS ?prop_count)
            WHERE {{
              GRAPH ?graph {{
                ?ke a aopo:KeyEvent ;
                    ?p ?o .
                {ke_prop_filter}
              }}
              FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }}
            GROUP BY ?graph ?ke
            ORDER BY ?graph ?ke
            """
            results_ke = run_sparql_query(query_ke)
        else:
            results_ke = []

        # Query KER property counts
        if ker_prop_count > 0:
            query_ker = f"""
            SELECT ?graph ?ker (COUNT(DISTINCT ?p) AS ?prop_count)
            WHERE {{
              GRAPH ?graph {{
                ?ker a aopo:KeyEventRelationship ;
                     ?p ?o .
                {ker_prop_filter}
              }}
              FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
            }}
            GROUP BY ?graph ?ker
            ORDER BY ?graph ?ker
            """
            results_ker = run_sparql_query(query_ker)
        else:
            results_ker = []
        
        # Step 3: Build completeness maps
        
        aop_completeness = {}
        for r in results_aop:
            version = r["graph"]["value"].split("/")[-1]
            entity_uri = r["aop"]["value"]
            prop_count = int(r["prop_count"]["value"])
            completeness = (prop_count / aop_prop_count) * 100 if aop_prop_count > 0 else 0
            
            if version not in aop_completeness:
                aop_completeness[version] = {}
            aop_completeness[version][entity_uri] = completeness
        
        ke_completeness = {}
        for r in results_ke:
            version = r["graph"]["value"].split("/")[-1]
            entity_uri = r["ke"]["value"]
            prop_count = int(r["prop_count"]["value"])
            completeness = (prop_count / ke_prop_count) * 100 if ke_prop_count > 0 else 0
            
            if version not in ke_completeness:
                ke_completeness[version] = {}
            ke_completeness[version][entity_uri] = completeness
        
        ker_completeness = {}
        for r in results_ker:
            version = r["graph"]["value"].split("/")[-1]
            entity_uri = r["ker"]["value"]
            prop_count = int(r["prop_count"]["value"])
            completeness = (prop_count / ker_prop_count) * 100 if ker_prop_count > 0 else 0
            
            if version not in ker_completeness:
                ker_completeness[version] = {}
            ker_completeness[version][entity_uri] = completeness
        
        # Step 4: Calculate composite scores WITH status
        completeness_data = []
        
        for version, aops in network_map.items():
            for aop_uri, network in aops.items():
                entity_scores = []
                
                # Add AOP entity completeness
                if version in aop_completeness and aop_uri in aop_completeness[version]:
                    entity_scores.append(aop_completeness[version][aop_uri])
                elif aop_prop_count > 0:
                    entity_scores.append(0)
                
                # Add KE completeness scores
                for ke_uri in network["kes"]:
                    if version in ke_completeness and ke_uri in ke_completeness[version]:
                        entity_scores.append(ke_completeness[version][ke_uri])
                    elif ke_prop_count > 0:
                        entity_scores.append(0)
                
                # Add KER completeness scores
                for ker_uri in network["kers"]:
                    if version in ker_completeness and ker_uri in ker_completeness[version]:
                        entity_scores.append(ker_completeness[version][ker_uri])
                    elif ker_prop_count > 0:
                        entity_scores.append(0)
                
                # Calculate composite score
                if entity_scores:
                    composite_score = sum(entity_scores) / len(entity_scores)
                    # Get status from network_map (already captured in Phase 1)
                    status = network["status"]

                    completeness_data.append({
                        "version": version,
                        "aop_uri": aop_uri,
                        "status": status,
                        "completeness": composite_score,
                        "entity_count": len(entity_scores)
                    })
        
        if not completeness_data:
            return create_fallback_plot("Composite AOP Completeness by OECD Status", "No completeness data available")
        
        # Create DataFrame
        df = pd.DataFrame(completeness_data)
        
        # Sort by version
        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt")
        
        # Cache data for CSV export
        _plot_data_cache['aop_completeness_boxplot_by_status'] = df.copy()

        # Dynamically extract all unique statuses and assign colors
        unique_statuses = sorted(df['status'].unique())

        # Use Plotly's qualitative color palette for consistent, distinguishable colors
        # Plotly Plotly palette provides 10 distinct colors
        plotly_colors = px.colors.qualitative.Plotly

        # Build dynamic color map
        status_color_map = {}
        for i, status in enumerate(unique_statuses):
            status_color_map[status] = plotly_colors[i % len(plotly_colors)]

        logger.info(f"Found {len(unique_statuses)} unique OECD statuses: {unique_statuses}")

        # Create boxplot with color grouping by status
        fig = px.box(
            df,
            x="version",
            y="completeness",
            color="status",
            labels={
                "completeness": "Composite Completeness (%)",
                "version": "Version",
                "status": "OECD Status"
            },
            color_discrete_map=status_color_map
        )
        
        fig.update_traces(
            marker=dict(opacity=0.6),
            line=dict(width=2)
        )
        
        fig.update_layout(
            margin=dict(l=50, r=20, t=50, b=100),
            yaxis=dict(title="Composite Completeness (%)", range=[0, 105]),
            xaxis=dict(title="Version", tickangle=-45),
            showlegend=True,
            legend=dict(title="OECD Status")
        )
        
        # Cache figure for image export
        _plot_figure_cache['aop_completeness_boxplot_by_status'] = fig
        
        return render_plot_html(fig)
    
    except Exception as e:
        logger.error(f"Error creating status-separated AOP completeness boxplot: {str(e)}")
        return create_fallback_plot("Composite AOP Completeness by OECD Status", f"Error: {str(e)}")


# plot_curation_progress removed — produced non-sensical output per UAT feedback


def plot_ontology_term_growth() -> tuple[str, str, pd.DataFrame]:
    """Show growth of unique ontology terms used across versions.

    Tracks the total count of unique ontology-linked IRI terms (from biological
    process and object annotations) across all versions.

    Uses per-version parallel SPARQL queries for efficiency.

    Returns:
        tuple: (absolute_html, delta_html, growth_df)
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = get_all_versions()
        if not versions:
            fallback = create_fallback_plot("Ontology Term Growth", "No versions available")
            return (fallback, fallback, pd.DataFrame())

        def _query_version(ver_info):
            """Count unique ontology terms for a single version."""
            graph_uri = ver_info['graph_uri']
            version = ver_info['version']

            query = f"""
            SELECT (COUNT(DISTINCT ?term) AS ?unique_terms)
            WHERE {{
                GRAPH <{graph_uri}> {{
                    {{
                        ?ke a aopo:KeyEvent ;
                            aopo:hasBiologicalEvent ?be .
                        ?be aopo:hasProcess ?term .
                    }}
                    UNION
                    {{
                        ?ke a aopo:KeyEvent ;
                            aopo:hasBiologicalEvent ?be .
                        ?be aopo:hasObject ?term .
                    }}
                }}
                FILTER(isIRI(?term))
            }}
            """

            results = run_sparql_query(query)
            count = int(results[0]["unique_terms"]["value"]) if results else 0
            return {"version": version, "Unique Terms": count}

        # Execute per-version queries in parallel
        rows = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_query_version, v): v for v in versions}
            for future in as_completed(futures):
                try:
                    row = future.result(timeout=60)
                    rows.append(row)
                except Exception as e:
                    ver = futures[future]
                    logger.warning(f"Ontology term growth query failed for {ver['version']}: {e}")
                    # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                    raise SparqlUnavailable(f"{type(e).__name__}: {e}") from e

        if not rows:
            fallback = create_fallback_plot("Ontology Term Growth", "No data collected")
            return (fallback, fallback, pd.DataFrame())

        df = pd.DataFrame(rows)
        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt").reset_index(drop=True)

        # --- Absolute line chart ---
        fig_abs = px.line(
            df,
            x="version",
            y="Unique Terms",
            markers=True,
            labels={"Unique Terms": "Unique Ontology Terms", "version": "Version"},
        )
        fig_abs.update_traces(line_color=BRAND_COLORS['blue'], marker_color=BRAND_COLORS['blue'])
        fig_abs.update_layout(
            margin=dict(l=50, r=20, t=50, b=80)
        )
        fig_abs.update_xaxes(tickangle=-45)

        # --- Delta bar chart ---
        df_delta = df.copy()
        df_delta["New Terms"] = df_delta["Unique Terms"].diff().fillna(0).astype(int)

        fig_delta = px.bar(
            df_delta,
            x="version",
            y="New Terms",
            labels={"New Terms": "New Terms Added", "version": "Version"},
        )
        fig_delta.update_traces(marker_color=BRAND_COLORS['blue'])
        fig_delta.update_layout(
            margin=dict(l=50, r=20, t=50, b=80)
        )
        fig_delta.update_xaxes(tickangle=-45)

        # Cache
        _plot_data_cache['ontology_term_growth_absolute'] = df.drop(columns=["version_dt"]).copy()
        _plot_data_cache['ontology_term_growth_delta'] = df_delta.drop(columns=["version_dt"]).copy()
        _plot_figure_cache['ontology_term_growth_absolute'] = fig_abs
        _plot_figure_cache['ontology_term_growth_delta'] = fig_delta

        return (
            render_plot_html(fig_abs),
            render_plot_html(fig_delta),
            df.drop(columns=["version_dt"])
        )

    except Exception as e:
        logger.error(f"Failed to generate ontology term growth plots: {str(e)}")
        fallback = create_fallback_plot("Ontology Term Growth", f"Error: {str(e)}")
        return (fallback, fallback, pd.DataFrame())


# ===========================================================================
# Organ-system coverage over time
# ===========================================================================


def _coverage_for_graph(graph_uri: str) -> set[tuple[str, str]]:
    """Return the set of (AOP IRI, organ-system bucket) pairs classified by
    any of Signals A, A' or B in a single snapshot graph.

    Signal C (keywords) is intentionally excluded from the trend version —
    keywords have no temporal signal beyond title rewrites and would
    distort the longitudinal picture.
    """
    query = f"""
    SELECT ?aop ?organ ?cell ?obj ?proc
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?aop a aopo:AdverseOutcomePathway ;
             aopo:has_key_event ?ke .
        OPTIONAL {{ ?ke aopo:OrganContext ?organ . }}
        OPTIONAL {{ ?ke aopo:CellTypeContext ?cell . }}
        OPTIONAL {{
          ?ke aopo:hasBiologicalEvent ?be .
          OPTIONAL {{ ?be aopo:hasObject ?obj . }}
          OPTIONAL {{ ?be aopo:hasProcess ?proc . }}
        }}
      }}
    }}
    """
    results = run_sparql_query(query) or []
    pairs: set[tuple[str, str]] = set()
    for r in results:
        aop = r.get("aop", {}).get("value")
        if not aop:
            continue
        for var in ("organ", "cell", "obj"):
            iri = r.get(var, {}).get("value")
            if iri:
                for bucket in classify_anatomy(iri):
                    pairs.add((aop, bucket))
                # Phenotype-as-object — HP/MP via UPHENO:0000001
                for bucket in classify_process(iri):
                    pairs.add((aop, bucket))
        proc = r.get("proc", {}).get("value")
        if proc:
            for bucket in classify_process(proc):
                pairs.add((aop, bucket))
    return pairs


def _query_aop_total(graph_uri: str) -> int:
    """Total number of AOPs in a snapshot — denominator for percentage trends."""
    query = f"""
    SELECT (COUNT(DISTINCT ?aop) AS ?n)
    WHERE {{
      GRAPH <{graph_uri}> {{ ?aop a aopo:AdverseOutcomePathway . }}
    }}
    """
    results = run_sparql_query(query) or []
    if not results:
        return 0
    try:
        return int(results[0]["n"]["value"])
    except (KeyError, ValueError, TypeError):
        return 0


def plot_organ_coverage_trends() -> tuple[str, str, pd.DataFrame]:
    """Organ-system coverage of AOPs across every quarterly snapshot.

    For each snapshot date, count the number of AOPs that classify into each
    organ-system bucket via Signals A/A'/B (the structured-anatomy signals).
    Produces a multi-line absolute plot and a percentage-of-snapshot view.

    Returns:
        tuple: (absolute_html, percentage_html, long_dataframe)
    """
    global _plot_data_cache, _plot_figure_cache

    try:
        versions = get_all_versions()
        if not versions:
            fallback = create_fallback_plot("Organ-System Coverage Trends", "No versions available")
            return (fallback, fallback, pd.DataFrame())

        def _query_version(ver_info):
            graph_uri = ver_info["graph_uri"]
            version = ver_info["version"]
            try:
                pairs = _coverage_for_graph(graph_uri)
                total_aops = _query_aop_total(graph_uri)
            except Exception as exc:
                logger.warning(f"Organ-coverage query failed for {version}: {exc}")
                # A failed query must fail the whole plot, not leave a partial one to be cached (#167).
                raise SparqlUnavailable(f"{type(exc).__name__}: {exc}") from exc

            seen_per_bucket: dict[str, set] = {b: set() for b in ORGAN_SYSTEM_BUCKETS}
            classified_aops: set = set()
            for aop, bucket in pairs:
                seen_per_bucket.setdefault(bucket, set()).add(aop)
                if bucket and bucket != NO_ANNOTATION_BUCKET:
                    classified_aops.add(aop)
            counts = {b: len(s) for b, s in seen_per_bucket.items() if b in ORGAN_SYSTEM_BUCKETS}
            # Derive 'No annotation' from total AOPs minus AOPs with any
            # classified bucket — matches the snapshot pie chart's
            # 'Unclassified' slice (see plot_latest_organ_coverage_pie).
            counts[NO_ANNOTATION_BUCKET] = max(0, total_aops - len(classified_aops))
            counts["version"] = version
            counts["__total_aops__"] = total_aops
            return counts

        rows = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_query_version, v): v for v in versions}
            for future in as_completed(futures):
                row = future.result(timeout=120)
                if row is not None:
                    rows.append(row)

        if not rows:
            fallback = create_fallback_plot("Organ-System Coverage Trends", "No data collected")
            return (fallback, fallback, pd.DataFrame())

        df = pd.DataFrame(rows)
        df["version_dt"] = pd.to_datetime(df["version"], errors="coerce")
        df = df.sort_values("version_dt").reset_index(drop=True)

        bucket_cols = [b for b in ORGAN_SYSTEM_BUCKETS if b in df.columns]
        if NO_ANNOTATION_BUCKET in df.columns:
            bucket_cols = bucket_cols + [NO_ANNOTATION_BUCKET]

        # Long form for the line chart
        long_df = df.melt(
            id_vars=["version", "version_dt"],
            value_vars=bucket_cols,
            var_name="Organ System",
            value_name="AOP count",
        )

        fig_abs = px.line(
            long_df,
            x="version",
            y="AOP count",
            color="Organ System",
            markers=True,
            color_discrete_sequence=BRAND_COLORS["palette"],
        )
        # 14 organ-system series exceed the palette's distinct hues, so pair each
        # with a marker glyph and thin the lines — the best feasible separation
        # short of dropping series (#plot-review).
        _apply_marker_symbols(fig_abs, size=6, line_width=1.4)
        fig_abs.update_layout(
            title="Organ-System Coverage of AOPs over Time",
            xaxis_title="Snapshot",
            yaxis_title="Number of AOPs (Signals A, A', B)",
            margin=dict(l=50, r=30, t=60, b=80),
        )
        fig_abs.update_xaxes(tickangle=-45)

        # Percentage view — each bucket's count divided by total AOPs in the snapshot.
        pct = df[bucket_cols].copy()
        totals = df["__total_aops__"].replace(0, pd.NA)
        pct = pct.div(totals, axis=0).fillna(0) * 100.0
        pct["version"] = df["version"].values
        pct_long = pct.melt(
            id_vars=["version"],
            value_vars=bucket_cols,
            var_name="Organ System",
            value_name="Percentage",
        )

        fig_pct = px.line(
            pct_long,
            x="version",
            y="Percentage",
            color="Organ System",
            markers=True,
            color_discrete_sequence=BRAND_COLORS["palette"],
        )
        _apply_marker_symbols(fig_pct, size=6, line_width=1.4)
        fig_pct.update_layout(
            title="Organ-System Coverage of AOPs over Time (% of snapshot)",
            xaxis_title="Snapshot",
            yaxis_title="% of AOPs in snapshot (Signals A, A', B)",
            margin=dict(l=50, r=30, t=60, b=80),
        )
        fig_pct.update_xaxes(tickangle=-45)
        fig_pct.update_yaxes(ticksuffix="%")

        export_df = df.drop(columns=["version_dt"])
        pct_export = pct[["version"] + bucket_cols].copy()
        pct_export["Total AOPs"] = df["__total_aops__"].values
        _plot_data_cache["organ_coverage_absolute"] = export_df
        _plot_data_cache["organ_coverage_percentage"] = pct_export
        _plot_figure_cache["organ_coverage_absolute"] = fig_abs
        _plot_figure_cache["organ_coverage_percentage"] = fig_pct

        return (render_plot_html(fig_abs), render_plot_html(fig_pct), export_df)

    except Exception as e:
        logger.error(f"Failed to generate organ-coverage trends: {str(e)}")
        fallback = create_fallback_plot("Organ-System Coverage Trends", f"Error: {str(e)}")
        return (fallback, fallback, pd.DataFrame())
