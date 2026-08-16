"""Domain lens — scope AOP-Wiki to a biological domain by ontology closure (issue #149).

The question this answers is the JRC's: *how strong is AOP-Wiki in the
cardiovascular field?* A domain is defined by a handful of ontology root terms
(anatomy, process, development, phenotype), expanded offline to their
descendants and intersected with the terms Key Events actually carry. Nothing
here matches on keywords or labels, and there is no per-AOP curation — swapping
the root set swaps the domain.

Three views:

- **Domain size** — AOPs per domain, so a field can be read against its
  neighbours rather than against an absolute number nobody can calibrate.
- **Curation depth** — property-tier presence for the domain cohort against all
  AOPs, reusing the same `property_labels.csv` tiers as the property-presence
  charts. This is where the cardiovascular case turned interesting: the subset
  is *below* average on Content and well *above* on Assessment, i.e. the
  evidence rationale is written and the structured fields are not.
- **OECD status** — how far the domain's AOPs have travelled through the OECD
  workflow, against the whole wiki.

Scoping accuracy, measured against the JRC cardiovascular expert review (80
candidate AOPs, 56 rated core) on the 2026-07-01 snapshot: this lens returns 41
AOPs, of which 28 were rated core — precision 28/41, recall 28/56. The
dashboard's organ-system *coverage* classifier, which is deliberately more
generous, returns 63 for one extra true positive (29/63, 29/56). The two exist
for different jobs and must not be conflated.

The domain vocabulary is a committed offline artefact,
``static/data/domain_lens_cache.json``, built by
``scripts/build_domain_lens_cache.py``. Ubergraph is never contacted at request
time.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import plotly.graph_objects as go

from .shared import (
    BRAND_COLORS,
    OECD_STATUS_ORDER,
    PROPERTY_TYPE_ORDER,
    _plot_data_cache,
    _plot_figure_cache,
    create_fallback_plot,
    get_properties_for_entity,
    pad_axis_for_outside_labels,
    render_plot_html,
    run_sparql_query,
)

logger = logging.getLogger(__name__)

OBO = "http://purl.obolibrary.org/obo/"
DEFAULT_DOMAIN = "Cardiovascular"

# Entity types scored by the tier view, in the order they are drawn. Stressors
# are left out: they are not owned by an AOP the way KEs and KERs are, so a
# "domain's stressors" would double-count every shared chemical.
_TIER_ENTITIES: Tuple[str, ...] = ("AOP", "KE", "KER")

# Predicates that carry an ontology term on a Key Event — the same set the
# offline builder resolves against, and the same one the coverage-holes plot
# uses. NOT the set the ontology-usage plots query: those go through
# aopo:hasBiologicalEvent only and never see OrganContext / CellTypeContext,
# which is where the anatomy lives.
_KE_ANNOTATION_PREDICATES: Tuple[str, ...] = (
    "http://aopkb.org/aop_ontology#OrganContext",
    "http://aopkb.org/aop_ontology#CellTypeContext",
    OBO + "PATO_0001241",   # biological object
    OBO + "GO_0008150",     # biological process
)

# VALUES blocks are chunked so a cohort query stays inside Virtuoso's parser
# limits; 400 URIs per block is what the property-presence queries use.
_URI_CHUNK = 400


# ---------------------------------------------------------------------------
# Placeholder handling
#
# Presence queries ask whether a triple exists, which counts "&nbsp;" and "TBD"
# as a filled field. That matters most for the Assessment tier, which is free
# text: measured on 2026-07-01, placeholders are 7.2% of Overall Assessment and
# 3.2% of Considerations for Applications.
#
# The list is deliberately short and typographic. "None identified", "no
# inconsistencies" and "No data." are curator ANSWERS, not placeholders, and are
# counted as present — stripping them would be a judgement about the science.
# ---------------------------------------------------------------------------

_PLACEHOLDER_TOKENS: frozenset = frozenset({
    # Compared after leading/trailing periods are stripped, so "n.a." arrives
    # here as "n.a".
    "", "-", "--", "---", "?", "n/a", "na", "n.a",
    "tbd", "to be determined", "x", "xx", "tba",
})

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_ENTITY_RE = re.compile(r"&(nbsp|#160|#xa0|amp|quot|apos|lt|gt);", re.IGNORECASE)

# Only short literals can be placeholders, so the presence query only ships
# values below this length back to Python. Anything longer is content.
_PLACEHOLDER_MAX_LEN = 64


def is_placeholder(value: str) -> bool:
    """True when a literal is a formatting artefact rather than a filled field.

    Strips HTML tags and the handful of entities AOP-Wiki's rich-text editor
    emits, then compares the residue against a closed token list. Long values
    are never placeholders and short-circuit to False.
    """
    if value is None:
        return True
    if len(value) > _PLACEHOLDER_MAX_LEN:
        return False
    cleaned = _ENTITY_RE.sub(" ", _HTML_TAG_RE.sub(" ", value))
    cleaned = cleaned.replace("\xa0", " ").strip().strip(".").strip()
    return cleaned.lower() in _PLACEHOLDER_TOKENS


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_DOMAIN_LENS_CACHE: Optional[dict] = None


def _load_domain_lens_cache() -> dict:
    """Lazy-load the offline domain vocabulary (issue #149).

    Loaded lazily and defensively: a missing or corrupt file degrades the
    domain plots to a fallback rather than taking the app down at import.
    """
    global _DOMAIN_LENS_CACHE
    if _DOMAIN_LENS_CACHE is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "static", "data", "domain_lens_cache.json",
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                _DOMAIN_LENS_CACHE = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load domain lens cache (%s): %s", path, e)
            _DOMAIN_LENS_CACHE = {}
    return _DOMAIN_LENS_CACHE


def available_domains() -> List[str]:
    """Domain names the cache can serve, for the selector and validation."""
    return sorted(_load_domain_lens_cache().get("domains", {}))


def _resolve_domain(domain: Optional[str]) -> Optional[str]:
    """Validate a requested domain, falling back to the default."""
    domains = available_domains()
    if not domains:
        return None
    if domain in domains:
        return domain
    return DEFAULT_DOMAIN if DEFAULT_DOMAIN in domains else domains[0]


def domain_term_iris(domain: str) -> List[str]:
    """Full IRIs of the in-use ontology terms that define a domain."""
    entry = _load_domain_lens_cache().get("domains", {}).get(domain, {})
    return [OBO + short_id for short_id in sorted(entry.get("terms", {}))]


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------


def _chunks(items: Sequence[str], size: int = _URI_CHUNK) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _resolve_target_graph(version: str = None) -> Optional[Tuple[str, str]]:
    """Return ``(graph_iri, version_label)`` for the given version, or None."""
    if version:
        return f"http://aopwiki.org/graph/{version}", version

    results = run_sparql_query("""
    SELECT ?graph
    WHERE {
        GRAPH ?graph { ?s a aopo:AdverseOutcomePathway . }
        FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
    }
    GROUP BY ?graph
    ORDER BY DESC(?graph)
    LIMIT 1
    """)
    if not results:
        return None
    graph_iri = results[0]["graph"]["value"]
    return graph_iri, graph_iri.rsplit("/", 1)[-1]


def _query_domain_members(target_graph: str, term_iris: Sequence[str]) -> Dict[str, Set[str]]:
    """Map AOP IRI → the domain terms its Key Events carry, in one snapshot."""
    if not term_iris:
        return {}
    predicate_list = ", ".join(f"<{p}>" for p in _KE_ANNOTATION_PREDICATES)
    members: Dict[str, Set[str]] = {}
    for chunk in _chunks(list(term_iris)):
        values = " ".join(f"<{t}>" for t in chunk)
        query = f"""
        SELECT DISTINCT ?aop ?term
        WHERE {{
            GRAPH <{target_graph}> {{
                VALUES ?term {{ {values} }}
                ?ke a aopo:KeyEvent ; ?p ?term .
                FILTER(?p IN ({predicate_list}))
                ?aop a aopo:AdverseOutcomePathway ; aopo:has_key_event ?ke .
            }}
        }}
        """
        for row in run_sparql_query(query, use_post=True) or []:
            aop = row.get("aop", {}).get("value")
            term = row.get("term", {}).get("value")
            if aop and term:
                members.setdefault(aop, set()).add(term)
    return members


def _query_aop_status(target_graph: str) -> Dict[str, str]:
    """AOP IRI → OECD status label, defaulting to 'No Status'."""
    query = f"""
    SELECT ?aop ?status
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway .
            OPTIONAL {{ ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?status . }}
        }}
    }}
    """
    out: Dict[str, str] = {}
    for row in run_sparql_query(query) or []:
        aop = row.get("aop", {}).get("value")
        if not aop:
            continue
        status = (row.get("status", {}).get("value") or "").strip()
        out[aop] = status or "No Status"
    return out


def _query_cohort_entities(
    target_graph: str, aop_uris: Sequence[str]
) -> Tuple[Set[str], Set[str]]:
    """KE and KER URIs belonging to a set of AOPs."""
    kes: Set[str] = set()
    kers: Set[str] = set()
    for chunk in _chunks(list(aop_uris)):
        values = " ".join(f"<{a}>" for a in chunk)
        query = f"""
        SELECT DISTINCT ?ke ?ker
        WHERE {{
            GRAPH <{target_graph}> {{
                VALUES ?aop {{ {values} }}
                OPTIONAL {{ ?aop aopo:has_key_event ?ke . }}
                OPTIONAL {{ ?aop aopo:has_key_event_relationship ?ker . }}
            }}
        }}
        """
        for row in run_sparql_query(query, use_post=True) or []:
            if "ke" in row:
                kes.add(row["ke"]["value"])
            if "ker" in row:
                kers.add(row["ker"]["value"])
    return kes, kers


_ENTITY_CLASSES: Dict[str, str] = {
    "AOP": "aopo:AdverseOutcomePathway",
    "KE": "aopo:KeyEvent",
    "KER": "aopo:KeyEventRelationship",
}


def _property_values(
    target_graph: str,
    entity_type: str,
    property_uris: Sequence[str],
    entity_uris: Optional[Sequence[str]],
) -> Dict[str, Set[str]]:
    """Property URI → the entities carrying a non-placeholder value for it.

    ``entity_uris=None`` scores every entity of the type in the graph (the
    all-AOPs baseline), which avoids shipping 2,338 KER URIs back into a VALUES
    block for a query the endpoint can answer from the class alone.

    Only literals short enough to *be* a placeholder are returned; anything
    longer is reported as the empty string and counted as present.
    """
    if not property_uris:
        return {}
    property_values = " ".join(f"<{p}>" for p in property_uris)
    have: Dict[str, Set[str]] = {p: set() for p in property_uris}

    if entity_uris is None:
        blocks: List[str] = [f"?s a {_ENTITY_CLASSES[entity_type]} ."]
    else:
        blocks = [
            f"VALUES ?s {{ {' '.join(f'<{u}>' for u in chunk)} }}"
            for chunk in _chunks(list(entity_uris))
        ]

    for block in blocks:
        query = f"""
        SELECT DISTINCT ?s ?p ?short
        WHERE {{
            GRAPH <{target_graph}> {{
                {block}
                VALUES ?p {{ {property_values} }}
                ?s ?p ?o .
            }}
            BIND(IF(isLiteral(?o) && STRLEN(STR(?o)) < {_PLACEHOLDER_MAX_LEN}, STR(?o), "") AS ?short)
        }}
        """
        for row in run_sparql_query(query, use_post=True) or []:
            subject = row.get("s", {}).get("value")
            prop = row.get("p", {}).get("value")
            if not subject or prop not in have:
                continue
            short = row.get("short", {}).get("value", "")
            if short and is_placeholder(short):
                continue
            have[prop].add(subject)
    return have


def _tier_scores(
    target_graph: str,
    cohort: Dict[str, Optional[Sequence[str]]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    """Mean property presence per ``entity/tier``, plus per-property detail.

    ``cohort`` maps entity type → the URIs to score, or None for "all entities
    of this type in the graph".

    Returns (tier scores, per-property scores, entity counts), all keyed by
    ``"<entity>/<name>"``.
    """
    tier_scores: Dict[str, float] = {}
    property_scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for entity_type in _TIER_ENTITIES:
        uris = cohort.get(entity_type)
        if uris is not None and not uris:
            continue

        grouped = get_properties_for_entity(entity_type)
        properties = [p for tier in PROPERTY_TYPE_ORDER for p in grouped.get(tier, [])]
        if not properties:
            continue

        present = _property_values(
            target_graph, entity_type, [p["uri"] for p in properties], uris
        )

        if uris is None:
            # Denominator for the whole-graph case: entities of this type that
            # exist at all. Anything with no triple on any scored property still
            # has to count, or the baseline flatters itself.
            rows = run_sparql_query(f"""
            SELECT (COUNT(DISTINCT ?s) AS ?count)
            WHERE {{ GRAPH <{target_graph}> {{ ?s a {_ENTITY_CLASSES[entity_type]} . }} }}
            """) or []
            total = int(rows[0]["count"]["value"]) if rows else 0
        else:
            total = len(uris)
        if not total:
            continue
        counts[entity_type] = total

        for tier in PROPERTY_TYPE_ORDER:
            tier_properties = grouped.get(tier, [])
            if not tier_properties:
                continue
            fractions = []
            for prop in tier_properties:
                fraction = len(present.get(prop["uri"], set())) / total
                property_scores[f"{entity_type}/{prop['label']}"] = round(100 * fraction, 1)
                fractions.append(fraction)
            tier_scores[f"{entity_type}/{tier}"] = round(
                100 * sum(fractions) / len(fractions), 1
            )

    return tier_scores, property_scores, counts


# ---------------------------------------------------------------------------
# Per-request memoisation
#
# The three domain plots each need the same cohort, and the tier view needs the
# expensive all-AOPs baseline. Keyed by graph (and domain) so the version
# selector stays correct; small enough to keep unbounded within a worker's life.
# ---------------------------------------------------------------------------

_MEMBER_CACHE: Dict[Tuple[str, str], Dict[str, Set[str]]] = {}
_BASELINE_CACHE: Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]] = {}


def _members(target_graph: str, domain: str) -> Dict[str, Set[str]]:
    key = (target_graph, domain)
    if key not in _MEMBER_CACHE:
        _MEMBER_CACHE[key] = _query_domain_members(target_graph, domain_term_iris(domain))
    return _MEMBER_CACHE[key]


def _baseline(target_graph: str):
    if target_graph not in _BASELINE_CACHE:
        _BASELINE_CACHE[target_graph] = _tier_scores(
            target_graph, {"AOP": None, "KE": None, "KER": None}
        )
    return _BASELINE_CACHE[target_graph]


def _cache_plot(stub: str, version_key: str, df: pd.DataFrame, fig) -> None:
    """Cache data + figure under both the versioned and bare keys.

    The bare key is what the generic ``/download/latest/<name>`` route falls
    back to, so a CSV or PNG export returns the domain the user is looking at
    rather than the default.
    """
    _plot_data_cache[f"{stub}_{version_key}"] = df
    _plot_data_cache[stub] = df
    _plot_figure_cache[f"{stub}_{version_key}"] = fig
    _plot_figure_cache[stub] = fig


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_latest_domain_coverage(version: str = None, domain: str = DEFAULT_DOMAIN) -> str:
    """AOPs per biological domain, selected domain highlighted (issue #149).

    Each bar counts the AOPs with at least one Key Event annotated with a term
    from that domain's ontology closure. Read it as "AOPs that make a
    machine-readable statement about this domain" — an AOP describing cardiac
    biology only in free text is not counted, which is the same blind spot the
    coverage-holes chart reports from the other side.
    """
    cache = _load_domain_lens_cache()
    domains = cache.get("domains", {})
    if not domains:
        return create_fallback_plot(
            "AOP Coverage by Biological Domain",
            "Domain lens cache missing — run scripts/build_domain_lens_cache.py",
        )
    domain = _resolve_domain(domain)

    target = _resolve_target_graph(version)
    if target is None:
        return create_fallback_plot("AOP Coverage by Biological Domain", "No graphs available")
    target_graph, version_label = target

    total_rows = run_sparql_query(f"""
    SELECT (COUNT(DISTINCT ?aop) AS ?count)
    WHERE {{ GRAPH <{target_graph}> {{ ?aop a aopo:AdverseOutcomePathway . }} }}
    """) or []
    total_aops = int(total_rows[0]["count"]["value"]) if total_rows else 0
    if not total_aops:
        return create_fallback_plot(
            "AOP Coverage by Biological Domain", f"No AOPs in snapshot {version_label}"
        )

    records = []
    for name in sorted(domains):
        members = _members(target_graph, name)
        entry = domains[name]
        records.append({
            "Domain": name,
            "AOPs": len(members),
            "Share of all AOPs (%)": round(100 * len(members) / total_aops, 1),
            "Ontology terms in use": len(entry.get("terms", {})),
            "Roots": ", ".join(r["label"] for r in entry.get("roots", [])),
            "Version": version_label,
        })

    df = pd.DataFrame(records).sort_values("AOPs", ascending=True)
    selected = df[df["Domain"] == domain]
    selected_count = int(selected["AOPs"].iloc[0]) if not selected.empty else 0

    fig = go.Figure(
        go.Bar(
            x=df["AOPs"],
            y=df["Domain"],
            orientation="h",
            text=df["AOPs"],
            textposition="outside",
            marker_color=[
                BRAND_COLORS["magenta"] if d == domain else BRAND_COLORS["primary"]
                for d in df["Domain"]
            ],
            customdata=df[["Share of all AOPs (%)", "Ontology terms in use", "Roots"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>%{x} AOPs (%{customdata[0]}% of all)<br>"
                "%{customdata[1]} ontology terms in use<br>"
                "roots: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    subtitle = (
        f"{domain}: {selected_count} of {total_aops} AOPs "
        f"({100 * selected_count / total_aops:.1f}%) in snapshot {version_label}."
        f"<br>Domains overlap — an AOP annotated in two of them is counted in both."
    )
    fig.update_layout(
        title={"text": f"AOP Coverage by Biological Domain<br><sub>{subtitle}</sub>"},
        xaxis_title="Number of AOPs",
        yaxis_title="",
        margin=dict(l=160, r=40, t=110, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_color=BRAND_COLORS["primary"],
        showlegend=False,
    )
    pad_axis_for_outside_labels(fig, axis="x")

    _cache_plot("latest_domain_coverage", version or "latest", df, fig)
    return render_plot_html(fig)


def plot_latest_domain_completeness(version: str = None, domain: str = DEFAULT_DOMAIN) -> str:
    """Property-tier presence for one domain against all AOPs (issue #149).

    Scored with the same ``property_labels.csv`` tiers as the property-presence
    charts, scoped by ``applies_to``, so a domain reads consistently with the
    rest of the dashboard instead of against a bespoke basket of counts.
    Placeholder values (``&nbsp;``, ``TBD``, ``N/A``) are stripped before
    counting — they are 7.2% of Overall Assessment on the 2026-07-01 snapshot.
    """
    cache = _load_domain_lens_cache()
    if not cache.get("domains"):
        return create_fallback_plot(
            "Curation Depth by Property Tier",
            "Domain lens cache missing — run scripts/build_domain_lens_cache.py",
        )
    domain = _resolve_domain(domain)

    target = _resolve_target_graph(version)
    if target is None:
        return create_fallback_plot("Curation Depth by Property Tier", "No graphs available")
    target_graph, version_label = target

    members = _members(target_graph, domain)
    if not members:
        return create_fallback_plot(
            "Curation Depth by Property Tier",
            f"No {domain} AOPs in snapshot {version_label}",
        )

    aop_uris = sorted(members)
    kes, kers = _query_cohort_entities(target_graph, aop_uris)
    domain_tiers, _, domain_counts = _tier_scores(
        target_graph, {"AOP": aop_uris, "KE": sorted(kes), "KER": sorted(kers)}
    )
    all_tiers, _, all_counts = _baseline(target_graph)

    keys = [
        f"{entity}/{tier}"
        for entity in _TIER_ENTITIES
        for tier in PROPERTY_TYPE_ORDER
        if f"{entity}/{tier}" in domain_tiers or f"{entity}/{tier}" in all_tiers
    ]
    if not keys:
        return create_fallback_plot(
            "Curation Depth by Property Tier", "No scorable properties for this cohort"
        )

    df = pd.DataFrame([
        {
            "Tier": key,
            "Entity": key.split("/", 1)[0],
            "Property tier": key.split("/", 1)[1],
            f"{domain} (%)": domain_tiers.get(key),
            "All AOPs (%)": all_tiers.get(key),
            "Difference (pp)": (
                round(domain_tiers[key] - all_tiers[key], 1)
                if key in domain_tiers and key in all_tiers else None
            ),
            "Version": version_label,
        }
        for key in keys
    ])

    order = list(reversed(keys))  # first tier at the top of a horizontal bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[all_tiers.get(k) for k in order],
        y=order,
        orientation="h",
        name="All AOPs",
        marker_color=BRAND_COLORS["primary"],
        hovertemplate="<b>%{y}</b><br>all AOPs: %{x}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[domain_tiers.get(k) for k in order],
        y=order,
        orientation="h",
        name=domain,
        marker_color=BRAND_COLORS["magenta"],
        hovertemplate="<b>%{y}</b><br>" + domain + ": %{x}%<extra></extra>",
    ))

    # Two sub-lines rather than one: the cohort sizes run past the ~700px
    # canvas on a single line (#plot-review).
    cohort_note = (
        f"{domain}: {domain_counts.get('AOP', 0)} AOPs, "
        f"{domain_counts.get('KE', 0)} KEs, {domain_counts.get('KER', 0)} KERs. "
        f"All AOPs: {all_counts.get('AOP', 0)}, {all_counts.get('KE', 0)}, "
        f"{all_counts.get('KER', 0)}."
    )
    fig.update_layout(
        title={
            "text": "Curation Depth by Property Tier<br>"
                    f"<sub>Mean share of entities carrying each tier's properties, "
                    f"snapshot {version_label}.<br>{cohort_note}</sub>"
        },
        barmode="group",
        xaxis_title="Entities carrying the property (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
        margin=dict(l=190, r=40, t=120, b=90),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_color=BRAND_COLORS["primary"],
        # Below the plot: the title already runs to three lines, and a top-right
        # legend lands on top of the second sub-line.
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )

    _cache_plot("latest_domain_completeness", version or "latest", df, fig)
    return render_plot_html(fig)


def plot_latest_domain_status(version: str = None, domain: str = DEFAULT_DOMAIN) -> str:
    """OECD status mix of a domain's AOPs against the whole wiki (issue #149).

    Percentages, because the two cohorts differ by an order of magnitude in
    size. Absence of a status overwhelmingly means "never entered the OECD
    process", not "failed" — it is displayed, never scored.
    """
    cache = _load_domain_lens_cache()
    if not cache.get("domains"):
        return create_fallback_plot(
            "OECD Status of Domain AOPs",
            "Domain lens cache missing — run scripts/build_domain_lens_cache.py",
        )
    domain = _resolve_domain(domain)

    target = _resolve_target_graph(version)
    if target is None:
        return create_fallback_plot("OECD Status of Domain AOPs", "No graphs available")
    target_graph, version_label = target

    statuses = _query_aop_status(target_graph)
    if not statuses:
        return create_fallback_plot(
            "OECD Status of Domain AOPs", f"No AOPs in snapshot {version_label}"
        )
    members = _members(target_graph, domain)
    if not members:
        return create_fallback_plot(
            "OECD Status of Domain AOPs", f"No {domain} AOPs in snapshot {version_label}"
        )

    def _mix(aops: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for aop in aops:
            label = statuses.get(aop, "No Status")
            counts[label] = counts.get(label, 0) + 1
        return counts

    domain_mix = _mix(members)
    all_mix = _mix(statuses)
    # Curated order first, then any status the order doesn't know about (older
    # snapshots carry retired EAGMST/TFHA labels) rather than dropping it.
    present = [s for s in OECD_STATUS_ORDER if s in domain_mix or s in all_mix]
    present += sorted((set(domain_mix) | set(all_mix)) - set(present))

    n_domain, n_all = len(members), len(statuses)
    df = pd.DataFrame([
        {
            "OECD status": status,
            f"{domain} AOPs": domain_mix.get(status, 0),
            f"{domain} (%)": round(100 * domain_mix.get(status, 0) / n_domain, 1),
            "All AOPs": all_mix.get(status, 0),
            "All AOPs (%)": round(100 * all_mix.get(status, 0) / n_all, 1),
            "Version": version_label,
        }
        for status in present
    ])

    fig = go.Figure()
    for cohort, mix, total in ((f"{domain} (n={n_domain})", domain_mix, n_domain),
                               (f"All AOPs (n={n_all})", all_mix, n_all)):
        fig.add_trace(go.Bar(
            x=present,
            y=[round(100 * mix.get(s, 0) / total, 1) for s in present],
            name=cohort,
            customdata=[[mix.get(s, 0)] for s in present],
            marker_color=(BRAND_COLORS["magenta"] if cohort.startswith(domain)
                          else BRAND_COLORS["primary"]),
            hovertemplate="<b>%{x}</b><br>%{y}% (%{customdata[0]} AOPs)<extra></extra>",
        ))

    fig.update_layout(
        title={
            # Wrapped onto a second sub-line so it doesn't run off the ~700px
            # canvas (#plot-review).
            "text": "OECD Status of Domain AOPs<br>"
                    f"<sub>Share of each cohort by OECD workflow status, snapshot "
                    f"{version_label}.<br>No status means the AOP never entered "
                    f"the OECD process.</sub>"
        },
        barmode="group",
        xaxis_title="",
        yaxis_title="Share of cohort (%)",
        margin=dict(l=70, r=40, t=120, b=100),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_color=BRAND_COLORS["primary"],
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
    )

    _cache_plot("latest_domain_status", version or "latest", df, fig)
    return render_plot_html(fig)


def serialise_domain_lens() -> dict:
    """Payload for ``/api/domain-lens`` — the vocabulary behind the plots.

    Exposes provenance, the root set per domain and every in-use term with its
    label, so a reviewer can audit the scope without re-running the build.
    """
    cache = _load_domain_lens_cache()
    return {
        "generated_at": cache.get("generated_at"),
        "default_domain": cache.get("default_domain", DEFAULT_DOMAIN),
        "source": cache.get("source", {}),
        "stats": cache.get("stats", {}),
        "placeholder_tokens": sorted(_PLACEHOLDER_TOKENS - {""}),
        "domains": cache.get("domains", {}),
    }
