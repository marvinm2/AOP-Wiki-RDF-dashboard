"""Version Diff Explorer: what changed between two snapshots (issue #188).

Three views of one comparison, version A ("against") to version B ("version"):

- **Summary**: per entity type, how many were added, removed, edited, changed
  only in their links, or changed only in their metadata.
- **OECD stages**: a two-column sankey of the AOPs that changed OECD stage
  (or appeared or disappeared) between A and B.
- **Worklist**: every changed entity, with the fields that changed and its
  AOP-Wiki link.

How a change is found. For every entity of the four types, one query per
graph returns a short hash of each of its triples, ``<predicate hash><triple
hash>``, concatenated (``FINGERPRINT_QUERY``). An entity whose multiset of
hashes differs between A and B changed, and the predicates whose hashes differ
are the fields that changed. This is exact up to hash collisions (6 + 10 hex
characters) and needs no ``dcterms:modified``, which Key Events do not have
and which, for the other types, moves without content changes (the 2024-04-01
licence wipe touched every AOP). The graphs hold no blank nodes, so a triple's
hash is the same in every graph it appears in.

Every query names one graph (``GRAPH <...>``), so the query store keys each on
that graph alone: the build stores them per version (the default comparison is
the previous version), and any pair of versions is then answered from the
store. Computed comparisons are kept in a small LRU.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from . import query_store
from .shared import (
    BRAND_COLORS,
    ENTITY_TYPE_CLASSES,
    SparqlUnavailable,
    _plot_data_cache,
    _plot_figure_cache,
    get_all_versions,
    render_plot_html,
    resolves_version,
    run_sparql_query_with_retry,
)
from .versions import GRAPH_PREFIX

logger = logging.getLogger(__name__)

# The comparison a page opens with, and the build stores: B against the version before it.
PREVIOUS = "previous"

CHANGE_KINDS = ("Added", "Removed", "Edited", "Links only", "Metadata only")
CHANGE_COLOURS = dict(zip(CHANGE_KINDS, (BRAND_COLORS["palette"][2], BRAND_COLORS["palette"][1],
                                         BRAND_COLORS["palette"][0], BRAND_COLORS["palette"][5],
                                         "#BFBFBF")))

# Predicates that tie an entity to others rather than describe it: a KE's
# dcterms:isPartOf changes when it joins another AOP, untouched itself.
LINK_PREDICATES = frozenset({
    "http://purl.org/dc/terms/isPartOf",
    "http://aopkb.org/aop_ontology#has_key_event",
    "http://aopkb.org/aop_ontology#has_key_event_relationship",
    "http://aopkb.org/aop_ontology#has_molecular_initiating_event",
    "http://aopkb.org/aop_ontology#has_adverse_outcome",
    "http://aopkb.org/aop_ontology#has_upstream_key_event",
    "http://aopkb.org/aop_ontology#has_downstream_key_event",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C54571",
})
# Bookkeeping AOP-Wiki sets itself; the 2024-04-01 wipe changed these on every AOP.
METADATA_PREDICATES = frozenset({
    "http://purl.org/dc/terms/modified",
    "http://purl.org/dc/terms/created",
    "http://purl.org/dc/terms/license",
    "http://purl.org/dc/terms/accessRights",
})

P_LEN, T_LEN = 6, 10

FINGERPRINT_QUERY = """SELECT ?type ?entity (GROUP_CONCAT(CONCAT(SUBSTR(MD5(STR(?p)), 1, 6),
       SUBSTR(MD5(CONCAT(STR(?p), " ", STR(?o))), 1, 10)); separator=" ") AS ?hashes)
WHERE {
  GRAPH __GRAPH_URI__ {
    VALUES ?type { aopo:AdverseOutcomePathway aopo:KeyEvent aopo:KeyEventRelationship nci:C54571 }
    ?entity a ?type ; ?p ?o .
  }
}
GROUP BY ?type ?entity"""

PREDICATE_QUERY = """SELECT DISTINCT ?p
WHERE {
  GRAPH __GRAPH_URI__ {
    VALUES ?type { aopo:AdverseOutcomePathway aopo:KeyEvent aopo:KeyEventRelationship nci:C54571 }
    ?entity a ?type ; ?p ?o .
  }
}"""

LABEL_QUERY = """SELECT ?entity (SAMPLE(?t) AS ?title) (GROUP_CONCAT(DISTINCT ?s; separator="|") AS ?statuses)
WHERE {
  GRAPH __GRAPH_URI__ {
    VALUES ?type { aopo:AdverseOutcomePathway aopo:KeyEvent aopo:KeyEventRelationship nci:C54571 }
    ?entity a ?type .
    OPTIONAL { ?entity dc:title ?t . }
    OPTIONAL { ?entity nci:C25688 ?s . }
  }
}
GROUP BY ?entity"""

QUERIES = (FINGERPRINT_QUERY, PREDICATE_QUERY, LABEL_QUERY)

WORKLIST_COLUMNS = ["Type", "ID", "Name", "Change", "Fields Changed", "OECD Stage A", "OECD Stage B", "Link"]
SUMMARY_COLUMNS = ["Entity Type", *CHANGE_KINDS, "In A", "In B"]


def _type_name(class_iri: str) -> str:
    local = class_iri.rsplit("#", 1)[-1]
    for name, prefixed in ENTITY_TYPE_CLASSES.items():
        if prefixed.split(":", 1)[1] == local:
            return name
    return local


def _labels() -> Dict[str, str]:
    """Readable names of predicates, from property_labels.csv."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "property_labels.csv")
    try:
        with open(path, newline="") as fh:
            return {row["uri"]: row["label"] for row in csv.DictReader(fh)}
    except OSError:
        return {}


# property_labels.csv names predicates as they read on an AOP; on a KE, KER
# or stressor dcterms:isPartOf is the AOPs it belongs to.
PROPERTY_LABELS = {**_labels(), "http://purl.org/dc/terms/isPartOf": "Part of AOP"}


def field_name(predicate: str) -> str:
    if predicate in PROPERTY_LABELS:
        return PROPERTY_LABELS[predicate]
    return predicate.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def predicate_hash(predicate: str) -> str:
    """The predicate part of a triple hash, as FINGERPRINT_QUERY computes it."""
    return hashlib.md5(predicate.encode()).hexdigest()[:P_LEN]


class Snapshot:
    """What one version contributes to a comparison."""

    def __init__(self, hashes: Dict[str, collections.Counter], types: Dict[str, str],
                 predicates: Dict[str, str], titles: Dict[str, str], statuses: Dict[str, str]):
        self.hashes = hashes          # entity -> Counter of triple hashes
        self.types = types            # entity -> "AOPs" | "KEs" | "KERs" | "Stressors"
        self.predicates = predicates  # predicate hash -> predicate IRI
        self.titles = titles
        self.statuses = statuses      # entity -> "|"-joined OECD statuses (AOPs that have one)


def _run(template: str, version: str) -> list:
    query = template.replace("__GRAPH_URI__", f"<{GRAPH_PREFIX}{version}>")
    rows = run_sparql_query_with_retry(query, use_post=True)
    if not rows:
        raise SparqlUnavailable(f"no rows for the version diff of {version}")
    return rows


def fetch_snapshot(version: str) -> Snapshot:
    hashes, types = {}, {}
    for r in _run(FINGERPRINT_QUERY, version):
        entity = r["entity"]["value"]
        hashes[entity] = collections.Counter(r["hashes"]["value"].split())
        types[entity] = _type_name(r["type"]["value"])
    predicates = {predicate_hash(r["p"]["value"]): r["p"]["value"] for r in _run(PREDICATE_QUERY, version)}
    titles, statuses = {}, {}
    for r in _run(LABEL_QUERY, version):
        entity = r["entity"]["value"]
        if "title" in r:
            titles[entity] = " ".join(r["title"]["value"].split())
        if r.get("statuses", {}).get("value"):
            statuses[entity] = r["statuses"]["value"]
    return Snapshot(hashes, types, predicates, titles, statuses)


def changed_predicates(a: collections.Counter, b: collections.Counter) -> List[str]:
    """Predicate hashes whose triples differ between two entity fingerprints."""
    by_a, by_b = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for token, n in a.items():
        by_a[token[:P_LEN]][token] += n
    for token, n in b.items():
        by_b[token[:P_LEN]][token] += n
    return sorted(h for h in set(by_a) | set(by_b) if by_a.get(h) != by_b.get(h))


def change_kind(predicates: List[str]) -> str:
    """Edited, unless only link or only metadata predicates changed."""
    if all(p in METADATA_PREDICATES for p in predicates):
        return "Metadata only"
    if all(p in LINK_PREDICATES or p in METADATA_PREDICATES for p in predicates):
        return "Links only"
    return "Edited"


def _entity_id(entity: str) -> str:
    return entity.rstrip("/").rsplit("/", 1)[-1]


def _link(entity: str) -> str:
    # identifiers.org IRIs resolve to the AOP-Wiki page.
    return entity.replace("http://", "https://", 1) if entity.startswith("http://identifiers.org") else entity


def _stage(statuses: Optional[str]) -> str:
    """The OECD stage of "|"-joined statuses; the work-plan flag is not a stage (see trends_plots)."""
    from .trends_plots import _oecd_stage
    return _oecd_stage(statuses.split("|") if statuses else [])


def diff_rows(a: Snapshot, b: Snapshot) -> pd.DataFrame:
    """One row per entity that was added, removed or changed from A to B."""
    order = {name: i for i, name in enumerate(ENTITY_TYPE_CLASSES)}
    kind_order = {k: i for i, k in enumerate(CHANGE_KINDS)}
    predicates = {**a.predicates, **b.predicates}
    rows = []
    for entity in set(a.hashes) | set(b.hashes):
        in_a, in_b = entity in a.hashes, entity in b.hashes
        fields: List[str] = []
        if in_a and in_b:
            if a.hashes[entity] == b.hashes[entity]:
                continue
            changed = [predicates.get(h, h) for h in changed_predicates(a.hashes[entity], b.hashes[entity])]
            kind = change_kind(changed)
            fields = sorted({field_name(p) for p in changed}, key=str.lower)
        else:
            kind = "Added" if in_b else "Removed"
        etype = (b if in_b else a).types[entity]
        is_aop = etype == "AOPs"
        rows.append({
            "Type": etype,
            "ID": _entity_id(entity),
            "Name": b.titles.get(entity) or a.titles.get(entity) or "",
            "Change": kind,
            "Fields Changed": "; ".join(fields),
            "OECD Stage A": _stage(a.statuses.get(entity)) if is_aop and in_a else "",
            "OECD Stage B": _stage(b.statuses.get(entity)) if is_aop and in_b else "",
            "Link": _link(entity),
        })
    rows.sort(key=lambda r: (order.get(r["Type"], 9), kind_order[r["Change"]],
                             int(r["ID"]) if r["ID"].isdigit() else 10**9, r["ID"]))
    return pd.DataFrame(rows, columns=WORKLIST_COLUMNS)


def summary_rows(worklist: pd.DataFrame, a: Snapshot, b: Snapshot) -> pd.DataFrame:
    rows = []
    for etype in ENTITY_TYPE_CLASSES:
        sub = worklist[worklist["Type"] == etype]
        row = {"Entity Type": etype}
        for kind in CHANGE_KINDS:
            row[kind] = int((sub["Change"] == kind).sum())
        row["In A"] = sum(1 for t in a.types.values() if t == etype)
        row["In B"] = sum(1 for t in b.types.values() if t == etype)
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def stage_links(worklist: pd.DataFrame, version_a: str, version_b: str) -> Tuple[pd.DataFrame, int]:
    """Sankey links of the AOPs that changed OECD stage, appeared or disappeared."""
    from .trends_plots import OECD_JOURNEY_COLUMNS
    aops = worklist[worklist["Type"] == "AOPs"]
    moved = aops[(aops["OECD Stage A"] != aops["OECD Stage B"])].copy()
    moved["OECD Stage A"] = moved["OECD Stage A"].replace("", f"Not in {version_a}")
    moved["OECD Stage B"] = moved["OECD Stage B"].replace("", f"Not in {version_b}")
    counts = moved.groupby(["OECD Stage A", "OECD Stage B"]).size()
    links = pd.DataFrame([
        {"From Snapshot": version_a, "To Snapshot": version_b, "From Status": x, "To Status": y, "AOPs": int(n)}
        for (x, y), n in counts.items()
    ], columns=OECD_JOURNEY_COLUMNS)
    return links, int(len(moved))


# --- the comparison, cached --------------------------------------------------

_LRU_SIZE = 12
_lru: "collections.OrderedDict[tuple, dict]" = collections.OrderedDict()
_lru_lock = threading.Lock()


def _freshness() -> object:
    """What a cached comparison is valid for: the store build it came from, or ten minutes live."""
    store = query_store.active_store()
    manifest = store.manifest() if store is not None else None
    return manifest.built_at if manifest is not None else int(time.time() // 600)


def resolve_against(version: str, against: Optional[str]) -> Optional[str]:
    """Version A for ``against``: a version older than B, else the one before B."""
    versions = sorted(v["version"] for v in get_all_versions())
    if against and against != PREVIOUS:
        if against not in versions:
            raise ValueError(f"unknown version {against}")
        return against
    older = [v for v in versions if v < version]
    return older[-1] if older else None


def _with_pair(df: pd.DataFrame, version_a: str, version_b: str) -> pd.DataFrame:
    """The frame with the pair it compares as its first columns, so a CSV says what it is."""
    out = df.copy()
    out.insert(0, "Version B", version_b)
    out.insert(0, "Version A", version_a)
    return out


def compare(version_a: str, version_b: str) -> dict:
    key = (version_a, version_b, _freshness())
    with _lru_lock:
        if key in _lru:
            _lru.move_to_end(key)
            return _lru[key]
    a, b = fetch_snapshot(version_a), fetch_snapshot(version_b)
    worklist = diff_rows(a, b)
    result = {"worklist": _with_pair(worklist, version_a, version_b),
              "summary": _with_pair(summary_rows(worklist, a, b), version_a, version_b),
              "stage_links": stage_links(worklist, version_a, version_b)}
    with _lru_lock:
        _lru[key] = result
        while len(_lru) > _LRU_SIZE:
            _lru.popitem(last=False)
    return result


# --- the three views -----------------------------------------------------------

def _pair_title(title: str, a: str, b: str) -> str:
    return f"{title}: {a} → {b}"


def _render_summary(summary: pd.DataFrame, a: str, b: str) -> go.Figure:
    types = list(summary["Entity Type"])[::-1]   # AOPs on top
    fig = go.Figure()
    for kind in CHANGE_KINDS:
        values = summary.set_index("Entity Type").loc[types, kind].tolist()
        fig.add_bar(y=types, x=values, name=kind, orientation="h", marker_color=CHANGE_COLOURS[kind],
                    hovertemplate=f"%{{y}}, {kind.lower()}: %{{x}}<extra></extra>")
    touched = int(summary[list(CHANGE_KINDS)].to_numpy().sum())
    fig.update_layout(
        barmode="stack",
        title=dict(text=_pair_title("What Changed", a, b)
                   + f"<br><sup>{touched} entities added, removed or changed; "
                     "an entity counts once, under the strongest kind of change</sup>"),
        # The legend hangs from just under the axis, so where it wraps (a
        # phone) it pushes the margin down instead of covering the ticks.
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, traceorder="normal"),
        height=380, margin=dict(l=90, r=30, t=80, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _message_figure(title: str, text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=BRAND_COLORS["primary"]))
    fig.update_layout(title=dict(text=title), height=260, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


PARTS = {
    # part: (plot id, title, columns of an empty frame)
    "summary": ("latest_version_diff", "What Changed", SUMMARY_COLUMNS),
    "stages": ("latest_version_diff_stages", "OECD Stage Moves", None),
    "worklist": ("latest_version_diff_worklist", "Changed Entities", WORKLIST_COLUMNS),
}


@resolves_version
def plot_latest_version_diff(version: str = None, against: str = PREVIOUS, part: str = "summary") -> str:
    """One view of the comparison of ``against`` (A) with ``version`` (B) (#188).

    ``part`` is "summary", "stages" or "worklist". Each caches its frame and
    figure under its plot id and B, in every branch: the plot takes settings,
    so every request renders it again and must never find another pair's
    result in the cache.
    """
    from .trends_plots import OECD_JOURNEY_COLUMNS, _render_oecd_status_journey
    plot_id, title, columns = PARTS[part]
    cache_key = f"{plot_id}_{version}"

    def publish(df: pd.DataFrame, fig: go.Figure) -> str:
        _plot_data_cache[cache_key] = df
        _plot_figure_cache[cache_key] = fig
        return render_plot_html(fig)

    def message(text: str) -> str:
        empty = pd.DataFrame(columns=["Version A", "Version B", *columns] if columns else OECD_JOURNEY_COLUMNS)
        return publish(empty, _message_figure(title, text))

    try:
        version_a = resolve_against(version, against)
    except ValueError as e:
        return message(str(e).capitalize())
    if version_a is None:
        return message(f"No version before {version} to compare with")
    if version_a >= version:
        return message(f"Version A ({version_a}) must be older than version B ({version})")
    result = compare(version_a, version)

    if part == "summary":
        return publish(result["summary"], _render_summary(result["summary"], version_a, version))

    if part == "stages":
        links, moved = result["stage_links"]
        if links.empty:
            return message(f"No AOP changed OECD stage from {version_a} to {version}")
        fig = _render_oecd_status_journey(links, moved, moved)
        fig.update_layout(title=dict(text=_pair_title(title, version_a, version)
                                     + f"<br><sup>{moved} AOPs changed stage, appeared or disappeared; "
                                       "AOPs that kept their stage are left out</sup>"), height=520)
        return publish(links, fig)

    # The page draws the worklist as a table from /api/v2/data.
    return publish(result["worklist"], _message_figure(_pair_title(title, version_a, version),
                                                       f"{len(result['worklist'])} changed entities: see the table"))
