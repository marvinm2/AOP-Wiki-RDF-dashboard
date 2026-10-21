"""AOP composition history: one AOP's Key Events in every snapshot (issue #190).

For one AOP (``?aop=<number>``), the strip on ``/aop`` has a row per Key Event
the AOP has ever had and a column per snapshot up to the selected version. A
cell shows the KE's role in the AOP in that snapshot (MIE, KE or AO), and the
KEs added to or removed from the AOP are marked. A first row gives the AOP's
own OECD stage per snapshot.

Two small queries per AOP, both over every snapshot graph with ``VALUES ?aop``:
the KEs it links with ``has_key_event``, ``has_molecular_initiating_event`` and
``has_adverse_outcome``, and its title and OECD status. They take about 0.1 s
on the endpoint, so they run on demand rather than in the store build; the
result is kept per AOP in a small LRU. KE titles are the newest ``dc:title``,
from the query the lifeline sends (in the query store already).
"""

from __future__ import annotations

import collections
import logging
import re
import threading
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from .diff_plots import _freshness, _message_figure
from .shared import (
    BRAND_COLORS,
    SparqlUnavailable,
    _plot_data_cache,
    _plot_figure_cache,
    get_all_versions,
    render_plot_html,
    resolves_version,
    run_sparql_query_with_retry,
)
from .trends_plots import _lifeline_labels, _oecd_stage

logger = logging.getLogger(__name__)

AOP_IRI = "https://identifiers.org/aop/"
PLOT_ID = "latest_aop_composition"
TITLE = "AOP Composition History"

COLUMNS = ["Element", "ID", "Title", "Role", "First Seen", "Last Seen", "Roles", "Stages"]
ABSENT = "-"
# "Roles" has one character per snapshot: the KE's role in the AOP, or "-" (the AOP row: "x" when the
# AOP exists). "Stages" (the AOP row only) is the AOP's OECD stage per snapshot, "|"-separated.
ROLE_CODES = {"MIE": "M", "KE": "K", "AO": "A"}
ROLE_NAMES = {code: name for name, code in ROLE_CODES.items()}
ROLE_ORDER = {"MIE": 0, "KE": 1, "AO": 2}
PREDICATE_ROLES = {
    "http://aopkb.org/aop_ontology#has_molecular_initiating_event": "MIE",
    "http://aopkb.org/aop_ontology#has_adverse_outcome": "AO",
    "http://aopkb.org/aop_ontology#has_key_event": "KE",
}

KE_QUERY = """SELECT ?graph ?p ?ke
WHERE {
  VALUES ?aop { __AOP__ }
  VALUES ?p { aopo:has_key_event aopo:has_molecular_initiating_event aopo:has_adverse_outcome }
  GRAPH ?graph { ?aop ?p ?ke . }
  FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
}"""

AOP_QUERY = """SELECT ?graph (SAMPLE(?t) AS ?title) (GROUP_CONCAT(DISTINCT ?s; separator="|") AS ?statuses)
WHERE {
  VALUES ?aop { __AOP__ }
  GRAPH ?graph {
    ?aop a aopo:AdverseOutcomePathway .
    OPTIONAL { ?aop dc:title ?t . }
    OPTIONAL { ?aop nci:C25688 ?s . }
  }
  FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
}
GROUP BY ?graph"""

QUERIES = (KE_QUERY, AOP_QUERY)


def aop_number(value) -> Optional[int]:
    """The AOP number in ``3``, ``AOP 3`` or its IRI; None for anything else."""
    m = re.fullmatch(r"\s*(?:AOP\s*|" + re.escape(AOP_IRI) + r")?(\d{1,6})\s*", str(value or ""), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _query(template: str, number: int) -> str:
    return template.replace("__AOP__", f"<{AOP_IRI}{number}>")


def _version(graph_uri: str) -> str:
    return graph_uri.rsplit("/", 1)[-1]


def _ke_label(uri: str) -> str:
    return f"KE {uri.rstrip('/').rsplit('/', 1)[-1]}"


def _ke_number(uri: str) -> int:
    tail = uri.rstrip("/").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else 10 ** 9


def role_in(predicates: set) -> str:
    """The role a KE plays given the predicates linking it: MIE before AO before KE."""
    roles = {PREDICATE_ROLES[p] for p in predicates if p in PREDICATE_ROLES}
    for role in ("MIE", "AO", "KE"):
        if role in roles:
            return role
    return "KE"


def composition_rows(number: int, ke_links: List[tuple], aop_rows: List[tuple],
                     versions: List[str], ke_titles: Dict[str, str]) -> pd.DataFrame:
    """The strip's rows: the AOP itself, then one per KE it ever had.

    ``ke_links`` holds ``(version, predicate, ke_iri)``; ``aop_rows`` holds
    ``(version, title, [statuses])``. Only ``versions`` (oldest first) count.
    KEs are ordered by their role in the newest snapshot that has them (MIE,
    KE, AO), then by the snapshot they joined, then by number.
    """
    wanted = set(versions)
    preds: Dict[str, Dict[str, set]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for version, predicate, ke in ke_links:
        if version in wanted:
            preds[ke][version].add(predicate)
    aop = {v: (title, statuses) for v, title, statuses in aop_rows if v in wanted}

    def span(codes: str):
        idx = [i for i, c in enumerate(codes) if c != ABSENT]
        return (versions[idx[0]], versions[idx[-1]], idx[0]) if idx else ("", "", len(codes))

    aop_codes = "".join("x" if v in aop else ABSENT for v in versions)
    first, last, _ = span(aop_codes)
    newest = aop[last] if last else ("", [])
    rows = [{"Element": "AOP", "ID": f"AOP {number}", "Title": newest[0],
             "Role": "AOP",
             "First Seen": first, "Last Seen": last, "Roles": aop_codes,
             "Stages": "|".join(_oecd_stage(aop[v][1]) if v in aop else "" for v in versions)}]

    kes = []
    for ke, by_version in preds.items():
        codes = "".join(ROLE_CODES[role_in(by_version[v])] if v in by_version else ABSENT for v in versions)
        first, last, start = span(codes)
        role = ROLE_NAMES[codes[versions.index(last)]]
        kes.append(((ROLE_ORDER[role], start, _ke_number(ke)),
                    {"Element": "KE", "ID": _ke_label(ke), "Title": ke_titles.get(ke, ""), "Role": role,
                     "First Seen": first, "Last Seen": last, "Roles": codes, "Stages": ""}))
    rows += [row for _, row in sorted(kes, key=lambda pair: pair[0])]
    return pd.DataFrame(rows, columns=COLUMNS)


def changes(codes: str) -> tuple[int, int]:
    """(times added, times removed) in a roles string, after its first snapshot."""
    added = removed = 0
    for before, now in zip(codes, codes[1:]):
        if before == ABSENT and now != ABSENT:
            added += 1
        elif before != ABSENT and now == ABSENT:
            removed += 1
    return added, removed


# --- fetched per AOP, cached ------------------------------------------------

_LRU_SIZE = 64
_lru: "collections.OrderedDict[tuple, tuple]" = collections.OrderedDict()
_lru_lock = threading.Lock()


def fetch_aop(number: int) -> tuple[list, list]:
    """``(ke_links, aop_rows)`` for one AOP over every snapshot, as composition_rows takes them."""
    key = (number, _freshness())
    with _lru_lock:
        if key in _lru:
            _lru.move_to_end(key)
            return _lru[key]
    ke_links = [(_version(r["graph"]["value"]), r["p"]["value"], r["ke"]["value"])
                for r in run_sparql_query_with_retry(_query(KE_QUERY, number))]
    aop_rows = []
    for r in run_sparql_query_with_retry(_query(AOP_QUERY, number)):
        statuses = [s for s in r.get("statuses", {}).get("value", "").split("|") if s]
        aop_rows.append((_version(r["graph"]["value"]), " ".join(r.get("title", {}).get("value", "").split()),
                         statuses))
    result = (ke_links, aop_rows)
    with _lru_lock:
        _lru[key] = result
        while len(_lru) > _LRU_SIZE:
            _lru.popitem(last=False)
    return result


_index_lock = threading.Lock()
_index: dict = {}


def aop_index() -> List[dict]:
    """Every AOP ever in AOP-Wiki, ``{number, title}`` by number: the picker's list."""
    key = _freshness()
    with _index_lock:
        if key in _index:
            return _index[key]
    titles = _lifeline_labels("AOPs")
    out = sorted(({"number": n, "title": t} for uri, t in titles.items()
                  if (n := aop_number(uri)) is not None), key=lambda a: a["number"])
    with _index_lock:
        _index.clear()
        _index[key] = out
    return out


# --- the figure -------------------------------------------------------------

def _render(df: pd.DataFrame, versions: List[str]) -> go.Figure:
    """The strip as a Plotly figure, for the PNG/SVG downloads (the page draws it with D3)."""
    aop, kes = df.iloc[0], df.iloc[1:]
    start = next((i for i, c in enumerate(aop["Roles"]) if c != ABSENT), 0)
    cols = versions[start:]
    roles = BRAND_COLORS["network_roles"]
    fig = go.Figure()
    labels = [f"{r.ID} {r.Title[:40]}" for r in kes.itertuples()]
    for code, name in ROLE_NAMES.items():
        xs, ys = [], []
        for label, codes in zip(labels, kes["Roles"]):
            for i, c in enumerate(codes[start:]):
                if c == code:
                    xs.append(cols[i])
                    ys.append(label)
        fig.add_scatter(x=xs, y=ys, mode="markers", name=name,
                        marker=dict(symbol="square", size=11, color=roles[name]),
                        hovertemplate="%{y}<br>%{x}: " + name + "<extra></extra>")
    for kind, symbol, colour in (("added", "triangle-right", BRAND_COLORS["primary"]),
                                 ("removed", "x", "#7F7F7F")):
        xs, ys = [], []
        for label, codes in zip(labels, kes["Roles"]):
            for i in range(start + 1, len(codes)):
                before, now = codes[i - 1], codes[i]
                if (kind == "added" and before == ABSENT and now != ABSENT) or \
                        (kind == "removed" and before != ABSENT and now == ABSENT):
                    xs.append(versions[i])
                    ys.append(label)
        fig.add_scatter(x=xs, y=ys, mode="markers", name=kind.capitalize(),
                        marker=dict(symbol=symbol, size=7, color=colour), hoverinfo="skip")
    added = sum(changes(c[start:])[0] for c in kes["Roles"])
    removed = sum(changes(c[start:])[1] for c in kes["Roles"])
    fig.update_layout(
        title=dict(text=f"{aop.ID}: {aop.Title[:80]}"
                        f"<br><sup>{len(kes)} Key Events in {len(cols)} snapshots up to {versions[-1]}; "
                        f"{added} added and {removed} removed after the AOP first appeared</sup>"),
        height=max(320, 140 + 22 * len(kes)), margin=dict(l=10, r=10, t=80, b=60),
        xaxis=dict(type="category", categoryorder="array", categoryarray=cols, tickangle=-45, title="Snapshot"),
        yaxis=dict(type="category", categoryorder="array", categoryarray=labels[::-1], automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@resolves_version
def plot_latest_aop_composition(version: str = None, aop: str = None) -> str:
    """One AOP's Key Events in every snapshot up to ``version`` (#190).

    Caches its frame and figure under the plot id and version in every branch:
    the plot takes a setting, so every request renders it again and must never
    find another AOP's result in the cache.
    """
    cache_key = f"{PLOT_ID}_{version}"

    def publish(df: pd.DataFrame, fig: go.Figure) -> str:
        _plot_data_cache[cache_key] = df
        _plot_figure_cache[cache_key] = fig
        return render_plot_html(fig)

    def message(text: str) -> str:
        return publish(pd.DataFrame(columns=COLUMNS), _message_figure(TITLE, text))

    number = aop_number(aop)
    if aop in (None, ""):
        return message("Pick an AOP to see its Key Events in every version")
    if number is None:
        return message(f"Not an AOP number: {aop}")
    versions = sorted(v["version"] for v in get_all_versions() if v["version"] <= version)
    try:
        ke_links, aop_rows = fetch_aop(number)
    except SparqlUnavailable:
        raise
    except Exception as e:
        logger.error(f"AOP composition history for AOP {number} failed: {e}")
        return message(f"Could not load AOP {number}: {e}")
    if not any(v in versions for v, _, _ in aop_rows):
        return message(f"AOP {number} is not in any version up to {version}")
    df = composition_rows(number, ke_links, aop_rows, versions, _lifeline_labels("KEs"))
    return publish(df, _render(df, versions))
