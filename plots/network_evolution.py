"""Network evolution: the KE/KER network of every snapshot on one layout (issue #189).

``/network`` lays out one version at a time, so the network's growth cannot be
followed: every version gets a fresh layout. This view lays out the union of
all versions once (Fruchterman-Reingold, fixed seed), so a Key Event keeps its
place, and the page shows the elements present in the version the scrubber
picks, marking those new since the version before it and those gone since.

No queries of its own: KE and KER presence per snapshot are the queries the
birth/death and lifeline views send, and the KE labels and KER endpoints the
lifeline's, so the query store already holds all of them.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from .shared import (
    BRAND_COLORS,
    SparqlUnavailable,
    _plot_data_cache,
    _plot_figure_cache,
    create_fallback_plot,
    fetch_entity_uris_by_version,
    get_all_versions,
    render_plot_html,
)
from .trends_plots import ABSENT, PRESENT, _lifeline_labels, ker_endpoints

logger = logging.getLogger(__name__)

COLUMNS = ["Element", "ID", "Label", "Source", "Target", "X", "Y", "First Seen", "Last Seen", "Presence"]
LAYOUT_SEED = 42
LAYOUT_SCALE = 1000


def _short(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]


def presence_string(uri: str, uris_by_version: Dict[str, Set[str]], versions: List[str]) -> str:
    return "".join(PRESENT if uri in uris_by_version.get(v, ()) else ABSENT for v in versions)


def union_graph(ke_by_version: Dict[str, Set[str]], ends: Dict[str, Tuple[str, str]]) -> nx.Graph:
    """Every KE ever seen, and an edge per KER ever seen, built in a fixed order."""
    G = nx.Graph()
    G.add_nodes_from(sorted(set().union(*ke_by_version.values()) if ke_by_version else ()))
    for ker in sorted(ends):
        up, down = ends[ker]
        G.add_edge(up, down)
    return G


def union_layout(G: nx.Graph) -> Dict[str, Tuple[float, float]]:
    """Positions for the union graph: the same input always gives the same picture."""
    if G.number_of_nodes() == 0:
        return {}
    pos = nx.spring_layout(G, seed=LAYOUT_SEED, scale=LAYOUT_SCALE)
    return {n: (round(float(x), 1), round(float(y), 1)) for n, (x, y) in pos.items()}


def evolution_rows(ke_by_version: Dict[str, Set[str]], ker_by_version: Dict[str, Set[str]],
                   ends: Dict[str, Tuple[str, str]], labels: Dict[str, str],
                   versions: List[str], positions: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """One row per KE (a node) and per KER (an edge) ever seen, with its presence per snapshot."""
    def seen(presence):
        idx = [i for i, c in enumerate(presence) if c == PRESENT]
        return (versions[idx[0]], versions[idx[-1]]) if idx else ("", "")

    rows = []
    for ke in sorted(set().union(*ke_by_version.values()) if ke_by_version else (), key=_sort_key):
        presence = presence_string(ke, ke_by_version, versions)
        first, last = seen(presence)
        x, y = positions.get(ke, (0.0, 0.0))
        rows.append({"Element": "KE", "ID": f"KE {_short(ke)}", "Label": labels.get(ke, ""),
                     "Source": "", "Target": "", "X": x, "Y": y,
                     "First Seen": first, "Last Seen": last, "Presence": presence})
    for ker in sorted(set().union(*ker_by_version.values()) if ker_by_version else (), key=_sort_key):
        if ker not in ends:
            continue   # a KER without both endpoints in any snapshot cannot be drawn
        up, down = ends[ker]
        presence = presence_string(ker, ker_by_version, versions)
        first, last = seen(presence)
        rows.append({"Element": "KER", "ID": f"KER {_short(ker)}", "Label": "",
                     "Source": f"KE {_short(up)}", "Target": f"KE {_short(down)}", "X": None, "Y": None,
                     "First Seen": first, "Last Seen": last, "Presence": presence})
    return pd.DataFrame(rows, columns=COLUMNS)


def _sort_key(uri: str):
    tail = _short(uri)
    return (0, int(tail), "") if tail.isdigit() else (1, 0, tail)


def _render(df: pd.DataFrame, versions: List[str]) -> go.Figure:
    """The latest snapshot on the shared layout, for the PNG/SVG downloads and the Plotly fallback."""
    i = len(versions) - 1
    nodes = df[(df["Element"] == "KE") & (df["Presence"].str[i] == PRESENT)]
    at = {r.ID: (r.X, r.Y) for r in df[df["Element"] == "KE"].itertuples()}
    xs, ys = [], []
    for r in df[(df["Element"] == "KER") & (df["Presence"].str[i] == PRESENT)].itertuples():
        if r.Source in at and r.Target in at:
            xs += [at[r.Source][0], at[r.Target][0], None]
            ys += [at[r.Source][1], at[r.Target][1], None]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color="#C8C8D0", width=0.6), hoverinfo="skip", name="KERs")
    fig.add_scatter(x=nodes["X"], y=nodes["Y"], mode="markers", name="KEs",
                    marker=dict(size=4, color=BRAND_COLORS["blue"]),
                    text=nodes["ID"] + ": " + nodes["Label"], hovertemplate="%{text}<extra></extra>")
    fig.update_layout(
        title=dict(text=f"KE Network on the Shared Layout, {versions[i]}"
                        f"<br><sup>{len(nodes)} KEs and {len(xs) // 3} KERs; every snapshot uses the same positions</sup>"),
        showlegend=False, height=640, margin=dict(l=10, r=10, t=70, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_network_evolution() -> tuple[str, pd.DataFrame]:
    """The network of every snapshot on one layout (#189); the page draws it with Cytoscape."""
    title = "Network Evolution"
    try:
        versions = sorted(v["version"] for v in get_all_versions())
        if len(versions) < 2:
            return create_fallback_plot(title, "Fewer than two snapshots"), pd.DataFrame(columns=COLUMNS)
        ke_by_version = fetch_entity_uris_by_version("KEs", versions)
        ker_by_version = fetch_entity_uris_by_version("KERs", versions)
        ends = ker_endpoints()
        if not ke_by_version or not ends:
            raise SparqlUnavailable("no KEs or KERs for the network evolution")
        labels = _lifeline_labels("KEs")
        positions = union_layout(union_graph(ke_by_version, ends))
        df = evolution_rows(ke_by_version, ker_by_version, ends, labels, versions, positions)
        fig = _render(df, versions)
        _plot_data_cache["network_evolution"] = df
        _plot_figure_cache["network_evolution"] = fig
        return render_plot_html(fig), df
    except SparqlUnavailable:
        raise
    except Exception as e:
        logger.error(f"Failed to build the network evolution: {e}")
        return create_fallback_plot(title, f"Error: {e}"), pd.DataFrame(columns=COLUMNS)
