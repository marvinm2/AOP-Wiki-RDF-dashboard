"""AOP-Wiki RDF Dashboard - Network Graph Construction and Analysis.

This module builds an interactive network graph from AOP-Wiki RDF data using
NetworkX. It queries the SPARQL endpoint for the latest version's KE-KER
topology, constructs a directed graph (KEs as nodes, KERs as edges), computes
centrality metrics (degree, betweenness, closeness, PageRank) and Louvain
community detection, then converts the result to Cytoscape.js JSON format for
frontend rendering.

Core Components:
    - Graph construction from 2 bulk SPARQL queries (KEs + KERs)
    - Centrality metrics computation (degree, betweenness, closeness, PageRank)
    - Louvain community detection with deterministic seeding
    - Cytoscape.js JSON conversion with VHP4Safety brand colors
    - Lazy computation with module-level caching

Data Flow:
    SPARQL endpoint -> 2 bulk queries -> NetworkX graph -> metrics + communities
    -> Cytoscape.js JSON -> cached for subsequent requests

Usage:
    >>> from plots.network import get_or_compute_network
    >>> data = get_or_compute_network()
    >>> elements = data['elements']  # Cytoscape.js JSON
    >>> metrics = data['metrics']    # List of metric dicts
"""

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from .shared import (
    BRAND_COLORS,
    _plot_data_cache,
    get_latest_version,
    run_sparql_query_with_retry,
)

logger = logging.getLogger(__name__)

# Module-level cache for lazy computation
_network_cache: Dict = {}


def build_aop_network() -> nx.Graph:
    """Build KE-KER network graph from SPARQL endpoint data.

    Queries the SPARQL endpoint with 2 bulk queries against the latest version
    graph to retrieve Key Events and Key Event Relationships. Builds a NetworkX
    undirected graph with KE nodes and KER edges.

    Returns:
        nx.Graph: NetworkX graph with KE node attributes (type, label, uri,
            wiki_url) and KER edge attributes (type='ker').
    """
    version = get_latest_version()
    target_graph = f"http://aopwiki.org/graph/{version}"
    logger.info(f"Building KE-KER network from graph: {target_graph}")

    # Query 1: Get all KEs with title and foaf:page
    ke_query = f"""
    SELECT ?ke
           (SAMPLE(?t) AS ?title)
           (SAMPLE(STR(?p)) AS ?page)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?ke a aopo:KeyEvent .
            OPTIONAL {{ ?ke dc:title ?t }}
            OPTIONAL {{ ?ke foaf:page ?p }}
        }}
    }}
    GROUP BY ?ke
    """

    # Query 2: Get KER edges with upstream/downstream KEs
    ker_query = f"""
    SELECT ?ker ?upstream ?downstream
    WHERE {{
        GRAPH <{target_graph}> {{
            ?ker a aopo:KeyEventRelationship ;
                 aopo:has_upstream_key_event ?upstream ;
                 aopo:has_downstream_key_event ?downstream .
        }}
    }}
    """

    # Build NetworkX graph
    G = nx.Graph()

    # Add KE nodes
    ke_results = run_sparql_query_with_retry(ke_query)
    for r in ke_results:
        ke_uri = r['ke']['value']
        ke_id = ke_uri.split('/')[-1]
        G.add_node(
            ke_id,
            type='KE',
            label=r.get('title', {}).get('value', ke_id),
            uri=ke_uri,
            wiki_url=r.get('page', {}).get('value', ''),
        )
    logger.info(f"Added {len(ke_results)} KE nodes")

    # Add KER edges (upstream KE -> downstream KE)
    ker_results = run_sparql_query_with_retry(ker_query)
    ker_count = 0
    for r in ker_results:
        up_id = r['upstream']['value'].split('/')[-1]
        down_id = r['downstream']['value'].split('/')[-1]
        if up_id in G and down_id in G:
            G.add_edge(up_id, down_id, type='ker')
            ker_count += 1
    logger.info(f"Added {ker_count} KER edges")

    logger.info(
        f"Network built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def detect_ke_roles(target_graph: str) -> Dict[str, str]:
    """Detect MIE/KE/AO roles for all KEs across all AOPs.

    Uses a single SPARQL query with UNION+GROUP_CONCAT to find which KEs are
    designated as MIE or AO in any AOP. Priority per D-02: MIE > AO > KE.
    Falls back to empty dict (all KE) if query returns no results (D-03).

    Args:
        target_graph: The RDF graph URI, e.g. "http://aopwiki.org/graph/2025-03"

    Returns:
        Dict mapping KE ID (string) -> role ('MIE' or 'AO').
        KEs not in the dict default to 'KE'.
    """
    role_query = f"""
    SELECT ?ke
           (GROUP_CONCAT(DISTINCT ?roleType; SEPARATOR=",") AS ?roles)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway .
            {{
                ?aop aopo:has_molecular_initiating_event ?ke .
                BIND("MIE" AS ?roleType)
            }}
            UNION
            {{
                ?aop aopo:has_adverse_outcome ?ke .
                BIND("AO" AS ?roleType)
            }}
        }}
    }}
    GROUP BY ?ke
    """
    results = run_sparql_query_with_retry(role_query)

    if not results:
        logger.warning("Role detection query returned no results; "
                       "falling back to all KE")
        return {}

    roles = {}
    for r in results:
        ke_uri = r['ke']['value']
        ke_id = ke_uri.split('/')[-1]
        role_str = r.get('roles', {}).get('value', '')

        # Priority: MIE > AO > KE (D-02)
        if 'MIE' in role_str:
            roles[ke_id] = 'MIE'
        elif 'AO' in role_str:
            roles[ke_id] = 'AO'
        # else: not in results = default KE (handled by caller)

    logger.info(f"Roles detected: {sum(1 for v in roles.values() if v == 'MIE')} MIE, "
                f"{sum(1 for v in roles.values() if v == 'AO')} AO")
    return roles


def compute_network_metrics(G: nx.Graph) -> Tuple[pd.DataFrame, List[Dict]]:
    """Compute centrality metrics and community detection on the network.

    Calculates degree centrality, betweenness centrality, closeness centrality,
    and PageRank for all nodes. Runs Louvain community detection with a fixed
    seed for deterministic results.

    Args:
        G: NetworkX graph as built by build_aop_network().

    Returns:
        Tuple of (metrics_df, communities_list) where:
            - metrics_df: DataFrame with columns id, label, type, uri,
                wiki_url, oecd_status, degree, betweenness, closeness,
                pagerank, community
            - communities_list: List of dicts with keys id, size, members
                (each member has id, label, type)

    Example:
        >>> G = build_aop_network()
        >>> metrics_df, communities = compute_network_metrics(G)
        >>> print(f"Communities found: {len(communities)}")
    """
    logger.info("Computing network metrics...")

    # Handle empty graph edge case
    if G.number_of_nodes() == 0:
        logger.warning("Empty graph, returning empty metrics")
        empty_df = pd.DataFrame(
            columns=['id', 'label', 'type', 'uri', 'wiki_url',
                     'degree', 'betweenness', 'closeness',
                     'pagerank', 'community']
        )
        return empty_df, []

    # Compute centrality metrics
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)

    # Community detection with fixed seed for determinism
    communities = nx.community.louvain_communities(G, seed=42)

    # Build node-to-community mapping
    community_map: Dict[str, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_map[node] = idx

    # Assemble metrics DataFrame
    rows = []
    for node in G.nodes():
        node_data = G.nodes[node]
        rows.append({
            'id': node,
            'label': node_data.get('label', node),
            'type': node_data.get('type', 'Unknown'),
            'uri': node_data.get('uri', ''),
            'wiki_url': node_data.get('wiki_url', ''),
            'degree': round(degree[node], 4),
            'betweenness': round(betweenness[node], 4),
            'closeness': round(closeness[node], 4),
            'pagerank': round(pagerank[node], 6),
            'community': community_map.get(node, -1),
        })

    metrics_df = pd.DataFrame(rows)

    # Build communities summary list
    communities_list = []
    for idx, community in enumerate(communities):
        members = []
        for node_id in community:
            node_data = G.nodes[node_id]
            members.append({
                'id': node_id,
                'label': node_data.get('label', node_id),
                'type': node_data.get('type', 'Unknown'),
            })
        communities_list.append({
            'id': idx,
            'size': len(community),
            'members': members,
        })

    logger.info(
        f"Metrics computed: {len(rows)} nodes, "
        f"{len(communities_list)} communities"
    )
    return metrics_df, communities_list


def graph_to_cytoscape_json(
    G: nx.Graph, metrics_df: pd.DataFrame,
    roles: Optional[Dict[str, str]] = None,
    positions: Optional[Dict] = None,
) -> List[Dict]:
    """Convert NetworkX graph and metrics to Cytoscape.js elements array.

    Builds a flat array of node and edge elements in Cytoscape.js JSON format.
    Nodes include all metric values and role-based color assignments using
    MIE/KE/AO network_roles from BRAND_COLORS. Positions are embedded at the
    element level (sibling to 'data') for Cytoscape preset layout.

    Args:
        G: NetworkX graph with node/edge attributes.
        metrics_df: DataFrame from compute_network_metrics() with centrality
            scores and community assignments.
        roles: Optional dict mapping KE ID -> role ('MIE' or 'AO').
            KEs not in dict default to 'KE'.
        positions: Optional dict mapping node ID -> (x, y) coordinates
            from NetworkX layout computation.

    Returns:
        List of Cytoscape.js element dicts. Node elements have 'data' with
        keys: id, label, type, degree, betweenness, closeness, pagerank,
        community, color, uri, wiki_url. Edge elements have 'data' with
        keys: source, target, type.

    Example:
        >>> elements = graph_to_cytoscape_json(G, metrics_df)
        >>> nodes = [e for e in elements if 'source' not in e['data']]
        >>> edges = [e for e in elements if 'source' in e['data']]
    """
    elements: List[Dict] = []

    # Build a lookup from metrics DataFrame for fast access
    metrics_lookup = {}
    for _, row in metrics_df.iterrows():
        metrics_lookup[row['id']] = row

    # Add nodes
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        row = metrics_lookup.get(node_id)

        if row is not None:
            community_idx = int(row['community'])
            node_role = (roles or {}).get(str(node_id), 'KE')
            role_colors = BRAND_COLORS['network_roles']
            node_color = role_colors.get(node_role, role_colors['KE'])

            element = {
                'data': {
                    'id': str(node_id),
                    'label': str(row['label']),
                    'type': node_role,
                    'degree': float(row['degree']),
                    'betweenness': float(row['betweenness']),
                    'closeness': float(row['closeness']),
                    'pagerank': float(row['pagerank']),
                    'community': community_idx,
                    'color': node_color,
                    'uri': str(row.get('uri', '')),
                    'wiki_url': str(row.get('wiki_url', '')),
                }
            }
            if positions and node_id in positions:
                pos = positions[node_id]
                element['position'] = {'x': float(pos[0]), 'y': float(pos[1])}
            elements.append(element)

    # Add edges
    for u, v, edge_data in G.edges(data=True):
        elements.append({
            'data': {
                'source': str(u),
                'target': str(v),
                'type': edge_data.get('type', 'unknown'),
            }
        })

    logger.info(
        f"Cytoscape JSON: {len([e for e in elements if 'source' not in e['data']])} nodes, "
        f"{len([e for e in elements if 'source' in e['data']])} edges"
    )
    return elements


def get_or_compute_network() -> Dict:
    """Get cached network data or compute it on first access.

    Lazy computation with module-level cache. On first call: builds the graph,
    computes all metrics, converts to Cytoscape.js JSON, and caches everything.
    On subsequent calls: returns cached data immediately.

    The metrics DataFrame is also cached in _plot_data_cache['network_metrics']
    for compatibility with the existing CSV export infrastructure.

    Returns:
        Dict with keys:
            - 'elements': Cytoscape.js elements array (list of dicts)
            - 'metrics': List of metric dicts (from DataFrame.to_dict('records'))
            - 'communities': Community summary list
            - 'version': Version string used for graph construction
            - 'stats': Dict with node/edge/AOP/KE/community counts

    Raises:
        Exception: Propagates errors from graph construction or metrics
            computation after logging them.

    Example:
        >>> data = get_or_compute_network()
        >>> print(f"Version: {data['version']}")
        >>> print(f"Nodes: {data['stats']['nodes']}")
    """
    global _network_cache

    if _network_cache:
        # Re-populate _plot_data_cache if the TTL-based entry has expired.
        # _network_cache has no TTL (lives forever), but _plot_data_cache
        # uses VersionedPlotCache with TTL=1800s. After expiry, the CSV
        # download endpoint can't find the metrics and returns 404.
        if 'network_metrics' not in _plot_data_cache:
            metrics_records = _network_cache.get('metrics', [])
            if metrics_records:
                _plot_data_cache['network_metrics'] = pd.DataFrame(metrics_records)
                logger.info("Re-populated _plot_data_cache['network_metrics'] from network cache")
        logger.info("Returning cached network data")
        return _network_cache

    try:
        logger.info("Computing network data (first request)...")

        # Get version for metadata
        version = get_latest_version()

        # Build graph
        G = build_aop_network()

        # Detect MIE/KE/AO roles
        target_graph = f"http://aopwiki.org/graph/{version}"
        roles = detect_ke_roles(target_graph)

        # Compute metrics and communities
        metrics_df, communities_list = compute_network_metrics(G)

        # Update metrics with role types for CSV export
        for idx, row in metrics_df.iterrows():
            node_id = row['id']
            metrics_df.at[idx, 'type'] = roles.get(str(node_id), 'KE')

        # Compute deterministic layout (D-06)
        positions = nx.spring_layout(G, seed=42, scale=1000)

        # Convert to Cytoscape.js JSON
        elements = graph_to_cytoscape_json(G, metrics_df, roles=roles, positions=positions)

        # Assemble result
        result = {
            'elements': elements,
            'metrics': metrics_df.to_dict('records'),
            'communities': communities_list,
            'version': version,
            'stats': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'ke_count': G.number_of_nodes(),
                'ker_count': G.number_of_edges(),
                'communities': len(communities_list),
                'mie_count': sum(1 for v in roles.values() if v == 'MIE'),
                'ao_count': sum(1 for v in roles.values() if v == 'AO'),
            },
        }

        # Cache the result
        _network_cache = result

        # Also cache metrics DataFrame for CSV export compatibility
        _plot_data_cache['network_metrics'] = metrics_df

        logger.info(
            f"Network data computed and cached: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
            f"{len(communities_list)} communities"
        )
        return result

    except Exception as e:
        logger.error(f"Error computing network data: {e}")
        raise
