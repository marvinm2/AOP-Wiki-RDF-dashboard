"""AOP-Wiki RDF Dashboard - Network Graph Construction and Analysis.

This module builds an interactive network graph from AOP-Wiki RDF data using
NetworkX. It queries the SPARQL endpoint for the latest version's AOP-KE-KER
topology, constructs a bipartite graph (AOPs and KEs as nodes, memberships and
KERs as edges), computes centrality metrics (degree, betweenness, closeness,
PageRank) and Louvain community detection, then converts the result to
Cytoscape.js JSON format for frontend rendering.

Core Components:
    - Graph construction from 4 bulk SPARQL queries
    - Centrality metrics computation (degree, betweenness, closeness, PageRank)
    - Louvain community detection with deterministic seeding
    - Cytoscape.js JSON conversion with VHP4Safety brand colors
    - Lazy computation with module-level caching

Data Flow:
    SPARQL endpoint -> 4 bulk queries -> NetworkX graph -> metrics + communities
    -> Cytoscape.js JSON -> cached for subsequent requests

Performance:
    - 4 SPARQL round-trips (bulk queries, not per-entity)
    - NetworkX metrics computed in <1s for ~2,000 node graph
    - Lazy-loaded on first /network page access (not at startup)
    - Cached in memory for subsequent requests

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
    """Build bipartite AOP-KE network graph from SPARQL endpoint data.

    Queries the SPARQL endpoint with 4 bulk queries against the latest version
    graph to retrieve AOPs, KEs, AOP-KE membership edges, and KER edges. Builds
    a NetworkX undirected graph with typed nodes and edges.

    The 4 queries are:
        1. AOPs with title, OECD status, and foaf:page
        2. KEs with title and foaf:page
        3. AOP-KE membership edges via aopo:has_key_event
        4. KER edges with upstream/downstream KEs

    Returns:
        nx.Graph: NetworkX graph with node attributes (type, label, uri,
            wiki_url, oecd_status) and edge attributes (type: 'membership'
            or 'ker').

    Example:
        >>> G = build_aop_network()
        >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    """
    version = get_latest_version()
    target_graph = f"http://aopwiki.org/graph/{version}"
    logger.info(f"Building AOP network from graph: {target_graph}")

    # Query 1: Get all AOPs with title, OECD status, and foaf:page
    aop_query = f"""
    SELECT ?aop
           (SAMPLE(?t) AS ?title)
           (SAMPLE(STR(?s)) AS ?status)
           (SAMPLE(STR(?p)) AS ?page)
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway .
            OPTIONAL {{ ?aop dc:title ?t }}
            OPTIONAL {{ ?aop <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25688> ?s }}
            OPTIONAL {{ ?aop foaf:page ?p }}
        }}
    }}
    GROUP BY ?aop
    """

    # Query 2: Get all KEs with title and foaf:page
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

    # Query 3: Get AOP-KE membership edges
    membership_query = f"""
    SELECT ?aop ?ke
    WHERE {{
        GRAPH <{target_graph}> {{
            ?aop a aopo:AdverseOutcomePathway ;
                 aopo:has_key_event ?ke .
        }}
    }}
    """

    # Query 4: Get KER edges with upstream/downstream KEs
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

    # Add AOP nodes
    aop_results = run_sparql_query_with_retry(aop_query)
    for r in aop_results:
        aop_uri = r['aop']['value']
        aop_id = aop_uri.split('/')[-1]
        G.add_node(
            aop_id,
            type='AOP',
            label=r.get('title', {}).get('value', aop_id),
            uri=aop_uri,
            wiki_url=r.get('page', {}).get('value', ''),
            oecd_status=r.get('status', {}).get('value', 'Unknown'),
        )
    logger.info(f"Added {len(aop_results)} AOP nodes")

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
            oecd_status='',
        )
    logger.info(f"Added {len(ke_results)} KE nodes")

    # Add membership edges (AOP -> KE)
    membership_results = run_sparql_query_with_retry(membership_query)
    membership_count = 0
    for r in membership_results:
        aop_id = r['aop']['value'].split('/')[-1]
        ke_id = r['ke']['value'].split('/')[-1]
        if aop_id in G and ke_id in G:
            G.add_edge(aop_id, ke_id, type='membership')
            membership_count += 1
    logger.info(f"Added {membership_count} membership edges")

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
                     'oecd_status', 'degree', 'betweenness', 'closeness',
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
            'oecd_status': node_data.get('oecd_status', ''),
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
    G: nx.Graph, metrics_df: pd.DataFrame
) -> List[Dict]:
    """Convert NetworkX graph and metrics to Cytoscape.js elements array.

    Builds a flat array of node and edge elements in Cytoscape.js JSON format.
    Nodes include all metric values and community color assignments from the
    VHP4Safety brand palette. AOP nodes use 'round-rectangle' shape and KE
    nodes use 'ellipse' shape.

    Args:
        G: NetworkX graph with node/edge attributes.
        metrics_df: DataFrame from compute_network_metrics() with centrality
            scores and community assignments.

    Returns:
        List of Cytoscape.js element dicts. Node elements have 'data' with
        keys: id, label, type, degree, betweenness, closeness, pagerank,
        community, color, uri, wiki_url, oecd_status, shape. Edge elements
        have 'data' with keys: source, target, type.

    Example:
        >>> elements = graph_to_cytoscape_json(G, metrics_df)
        >>> nodes = [e for e in elements if 'source' not in e['data']]
        >>> edges = [e for e in elements if 'source' in e['data']]
    """
    palette = BRAND_COLORS['palette']
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
            community_color = palette[community_idx % len(palette)]
            node_type = row['type']

            elements.append({
                'data': {
                    'id': str(node_id),
                    'label': str(row['label']),
                    'type': node_type,
                    'degree': float(row['degree']),
                    'betweenness': float(row['betweenness']),
                    'closeness': float(row['closeness']),
                    'pagerank': float(row['pagerank']),
                    'community': community_idx,
                    'color': community_color,
                    'uri': str(row.get('uri', '')),
                    'wiki_url': str(row.get('wiki_url', '')),
                    'oecd_status': str(row.get('oecd_status', '')),
                    'shape': 'round-rectangle' if node_type == 'AOP' else 'ellipse',
                }
            })

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
        logger.info("Returning cached network data")
        return _network_cache

    try:
        logger.info("Computing network data (first request)...")

        # Get version for metadata
        version = get_latest_version()

        # Build graph
        G = build_aop_network()

        # Compute metrics and communities
        metrics_df, communities_list = compute_network_metrics(G)

        # Convert to Cytoscape.js JSON
        elements = graph_to_cytoscape_json(G, metrics_df)

        # Count node types
        aop_count = sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'AOP')
        ke_count = sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'KE')

        # Assemble result
        result = {
            'elements': elements,
            'metrics': metrics_df.to_dict('records'),
            'communities': communities_list,
            'version': version,
            'stats': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'aop_count': aop_count,
                'ke_count': ke_count,
                'communities': len(communities_list),
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
