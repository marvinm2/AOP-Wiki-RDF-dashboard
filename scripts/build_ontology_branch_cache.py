#!/usr/bin/env python3
"""Build the ontology-branch cache backing the coverage-holes plot (issue #151).

The dashboard's ontology views all report which terms *are* used. This artefact
supports the inverse question: given a branch of an ontology, which parts of it
does AOP-Wiki never touch? The worked example is valvulopathy — `UBERON:0000946`
(cardiac valve) and `HP:0001654` (abnormal heart valve morphology) are used zero
times anywhere in AOP-Wiki, which is the same conclusion a JRC expert panel
reached by reading 80 candidate AOPs.

Policy choices worth knowing before you change anything here:

- **Descendants come from Ubergraph's `<…/redundant>` graph**, where the OBO
  relation closure is pre-materialised. A single one-hop triple pattern
  therefore returns the whole subtree — no property path, ~0.8s for the 1,936
  terms under the circulatory system. The `<…/nonredundant>` graph gives the
  *asserted* edges, which is what we need for depth and tree structure.

- **Branches are restricted to UBERON and CL.** Ubergraph's closure mixes
  species-specific anatomy ontologies into UBERON's branches and they dominate:
  only 12% of the 34,808 terms under `nervous system` are UBERON or CL, the
  rest being FBbt (Drosophila), PCL and the mouse brain atlas. AOP-Wiki
  annotates with none of them, so without this filter the plot reports fly
  neuroanatomy as an AOP-Wiki coverage gap.

- **Depth is capped** (`--max-depth`, default 3). Deeper holes are leaf-level
  gaps nobody annotates against, and the cap keeps the committed artefact
  tractable.

- **The cache stores the ontology, not the answer.** Which terms are *used*
  varies by snapshot (the dashboard has a version selector), so the used-vs-
  unused diff is computed at request time. What is cached is the branch
  structure plus, for every term AOP-Wiki has ever used, which candidates sit
  above it — enough to resolve any snapshot without touching Ubergraph.

- **Term usage is read from the `collect_all_terms()` predicate set**
  (OrganContext, CellTypeContext, hasBiologicalEvent/hasObject|hasProcess), NOT
  the set the ontology-usage plots query. Those omit OrganContext and
  CellTypeContext, which is where the anatomy lives — 4 of the 10 in-use
  circulatory-system terms are reachable only that way.

Usage:
    cd AOP-Wiki-RDF-dashboard/
    python scripts/build_ontology_branch_cache.py
    # → writes static/data/ontology_branch_cache.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# Reuse the SPARQL helper, endpoints and curated anchors from the organ-system
# builder so the two artefacts cannot drift apart. Sibling import, as in
# scripts/audit_closure_paths.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_organ_system_cache import (  # noqa: E402
    ANCHOR_LABELS,
    ANCHORS,
    AOPWIKI_SPARQL,
    OBO,
    UBERGRAPH_SPARQL,
    _short,
    collect_all_terms,
    sparql_select,
)

REDUNDANT_GRAPH = "http://reasoner.renci.org/redundant"
NONREDUNDANT_GRAPH = "http://reasoner.renci.org/nonredundant"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
BFO_PART_OF = OBO + "BFO_0000050"
RO_NEVER_IN_TAXON = OBO + "RO_0002161"

DEFAULT_MAX_DEPTH = 3
LABEL_CHUNK = 200

# Clades that contain vertebrates. A term declared `never_in_taxon` any of
# these cannot occur in the animals AOP-Wiki is about, so reporting it as a
# coverage hole is noise — it removes "open circulatory system" (128 terms),
# "insect embryonic/larval circulatory system" (54) and "hemolymph" from the
# cardiovascular result. The filter is PARTIAL: invertebrate terms carrying no
# such axiom survive it (e.g. "circulatory system dorsal vessel"), which the
# methodology note discloses.
VERTEBRATE_CLADES: Set[str] = {
    "NCBITaxon_33511",   # Deuterostomia
    "NCBITaxon_7711",    # Chordata
    "NCBITaxon_7742",    # Vertebrata
    "NCBITaxon_33213",   # Bilateria
    "NCBITaxon_33208",   # Metazoa
}

# UBERON_6* is the Drosophila (FBbt-derived) subset of UBERON. Those terms
# carry `in_taxon` rather than `never_in_taxon`, so the clade rule above does
# not catch them — "insect embryonic/larval circulatory system" survived it in
# testing. The numeric-range convention is stable enough to filter on directly.
INSECT_IRI_PREFIX = "UBERON_6"

# Ubergraph's closure mixes species-specific anatomy ontologies into UBERON's
# branches, and they dominate: of the 34,808 terms under `nervous system`, only
# 12% are UBERON or CL — the rest is FBbt (Drosophila, 21,719), PCL (7,477) and
# MBA (mouse brain atlas, 1,327). AOP-Wiki annotates with none of them (its
# anatomy vocabulary is UBERON, CL and a little FMA), so counting them would
# report fly neuroanatomy as an AOP-Wiki coverage gap and inflate every subtree
# size. Restrict branches to the namespaces the resource actually uses.
BRANCH_NAMESPACES: Tuple[str, ...] = ("UBERON_", "CL_")


def _in_branch_namespace(iri: str) -> bool:
    return _short(iri).startswith(BRANCH_NAMESPACES)


def _is_non_vertebrate(short_id: str, never_in_taxon: Set[str]) -> bool:
    """True when a term cannot describe the animals AOP-Wiki is about."""
    if short_id.startswith(INSECT_IRI_PREFIX):
        return True
    return bool(never_in_taxon & VERTEBRATE_CLADES)


def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Ubergraph queries
# ---------------------------------------------------------------------------


def fetch_descendants(root: str) -> Set[str]:
    """Every term under `root` via subClassOf or part_of.

    One hop against the pre-materialised closure graph, so this is a single
    cheap query rather than a property path.
    """
    query = f"""
    SELECT DISTINCT ?term WHERE {{
      GRAPH <{REDUNDANT_GRAPH}> {{
        {{ ?term <{RDFS_SUBCLASS}> <{OBO}{root}> }}
        UNION
        {{ ?term <{BFO_PART_OF}> <{OBO}{root}> }}
      }}
    }}
    """
    return {
        r["term"]["value"]
        for r in sparql_select(UBERGRAPH_SPARQL, query)
        if _in_branch_namespace(r["term"]["value"])
    }


def fetch_direct_edges(root: str) -> List[Tuple[str, str]]:
    """Asserted child->parent edges inside the branch, for depth and structure.

    Restricted to OBO IRIs on both ends, and to children that are within the
    root's closure, so the result describes this branch only.
    """
    query = f"""
    SELECT ?child ?parent WHERE {{
      GRAPH <{NONREDUNDANT_GRAPH}> {{
        ?child ?rel ?parent .
        VALUES ?rel {{ <{RDFS_SUBCLASS}> <{BFO_PART_OF}> }}
        FILTER(STRSTARTS(STR(?child), "{OBO}"))
        FILTER(STRSTARTS(STR(?parent), "{OBO}"))
      }}
      GRAPH <{REDUNDANT_GRAPH}> {{
        {{ ?child <{RDFS_SUBCLASS}> <{OBO}{root}> }}
        UNION
        {{ ?child <{BFO_PART_OF}> <{OBO}{root}> }}
      }}
    }}
    """
    return [
        (r["child"]["value"], r["parent"]["value"])
        for r in sparql_select(UBERGRAPH_SPARQL, query)
        if _in_branch_namespace(r["child"]["value"])
        and _in_branch_namespace(r["parent"]["value"])
    ]


def fetch_labels(iris: Iterable[str]) -> Dict[str, str]:
    """rdfs:label for the given IRIs.

    Ubergraph is the only workable source here: the dashboard's usual label
    path is `dc:title` on the AOP-Wiki graph, which by construction only covers
    terms that ARE used — never the unused ones this cache exists to name.
    """
    out: Dict[str, str] = {}
    ordered = sorted(iris)
    for chunk in _chunks(ordered, LABEL_CHUNK):
        values = " ".join(f"<{t}>" for t in chunk)
        query = f"""
        SELECT ?term ?label WHERE {{
          VALUES ?term {{ {values} }}
          ?term <{RDFS_LABEL}> ?label .
        }}
        """
        for row in sparql_select(UBERGRAPH_SPARQL, query):
            out.setdefault(row["term"]["value"], row["label"]["value"])
        time.sleep(0.5)  # be polite — Ubergraph is a shared free service
    return out


def fetch_never_in_taxon(iris: Iterable[str]) -> Dict[str, Set[str]]:
    """`RO:0002161` (never in taxon) assertions, for the vertebrate filter."""
    out: Dict[str, Set[str]] = defaultdict(set)
    ordered = sorted(iris)
    for chunk in _chunks(ordered, LABEL_CHUNK):
        values = " ".join(f"<{t}>" for t in chunk)
        query = f"""
        SELECT ?term ?taxon WHERE {{
          VALUES ?term {{ {values} }}
          ?term <{RO_NEVER_IN_TAXON}> ?taxon .
        }}
        """
        for row in sparql_select(UBERGRAPH_SPARQL, query):
            out[row["term"]["value"]].add(_short(row["taxon"]["value"]))
        time.sleep(0.5)  # be polite — Ubergraph is a shared free service
    return out


# ---------------------------------------------------------------------------
# Branch structure
# ---------------------------------------------------------------------------


def build_structure(
    roots: List[str], nodes: Set[str], edges: List[Tuple[str, str]]
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, int]]:
    """Return (children, parents, depth) restricted to the branch.

    Depth is measured from the real anchor root(s), which sit at 0. A bucket
    with several anchors (Hepatobiliary = liver + gallbladder + common bile
    duct) is walked as one branch by seeding the BFS from all of them.
    """
    children: Dict[str, Set[str]] = defaultdict(set)
    parents: Dict[str, Set[str]] = defaultdict(set)
    for child, parent in edges:
        if child in nodes and parent in nodes and child != parent:
            children[parent].add(child)
            parents[child].add(parent)

    depth: Dict[str, int] = {r: 0 for r in roots}
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        for child in children[node]:
            if child not in depth:
                depth[child] = depth[node] + 1
                queue.append(child)
    return children, parents, depth


def subtree_sizes(children: Dict[str, Set[str]], wanted: Set[str]) -> Dict[str, int]:
    """Size of each wanted node's subtree, counting distinct reachable terms.

    The ontology is a DAG (810 of the 1,936 circulatory-system terms have more
    than one parent), so a subtree is the reachable SET, not a sum over
    children — summing would double-count every multi-parent term.
    """
    memo: Dict[str, Set[str]] = {}

    def reachable(node: str, stack: Set[str]) -> Set[str]:
        if node in memo:
            return memo[node]
        if node in stack:  # cycle guard; OBO should not have them, but do not hang
            return {node}
        stack.add(node)
        acc = {node}
        for child in children[node]:
            acc |= reachable(child, stack)
        stack.discard(node)
        memo[node] = acc
        return acc

    sys.setrecursionlimit(100000)
    return {n: len(reachable(n, set())) for n in wanted}


def ancestors_within(
    term: str, parents: Dict[str, Set[str]], candidates: Set[str]
) -> Set[str]:
    """Candidate nodes above `term` (inclusive when the term is a candidate)."""
    seen: Set[str] = set()
    out: Set[str] = set()
    queue = deque([term])
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        if node in candidates:
            out.add(node)
        queue.extend(parents[node])
    return out


# ---------------------------------------------------------------------------
# Cache assembly
# ---------------------------------------------------------------------------


def build_cache(max_depth: int = DEFAULT_MAX_DEPTH) -> Dict:
    started = time.time()

    print("[1/4] collecting terms used across all AOP-Wiki snapshots …", file=sys.stderr)
    used_by_ontology = collect_all_terms()
    used_terms: Set[str] = set()
    for group in used_by_ontology.values():
        used_terms |= group
    print(f"  {len(used_terms)} distinct OBO terms ever used", file=sys.stderr)

    print(f"[2/4] resolving branch structure for {len(ANCHORS)} branches …", file=sys.stderr)
    branches: Dict[str, Dict] = {}
    all_candidates: Set[str] = set()
    per_branch_raw: Dict[str, Dict] = {}

    for bucket, anchors in sorted(ANCHORS.items()):
        merged_nodes: Set[str] = set()
        merged_edges: List[Tuple[str, str]] = []
        roots: List[str] = []
        for anchor in anchors:
            root_iri = OBO + anchor
            roots.append(root_iri)
            nodes = fetch_descendants(anchor)
            nodes.add(root_iri)
            merged_nodes |= nodes
            merged_edges.extend(fetch_direct_edges(anchor))
            time.sleep(0.5)  # be polite — Ubergraph is a shared free service

        children, parents, depth = build_structure(roots, merged_nodes, merged_edges)
        candidates = {n for n, d in depth.items() if 0 < d <= max_depth}
        all_candidates |= candidates
        per_branch_raw[bucket] = {
            "roots": roots,
            "nodes": merged_nodes,
            "children": children,
            "parents": parents,
            "depth": depth,
            "candidates": candidates,
        }
        print(
            f"  {bucket:24} {len(merged_nodes):6} terms  {len(candidates):5} candidates",
            file=sys.stderr,
        )

    print(f"[3/4] fetching labels + taxon constraints for {len(all_candidates)} candidates …",
          file=sys.stderr)
    labels = fetch_labels(all_candidates)
    never_in_taxon = fetch_never_in_taxon(all_candidates)
    print(f"  {len(labels)} labelled, {len(never_in_taxon)} with never_in_taxon", file=sys.stderr)

    print("[4/4] assembling branch payloads …", file=sys.stderr)
    total_filtered = 0
    for bucket, raw in sorted(per_branch_raw.items()):
        children = raw["children"]
        parents = raw["parents"]
        depth = raw["depth"]
        candidates = raw["candidates"]
        nodes = raw["nodes"]

        sizes = subtree_sizes(children, candidates)

        kept: Dict[str, Dict] = {}
        for node in sorted(candidates):
            if _is_non_vertebrate(_short(node), never_in_taxon.get(node, set())):
                total_filtered += 1
                continue
            kept[_short(node)] = {
                "label": labels.get(node, _short(node)),
                "subtree_size": sizes.get(node, 1),
                "depth": depth[node],
                "parents": sorted(
                    _short(p) for p in parents[node] if p in candidates
                ),
            }

        # Which candidates sit above each ever-used term. This is what lets the
        # runtime resolve any snapshot without re-querying Ubergraph.
        kept_iris = {OBO + s for s in kept}
        used_ancestors: Dict[str, List[str]] = {}
        for term in sorted(used_terms & nodes):
            above = ancestors_within(term, parents, kept_iris)
            if above:
                used_ancestors[_short(term)] = sorted(_short(a) for a in above)

        branches[bucket] = {
            "roots": [_short(r) for r in raw["roots"]],
            "root_labels": [ANCHOR_LABELS.get(_short(r), _short(r)) for r in raw["roots"]],
            "branch_size": len(nodes),
            "candidates": dict(sorted(kept.items())),
            "used_term_ancestors": dict(sorted(used_ancestors.items())),
            "used_terms_in_branch": sorted(_short(t) for t in (used_terms & nodes)),
        }

    cache = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "aopwiki_sparql": AOPWIKI_SPARQL,
            "ubergraph_sparql": UBERGRAPH_SPARQL,
            "descendant_relation": "subClassOf | part_of, one hop against <reasoner.renci.org/redundant>",
            "structure_relation": "asserted subClassOf | part_of from <reasoner.renci.org/nonredundant>",
            "usage_predicates": (
                "aopo:OrganContext, aopo:CellTypeContext, "
                "aopo:hasBiologicalEvent/aopo:hasObject, aopo:hasBiologicalEvent/aopo:hasProcess "
                "(all AOP-Wiki snapshot graphs)"
            ),
            "branch_namespaces": list(BRANCH_NAMESPACES),
            "max_depth": max_depth,
            "non_vertebrate_filter": (
                f"drop candidates with RO:0002161 (never_in_taxon) in {sorted(VERTEBRATE_CLADES)}, "
                f"or in the Drosophila-derived {INSECT_IRI_PREFIX}* range; partial — other "
                "invertebrate terms lacking the axiom survive (e.g. UBERON_0015231 "
                "circulatory system dorsal vessel)"
            ),
        },
        "branches": branches,
        "stats": {
            "branches": len(branches),
            "candidates_total": sum(len(b["candidates"]) for b in branches.values()),
            "candidates_filtered_non_vertebrate": total_filtered,
            "used_terms_ever": len(used_terms),
            "max_depth": max_depth,
            "elapsed_s": round(time.time() - started, 1),
        },
    }
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="static/data/ontology_branch_cache.json",
        help="Output JSON file (default: static/data/ontology_branch_cache.json)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Report holes down to this depth from the branch root (default: {DEFAULT_MAX_DEPTH})",
    )
    args = parser.parse_args()

    cache = build_cache(max_depth=args.max_depth)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    s = cache["stats"]
    print(
        f"\nWrote {out_path}  "
        f"{s['branches']} branches  "
        f"{s['candidates_total']} candidates  "
        f"({s['candidates_filtered_non_vertebrate']} non-vertebrate dropped)  "
        f"({s['elapsed_s']}s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
