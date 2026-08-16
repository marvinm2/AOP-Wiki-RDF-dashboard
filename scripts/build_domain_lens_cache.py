#!/usr/bin/env python3
"""Build the domain-lens cache: ontology roots → the terms AOP-Wiki uses (issue #149).

The dashboard already reports organ-system *coverage* via a classifier that is
deliberately generous — it carries editorial overrides and resolves phenotypes
through UPHENO, because a coverage bar should not leave AOPs unclassified. A
domain lens needs the opposite bias. Measured against the JRC cardiovascular
expert review (80 candidate AOPs, 56 rated core) on the 2026-07-01 snapshot:

    root closure (this file)     41 AOPs   precision 28/41   recall 28/56
    organ-system classifier      63 AOPs   precision 29/63   recall 29/56

The classifier buys one extra true positive for 22 extra false ones, so the
domain lens gets its own scoping rule: a small set of ontology roots, expanded
to their descendants, intersected with the terms Key Events actually carry.
No keyword list, no synonyms, no per-term curation — the only hand-written
input is the root set below.

Policy choices worth knowing before you change anything here:

- **Descendants come from Ubergraph's `<…/redundant>` graph**, where the OBO
  relation closure is pre-materialised, so one triple pattern returns a whole
  subtree. Same source and same reasoning as
  `scripts/build_ontology_branch_cache.py`.

- **Every domain gets anatomy *and* process/phenotype roots.** An anatomy-only
  domain under-counts badly: on 2026-07-01 the cardiovascular UBERON branch
  alone reaches 34 of the 41 AOPs, and the 7 it misses (31, 53, 200, 237, 392,
  516, 614) are annotated only with GO processes or MP/HP phenotypes.

- **Each root's closure is restricted to its own namespace family.** Ubergraph
  mixes species-specific anatomy into UBERON's branches and it dominates —
  only 12% of the 34,808 terms under `nervous system` are UBERON or CL, the
  rest FBbt (Drosophila), PCL and the mouse brain atlas. AOP-Wiki annotates
  with none of them. The `UBERON_6*` (Drosophila-derived) range is dropped for
  the same reason.

- **MeSH is deliberately not covered.** AOP-Wiki uses 70 MeSH descriptors and
  MeSH is not in Ubergraph, so it would need a second endpoint (NLM) at build
  time. Measured 2026-08-16: exactly 3 of the 70 sit under tree number C14
  (Cardiovascular Diseases) — Arrhythmias, Bradycardia, Atrioventricular Block
  — and every AOP carrying one is already in the closure result. The leg costs
  a second service dependency and returns zero AOPs, so it is left out and the
  methodology note says so.

- **The cache stores the term set, not the answer.** Which AOPs are in a domain
  varies by snapshot (the dashboard has a version selector), so the AOP-level
  query runs at request time against the selected graph. What is cached is the
  domain vocabulary, resolved once against every snapshot's term usage.

Usage:
    cd AOP-Wiki-RDF-dashboard/
    python scripts/build_domain_lens_cache.py
    # → writes static/data/domain_lens_cache.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# Reuse the SPARQL helper and endpoints from the organ-system builder so the
# artefacts cannot drift apart. Sibling import, as in build_ontology_branch_cache.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_organ_system_cache import (  # noqa: E402
    AOPWIKI_SPARQL,
    OBO,
    UBERGRAPH_SPARQL,
    _short,
    sparql_select,
)

REDUNDANT_GRAPH = "http://reasoner.renci.org/redundant"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
BFO_PART_OF = OBO + "BFO_0000050"

LABEL_CHUNK = 200

# Predicates that carry an ontology term on a Key Event. Same set as the
# coverage-holes plot and NOT the one the ontology-usage plots query: those go
# through aopo:hasBiologicalEvent only and never see OrganContext /
# CellTypeContext, which is where the anatomy lives. The flat obo:PATO_0001241
# / obo:GO_0008150 predicates are the shortcut form of hasBiologicalEvent's
# hasObject / hasProcess — verified equivalent on 2026-07-01 (494 and 559
# distinct terms either way).
KE_ANNOTATION_PREDICATES: Tuple[str, ...] = (
    "http://aopkb.org/aop_ontology#OrganContext",
    "http://aopkb.org/aop_ontology#CellTypeContext",
    OBO + "PATO_0001241",   # biological object
    OBO + "GO_0008150",     # biological process
)

# UBERON_6* is the Drosophila (FBbt-derived) subset of UBERON — see the module
# docstring. Dropped from every anatomy closure.
INSECT_IRI_PREFIX = "UBERON_6"

# Which namespaces a root's closure may contribute. Anatomy roots also admit
# cell types, because AOP-Wiki annotates CellTypeContext with CL and those
# terms sit under UBERON anatomy in the closure.
NAMESPACE_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "UBERON": ("UBERON_", "CL_"),
    "CL": ("UBERON_", "CL_"),
    "GO": ("GO_",),
    "MP": ("MP_",),
    "HP": ("HP_",),
}

# ---------------------------------------------------------------------------
# Domain presets. The ONLY hand-written input in the whole artefact.
#
# Each domain gets four kinds of root where the ontologies provide one:
# anatomy (UBERON), physiological process (GO), development (GO) and abnormal
# phenotype (MP + HP). The cardiovascular set is the one the JRC prototype was
# validated with; the others follow the same template so a domain's numbers
# stay comparable to its neighbours'.
# ---------------------------------------------------------------------------

DOMAIN_ROOTS: Dict[str, List[str]] = {
    "Cardiovascular": [
        "UBERON_0001009",  # circulatory system
        "GO_0003013",      # circulatory system process
        "GO_0072359",      # circulatory system development
        "MP_0005385",      # cardiovascular system phenotype
        "HP_0001626",      # Abnormality of the cardiovascular system
    ],
    "Hepatobiliary": [
        "UBERON_0002423",  # hepatobiliary system
        "GO_0061007",      # hepaticobiliary system process
        "GO_0001889",      # liver development
        "MP_0005370",      # liver/biliary system phenotype
        "HP_0001392",      # Abnormality of the liver
    ],
    "Nervous": [
        "UBERON_0001016",  # nervous system
        "GO_0050877",      # nervous system process
        "GO_0007399",      # nervous system development
        "MP_0003631",      # nervous system phenotype
        "HP_0000707",      # Abnormality of the nervous system
    ],
    "Renal/Urinary": [
        "UBERON_0001008",  # renal system
        "GO_0003014",      # renal system process
        "GO_0072001",      # renal system development
        "MP_0005367",      # renal/urinary system phenotype
        "HP_0000079",      # Abnormality of the urinary system
    ],
    "Reproductive": [
        "UBERON_0000990",  # reproductive system
        "GO_0003006",      # developmental process involved in reproduction
        "GO_0061458",      # reproductive system development
        "MP_0005389",      # reproductive system phenotype
        "HP_0000078",      # Abnormality of the genital system
    ],
    "Respiratory": [
        "UBERON_0001004",  # respiratory system
        "GO_0003016",      # respiratory system process
        "GO_0060541",      # respiratory system development
        "MP_0005388",      # respiratory system phenotype
        "HP_0002086",      # Abnormality of the respiratory system
    ],
}

DEFAULT_DOMAIN = "Cardiovascular"


def _namespace_of(short_id: str) -> str:
    return short_id.split("_", 1)[0]


def _allowed_namespaces(root: str) -> Tuple[str, ...]:
    return NAMESPACE_FAMILIES.get(_namespace_of(root), (_namespace_of(root) + "_",))


def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Ubergraph
# ---------------------------------------------------------------------------


def fetch_descendants(root: str) -> Set[str]:
    """Every term under `root` via subClassOf or part_of, own namespace family only.

    One hop against the pre-materialised closure graph, so this is a single
    cheap query rather than a property path.
    """
    allowed = _allowed_namespaces(root)
    query = f"""
    SELECT DISTINCT ?term WHERE {{
      GRAPH <{REDUNDANT_GRAPH}> {{
        {{ ?term <{RDFS_SUBCLASS}> <{OBO}{root}> }}
        UNION
        {{ ?term <{BFO_PART_OF}> <{OBO}{root}> }}
      }}
    }}
    """
    out: Set[str] = set()
    for row in sparql_select(UBERGRAPH_SPARQL, query):
        short = _short(row["term"]["value"])
        if short.startswith(INSECT_IRI_PREFIX):
            continue
        if short.startswith(allowed):
            out.add(row["term"]["value"])
    return out


def fetch_labels(iris: Iterable[str]) -> Dict[str, str]:
    """rdfs:label for the given IRIs.

    AOP-Wiki's own dc:title on the term would also work here, but Ubergraph is
    the definitional source and keeps labels consistent with the coverage-holes
    cache, which names terms nobody has annotated.
    """
    out: Dict[str, str] = {}
    for chunk in _chunks(sorted(iris), LABEL_CHUNK):
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


# ---------------------------------------------------------------------------
# AOP-Wiki
# ---------------------------------------------------------------------------


def collect_used_terms() -> Dict[str, int]:
    """Every OBO term annotated on a Key Event in ANY snapshot → snapshot count.

    Resolving against all snapshots rather than the latest one keeps the
    version selector honest: a term that was dropped from AOP-Wiki in 2021
    still has to be in the domain vocabulary for the 2020 graphs to answer.
    """
    predicate_list = ", ".join(f"<{p}>" for p in KE_ANNOTATION_PREDICATES)
    query = f"""
    PREFIX aopo: <http://aopkb.org/aop_ontology#>
    SELECT ?term (COUNT(DISTINCT ?graph) AS ?graphs) WHERE {{
      GRAPH ?graph {{
        ?ke a aopo:KeyEvent ; ?p ?term .
        FILTER(?p IN ({predicate_list}))
      }}
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
      FILTER(STRSTARTS(STR(?term), "{OBO}"))
    }}
    GROUP BY ?term
    """
    return {
        r["term"]["value"]: int(r["graphs"]["value"])
        for r in sparql_select(AOPWIKI_SPARQL, query)
    }


# ---------------------------------------------------------------------------
# Cache assembly
# ---------------------------------------------------------------------------


def build_cache() -> Dict:
    started = time.time()

    print("[1/3] collecting terms used on Key Events across all snapshots …", file=sys.stderr)
    used = collect_used_terms()
    print(f"  {len(used)} distinct OBO terms ever annotated", file=sys.stderr)

    print(f"[2/3] expanding {sum(len(r) for r in DOMAIN_ROOTS.values())} roots "
          f"over {len(DOMAIN_ROOTS)} domains …", file=sys.stderr)
    closures: Dict[str, Dict[str, Set[str]]] = {}
    for domain, roots in sorted(DOMAIN_ROOTS.items()):
        per_root: Dict[str, Set[str]] = {}
        for root in roots:
            root_iri = OBO + root
            nodes = fetch_descendants(root)
            nodes.add(root_iri)
            per_root[root] = nodes
            time.sleep(0.5)  # be polite — Ubergraph is a shared free service
        closures[domain] = per_root
        total = len(set().union(*per_root.values()))
        in_use = len({t for t in set().union(*per_root.values()) if t in used})
        print(f"  {domain:16} {total:7} terms in closure, {in_use:4} of them in use",
              file=sys.stderr)

    wanted: Set[str] = set()
    for per_root in closures.values():
        for nodes in per_root.values():
            wanted |= {t for t in nodes if t in used}
    all_roots = {OBO + r for roots in DOMAIN_ROOTS.values() for r in roots}
    print(f"[3/3] labelling {len(wanted | all_roots)} terms …", file=sys.stderr)
    labels = fetch_labels(wanted | all_roots)

    domains: Dict[str, Dict] = {}
    for domain, per_root in sorted(closures.items()):
        roots_of: Dict[str, List[str]] = defaultdict(list)
        for root, nodes in per_root.items():
            for term in nodes:
                if term in used:
                    roots_of[term].append(root)

        terms = {
            _short(term): {
                "label": labels.get(term, _short(term)),
                "roots": sorted(roots_of[term]),
                "snapshots": used[term],
            }
            for term in sorted(roots_of)
        }
        by_namespace: Dict[str, int] = defaultdict(int)
        for short_id in terms:
            by_namespace[_namespace_of(short_id)] += 1

        domains[domain] = {
            "roots": [
                {
                    "id": root,
                    "label": labels.get(OBO + root, root),
                    "closure_size": len(per_root[root]),
                    "terms_in_use": sum(1 for t in per_root[root] if t in used),
                }
                for root in DOMAIN_ROOTS[domain]
            ],
            "closure_size": len(set().union(*per_root.values())),
            "terms": terms,
            "terms_by_namespace": dict(sorted(by_namespace.items())),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "default_domain": DEFAULT_DOMAIN,
        "source": {
            "aopwiki_sparql": AOPWIKI_SPARQL,
            "ubergraph_sparql": UBERGRAPH_SPARQL,
            "descendant_relation": (
                "subClassOf | part_of, one hop against <reasoner.renci.org/redundant>"
            ),
            "usage_predicates": list(KE_ANNOTATION_PREDICATES),
            "usage_scope": "every http://aopwiki.org/graph/* snapshot, not only the latest",
            "namespace_families": {k: list(v) for k, v in NAMESPACE_FAMILIES.items()},
            "excluded": (
                f"{INSECT_IRI_PREFIX}* (Drosophila-derived UBERON subset); MeSH "
                "(not in Ubergraph — 3 of the 70 in-use descriptors are under tree "
                "number C14 and every AOP carrying one is already reached through "
                "the OBO closure, measured 2026-08-16)"
            ),
        },
        "domains": domains,
        "stats": {
            "domains": len(domains),
            "roots": sum(len(r) for r in DOMAIN_ROOTS.values()),
            "used_terms_ever": len(used),
            "terms_in_any_domain": len({t for d in domains.values() for t in d["terms"]}),
            "elapsed_s": round(time.time() - started, 1),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="static/data/domain_lens_cache.json",
        help="Output JSON file (default: static/data/domain_lens_cache.json)",
    )
    args = parser.parse_args()

    cache = build_cache()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    s = cache["stats"]
    print(
        f"\nWrote {out_path}  "
        f"{s['domains']} domains  "
        f"{s['roots']} roots  "
        f"{s['terms_in_any_domain']} in-use terms mapped  "
        f"({s['elapsed_s']}s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
