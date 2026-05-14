"""Build the ontology-driven organ-system cache shipped with the dashboard.

For every UBERON / CL / GO term that appears in any AOP-Wiki RDF snapshot,
resolve its organ-system bucket(s) via Ubergraph and write the result to
``static/data/organ_system_cache.json``. The dashboard's classifier loads
this file at startup and uses it instead of any hardcoded dictionary.

UBERON / CL terms are classified by transitive ``part_of`` / ``subClassOf``
to one or more curated *anchor* IRIs (one or two per bucket). GO BP terms are
classified *only* via ``RO:0002296 results_in_development_of`` to a UBERON
anchor — terms without such an axiom stay unclassified, which is the explicit
policy: avoid hand-curated GO → organ bridges.

A tiny override layer (kept in this file, not in the ontology) handles a
handful of terms where the project's editorial choice diverges from UBERON.
The override layer is applied after the ontology resolution and is recorded
in the output JSON so it shows up in the methodology API.

Usage:
    cd AOP-Wiki-RDF-dashboard/
    python scripts/build_organ_system_cache.py
    # → writes static/data/organ_system_cache.json
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

import requests

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

AOPWIKI_SPARQL = "https://aopwiki-multirdf.vhp4safety.nl/sparql"
UBERGRAPH_SPARQL = "https://ubergraph.apps.renci.org/sparql"

OBO = "http://purl.obolibrary.org/obo/"
BFO_PART_OF = OBO + "BFO_0000050"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RO_RESULTS_IN_DEV_OF = OBO + "RO_0002296"

HTTP_TIMEOUT = 90


# ---------------------------------------------------------------------------
# Bucket anchors. An anchor is a UBERON IRI; a term belongs to a bucket if it
# is (transitively) `part_of`/`subClassOf` any anchor. Anchors can be shared
# across buckets — e.g. pancreas is an anchor for both Digestive and
# Endocrine because it is both an accessory digestive organ and an endocrine
# gland.
# ---------------------------------------------------------------------------

ANCHORS: Dict[str, List[str]] = {
    "Nervous":               ["UBERON_0001016"],
    "Cardiovascular":        ["UBERON_0004535", "UBERON_0001009"],
    "Respiratory":           ["UBERON_0001004"],
    "Digestive":             ["UBERON_0001007", "UBERON_0001555", "UBERON_0001264"],
    "Hepatobiliary":         ["UBERON_0002107", "UBERON_0001173", "UBERON_0001174"],
    "Renal/Urinary":         ["UBERON_0001008"],
    "Reproductive":          ["UBERON_0000990"],
    "Endocrine":             ["UBERON_0000949", "UBERON_0001264"],
    "Immune/Haematopoietic": ["UBERON_0002405", "UBERON_0002390"],
    "Integumentary":         ["UBERON_0002416"],
    "Musculoskeletal":       ["UBERON_0002204", "UBERON_0002385"],
    "Sensory":               ["UBERON_0001032", "UBERON_0001846", "UBERON_0001690", "UBERON_0000970"],
}

# Human-readable anchor labels — written into the cache JSON for transparency.
ANCHOR_LABELS: Dict[str, str] = {
    "UBERON_0001016": "nervous system",
    "UBERON_0004535": "cardiovascular system",
    "UBERON_0001009": "circulatory system",
    "UBERON_0001004": "respiratory system",
    "UBERON_0001007": "digestive system",
    "UBERON_0001555": "digestive tract",
    "UBERON_0001264": "pancreas",
    "UBERON_0002107": "liver",
    "UBERON_0001173": "gallbladder",
    "UBERON_0001174": "common bile duct",
    "UBERON_0001008": "renal system",
    "UBERON_0000990": "reproductive system",
    "UBERON_0000949": "endocrine system",
    "UBERON_0002405": "immune system",
    "UBERON_0002390": "haematopoietic system",
    "UBERON_0002416": "integumental system",
    "UBERON_0002204": "musculoskeletal system",
    "UBERON_0002385": "muscle tissue",
    "UBERON_0001032": "sensory system",
    "UBERON_0001846": "inner ear",
    "UBERON_0001690": "ear",
    "UBERON_0000970": "eye",
}

# Project-specific overrides applied AFTER ontology resolution. Keep tiny and
# justified — every entry is an editorial decision the maintainer owns.
OVERRIDES: Dict[str, Dict[str, List[str]]] = {
    "uberon": {
        # Toxicology framing: blood-borne / haemodynamic effects often
        # categorise as cardiovascular in regulatory toxicology, even though
        # UBERON places blood structurally under the haematopoietic system.
        # Keep both — the AOP is multi-organ.
        "UBERON_0000178": ["Cardiovascular", "Immune/Haematopoietic"],
    },
    "cl": {},
    "go": {},
}

ANCHOR_TO_BUCKETS: Dict[str, Set[str]] = defaultdict(set)
for _bucket, _anchors in ANCHORS.items():
    for _anchor in _anchors:
        ANCHOR_TO_BUCKETS[_anchor].add(_bucket)
ALL_ANCHORS: List[str] = sorted(ANCHOR_TO_BUCKETS.keys())


# ---------------------------------------------------------------------------
# SPARQL helper
# ---------------------------------------------------------------------------


def sparql_select(
    endpoint: str,
    query: str,
    *,
    timeout: int = HTTP_TIMEOUT,
    retries: int = 4,
    backoff: float = 2.0,
) -> List[Dict]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(
                endpoint,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["results"]["bindings"]
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc
            if attempt == retries - 1:
                raise
            sleep_for = backoff * (2 ** attempt)
            print(
                f"  ! {endpoint.split('//')[1].split('/')[0]} attempt {attempt + 1}/{retries} "
                f"failed ({exc}); retrying in {sleep_for:.0f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    if last_err:
        raise last_err
    return []


def _short(iri: str) -> str:
    return iri.split("/")[-1] if "/" in iri else iri


# ---------------------------------------------------------------------------
# Step 1 — collect terms across ALL AOP-Wiki snapshots
# ---------------------------------------------------------------------------


def collect_all_terms() -> Dict[str, Set[str]]:
    """Return every UBERON / CL / GO IRI used in any AOP-Wiki RDF snapshot."""
    query = """
    PREFIX aopo: <http://aopkb.org/aop_ontology#>
    SELECT DISTINCT ?term WHERE {
      GRAPH ?graph {
        ?aop a aopo:AdverseOutcomePathway ;
             aopo:has_key_event ?ke .
        { ?ke aopo:OrganContext ?term . }
        UNION { ?ke aopo:CellTypeContext ?term . }
        UNION { ?ke aopo:hasBiologicalEvent ?be . ?be aopo:hasObject ?term . }
        UNION { ?ke aopo:hasBiologicalEvent ?be . ?be aopo:hasProcess ?term . }
      }
      FILTER(STRSTARTS(STR(?graph), "http://aopwiki.org/graph/"))
      FILTER(STRSTARTS(STR(?term), "http://purl.obolibrary.org/obo/"))
    }
    """
    rows = sparql_select(AOPWIKI_SPARQL, query)
    by_onto: Dict[str, Set[str]] = {"uberon": set(), "cl": set(), "go": set()}
    for r in rows:
        iri = r["term"]["value"]
        short = _short(iri)
        if short.startswith("UBERON_"):
            by_onto["uberon"].add(iri)
        elif short.startswith("CL_"):
            by_onto["cl"].add(iri)
        elif short.startswith("GO_"):
            by_onto["go"].add(iri)
    return by_onto


# ---------------------------------------------------------------------------
# Step 2 — Ubergraph resolvers
# ---------------------------------------------------------------------------


def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def resolve_anatomy_buckets(terms: List[str]) -> Dict[str, Set[str]]:
    """Map each UBERON / CL term to the set of anchor IRIs it sits under."""
    if not terms:
        return {}

    out: Dict[str, Set[str]] = {t: set() for t in terms}
    anchor_values = " ".join(f"<{OBO}{a}>" for a in ALL_ANCHORS)

    for chunk in _chunks(terms, 40):
        term_values = " ".join(f"<{t}>" for t in chunk)
        query = f"""
        PREFIX BFO: <{OBO}BFO_>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?term ?anchor WHERE {{
          VALUES ?term {{ {term_values} }}
          VALUES ?anchor {{ {anchor_values} }}
          ?term (BFO:0000050|rdfs:subClassOf)* ?anchor .
        }}
        """
        rows = sparql_select(UBERGRAPH_SPARQL, query)
        for r in rows:
            out.setdefault(r["term"]["value"], set()).add(r["anchor"]["value"])
        time.sleep(0.5)  # be polite — Ubergraph is a shared free service
    return out


def resolve_go_via_results_in_development(go_terms: List[str]) -> Dict[str, Set[str]]:
    """Map each GO BP IRI to UBERON IRIs it 'results in development of'."""
    if not go_terms:
        return {}
    out: Dict[str, Set[str]] = {t: set() for t in go_terms}
    for chunk in _chunks(go_terms, 40):
        term_values = " ".join(f"<{t}>" for t in chunk)
        query = f"""
        PREFIX RO: <{OBO}RO_>
        SELECT DISTINCT ?go ?uberon WHERE {{
          VALUES ?go {{ {term_values} }}
          ?go RO:0002296 ?uberon .
          FILTER(STRSTARTS(STR(?uberon), "{OBO}UBERON_"))
        }}
        """
        rows = sparql_select(UBERGRAPH_SPARQL, query)
        for r in rows:
            out.setdefault(r["go"]["value"], set()).add(r["uberon"]["value"])
        time.sleep(0.5)
    return out


def anchors_to_buckets(anchor_iris: Set[str]) -> Set[str]:
    buckets: Set[str] = set()
    for iri in anchor_iris:
        buckets |= ANCHOR_TO_BUCKETS.get(_short(iri), set())
    return buckets


# ---------------------------------------------------------------------------
# Step 3 — assemble the cache
# ---------------------------------------------------------------------------


def _to_short_buckets(term_to_anchors: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Project resolved-anchor sets onto bucket-name lists, keyed by short IRI."""
    out: Dict[str, List[str]] = {}
    for iri, anchors in term_to_anchors.items():
        buckets = anchors_to_buckets(anchors)
        if buckets:
            out[_short(iri)] = sorted(buckets)
    return out


def apply_overrides(cache_section: Dict[str, List[str]], overrides: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
    """Merge overrides into a cache section. Returns (merged, applied_log)."""
    applied: Dict[str, Dict] = {}
    for term, override_buckets in overrides.items():
        before = cache_section.get(term)
        cache_section[term] = sorted(set(override_buckets) | set(before or []))
        applied[term] = {
            "ontology": sorted(before) if before else [],
            "override": sorted(override_buckets),
            "final": cache_section[term],
        }
    return cache_section, applied


def build_cache() -> Dict:
    started = time.time()
    print("[1/4] collecting terms across all AOP-Wiki snapshots …", file=sys.stderr)
    terms = collect_all_terms()
    print(
        f"  UBERON={len(terms['uberon'])}  CL={len(terms['cl'])}  GO={len(terms['go'])}",
        file=sys.stderr,
    )

    print("[2/4] resolving UBERON + CL via Ubergraph …", file=sys.stderr)
    uberon_anchors = resolve_anatomy_buckets(sorted(terms["uberon"]))
    cl_anchors = resolve_anatomy_buckets(sorted(terms["cl"]))

    print("[3/4] resolving GO BP via RO:0002296 results_in_development_of …", file=sys.stderr)
    go_to_uberon = resolve_go_via_results_in_development(sorted(terms["go"]))
    unique_uberon_targets = sorted({u for s in go_to_uberon.values() for u in s})
    go_target_anchors = resolve_anatomy_buckets(unique_uberon_targets) if unique_uberon_targets else {}
    go_buckets: Dict[str, Set[str]] = {}
    for go, uberons in go_to_uberon.items():
        bs: Set[str] = set()
        for u in uberons:
            bs |= anchors_to_buckets(go_target_anchors.get(u, set()))
        if bs:
            go_buckets[go] = bs

    print("[4/4] applying overrides + assembling JSON …", file=sys.stderr)
    uberon_short = _to_short_buckets(uberon_anchors)
    cl_short = _to_short_buckets(cl_anchors)
    go_short = {_short(iri): sorted(bs) for iri, bs in go_buckets.items()}

    uberon_short, ub_log = apply_overrides(uberon_short, OVERRIDES["uberon"])
    cl_short, cl_log = apply_overrides(cl_short, OVERRIDES["cl"])
    go_short, go_log = apply_overrides(go_short, OVERRIDES["go"])

    cache = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "aopwiki_sparql": AOPWIKI_SPARQL,
            "ubergraph_sparql": UBERGRAPH_SPARQL,
            "uberon_cl_relation": "(part_of | subClassOf)*",
            "go_relation": "RO:0002296 (results_in_development_of)",
        },
        "anchors": {
            bucket: [{"iri": a, "label": ANCHOR_LABELS.get(a, "")} for a in anchors]
            for bucket, anchors in ANCHORS.items()
        },
        "overrides": {
            "uberon": ub_log,
            "cl": cl_log,
            "go": go_log,
        },
        "uberon": dict(sorted(uberon_short.items())),
        "cl":     dict(sorted(cl_short.items())),
        "go":     dict(sorted(go_short.items())),
        "stats": {
            "uberon_terms_in_snapshots": len(terms["uberon"]),
            "uberon_terms_classified":   len(uberon_short),
            "cl_terms_in_snapshots":     len(terms["cl"]),
            "cl_terms_classified":       len(cl_short),
            "go_terms_in_snapshots":     len(terms["go"]),
            "go_terms_classified":       len(go_short),
            "elapsed_s":                 round(time.time() - started, 1),
        },
    }
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="static/data/organ_system_cache.json",
        help="Output JSON file (default: static/data/organ_system_cache.json)",
    )
    args = parser.parse_args()

    cache = build_cache()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    s = cache["stats"]
    print(
        f"\nWrote {out_path}  "
        f"UBERON {s['uberon_terms_classified']}/{s['uberon_terms_in_snapshots']}  "
        f"CL {s['cl_terms_classified']}/{s['cl_terms_in_snapshots']}  "
        f"GO {s['go_terms_classified']}/{s['go_terms_in_snapshots']}  "
        f"({s['elapsed_s']}s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
