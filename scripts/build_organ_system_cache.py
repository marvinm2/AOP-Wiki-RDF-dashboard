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
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

# Import the Signal-C regex from the runtime classifier so the label-regex
# bridge stays in sync. KEYWORD_PATTERNS lives next to the dashboard code.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plots.organ_systems import KEYWORD_PATTERNS  # noqa: E402

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

AOPWIKI_SPARQL = "https://aopwiki-multirdf.vhp4safety.nl/sparql"
UBERGRAPH_SPARQL = "https://ubergraph.apps.renci.org/sparql"

OBO = "http://purl.obolibrary.org/obo/"
BFO_PART_OF = OBO + "BFO_0000050"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RO_RESULTS_IN_DEV_OF = OBO + "RO_0002296"
# UPHENO:0000001 is the Unified Phenotype Ontology's "has phenotype affecting"
# predicate — links HP / MP terms (abnormal phenotypes) to the UBERON anatomy
# they affect, with Ubergraph's closure giving multiple parent contexts.
UPHENO_PHENOTYPE_OF = OBO + "UPHENO_0000001"

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
    "hp": {},
    "mp": {},
    "fma": {},
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
    """Return every UBERON / CL / GO / HP / MP / FMA IRI used in any AOP-Wiki RDF snapshot."""
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
    by_onto: Dict[str, Set[str]] = {
        "uberon": set(), "cl": set(), "go": set(), "hp": set(), "mp": set(), "fma": set(),
    }
    for r in rows:
        iri = r["term"]["value"]
        short = _short(iri)
        if short.startswith("UBERON_"):
            by_onto["uberon"].add(iri)
        elif short.startswith("CL_"):
            by_onto["cl"].add(iri)
        elif short.startswith("GO_"):
            by_onto["go"].add(iri)
        elif short.startswith("HP_"):
            by_onto["hp"].add(iri)
        elif short.startswith("MP_"):
            by_onto["mp"].add(iri)
        elif short.startswith("FMA_"):
            by_onto["fma"].add(iri)
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


def build_anchor_ancestor_graph() -> Dict[str, Set[str]]:
    """For each anchor, return the set of OTHER anchors that are its ancestors.

    "Ancestor" = reachable via ``(BFO:0000050 | rdfs:subClassOf)+`` (strict —
    excludes self-match). Result feeds :func:`prune_to_specific_anchors`.

    Example for the UBERON snapshot the dashboard ships against:
      liver (UBERON_0002107) → {digestive system, endocrine system}
      pancreas (UBERON_0001264) → {digestive system, endocrine system}
      gallbladder (UBERON_0001173) → {digestive system, common bile duct}

    The query is one SPARQL round-trip with all anchors on both sides.
    """
    anchor_iris = [f"<{OBO}{a}>" for a in ALL_ANCHORS]
    values_block = " ".join(anchor_iris)
    query = f"""
    PREFIX BFO: <{OBO}BFO_>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?desc ?anc WHERE {{
      VALUES ?desc {{ {values_block} }}
      VALUES ?anc  {{ {values_block} }}
      ?desc (BFO:0000050|rdfs:subClassOf)+ ?anc .
      FILTER(?desc != ?anc)
    }}
    """
    rows = sparql_select(UBERGRAPH_SPARQL, query)
    graph: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        graph[_short(r["desc"]["value"])].add(_short(r["anc"]["value"]))
    return dict(graph)


def prune_to_specific_anchors(
    reached: Set[str],
    anchor_ancestors: Dict[str, Set[str]],
) -> Set[str]:
    """Drop any anchor in ``reached`` that is an ancestor of another reached anchor.

    ``reached`` and the dict keys/values are **short** IRIs (``UBERON_xxx``).
    The intent: a term reaching {liver, digestive_system, endocrine_system}
    keeps only ``liver`` — the other two are ancestors of liver in UBERON and
    therefore over-general for this term's classification.

    Pancreas is an anchor for both Digestive and Endocrine. Pancreas reaches
    {pancreas, digestive_system, endocrine_system}; pruning keeps {pancreas},
    which projects to {Digestive, Endocrine} via the dual-anchor mapping. The
    intentional dual-org is preserved.
    """
    reached = set(reached)
    to_remove: Set[str] = set()
    for d in reached:
        to_remove |= anchor_ancestors.get(d, set()) & reached
    return reached - to_remove


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


def resolve_fma_via_uberon_xref(fma_terms: List[str]) -> Dict[str, Set[str]]:
    """Map each FMA IRI to the UBERON IRIs that cross-reference it.

    Two stages:

    1. **Ubergraph reverse direction** — ``?uberon oboInOwl:hasDbXref "FMA:nnnn"``.
       Misses most FMA IDs because Ubergraph's UBERON snapshot does not include
       FMA xrefs for the specific IDs AOP-Wiki uses.

    2. **OLS REST fallback** — for any FMA IRI still unresolved, query OLS for
       the term's xrefs and pull any UBERON-prefixed cross-reference. Cached
       under ``static/data/fma-ols-xrefs.json`` to avoid re-fetching on rebuilds.
    """
    if not fma_terms:
        return {}
    out: Dict[str, Set[str]] = {t: set() for t in fma_terms}

    # ---- Stage 1: Ubergraph reverse xref (UBERON → FMA literal) ------------
    xref_strings: Dict[str, str] = {
        iri: "FMA:" + _short(iri).split("_", 1)[-1] for iri in fma_terms
    }
    inverse: Dict[str, str] = {v: k for k, v in xref_strings.items()}

    for chunk in _chunks(list(xref_strings.values()), 40):
        xref_values = " ".join(f'"{x}"' for x in chunk)
        query = f"""
        PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
        SELECT DISTINCT ?xref ?uberon WHERE {{
          VALUES ?xref {{ {xref_values} }}
          ?uberon oboInOwl:hasDbXref ?xref .
          FILTER(STRSTARTS(STR(?uberon), "{OBO}UBERON_"))
        }}
        """
        rows = sparql_select(UBERGRAPH_SPARQL, query)
        for r in rows:
            xref = r["xref"]["value"]
            fma_iri = inverse.get(xref)
            if fma_iri:
                out[fma_iri].add(r["uberon"]["value"])
        time.sleep(0.5)

    return out


def _ols_fma_label(fma_iri: str) -> str:
    """OLS REST: fetch the ``rdfs:label`` for an FMA term.

    AOP-Wiki uses ``http://purl.obolibrary.org/obo/FMA_7210`` IRIs but FMA's
    canonical scheme in OLS is ``http://purl.org/sig/ont/fma/fma7210``. The
    obo/FMA_ form does not resolve in OLS; the sig/ont form does. We convert
    on the fly.

    OLS v2 ``/api/v2/ontologies/fma/classes?iri=...`` returns ``label`` as a
    JSON array (rarely scalar). Returns empty string if not found.
    """
    from urllib.parse import quote

    short = _short(fma_iri)  # e.g. "FMA_7210"
    fma_id = short.split("_", 1)[-1] if "_" in short else short
    canonical = f"http://purl.org/sig/ont/fma/fma{fma_id}"
    url = (
        "https://www.ebi.ac.uk/ols4/api/v2/ontologies/fma/classes"
        f"?iri={quote(canonical, safe='')}"
    )
    try:
        r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
        if r.status_code == 404:
            return ""
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"  ! OLS lookup failed for {short}: {exc}", file=sys.stderr)
        return ""

    elements = data.get("elements") or []
    if not elements:
        return ""
    label = elements[0].get("label")
    if isinstance(label, list):
        return label[0] if label else ""
    return label or ""


def resolve_fma_via_label(fma_iris: List[str]) -> Dict[str, Tuple[str, Set[str], List[str]]]:
    """Fallback FMA resolver: fetch label from OLS, apply Signal-C regex.

    Mirrors :func:`resolve_phenotype_via_label` for FMA terms. Useful because
    the canonical Ubergraph reverse-xref returns 0/21 matches and OLS has no
    UBERON cross-references on FMA terms (xref direction is one-way:
    UBERON → FMA, never the inverse). Label-matching converts "testis" to
    the Reproductive bucket regardless of the cross-reference gap.

    Cached under ``static/data/fma-ols-labels.json``.
    """
    if not fma_iris:
        return {}
    cache_path = Path("static/data/fma-ols-labels.json")
    cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}

    out: Dict[str, Tuple[str, Set[str], List[str]]] = {}
    for iri in fma_iris:
        short = _short(iri)
        if short in cache:
            label = cache[short]
        else:
            label = _ols_fma_label(iri)
            cache[short] = label
            time.sleep(0.3)  # polite to OLS
        if not label:
            continue
        matched_buckets: Set[str] = set()
        matched_patterns: List[str] = []
        for bucket, patterns in _COMPILED_KEYWORDS.items():
            for pattern in patterns:
                if pattern.search(label):
                    matched_buckets.add(bucket)
                    matched_patterns.append(f"{bucket}:{pattern.pattern}")
                    break
        if matched_buckets:
            out[iri] = (label, matched_buckets, matched_patterns)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    return out


# Compile the Signal-C bucket regex once for reuse.
_COMPILED_KEYWORDS: Dict[str, List[re.Pattern]] = {
    bucket: [re.compile(p, re.IGNORECASE) for p in patterns]
    for bucket, patterns in KEYWORD_PATTERNS.items()
}


def resolve_phenotype_via_label(phen_terms: List[str]) -> Dict[str, Tuple[str, Set[str], List[str]]]:
    """Map each HP / MP IRI to bucket(s) via its ``rdfs:label`` + Signal-C regex.

    This replaces the previous ``UPHENO:0000001`` approach, which Ubergraph
    materialises as a generic ancestor closure that returns the *same* anatomy
    set for ``HP_0001397`` (hepatic cirrhosis) and ``HP_0001414`` (renal
    hypoplasia). The label-regex bridge gives mono-organ classification for
    phenotypes whose name names an organ system.

    Returns ``{phen_iri: (label, {bucket, ...}, [pattern, ...])}``. Phenotypes
    whose label matches no pattern are absent from the result (≡ unclassified).
    """
    if not phen_terms:
        return {}

    labels: Dict[str, str] = {}
    rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
    for chunk in _chunks(phen_terms, 80):
        term_values = " ".join(f"<{t}>" for t in chunk)
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?term ?label WHERE {{
          VALUES ?term {{ {term_values} }}
          ?term rdfs:label ?label .
          FILTER(LANG(?label) = "" || LANG(?label) = "en")
        }}
        """
        rows = sparql_select(UBERGRAPH_SPARQL, query)
        for r in rows:
            iri = r["term"]["value"]
            lbl = r["label"]["value"]
            # Prefer the first non-empty English label.
            labels.setdefault(iri, lbl)
        time.sleep(0.5)

    out: Dict[str, Tuple[str, Set[str], List[str]]] = {}
    for iri in phen_terms:
        label = labels.get(iri, "")
        if not label:
            continue
        matched_buckets: Set[str] = set()
        matched_patterns: List[str] = []
        for bucket, patterns in _COMPILED_KEYWORDS.items():
            for pattern in patterns:
                if pattern.search(label):
                    matched_buckets.add(bucket)
                    matched_patterns.append(f"{bucket}:{pattern.pattern}")
                    break  # one pattern per bucket is enough
        if matched_buckets:
            out[iri] = (label, matched_buckets, matched_patterns)
    return out


def anchors_to_buckets(anchor_iris: Set[str]) -> Set[str]:
    buckets: Set[str] = set()
    for iri in anchor_iris:
        buckets |= ANCHOR_TO_BUCKETS.get(_short(iri), set())
    return buckets


# ---------------------------------------------------------------------------
# Step 3 — assemble the cache
# ---------------------------------------------------------------------------


def _to_short_buckets(
    term_to_anchors: Dict[str, Set[str]],
    anchor_ancestors: Optional[Dict[str, Set[str]]] = None,
    pruning_log: Optional[Dict[str, Dict]] = None,
) -> Dict[str, List[str]]:
    """Project resolved-anchor sets onto bucket-name lists, keyed by short IRI.

    If ``anchor_ancestors`` is provided, apply :func:`prune_to_specific_anchors`
    before projection — drop anchors that are ancestors of more specific
    anchors reached for the same term. This collapses e.g. liver-derived terms
    from {Digestive, Endocrine, Hepatobiliary} to {Hepatobiliary} only.

    If ``pruning_log`` dict is supplied, populate it with per-term records of
    the before/after anchor sets for transparency in the methodology API.
    """
    out: Dict[str, List[str]] = {}
    for iri, anchors in term_to_anchors.items():
        short_anchors = {_short(a) for a in anchors}
        if anchor_ancestors is not None and len(short_anchors) > 1:
            pruned = prune_to_specific_anchors(short_anchors, anchor_ancestors)
            if pruning_log is not None and pruned != short_anchors:
                pruning_log[_short(iri)] = {
                    "before": sorted(short_anchors),
                    "after": sorted(pruned),
                    "dropped": sorted(short_anchors - pruned),
                }
            short_anchors = pruned
        # Re-prefix to OBO IRIs for the existing anchors_to_buckets() helper.
        buckets = anchors_to_buckets({OBO + a for a in short_anchors})
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
    print("[1/5] collecting terms across all AOP-Wiki snapshots …", file=sys.stderr)
    terms = collect_all_terms()
    print(
        f"  UBERON={len(terms['uberon'])}  CL={len(terms['cl'])}  GO={len(terms['go'])}  "
        f"HP={len(terms['hp'])}  MP={len(terms['mp'])}  FMA={len(terms['fma'])}",
        file=sys.stderr,
    )

    print("[2/5] building anchor-ancestor graph from Ubergraph …", file=sys.stderr)
    anchor_ancestors = build_anchor_ancestor_graph()
    print(
        f"  {sum(len(v) for v in anchor_ancestors.values())} ancestor pairs across "
        f"{len(ALL_ANCHORS)} anchors",
        file=sys.stderr,
    )

    print("[3/5] resolving UBERON + CL via Ubergraph (with specificity pruning) …", file=sys.stderr)
    uberon_anchors = resolve_anatomy_buckets(sorted(terms["uberon"]))
    cl_anchors = resolve_anatomy_buckets(sorted(terms["cl"]))

    print(
        "[4/5] resolving GO via RO:0002296, HP/MP via label-regex, FMA via "
        "Ubergraph xref + OLS fallback …",
        file=sys.stderr,
    )
    go_to_uberon = resolve_go_via_results_in_development(sorted(terms["go"]))
    fma_to_uberon = resolve_fma_via_uberon_xref(sorted(terms["fma"]))
    phenotype_labels = resolve_phenotype_via_label(
        sorted(terms["hp"]) + sorted(terms["mp"])
    )
    # FMA terms unresolved via UBERON xref → label-regex fallback (OLS).
    fma_unresolved = [iri for iri, uberons in fma_to_uberon.items() if not uberons]
    fma_labels = resolve_fma_via_label(fma_unresolved)
    print(
        f"  FMA via xref: {sum(1 for v in fma_to_uberon.values() if v)}, "
        f"FMA via label: {len(fma_labels)}, "
        f"FMA unresolved: {len(fma_to_uberon) - sum(1 for v in fma_to_uberon.values() if v) - len(fma_labels)}",
        file=sys.stderr,
    )

    # GO / FMA-via-xref still anchor through UBERON; resolve UBERON targets
    # via the same closure + specificity pruning as direct UBERON terms.
    unique_uberon_targets = sorted(
        {u for s in go_to_uberon.values() for u in s}
        | {u for s in fma_to_uberon.values() for u in s}
    )
    process_target_anchors = (
        resolve_anatomy_buckets(unique_uberon_targets) if unique_uberon_targets else {}
    )

    def _project_through_uberon(term_to_uberon: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """For each term, union the pruned bucket sets of its UBERON targets."""
        out: Dict[str, Set[str]] = {}
        for term, uberons in term_to_uberon.items():
            bs: Set[str] = set()
            for u in uberons:
                anchors = {_short(a) for a in process_target_anchors.get(u, set())}
                if len(anchors) > 1:
                    anchors = prune_to_specific_anchors(anchors, anchor_ancestors)
                bs |= anchors_to_buckets({OBO + a for a in anchors})
            if bs:
                out[term] = bs
        return out

    go_buckets = _project_through_uberon(go_to_uberon)
    fma_buckets = _project_through_uberon(fma_to_uberon)
    # Add FMA terms resolved via label regex.
    fma_label_log: Dict[str, Dict] = {}
    for iri, (label, buckets, patterns) in fma_labels.items():
        fma_buckets[iri] = buckets
        fma_label_log[_short(iri)] = {
            "label": label,
            "buckets": sorted(buckets),
            "matched_patterns": patterns,
        }

    # Phenotypes get their buckets directly from label regex — no anchor walk.
    hp_short: Dict[str, List[str]] = {}
    mp_short: Dict[str, List[str]] = {}
    phenotype_pattern_log: Dict[str, Dict] = {}
    for iri, (label, buckets, patterns) in phenotype_labels.items():
        short = _short(iri)
        phenotype_pattern_log[short] = {
            "label": label,
            "buckets": sorted(buckets),
            "matched_patterns": patterns,
        }
        if short.startswith("HP_"):
            hp_short[short] = sorted(buckets)
        elif short.startswith("MP_"):
            mp_short[short] = sorted(buckets)

    print("[5/5] applying pruning + overrides + assembling JSON …", file=sys.stderr)
    uberon_pruning: Dict[str, Dict] = {}
    cl_pruning: Dict[str, Dict] = {}
    uberon_short = _to_short_buckets(uberon_anchors, anchor_ancestors, uberon_pruning)
    cl_short = _to_short_buckets(cl_anchors, anchor_ancestors, cl_pruning)
    go_short = {_short(iri): sorted(bs) for iri, bs in go_buckets.items()}
    fma_short = {_short(iri): sorted(bs) for iri, bs in fma_buckets.items()}

    uberon_short, ub_log = apply_overrides(uberon_short, OVERRIDES["uberon"])
    cl_short, cl_log = apply_overrides(cl_short, OVERRIDES["cl"])
    go_short, go_log = apply_overrides(go_short, OVERRIDES["go"])
    hp_short, hp_log = apply_overrides(hp_short, OVERRIDES["hp"])
    mp_short, mp_log = apply_overrides(mp_short, OVERRIDES["mp"])
    fma_short, fma_log = apply_overrides(fma_short, OVERRIDES["fma"])

    cache = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "aopwiki_sparql": AOPWIKI_SPARQL,
            "ubergraph_sparql": UBERGRAPH_SPARQL,
            "uberon_cl_relation": "(part_of | subClassOf)*  (then anchor-specificity pruning)",
            "go_relation": "RO:0002296 (results_in_development_of) → UBERON → anchor (with specificity pruning)",
            "hp_mp_relation": "rdfs:label → Signal-C bucket regex match",
            "fma_relation": "Ubergraph UBERON oboInOwl:hasDbXref \"FMA:nnnn\" reverse + OLS REST forward fallback",
        },
        "anchors": {
            bucket: [{"iri": a, "label": ANCHOR_LABELS.get(a, "")} for a in anchors]
            for bucket, anchors in ANCHORS.items()
        },
        "anchor_ancestors": {
            anchor: sorted(ancestors) for anchor, ancestors in sorted(anchor_ancestors.items())
        },
        "specificity_pruning": {
            "uberon": uberon_pruning,
            "cl": cl_pruning,
        },
        "phenotype_label_matches": phenotype_pattern_log,
        "fma_label_matches": fma_label_log,
        "overrides": {
            "uberon": ub_log,
            "cl": cl_log,
            "go": go_log,
            "hp": hp_log,
            "mp": mp_log,
            "fma": fma_log,
        },
        "uberon": dict(sorted(uberon_short.items())),
        "cl":     dict(sorted(cl_short.items())),
        "go":     dict(sorted(go_short.items())),
        "hp":     dict(sorted(hp_short.items())),
        "mp":     dict(sorted(mp_short.items())),
        "fma":    dict(sorted(fma_short.items())),
        "stats": {
            "uberon_terms_in_snapshots": len(terms["uberon"]),
            "uberon_terms_classified":   len(uberon_short),
            "uberon_terms_pruned":       len(uberon_pruning),
            "cl_terms_in_snapshots":     len(terms["cl"]),
            "cl_terms_classified":       len(cl_short),
            "cl_terms_pruned":           len(cl_pruning),
            "go_terms_in_snapshots":     len(terms["go"]),
            "go_terms_classified":       len(go_short),
            "hp_terms_in_snapshots":     len(terms["hp"]),
            "hp_terms_classified":       len(hp_short),
            "mp_terms_in_snapshots":     len(terms["mp"]),
            "mp_terms_classified":       len(mp_short),
            "fma_terms_in_snapshots":    len(terms["fma"]),
            "fma_terms_classified":      len(fma_short),
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
        f"HP {s['hp_terms_classified']}/{s['hp_terms_in_snapshots']}  "
        f"MP {s['mp_terms_classified']}/{s['mp_terms_in_snapshots']}  "
        f"FMA {s['fma_terms_classified']}/{s['fma_terms_in_snapshots']}  "
        f"({s['elapsed_s']}s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
