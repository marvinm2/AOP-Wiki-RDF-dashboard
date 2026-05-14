"""Diagnostic probe — print Ubergraph closure paths for the organ-system audit.

For a hand-picked watchlist of UBERON / CL / HP / MP terms (the worst
multi-bucket offenders plus a few mono-bucket controls), this script queries
Ubergraph with three closure variants and prints which anchors each variant
reaches. The output is the evidence base for picking the right closure rule
in scripts/build_organ_system_cache.py.

Three closure variants on UBERON / CL:
  - partof   : BFO:0000050+ only (no subClassOf)
  - mixed    : (BFO:0000050 | rdfs:subClassOf)+ (current production)
  - bounded  : (BFO:0000050 | rdfs:subClassOf){0,3} emulated as UNION

HP / MP go through UPHENO:0000001 first, then resolve the UBERON targets via
each variant. Output is markdown-style text, saved to
static/data/audit-closure-paths-baseline.txt.

Usage:
    cd AOP-Wiki-RDF-dashboard/
    python scripts/audit_closure_paths.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Re-use the SPARQL helper, anchors, and OBO prefix from the cache builder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_organ_system_cache import (  # noqa: E402
    ALL_ANCHORS,
    ANCHOR_LABELS,
    ANCHOR_TO_BUCKETS,
    OBO,
    UBERGRAPH_SPARQL,
    sparql_select,
    _short,
)

# ---------------------------------------------------------------------------
# Watchlist — 9 offenders + 3 mono-bucket controls
# ---------------------------------------------------------------------------

WATCHLIST: List[Tuple[str, str, str]] = [
    # (short_id, prefix, description)
    ("UBERON_0001280", "UBERON", "pancreatic duct — 3 buckets, deliberate (dual-anchor)"),
    ("UBERON_0002107", "UBERON", "liver — control, expect Hepatobiliary only"),
    ("UBERON_0000948", "UBERON", "heart — control, expect Cardiovascular only"),
    ("CL_0000182",     "CL",     "hepatocyte — 3 buckets, suspect subClassOf chain"),
    ("CL_0000632",     "CL",     "pancreatic acinar cell — 3 buckets, dual-anchor + closure"),
    ("CL_0002538",     "CL",     "intestinal epithelial cell — 3 buckets, false-positive Endocrine?"),
    ("CL_0002196",     "CL",     "pancreatic ductal cell — 3 buckets"),
    ("CL_0000235",     "CL",     "macrophage — control, expect Immune/Haematopoietic"),
    ("HP_0001397",     "HP",     "hepatic cirrhosis — 3 buckets via UPHENO"),
    ("HP_0001414",     "HP",     "renal hypoplasia — 3 buckets, WRONG (should be Renal only)"),
    ("MP_0010019",     "MP",     "abnormal cardiac morphology — 4 buckets, mostly wrong"),
    ("MP_0001860",     "MP",     "abnormal liver carbohydrate metabolism — 4 buckets"),
]


# ---------------------------------------------------------------------------
# Three closure variant queries
# ---------------------------------------------------------------------------

def reach_partof(term_iri: str) -> Set[str]:
    """Anchors reachable via BFO:0000050* (part_of, including 0-hop self-match)."""
    anchor_values = " ".join(f"<{OBO}{a}>" for a in ALL_ANCHORS)
    query = f"""
    PREFIX BFO: <{OBO}BFO_>
    SELECT DISTINCT ?anchor WHERE {{
      VALUES ?anchor {{ {anchor_values} }}
      <{term_iri}> BFO:0000050* ?anchor .
    }}
    """
    rows = sparql_select(UBERGRAPH_SPARQL, query)
    return {r["anchor"]["value"] for r in rows}


def reach_mixed(term_iri: str) -> Set[str]:
    """Anchors reachable via (BFO:0000050 | rdfs:subClassOf)* — production baseline."""
    anchor_values = " ".join(f"<{OBO}{a}>" for a in ALL_ANCHORS)
    query = f"""
    PREFIX BFO: <{OBO}BFO_>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?anchor WHERE {{
      VALUES ?anchor {{ {anchor_values} }}
      <{term_iri}> (BFO:0000050|rdfs:subClassOf)* ?anchor .
    }}
    """
    rows = sparql_select(UBERGRAPH_SPARQL, query)
    return {r["anchor"]["value"] for r in rows}


def reach_bounded(term_iri: str, max_hops: int = 3) -> Set[str]:
    """Anchors reachable via (part_of | subClassOf){0,max_hops} — emulated as UNION."""
    anchor_values = " ".join(f"<{OBO}{a}>" for a in ALL_ANCHORS)
    # Build UNION of fixed-length paths 0..max_hops
    p = "(BFO:0000050|rdfs:subClassOf)"
    union_blocks = ["{ <%s> ?anchor . FILTER(?anchor = <%s>) }" % (term_iri, "<placeholder>")]
    union_blocks = []
    union_blocks.append(f"{{ FILTER(<{term_iri}> = ?anchor) }}")  # 0-hop self-match
    accum = ""
    for k in range(1, max_hops + 1):
        accum = (accum + "/" + p) if accum else p
        union_blocks.append(f"{{ <{term_iri}> {accum} ?anchor . }}")
    union_clause = " UNION ".join(union_blocks)
    query = f"""
    PREFIX BFO: <{OBO}BFO_>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?anchor WHERE {{
      VALUES ?anchor {{ {anchor_values} }}
      {union_clause}
    }}
    """
    rows = sparql_select(UBERGRAPH_SPARQL, query)
    return {r["anchor"]["value"] for r in rows}


def upheno_targets(phen_iri: str) -> Set[str]:
    """UBERON targets a phenotype affects via UPHENO:0000001."""
    query = f"""
    PREFIX UPHENO: <{OBO}UPHENO_>
    SELECT DISTINCT ?uberon WHERE {{
      <{phen_iri}> UPHENO:0000001 ?uberon .
      FILTER(STRSTARTS(STR(?uberon), "{OBO}UBERON_"))
    }}
    """
    rows = sparql_select(UBERGRAPH_SPARQL, query)
    return {r["uberon"]["value"] for r in rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def anchors_to_buckets(anchor_iris: Set[str]) -> Set[str]:
    buckets: Set[str] = set()
    for iri in anchor_iris:
        buckets |= ANCHOR_TO_BUCKETS.get(_short(iri), set())
    return buckets


def fmt_anchors(anchors: Set[str]) -> str:
    if not anchors:
        return "(none)"
    return ", ".join(sorted(
        f"{_short(a)} ({ANCHOR_LABELS.get(_short(a), '?')})"
        for a in anchors
    ))


def fmt_buckets(buckets: Set[str]) -> str:
    return "{" + ", ".join(sorted(buckets)) + "}" if buckets else "{}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def probe_anatomy_term(short_id: str) -> Dict[str, Set[str]]:
    iri = OBO + short_id
    partof = reach_partof(iri)
    time.sleep(0.3)
    mixed = reach_mixed(iri)
    time.sleep(0.3)
    bounded = reach_bounded(iri)
    time.sleep(0.3)
    return {"partof": partof, "mixed": mixed, "bounded": bounded}


def probe_phenotype_term(short_id: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    iri = OBO + short_id
    targets = upheno_targets(iri)
    time.sleep(0.3)
    # Resolve each target via all three variants; union to produce a per-variant anchor set.
    by_variant: Dict[str, Set[str]] = {"partof": set(), "mixed": set(), "bounded": set()}
    for t in sorted(targets):
        by_variant["partof"]  |= reach_partof(t);  time.sleep(0.2)
        by_variant["mixed"]   |= reach_mixed(t);   time.sleep(0.2)
        by_variant["bounded"] |= reach_bounded(t); time.sleep(0.2)
    return targets, by_variant


def render_section(short_id: str, prefix: str, description: str, lines: List[str]) -> None:
    lines.append(f"## {short_id} — {description}")
    lines.append("")
    if prefix in {"UBERON", "CL"}:
        results = probe_anatomy_term(short_id)
        for variant in ("partof", "mixed", "bounded"):
            anchors = results[variant]
            buckets = anchors_to_buckets(anchors)
            lines.append(f"- **{variant:<8}** → buckets={fmt_buckets(buckets)}  ({len(buckets)} buckets)")
            lines.append(f"  - anchors: {fmt_anchors(anchors)}")
    elif prefix in {"HP", "MP"}:
        targets, by_variant = probe_phenotype_term(short_id)
        if not targets:
            lines.append(f"- UPHENO:0000001 returned **no UBERON targets** — phenotype has no anatomy axiom.")
        else:
            lines.append(f"- UPHENO:0000001 targets ({len(targets)}):")
            for t in sorted(targets):
                lines.append(f"  - {_short(t)}")
            lines.append("")
            for variant in ("partof", "mixed", "bounded"):
                anchors = by_variant[variant]
                buckets = anchors_to_buckets(anchors)
                lines.append(f"- **{variant:<8}** (union over targets) → buckets={fmt_buckets(buckets)}  ({len(buckets)} buckets)")
                lines.append(f"  - anchors: {fmt_anchors(anchors)}")
    lines.append("")


def main() -> int:
    out_lines: List[str] = [
        "# Closure-paths audit baseline",
        "",
        "Probe of Ubergraph anchor reachability under three closure variants:",
        "",
        "- `partof`   = `BFO:0000050*` (part_of only, includes 0-hop self-match)",
        "- `mixed`    = `(BFO:0000050 | rdfs:subClassOf)*` (current production baseline)",
        "- `bounded`  = `(BFO:0000050 | rdfs:subClassOf){0,3}` (emulated UNION, depth ≤3)",
        "",
        "HP / MP terms first resolve to UBERON via `UPHENO:0000001`, then each",
        "UBERON target is walked with the variant. Output is union over targets.",
        "",
        "Generated by `scripts/audit_closure_paths.py`.",
        "",
        "---",
        "",
    ]

    for short_id, prefix, description in WATCHLIST:
        print(f"  probing {short_id} ({description}) …", file=sys.stderr)
        render_section(short_id, prefix, description, out_lines)

    out_path = Path("static/data/audit-closure-paths-baseline.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nWrote {out_path} ({len(out_lines)} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
