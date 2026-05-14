"""Diagnostic probe — what relation actually distinguishes HP phenotypes?

HP_0001397 (hepatic cirrhosis, liver-specific) and HP_0001414 (renal hypoplasia,
kidney-specific) currently return the *same* 35 UBERON targets via
``UPHENO:0000001``. This script dumps every (predicate, UBERON-object) pair for
both phenotypes so we can pick a more specific relation.

Usage:
    cd AOP-Wiki-RDF-dashboard/
    python scripts/audit_phenotype_relations.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_organ_system_cache import OBO, UBERGRAPH_SPARQL, sparql_select, _short  # noqa: E402


WATCHLIST = [
    ("HP_0001397", "hepatic cirrhosis — should hit liver only"),
    ("HP_0001414", "renal hypoplasia — should hit kidney only"),
    ("MP_0010019", "abnormal cardiac morphology — should hit heart only"),
    ("MP_0001860", "abnormal liver carbohydrate metabolism — liver only"),
]


def all_uberon_via(pred_filter_clause: str, term_iri: str) -> Dict[str, Set[str]]:
    """For a single phenotype, return {predicate_short: {uberon_short, ...}}."""
    query = f"""
    SELECT DISTINCT ?p ?o WHERE {{
      <{term_iri}> ?p ?o .
      FILTER(STRSTARTS(STR(?o), "{OBO}UBERON_"))
      {pred_filter_clause}
    }}
    """
    rows = sparql_select(UBERGRAPH_SPARQL, query)
    out: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        out[_short(r["p"]["value"])].add(_short(r["o"]["value"]))
    return out


def main() -> int:
    lines = []
    lines.append("# Phenotype-relation probe")
    lines.append("")
    lines.append("For each watchlist phenotype, list every direct predicate that points to a UBERON term.")
    lines.append("Compare across phenotypes to find a relation that actually distinguishes them.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-phenotype dump
    by_term: Dict[str, Dict[str, Set[str]]] = {}
    for short_id, desc in WATCHLIST:
        print(f"  probing {short_id} ({desc}) …", file=sys.stderr)
        iri = OBO + short_id
        result = all_uberon_via("", iri)
        by_term[short_id] = result
        lines.append(f"## {short_id} — {desc}")
        lines.append("")
        if not result:
            lines.append("- (no direct UBERON-valued predicates)")
        else:
            for pred in sorted(result):
                ubs = sorted(result[pred])
                lines.append(f"- **{pred}** ({len(ubs)} UBERON terms)")
                if len(ubs) <= 8:
                    for u in ubs:
                        lines.append(f"  - {u}")
                else:
                    lines.append(f"  - {', '.join(ubs[:8])}, … (+{len(ubs) - 8} more)")
        lines.append("")

    # Cross-comparison
    lines.append("---")
    lines.append("")
    lines.append("## Predicate cross-comparison (intersection across all watchlist phenotypes)")
    lines.append("")
    lines.append("If a predicate returns DIFFERENT UBERON sets for different phenotypes, it's useful.")
    lines.append("If it returns the same set, it's a generic ancestor closure relation.")
    lines.append("")

    all_preds: Set[str] = set()
    for term_data in by_term.values():
        all_preds |= set(term_data.keys())

    for pred in sorted(all_preds):
        lines.append(f"### {pred}")
        lines.append("")
        for short_id, _desc in WATCHLIST:
            ubs = by_term[short_id].get(pred, set())
            lines.append(f"- {short_id}: {len(ubs)} UBERON terms — {sorted(ubs)[:6]}…")
        lines.append("")

    out_path = Path("static/data/audit-phenotype-relations.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
