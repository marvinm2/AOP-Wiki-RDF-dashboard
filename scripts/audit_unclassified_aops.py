"""Audit the AOPs that fall into the No-annotation bucket under scope="all".

Each unclassified AOP is categorised by *why* the classifier didn't place it,
so the bucket of ~160 can be triaged into genuine no-annotation cases vs
classifier false-negatives (FMA IRIs absent from the cache, non-vertebrate
UBERON terms, behavioural HP/MP terms, etc.). Writes a markdown report plus a
JSON sidecar to ``docs/audits/``.

Usage:
    cd AOP-Wiki-RDF-dashboard/
    SPARQL_ENDPOINT=https://aopwiki-multirdf.vhp4safety.nl/sparql \\
        python scripts/audit_unclassified_aops.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same paths the runtime classifier uses, so the audit reflects exactly what
# users see on the dashboard.
from plots.latest_plots import (  # noqa: E402
    _get_coverage_dataframe,
    _aggregate_per_aop,
)
from plots.organ_systems import (  # noqa: E402
    NO_ANNOTATION_BUCKET,
    classify_anatomy,
    classify_process,
    classify_text,
    _UBERON_BUCKETS,
    _CL_BUCKETS,
    _GO_BUCKETS,
    _HP_BUCKETS,
    _MP_BUCKETS,
    _FMA_BUCKETS,
)


SUFFIX_PREFIXES = ("UBERON_", "CL_", "FMA_", "GO_", "HP_", "MP_")


def _suffix(iri: str) -> str:
    return iri.rsplit("/", 1)[-1] if iri else ""


def _ontology(suffix: str) -> str:
    for p in SUFFIX_PREFIXES:
        if suffix.startswith(p):
            return p.rstrip("_")
    return "OTHER"


def _bucket_lookup(suffix: str):
    """Returns the cache dict for this ontology so we can tell 'in cache, empty
    set' (over-generic / non-vertebrate term) from 'not in cache' (missing
    Ubergraph mapping)."""
    if suffix.startswith("UBERON_"):
        return _UBERON_BUCKETS
    if suffix.startswith("CL_"):
        return _CL_BUCKETS
    if suffix.startswith("FMA_"):
        return _FMA_BUCKETS
    if suffix.startswith("GO_"):
        return _GO_BUCKETS
    if suffix.startswith("HP_"):
        return _HP_BUCKETS
    if suffix.startswith("MP_"):
        return _MP_BUCKETS
    return None


def categorise_aop(granular_rows: list[dict], title: str) -> dict:
    """Inspect the per-AOP signal-source rows and return a category + evidence."""
    # Collect IRIs the classifier looked at, per source column.
    iris = set()
    for r in granular_rows:
        for col in ("organ", "cell", "obj", "proc"):
            v = (r.get(col) or {}).get("value") if isinstance(r.get(col), dict) else r.get(col)
            if v:
                iris.add(v)

    if not iris:
        # No raw IRIs on any KE *and* title didn't match Signal C (otherwise
        # the AOP would have been classified under scope="all"). Genuine
        # no-annotation case.
        return {
            "category": "genuinely_unannotated",
            "iris": [],
            "ontology_breakdown": {},
            "title_signal_c": sorted(classify_text(title)),
        }

    # We have IRIs but no bucket emerged. Break them down by ontology and by
    # whether each IRI is in the cache at all (vs in cache but with an empty
    # bucket set).
    in_cache_empty = defaultdict(list)  # ontology -> [IRI]
    not_in_cache = defaultdict(list)    # ontology -> [IRI]
    unknown_ontology = []
    for iri in iris:
        sfx = _suffix(iri)
        ont = _ontology(sfx)
        if ont == "OTHER":
            unknown_ontology.append(iri)
            continue
        cache = _bucket_lookup(sfx)
        if cache is None:
            unknown_ontology.append(iri)
            continue
        if sfx in cache:
            if not cache[sfx]:  # in cache but empty bucket set (over-generic)
                in_cache_empty[ont].append(iri)
            else:
                # In cache *and* has buckets — but somehow didn't classify this AOP.
                # That shouldn't happen; record as anomaly.
                in_cache_empty[f"{ont} (ANOMALY: cache says {sorted(cache[sfx])})"].append(iri)
        else:
            not_in_cache[ont].append(iri)

    ontology_breakdown = {
        "in_cache_empty": {k: sorted(set(v)) for k, v in in_cache_empty.items()},
        "not_in_cache": {k: sorted(set(v)) for k, v in not_in_cache.items()},
        "unknown_ontology": sorted(set(unknown_ontology)),
    }

    # Pick a primary category by precedence of "most likely tooling fix".
    if not_in_cache.get("FMA"):
        category = "fma_missing_from_cache"
    elif not_in_cache.get("UBERON"):
        category = "uberon_missing_from_cache"
    elif in_cache_empty.get("UBERON"):
        category = "uberon_generic_or_non_vertebrate"
    elif in_cache_empty.get("CL") or not_in_cache.get("CL"):
        category = "cl_unmapped"
    elif in_cache_empty.get("HP") or in_cache_empty.get("MP") or not_in_cache.get("HP") or not_in_cache.get("MP"):
        category = "phenotype_no_anatomy_axiom"
    elif in_cache_empty.get("GO") or not_in_cache.get("GO"):
        category = "go_no_ro_axiom"
    elif unknown_ontology:
        category = "exotic_ontology"
    else:
        category = "other"

    return {
        "category": category,
        "iris": sorted(iris),
        "ontology_breakdown": ontology_breakdown,
        "title_signal_c": sorted(classify_text(title)),
    }


CATEGORY_LABELS = {
    "genuinely_unannotated":         "Genuinely unannotated (no IRIs on any KE, title doesn't match Signal C)",
    "fma_missing_from_cache":        "Has FMA IRIs absent from the cache — likely older FMA IDs not in Ubergraph",
    "uberon_missing_from_cache":     "Has UBERON IRIs absent from the cache — likely UBERON version skew",
    "uberon_generic_or_non_vertebrate":  "UBERON IRIs present in cache but classify to no bucket — generic anatomy or non-vertebrate (swim bladder, eggshell, generic 'organ')",
    "cl_unmapped":                   "CL (cell type) IRIs that don't classify to any organ bucket",
    "phenotype_no_anatomy_axiom":    "HP/MP terms with no anatomy axiom — typically behavioural phenotypes by design",
    "go_no_ro_axiom":                "GO biological-process IRIs without an RO:0002296 axiom to a UBERON anchor",
    "exotic_ontology":               "Annotations from ontologies the classifier doesn't currently recognise",
    "other":                         "Other (no clear primary cause)",
}


def main() -> int:
    print("Pulling granular coverage rows from the live endpoint…", flush=True)
    granular, aop_universe, version_label = _get_coverage_dataframe(version=None)
    per_pair = _aggregate_per_aop(granular, aop_universe, scope="all")

    unclassified = per_pair[per_pair["Organ System"] == NO_ANNOTATION_BUCKET]
    n_unclassified = len(unclassified)
    print(f"Found {n_unclassified} unclassified AOPs under scope=all (graph: {version_label})", flush=True)

    # Group the granular SPARQL rows by AOP so we can look at the raw IRIs the
    # classifier saw per AOP. The granular DF here is the *post-_compute_coverage_rows*
    # form which only has rows for IRIs that classified — useless for our audit.
    # We need the raw query rows back. Re-run that pull.
    from plots.latest_plots import _query_aop_signal_rows
    # _get_coverage_dataframe resolves and caches the graph URI internally;
    # re-resolve it the same way:
    target_graph = f"http://aopwiki.org/graph/{version_label}"
    raw_rows = _query_aop_signal_rows(target_graph)

    per_aop_raw: dict[str, list[dict]] = defaultdict(list)
    for r in raw_rows:
        aop = r.get("aop", {}).get("value")
        if aop:
            per_aop_raw[aop].append(r)

    audit: list[dict] = []
    for _, row in unclassified.iterrows():
        aop = row["AOP"]
        title = row.get("AOP Title", "") or aop_universe.get(aop, "")
        result = categorise_aop(per_aop_raw.get(aop, []), title)
        audit.append({
            "aop": aop,
            "title": title,
            **result,
        })

    cat_counts = Counter(a["category"] for a in audit)
    print("\nCategory breakdown:")
    for cat, n in cat_counts.most_common():
        print(f"  {n:>4}  {cat}")

    # Write outputs
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    md_path = out_dir / f"unclassified_audit_{today}.md"
    json_path = out_dir / f"unclassified_audit_{today}.json"

    json_path.write_text(json.dumps({
        "generated": today,
        "snapshot_version": version_label,
        "scope": "all",
        "total_unclassified": n_unclassified,
        "category_counts": dict(cat_counts),
        "aops": audit,
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote {json_path}")

    # Markdown report
    lines = []
    lines.append(f"# Unclassified-AOP audit — {today}")
    lines.append("")
    lines.append(f"- **Snapshot:** `{version_label}`")
    lines.append(f"- **Scope:** `all` (every KE counts toward Signals A/A'/B; Signal C runs on AOP titles regardless)")
    lines.append(f"- **Total unclassified:** **{n_unclassified}** AOPs (`Organ System == \"{NO_ANNOTATION_BUCKET}\"`)")
    lines.append("")
    lines.append("Each AOP below has been categorised by the *primary* reason it didn't classify. The categorisation is heuristic — an AOP with FMA-not-in-cache AND title-matches-Signal-C will be reported under FMA, because that's the more actionable tooling fix.")
    lines.append("")
    lines.append("## Category counts")
    lines.append("")
    lines.append("| Count | Category |")
    lines.append("|------:|----------|")
    for cat, n in cat_counts.most_common():
        lines.append(f"| {n} | **{cat}** — {CATEGORY_LABELS.get(cat, cat)} |")
    lines.append("")

    # Per-category sections with up to 5 example AOPs each
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for a in audit:
        by_cat[a["category"]].append(a)

    for cat, n in cat_counts.most_common():
        lines.append(f"## {cat}  ({n} AOPs)")
        lines.append("")
        lines.append(f"_{CATEGORY_LABELS.get(cat, cat)}_")
        lines.append("")
        examples = by_cat[cat][:5]
        for a in examples:
            aop_id = a["aop"].rsplit("/", 1)[-1]
            lines.append(f"- [`{aop_id}`]({a['aop']}) — {a['title'] or '(no title)'}")
            iris = a.get("iris", [])
            if iris:
                preview = ", ".join(_suffix(i) for i in iris[:6])
                if len(iris) > 6:
                    preview += f", … (+{len(iris)-6} more)"
                lines.append(f"  - IRIs: {preview}")
            sigc = a.get("title_signal_c", [])
            if sigc:
                lines.append(f"  - Title would match Signal C: {', '.join(sigc)}")
        if n > 5:
            lines.append(f"")
            lines.append(f"_…and {n - 5} more — see the JSON sidecar for the complete list._")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    # Cross-cutting finding: any AOP with `title_signal_c` populated had a
    # title that would match the Signal C regex, yet Signal C didn't fire in
    # production. That happens when the AOP has zero rows in
    # `_query_aop_signal_rows` (which requires `aopo:has_key_event`) — Signal C
    # is only seeded inside that loop, so AOPs without KEs in the SPARQL pull
    # never get Signal C even when their title would otherwise classify.
    sigc_missed = [a for a in audit if a.get("title_signal_c")]
    if sigc_missed:
        lines.append(f"- **Signal C false-negative ({len(sigc_missed)} AOPs):** these AOPs are listed as unclassified, but their title *would* match the Signal C keyword set. They appear unclassified because they have no Key-Event rows in `_query_aop_signal_rows` (which requires `aopo:has_key_event`), and Signal C is seeded inside that loop. Look for the `Title would match Signal C:` hint in the categorised lists above. Fixing this is a small classifier-side change: seed Signal C from `_query_aop_universe` (which always has the title) instead of from the KE-driven query.")
        lines.append("")
    lines.append("- The `genuinely_unannotated` category is the only one that is **not** a tooling false-negative — those AOPs really do lack any organ/cell/object/process triples on every KE, *and* their title doesn't match the Signal C keyword set.")
    lines.append("- For `fma_missing_from_cache`, the fix is regenerating the cache against a newer FMA snapshot or adding a manual FMA→bucket override layer.")
    lines.append("- For `uberon_generic_or_non_vertebrate`, candidates for a dedicated 'Other-taxa / non-vertebrate' bucket include swim bladder, eggshell, cuticle, generic 'organ'/'tissue' terms.")
    lines.append("- For `phenotype_no_anatomy_axiom`, the policy is intentional: behavioural HP/MP terms have no anatomy axiom and we don't fall back to keyword bridges at the term level.")
    lines.append("")
    lines.append("Full per-AOP breakdown: `" + json_path.name + "`.")

    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
