"""Measure per-AOP mono-annotation and unclassified rates against the live SPARQL endpoint.

Loads the current organ_system_cache.json (via the runtime classifier), runs
the production SPARQL queries to build the granular coverage dataframe, then
reports the mono-rate, multi-rate, and unclassified-rate under all three scopes
(``all``, ``apical``, ``ao``).

Usage:
    cd AOP-Wiki-RDF-dashboard/
    python scripts/measure_coverage_metrics.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the dashboard's own functions so we measure exactly what users see.
from plots.latest_plots import (  # noqa: E402
    _get_coverage_dataframe,
    _aggregate_per_aop,
)
from plots.organ_systems import NO_ANNOTATION_BUCKET  # noqa: E402


def report(scope: str) -> dict:
    granular, aop_universe, version_label = _get_coverage_dataframe(version=None)
    per_pair = _aggregate_per_aop(granular, aop_universe, scope)

    # Count per-AOP unique buckets, excluding the No-annotation sentinel.
    real = per_pair[
        (per_pair["Organ System"] != NO_ANNOTATION_BUCKET)
        & per_pair["Best Signal"].notna()
    ]
    counts_per_aop = real.groupby("AOP")["Organ System"].nunique()

    total_aops = len(aop_universe)
    classified = len(counts_per_aop)
    unclassified = total_aops - classified

    mono = int((counts_per_aop == 1).sum())
    two = int((counts_per_aop == 2).sum())
    three = int((counts_per_aop == 3).sum())
    four_plus = int((counts_per_aop >= 4).sum())

    return {
        "version_label": version_label,
        "scope": scope,
        "total_aops": total_aops,
        "classified": classified,
        "unclassified": unclassified,
        "mono": mono,
        "two_buckets": two,
        "three_buckets": three,
        "four_plus": four_plus,
        "mono_rate_of_total": mono / total_aops if total_aops else 0,
        "mono_rate_of_classified": mono / classified if classified else 0,
        "unclassified_rate": unclassified / total_aops if total_aops else 0,
    }


def fmt(stats: dict) -> str:
    return (
        f"\nscope='{stats['scope']}'  (version: {stats['version_label']})\n"
        f"  total AOPs           = {stats['total_aops']}\n"
        f"  classified           = {stats['classified']}  ({stats['classified']/stats['total_aops']*100:.1f}%)\n"
        f"  unclassified         = {stats['unclassified']}  ({stats['unclassified_rate']*100:.1f}%)\n"
        f"  mono-annotated       = {stats['mono']}  "
        f"({stats['mono_rate_of_total']*100:.1f}% of total, "
        f"{stats['mono_rate_of_classified']*100:.1f}% of classified)\n"
        f"  2 buckets            = {stats['two_buckets']}\n"
        f"  3 buckets            = {stats['three_buckets']}\n"
        f"  4+ buckets           = {stats['four_plus']}\n"
    )


def main() -> int:
    print("=" * 60)
    print("Coverage metrics — current cache vs. live SPARQL endpoint")
    print("=" * 60)
    for scope in ("all", "apical", "ao"):
        print(fmt(report(scope)))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
