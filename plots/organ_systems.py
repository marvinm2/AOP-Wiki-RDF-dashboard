"""Organ-system classification for the AOP coverage plots.

Ontology-driven: the UBERON / CL / GO → bucket mapping is built offline by
``scripts/build_organ_system_cache.py`` (which queries Ubergraph) and shipped
as ``static/data/organ_system_cache.json``. This module loads that cache at
import time and exposes the classification functions used by the plots.

Four signal sources are supported (highest to lowest reproducibility):

    Signal A  — aopo:OrganContext (UBERON) and aopo:CellTypeContext (CL)
                on member Key Events
    Signal A' — aopo:hasObject UBERON / CL inside aopo:hasBiologicalEvent
    Signal B  — aopo:hasProcess GO biological-process IRIs that carry an
                ``RO:0002296 results_in_development_of`` axiom to a UBERON
                anchor (only). No hand-curated GO bridge.
    Signal C  — regex on AOP/AO title text (exploratory; not for regulatory
                use)

A small editorial override layer baked into the cache covers the handful of
terms where the project's preferred classification diverges from UBERON
(e.g. blood is reported under BOTH Cardiovascular and Immune/Haematopoietic).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

ORGAN_SYSTEM_BUCKETS: Tuple[str, ...] = (
    "Nervous",
    "Cardiovascular",
    "Respiratory",
    "Digestive",
    "Hepatobiliary",
    "Renal/Urinary",
    "Reproductive",
    "Endocrine",
    "Immune/Haematopoietic",
    "Integumentary",
    "Musculoskeletal",
    "Sensory",
    "Other/Unclassified",
)

NO_ANNOTATION_BUCKET = "No annotation"

# ---------------------------------------------------------------------------
# Cache loader. The cache JSON ships in static/data/. If missing we fall back
# to empty maps + a logged warning — the dashboard then under-classifies
# visibly rather than silently failing.
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _THIS_DIR.parent / "static" / "data" / "organ_system_cache.json"


def _load_cache() -> Dict:
    if not _CACHE_PATH.exists():
        logger.warning(
            "organ_system_cache.json not found at %s — coverage plots will be "
            "empty until scripts/build_organ_system_cache.py is run.",
            _CACHE_PATH,
        )
        return {
            "uberon": {}, "cl": {}, "go": {},
            "anchors": {}, "overrides": {},
            "generated_at": None, "source": {}, "stats": {},
        }
    return json.loads(_CACHE_PATH.read_text())


_CACHE = _load_cache()

# ``UBERON_0000955`` → ``{"Nervous"}``-style lookups. Stored as sets internally
# for cheap membership checks; the JSON ships lists.
_UBERON_BUCKETS: Dict[str, Set[str]] = {
    k: set(v) for k, v in _CACHE.get("uberon", {}).items()
}
_CL_BUCKETS: Dict[str, Set[str]] = {
    k: set(v) for k, v in _CACHE.get("cl", {}).items()
}
_GO_BUCKETS: Dict[str, Set[str]] = {
    k: set(v) for k, v in _CACHE.get("go", {}).items()
}


def cache_metadata() -> Dict:
    """Expose cache provenance for the methodology API."""
    return {
        "generated_at": _CACHE.get("generated_at"),
        "source": _CACHE.get("source", {}),
        "stats": _CACHE.get("stats", {}),
        "anchors": _CACHE.get("anchors", {}),
        "overrides": _CACHE.get("overrides", {}),
    }


# ---------------------------------------------------------------------------
# Signal C — keyword regex (Python-only; no ontology involved). The patterns
# are the lowest-confidence signal and live here rather than in the cache.
# ---------------------------------------------------------------------------

KEYWORD_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "Nervous": (
        r"\bneuro", r"\bneural\b", r"\bbrain\b", r"\bcortex\b", r"\bcerebr",
        r"\bdopamin", r"\bsynap", r"\bcogniti", r"\bmemory\b", r"\blearning\b",
        r"\bmyelin",
    ),
    "Cardiovascular": (
        r"\bcardio", r"\bcardiac\b", r"\bheart\b", r"\bvascular\b",
        r"\bblood pressure\b", r"\bhypertens", r"\bathero", r"\bischemi",
    ),
    "Respiratory": (
        r"\bpulmon", r"\blung\b", r"\brespirator", r"\basthma\b",
        r"\bfibrosis of\s+the\s+lung\b",
    ),
    "Digestive": (
        r"\bintestin", r"\bgastro", r"\bcolitis\b", r"\bgut\b",
    ),
    "Hepatobiliary": (
        r"\bhepat", r"\bliver\b", r"\bbiliary\b", r"\bsteatosis\b",
        r"\bcirrhosis\b", r"\bcholestasis\b",
    ),
    "Renal/Urinary": (
        r"\brenal\b", r"\bkidney\b", r"\bnephro", r"\burinary\b",
    ),
    "Reproductive": (
        r"\breproduct", r"\bovari", r"\btestis\b", r"\btesticular\b",
        r"\bspermat", r"\boocyte\b", r"\bfertilit", r"\bfetal\b",
        r"\bpregnan", r"\bestrogen", r"\bandrogen",
    ),
    "Endocrine": (
        r"\bthyroid\b", r"\badrenal\b", r"\bpituitar", r"\bhormone\b",
        r"\bendocrine\b", r"\binsulin\b", r"\bdiabet",
    ),
    "Immune/Haematopoietic": (
        r"\bimmun", r"\bhematopoie", r"\bhaematopoie", r"\blymph",
        r"\bspleen\b", r"\binflammat",
    ),
    "Integumentary": (
        r"\bskin\b", r"\bepiderm", r"\bdermal\b",
    ),
    "Musculoskeletal": (
        r"\bskeletal\b", r"\bmuscle\b", r"\bbone\b", r"\bosteo",
        r"\bcartilage\b",
    ),
    "Sensory": (
        r"\bocular\b", r"\beye\b", r"\bretin", r"\bcornea", r"\bvisual\b",
        r"\bauditory\b", r"\botic\b", r"\bhearing loss\b",
    ),
}

_COMPILED_KEYWORDS: Dict[str, Tuple[re.Pattern, ...]] = {
    bucket: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for bucket, patterns in KEYWORD_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Public classification helpers
# ---------------------------------------------------------------------------


def _iri_suffix(iri: str) -> str:
    if not iri:
        return iri
    return iri.rsplit("/", 1)[-1] if "/" in iri else iri


def classify_anatomy(iri: str) -> Set[str]:
    """Return the set of buckets a UBERON or CL IRI maps to.

    Empty set means "not classified" — either the term is not in UBERON/CL or
    its closure does not reach any anchor. A single term may map to several
    buckets (e.g. pancreas → Digestive + Endocrine).
    """
    suffix = _iri_suffix(iri)
    if suffix.startswith("UBERON_"):
        return _UBERON_BUCKETS.get(suffix, set())
    if suffix.startswith("CL_"):
        return _CL_BUCKETS.get(suffix, set())
    return set()


def classify_go_bp(iri: str) -> Set[str]:
    """Return the bucket set a GO BP IRI maps to via RO:0002296 to anatomy.

    Empty set means the GO term has no formal anatomy axiom — by policy we
    do NOT fall back to a hand-curated bridge.
    """
    suffix = _iri_suffix(iri)
    if not suffix.startswith("GO_"):
        return set()
    return _GO_BUCKETS.get(suffix, set())


def classify_text(text: str) -> Set[str]:
    """Return the set of buckets a text matches via Signal-C regex."""
    if not text:
        return set()
    matched: Set[str] = set()
    for bucket, patterns in _COMPILED_KEYWORDS.items():
        for pattern in patterns:
            if pattern.search(text):
                matched.add(bucket)
                break
    return matched


SIGNAL_ORDER: Tuple[str, ...] = ("A", "A'", "B", "C")


def best_signal(signals: Iterable[str]) -> Optional[str]:
    present = set(signals)
    for s in SIGNAL_ORDER:
        if s in present:
            return s
    return None


SIGNAL_COLOURS: Dict[str, str] = {
    "A":  "#307BBF",
    "A'": "#29235C",
    "B":  "#E6007E",
    "C":  "#9c9c9c",
}


# ---------------------------------------------------------------------------
# Serialisation for /api/organ-system-buckets — feeds the methodology note.
# ---------------------------------------------------------------------------


def serialise_for_methodology() -> Dict:
    """Build the JSON payload served at ``/api/organ-system-buckets``.

    Exposes everything that drives classification: the buckets, the ontology
    anchors used, the override layer, the resolved term-to-bucket maps from
    the cache, and the Signal-C keyword regexes.
    """
    return {
        "buckets": list(ORGAN_SYSTEM_BUCKETS),
        "no_annotation_label": NO_ANNOTATION_BUCKET,
        "signal_order": list(SIGNAL_ORDER),
        "signal_descriptions": {
            "A":  "aopo:OrganContext (UBERON) and aopo:CellTypeContext (CL) "
                  "directly on Key Events — resolved via Ubergraph "
                  "(part_of | subClassOf)* closure to curated bucket anchors.",
            "A'": "aopo:hasObject UBERON/CL inside aopo:hasBiologicalEvent — "
                  "same Ubergraph resolution as Signal A.",
            "B":  "aopo:hasProcess GO biological-process IRIs that carry an "
                  "RO:0002296 (results_in_development_of) axiom to a UBERON "
                  "term that resolves to a bucket. Terms without such an "
                  "axiom are NOT classified.",
            "C":  "Regex on AOP/AO title text (EXPLORATORY — not for "
                  "regulatory use). Patterns listed below.",
        },
        "cache_metadata": cache_metadata(),
        "uberon": {k: sorted(v) for k, v in _UBERON_BUCKETS.items()},
        "cl":     {k: sorted(v) for k, v in _CL_BUCKETS.items()},
        "go_bp":  {k: sorted(v) for k, v in _GO_BUCKETS.items()},
        "keyword_patterns": {b: list(p) for b, p in KEYWORD_PATTERNS.items()},
    }
