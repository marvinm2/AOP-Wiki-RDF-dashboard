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
_HP_BUCKETS: Dict[str, Set[str]] = {
    k: set(v) for k, v in _CACHE.get("hp", {}).items()
}
_MP_BUCKETS: Dict[str, Set[str]] = {
    k: set(v) for k, v in _CACHE.get("mp", {}).items()
}
_FMA_BUCKETS: Dict[str, Set[str]] = {
    k: set(v) for k, v in _CACHE.get("fma", {}).items()
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
        r"\bfibrosis of\s+the\s+lung\b", r"\bnasal\b", r"\bnose\b",
        r"\bbronch", r"\btrache", r"\balveol",
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
        r"\bspermat", r"\bsperm\b", r"\boocyte\b", r"\bovum\b",
        r"\bfertilit", r"\bfetal\b", r"\bpregnan",
        r"\bestrogen", r"\bandrogen", r"\bendometri", r"\buter",
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
        r"\blens\b", r"\bcochlea", r"\bolfact",
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
    """Return the set of buckets a UBERON / CL / FMA IRI maps to.

    Empty set means "not classified". UBERON / CL come from the standard
    ``(part_of | subClassOf)*`` closure to anchors; FMA is resolved by
    finding a UBERON term that cross-references the FMA ID via
    ``oboInOwl:hasDbXref`` and then taking that UBERON term's bucket(s).
    """
    suffix = _iri_suffix(iri)
    if suffix.startswith("UBERON_"):
        return _UBERON_BUCKETS.get(suffix, set())
    if suffix.startswith("CL_"):
        return _CL_BUCKETS.get(suffix, set())
    if suffix.startswith("FMA_"):
        return _FMA_BUCKETS.get(suffix, set())
    return set()


def classify_process(iri: str) -> Set[str]:
    """Return the bucket set a process/phenotype IRI maps to.

    Covers three ontology relations, all anchored in UBERON anatomy:

    - GO biological-process via ``RO:0002296`` (results_in_development_of)
    - HP (Human Phenotype) via ``UPHENO:0000001`` (has phenotype affecting)
    - MP (Mammalian Phenotype) via ``UPHENO:0000001``

    Empty set means the term has no formal anatomy axiom — by policy we do not
    fall back to hand-curated bridges.
    """
    suffix = _iri_suffix(iri)
    if suffix.startswith("GO_"):
        return _GO_BUCKETS.get(suffix, set())
    if suffix.startswith("HP_"):
        return _HP_BUCKETS.get(suffix, set())
    if suffix.startswith("MP_"):
        return _MP_BUCKETS.get(suffix, set())
    return set()


# Backwards-compatible alias — older code paths import classify_go_bp.
def classify_go_bp(iri: str) -> Set[str]:
    return classify_process(iri)


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
                  "A'": "aopo:hasObject UBERON / CL / FMA inside aopo:hasBiologicalEvent. "
                  "UBERON/CL resolve via the same anchor closure as Signal A; FMA terms "
                  "resolve via UBERON's oboInOwl:hasDbXref reverse-lookup (UBERON_X "
                  "hasDbXref \"FMA:nnnn\") and then the same anchor closure.",
            "B":  "Process / phenotype signal. aopo:hasProcess (and hasObject) "
                  "terms from GO BP (via RO:0002296 results_in_development_of), "
                  "HP (via UPHENO:0000001 has phenotype affecting), and MP "
                  "(via UPHENO:0000001). HP/MP carry the bulk of apical-endpoint "
                  "anatomy because abnormal phenotypes are explicitly anchored "
                  "in UBERON. Terms without a formal anatomy axiom are NOT "
                  "classified.",
            "C":  "Regex on AOP/AO title text (EXPLORATORY — not for "
                  "regulatory use). Patterns listed below.",
        },
        "cache_metadata": cache_metadata(),
        "uberon": {k: sorted(v) for k, v in _UBERON_BUCKETS.items()},
        "cl":     {k: sorted(v) for k, v in _CL_BUCKETS.items()},
        "go_bp":  {k: sorted(v) for k, v in _GO_BUCKETS.items()},
        "hp":     {k: sorted(v) for k, v in _HP_BUCKETS.items()},
        "mp":     {k: sorted(v) for k, v in _MP_BUCKETS.items()},
        "fma":    {k: sorted(v) for k, v in _FMA_BUCKETS.items()},
        "keyword_patterns": {b: list(p) for b, p in KEYWORD_PATTERNS.items()},
    }
