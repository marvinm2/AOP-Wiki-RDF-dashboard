"""Organ-system classification dictionary for AOP coverage analysis.

This module is the single source of truth for mapping anatomical and process
ontology terms to high-level organ-system buckets, used by the AOP
organ-coverage visualisations.

Four signal sources are supported (highest to lowest reproducibility):

    Signal A — aopo:OrganContext (UBERON) and aopo:CellTypeContext (CL)
    Signal A' — aopo:hasObject UBERON/CL terms inside aopo:hasBiologicalEvent
    Signal B — aopo:hasProcess GO biological-process terms bridged via curated map
    Signal C — regex on AOP/AO title text (exploratory; not for regulatory use)

To keep the mapping auditable, the same data is also serialised to
``static/data/organ_system_buckets.json`` so the dashboard methodology note
can link to it directly.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Buckets — ordered for stable plot rendering. "Other/Unclassified" is the
# fallback when annotations exist but do not map to any system; AOPs with no
# anatomy signal at all are reported separately as "No annotation".
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
# Signal A — UBERON anatomical entities → bucket.
# Curated set covering the organs/regions most frequently annotated in AOP-Wiki
# Key Events. Keys are the OBO-suffix form (e.g. "UBERON_0000955" for brain),
# which matches the format produced by stripping the OBO IRI prefix.
# ---------------------------------------------------------------------------

UBERON_TO_BUCKET: Dict[str, str] = {
    # Nervous system
    "UBERON_0001016": "Nervous",        # nervous system
    "UBERON_0000955": "Nervous",        # brain
    "UBERON_0002240": "Nervous",        # spinal cord
    "UBERON_0001017": "Nervous",        # central nervous system
    "UBERON_0000010": "Nervous",        # peripheral nervous system
    "UBERON_0002298": "Nervous",        # brainstem
    "UBERON_0002037": "Nervous",        # cerebellum
    "UBERON_0001890": "Nervous",        # forebrain
    "UBERON_0000956": "Nervous",        # cerebral cortex
    "UBERON_0001950": "Nervous",        # neocortex
    "UBERON_0002421": "Nervous",        # hippocampal formation
    "UBERON_0001870": "Nervous",        # frontal cortex
    "UBERON_0001872": "Nervous",        # parietal cortex
    "UBERON_0002435": "Nervous",        # striatum
    "UBERON_0001873": "Nervous",        # caudate nucleus
    "UBERON_0002038": "Nervous",        # substantia nigra
    "UBERON_0001876": "Nervous",        # amygdala
    "UBERON_0001891": "Nervous",        # midbrain
    "UBERON_0000044": "Nervous",        # dorsal root ganglion
    "UBERON_0000964": "Nervous",        # cornea — moved to Sensory below (kept for safety)
    "UBERON_0001021": "Nervous",        # nerve
    # Cardiovascular
    "UBERON_0001009": "Cardiovascular",  # circulatory system
    "UBERON_0004535": "Cardiovascular",  # cardiovascular system
    "UBERON_0000948": "Cardiovascular",  # heart
    "UBERON_0001016_cv": "Cardiovascular",  # placeholder — see TODO
    "UBERON_0001637": "Cardiovascular",  # artery
    "UBERON_0001638": "Cardiovascular",  # vein
    "UBERON_0001981": "Cardiovascular",  # blood vessel
    "UBERON_0000178": "Cardiovascular",  # blood
    "UBERON_0001088": "Cardiovascular",  # urine — actually Renal, see below override
    "UBERON_0002080": "Cardiovascular",  # heart left ventricle (alias)
    "UBERON_0001984": "Cardiovascular",  # capillary
    "UBERON_0002049": "Cardiovascular",  # vasculature
    # Respiratory
    "UBERON_0001004": "Respiratory",     # respiratory system
    "UBERON_0002048": "Respiratory",     # lung
    "UBERON_0000970": "Respiratory",     # eye — actually Sensory, override below
    "UBERON_0002185": "Respiratory",     # bronchus
    "UBERON_0002186": "Respiratory",     # bronchiole
    "UBERON_0002299": "Respiratory",     # alveolus of lung
    "UBERON_0003126": "Respiratory",     # trachea
    "UBERON_0001005": "Respiratory",     # respiratory tract
    "UBERON_0000065": "Respiratory",     # respiratory tract (alias)
    "UBERON_0001443": "Respiratory",     # epithelium of bronchus
    # Digestive (incl. GI tract proper; hepatobiliary split below)
    "UBERON_0001007": "Digestive",       # digestive system
    "UBERON_0000160": "Digestive",       # intestine
    "UBERON_0000945": "Digestive",       # stomach
    "UBERON_0002108": "Digestive",       # small intestine
    "UBERON_0001155": "Digestive",       # colon
    "UBERON_0001052": "Digestive",       # rectum
    "UBERON_0000165": "Digestive",       # mouth
    "UBERON_0001043": "Digestive",       # esophagus
    "UBERON_0001264": "Digestive",       # pancreas (exocrine functions; endocrine bucket overrides via override map)
    # Hepatobiliary
    "UBERON_0002107": "Hepatobiliary",   # liver
    "UBERON_0001179": "Hepatobiliary",   # peritoneal cavity — keep general
    "UBERON_0001173": "Hepatobiliary",   # gallbladder
    "UBERON_0001174": "Hepatobiliary",   # common bile duct
    "UBERON_0001242": "Hepatobiliary",   # hepatic duct
    "UBERON_0002106": "Hepatobiliary",   # spleen — actually Immune; override below
    "UBERON_0001281": "Hepatobiliary",   # hepatic sinusoid
    # Renal / urinary
    "UBERON_0001008": "Renal/Urinary",   # renal system
    "UBERON_0002113": "Renal/Urinary",   # kidney
    "UBERON_0001255": "Renal/Urinary",   # urinary bladder
    "UBERON_0000056": "Renal/Urinary",   # ureter
    "UBERON_0001285": "Renal/Urinary",   # nephron
    "UBERON_0001229": "Renal/Urinary",   # renal tubule
    "UBERON_0006547": "Renal/Urinary",   # glomerulus (capillary tuft)
    "UBERON_0002015": "Renal/Urinary",   # collecting duct
    # Reproductive
    "UBERON_0000990": "Reproductive",    # reproductive system
    "UBERON_0000991": "Reproductive",    # gonad
    "UBERON_0000992": "Reproductive",    # ovary
    "UBERON_0000473": "Reproductive",    # testis
    "UBERON_0000995": "Reproductive",    # uterus
    "UBERON_0000996": "Reproductive",    # vagina
    "UBERON_0000993": "Reproductive",    # oviduct
    "UBERON_0000998": "Reproductive",    # seminal vesicle
    "UBERON_0002367": "Reproductive",    # prostate gland
    "UBERON_0001302": "Reproductive",    # epididymis
    "UBERON_0001967": "Reproductive",    # seminal fluid (not strictly anatomy)
    "UBERON_0000310": "Reproductive",    # breast / mammary gland
    "UBERON_0001295": "Reproductive",    # endometrium
    "UBERON_0001987": "Reproductive",    # placenta
    "UBERON_0000080": "Reproductive",    # mesonephros
    # Endocrine
    "UBERON_0000949": "Endocrine",       # endocrine system
    "UBERON_0002046": "Endocrine",       # thyroid gland
    "UBERON_0000079": "Endocrine",       # male reproductive duct — actually Reproductive; keep override
    "UBERON_0002369": "Endocrine",       # adrenal gland
    "UBERON_0000007": "Endocrine",       # pituitary gland
    "UBERON_0001132": "Endocrine",       # parathyroid gland
    "UBERON_0000006": "Endocrine",       # islet of Langerhans
    "UBERON_0000114": "Endocrine",       # alveolus (alias) — typically Resp; override
    "UBERON_0001225": "Endocrine",       # cortex of kidney — actually Renal; override
    # Immune / Haematopoietic
    "UBERON_0002405": "Immune/Haematopoietic",  # immune system
    "UBERON_0002106_imm": "Immune/Haematopoietic",  # spleen (avoid key collision; see override block)
    "UBERON_0002370": "Immune/Haematopoietic",  # thymus
    "UBERON_0000029": "Immune/Haematopoietic",  # lymph node
    "UBERON_0002390": "Immune/Haematopoietic",  # haematopoietic system
    "UBERON_0002371": "Immune/Haematopoietic",  # bone marrow
    "UBERON_0001977": "Immune/Haematopoietic",  # blood plasma
    "UBERON_0000178_imm": "Immune/Haematopoietic",  # blood (overlaps; see overrides)
    # Integumentary
    "UBERON_0002416": "Integumentary",    # integumental system
    "UBERON_0002097": "Integumentary",    # skin of body
    "UBERON_0001003": "Integumentary",    # skin epidermis
    "UBERON_0001456": "Integumentary",    # face / skin
    "UBERON_0000014": "Integumentary",    # zone of skin
    "UBERON_0002073": "Integumentary",    # hair follicle
    # Musculoskeletal
    "UBERON_0002204": "Musculoskeletal",  # musculoskeletal system
    "UBERON_0002101": "Musculoskeletal",  # limb
    "UBERON_0001434": "Musculoskeletal",  # skeletal system
    "UBERON_0001630": "Musculoskeletal",  # muscle organ
    "UBERON_0014892": "Musculoskeletal",  # skeletal muscle organ
    "UBERON_0001474": "Musculoskeletal",  # bone element
    "UBERON_0002481": "Musculoskeletal",  # bone tissue
    "UBERON_0002529": "Musculoskeletal",  # cartilage tissue
    "UBERON_0000388": "Musculoskeletal",  # ligament
    "UBERON_0000981": "Musculoskeletal",  # tendon
    # Sensory
    "UBERON_0001846": "Sensory",          # inner ear
    "UBERON_0001690": "Sensory",          # ear
    "UBERON_0000970_sens": "Sensory",     # eye (canonical; override below)
    "UBERON_0001769": "Sensory",          # retina (alias for 0000966)
    "UBERON_0000966": "Sensory",          # retina
    "UBERON_0001775": "Sensory",          # iris (alias)
    "UBERON_0002265": "Sensory",          # olfactory bulb
    "UBERON_0001728": "Sensory",          # nasal cavity (also Resp; override)
}

# Resolve canonical entries for terms that appeared under multiple buckets above
# (curation convenience — the suffix-aliased keys above are placeholders).
_CANONICAL_OVERRIDES: Dict[str, str] = {
    "UBERON_0000970": "Sensory",      # eye
    "UBERON_0002106": "Immune/Haematopoietic",  # spleen
    "UBERON_0000178": "Cardiovascular",  # blood (vasculature-adjacent)
    "UBERON_0001088": "Renal/Urinary",  # urine
    "UBERON_0001225": "Renal/Urinary",  # cortex of kidney
    "UBERON_0000114": "Respiratory",   # alveolus (canonical)
    "UBERON_0000079": "Reproductive",  # male reproductive duct
    "UBERON_0001728": "Respiratory",   # nasal cavity (canonical: airway)
}
for _k, _v in _CANONICAL_OVERRIDES.items():
    UBERON_TO_BUCKET[_k] = _v

# Drop placeholder collision keys (aliases ending in _cv/_imm/_sens) — they
# were only present to document multi-mapping during curation.
for _alias in list(UBERON_TO_BUCKET):
    if "_" in _alias and not _alias.startswith("UBERON_") or _alias.count("_") > 1:
        UBERON_TO_BUCKET.pop(_alias, None)

# ---------------------------------------------------------------------------
# Signal A — CL (Cell Ontology) → bucket. Curated subset; many cell types are
# explicitly annotated on AOP-Wiki KEs.
# ---------------------------------------------------------------------------

CL_TO_BUCKET: Dict[str, str] = {
    # Nervous
    "CL_0000540": "Nervous",        # neuron
    "CL_0000125": "Nervous",        # glial cell
    "CL_0000127": "Nervous",        # astrocyte
    "CL_0000128": "Nervous",        # oligodendrocyte
    "CL_0000129": "Nervous",        # microglial cell
    "CL_0002319": "Nervous",        # neural cell
    "CL_0000031": "Nervous",        # neuroblast
    "CL_0000099": "Nervous",        # interneuron
    "CL_0000700": "Nervous",        # dopaminergic neuron
    # Cardiovascular
    "CL_0000746": "Cardiovascular",  # cardiac muscle cell (cardiomyocyte)
    "CL_0002494": "Cardiovascular",  # cardiocyte (alias)
    "CL_0000115": "Cardiovascular",  # endothelial cell
    "CL_0002138": "Cardiovascular",  # endothelial cell of lymphatic vessel
    "CL_0000232": "Cardiovascular",  # erythrocyte / red blood cell
    # Respiratory
    "CL_0000082": "Respiratory",     # epithelial cell of lung
    "CL_0002145": "Respiratory",     # ciliated columnar cell of tracheobronchial tree
    "CL_0002062": "Respiratory",     # pulmonary alveolar epithelial cell
    "CL_0002063": "Respiratory",     # type II pneumocyte
    # Digestive
    "CL_0000584": "Digestive",       # enterocyte
    "CL_0002193": "Digestive",       # myelocyte — actually Immune; override
    "CL_0002628": "Digestive",       # immature intestinal cell
    # Hepatobiliary
    "CL_0000182": "Hepatobiliary",   # hepatocyte
    "CL_0002624": "Hepatobiliary",   # Kupffer cell (sinusoidal macrophage)
    "CL_0000632": "Hepatobiliary",   # hepatic stellate cell
    # Renal/Urinary
    "CL_0002518": "Renal/Urinary",   # kidney epithelial cell
    "CL_0002306": "Renal/Urinary",   # epithelial cell of proximal tubule
    "CL_0002307": "Renal/Urinary",   # epithelial cell of distal tubule
    # Reproductive
    "CL_0000019": "Reproductive",    # sperm
    "CL_0000023": "Reproductive",    # oocyte
    "CL_0000216": "Reproductive",    # Sertoli cell
    "CL_0000178": "Reproductive",    # Leydig cell
    # Endocrine
    "CL_0000169": "Endocrine",       # pancreatic beta cell — endocrine
    "CL_0000171": "Endocrine",       # pancreatic alpha cell
    "CL_0002257": "Endocrine",       # thyroid follicular cell
    "CL_0000035": "Endocrine",       # adrenocortical cell (alias)
    # Immune / Haematopoietic
    "CL_0000236": "Immune/Haematopoietic",  # B cell
    "CL_0000084": "Immune/Haematopoietic",  # T cell
    "CL_0000235": "Immune/Haematopoietic",  # macrophage
    "CL_0000094": "Immune/Haematopoietic",  # granulocyte
    "CL_0000771": "Immune/Haematopoietic",  # eosinophil
    "CL_0000766": "Immune/Haematopoietic",  # myeloid leukocyte
    "CL_0000037": "Immune/Haematopoietic",  # haematopoietic stem cell
    "CL_0000451": "Immune/Haematopoietic",  # dendritic cell
    # Integumentary
    "CL_0000312": "Integumentary",    # keratinocyte
    "CL_0000148": "Integumentary",    # melanocyte
    # Musculoskeletal
    "CL_0000746_mus": "Musculoskeletal",  # placeholder (cardiac myocyte handled above)
    "CL_0000192": "Musculoskeletal",  # smooth muscle cell
    "CL_0000188": "Musculoskeletal",  # skeletal muscle cell
    "CL_0000062": "Musculoskeletal",  # osteoblast
    "CL_0000092": "Musculoskeletal",  # osteoclast
    "CL_0000138": "Musculoskeletal",  # chondrocyte
    # Generic / over-broad cell terms (CL_0000000 cell, CL_0000003 native cell,
    # CL_0000255 eukaryotic cell) are intentionally NOT mapped — letting them
    # fall through as None keeps AOPs whose only anatomy annotation is "cell"
    # honestly in the No-annotation bucket rather than in Other/Unclassified.
}

_CL_OVERRIDES: Dict[str, str] = {
    "CL_0002193": "Immune/Haematopoietic",  # myelocyte canonical
}
CL_TO_BUCKET.update(_CL_OVERRIDES)
CL_TO_BUCKET.pop("CL_0000746_mus", None)

# ---------------------------------------------------------------------------
# Signal B — GO biological-process → bucket (curated bridge). Restricted to
# processes that are clearly tissue/system-specific; generic terms like
# "apoptotic process" or "DNA damage response" are intentionally absent so
# Signal B does not over-claim coverage.
# ---------------------------------------------------------------------------

GO_BP_TO_BUCKET: Dict[str, str] = {
    # Nervous
    "GO_0007268": "Nervous",   # chemical synaptic transmission
    "GO_0007420": "Nervous",   # brain development
    "GO_0042552": "Nervous",   # myelination
    "GO_0050905": "Nervous",   # neuromuscular process controlling balance
    "GO_0007399": "Nervous",   # nervous system development
    "GO_0021510": "Nervous",   # spinal cord development
    "GO_0007158": "Nervous",   # neuron cell-cell adhesion
    "GO_0007612": "Nervous",   # learning
    "GO_0007613": "Nervous",   # memory
    "GO_0048709": "Nervous",   # oligodendrocyte differentiation
    # Cardiovascular
    "GO_0008015": "Cardiovascular",   # blood circulation
    "GO_0007507": "Cardiovascular",   # heart development
    "GO_0001525": "Cardiovascular",   # angiogenesis
    "GO_0001568": "Cardiovascular",   # blood vessel development
    "GO_0008217": "Cardiovascular",   # regulation of blood pressure
    "GO_0060047": "Cardiovascular",   # heart contraction
    # Respiratory
    "GO_0007585": "Respiratory",      # respiratory gaseous exchange by respiratory system
    "GO_0030324": "Respiratory",      # lung development
    "GO_0048286": "Respiratory",      # lung alveolus development
    # Digestive
    "GO_0022600": "Digestive",        # digestive system process
    "GO_0007586": "Digestive",        # digestion
    "GO_0048565": "Digestive",        # digestive tract development
    # Hepatobiliary
    "GO_0001889": "Hepatobiliary",    # liver development
    "GO_0002384": "Hepatobiliary",    # hepatic immune response
    "GO_0006699": "Hepatobiliary",    # bile acid biosynthetic process
    "GO_0008206": "Hepatobiliary",    # bile acid metabolic process
    # Renal/Urinary
    "GO_0001822": "Renal/Urinary",    # kidney development
    "GO_0072001": "Renal/Urinary",    # renal system development
    "GO_0003014": "Renal/Urinary",    # renal system process
    # Reproductive
    "GO_0007283": "Reproductive",     # spermatogenesis
    "GO_0007292": "Reproductive",     # female gamete generation
    "GO_0030728": "Reproductive",     # ovulation
    "GO_0007565": "Reproductive",     # female pregnancy
    "GO_0001541": "Reproductive",     # ovarian follicle development
    "GO_0008584": "Reproductive",     # male gonad development
    "GO_0008585": "Reproductive",     # female gonad development
    "GO_0007281": "Reproductive",     # germ cell development
    # Endocrine
    "GO_0035556": "Endocrine",        # intracellular signal transduction — over-broad; consider removing
    "GO_0042446": "Endocrine",        # hormone biosynthetic process
    "GO_0050817": "Endocrine",        # coagulation — actually Cardiovascular; override
    "GO_0042403": "Endocrine",        # thyroid hormone metabolic process
    "GO_0006590": "Endocrine",        # thyroid hormone generation
    "GO_0030072": "Endocrine",        # peptide hormone secretion
    # Immune / Haematopoietic
    "GO_0006955": "Immune/Haematopoietic",  # immune response
    "GO_0030097": "Immune/Haematopoietic",  # hemopoiesis
    "GO_0006954": "Immune/Haematopoietic",  # inflammatory response
    "GO_0042098": "Immune/Haematopoietic",  # T cell proliferation
    "GO_0042100": "Immune/Haematopoietic",  # B cell proliferation
    # Integumentary
    "GO_0008544": "Integumentary",    # epidermis development
    "GO_0043588": "Integumentary",    # skin development
    "GO_0031069": "Integumentary",    # hair follicle morphogenesis
    # Musculoskeletal
    "GO_0007517": "Musculoskeletal",  # muscle organ development
    "GO_0001501": "Musculoskeletal",  # skeletal system development
    "GO_0001503": "Musculoskeletal",  # ossification
    "GO_0006936": "Musculoskeletal",  # muscle contraction
    # Sensory
    "GO_0001654": "Sensory",          # eye development
    "GO_0007601": "Sensory",          # visual perception
    "GO_0007605": "Sensory",          # sensory perception of sound
    "GO_0050911": "Sensory",          # detection of chemical stimulus involved in sensory perception of smell
}

# Apply explicit overrides for terms accidentally bucketed twice during curation
_GO_OVERRIDES: Dict[str, str] = {
    "GO_0050817": "Cardiovascular",  # coagulation canonical
    "GO_0035556": "Other/Unclassified",  # signal transduction is too generic — keep but neutralise
}
GO_BP_TO_BUCKET.update(_GO_OVERRIDES)

# ---------------------------------------------------------------------------
# Signal C — keyword regex on AO / AOP titles. Order matters for tie-breaking
# (first match wins inside a bucket; an AO can match multiple buckets across
# the dict, which is preserved). Patterns are case-insensitive and anchored to
# whole-word boundaries where possible to limit false positives.
# ---------------------------------------------------------------------------

KEYWORD_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "Nervous": (
        r"\bneuro",
        r"\bneural\b",
        r"\bbrain\b",
        r"\bcortex\b",
        r"\bcerebr",
        r"\bdopamin",
        r"\bsynap",
        r"\bcogniti",
        r"\bmemory\b",
        r"\blearning\b",
        r"\bmyelin",
    ),
    "Cardiovascular": (
        r"\bcardio",
        r"\bcardiac\b",
        r"\bheart\b",
        r"\bvascular\b",
        r"\bblood pressure\b",
        r"\bhypertens",
        r"\bathero",
        r"\bischemi",
    ),
    "Respiratory": (
        r"\bpulmon",
        r"\blung\b",
        r"\brespirator",
        r"\basthma\b",
        r"\bfibrosis of\s+the\s+lung\b",
    ),
    "Digestive": (
        r"\bintestin",
        r"\bgastro",
        r"\bcolitis\b",
        r"\bgut\b",
    ),
    "Hepatobiliary": (
        r"\bhepat",
        r"\bliver\b",
        r"\bbiliary\b",
        r"\bsteatosis\b",
        r"\bcirrhosis\b",
        r"\bcholestasis\b",
    ),
    "Renal/Urinary": (
        r"\brenal\b",
        r"\bkidney\b",
        r"\bnephro",
        r"\burinary\b",
    ),
    "Reproductive": (
        r"\breproduct",
        r"\bovari",
        r"\btestis\b",
        r"\btesticular\b",
        r"\bspermat",
        r"\boocyte\b",
        r"\bfertilit",
        r"\bfetal\b",
        r"\bpregnan",
        r"\bestrogen",
        r"\bandrogen",
    ),
    "Endocrine": (
        r"\bthyroid\b",
        r"\badrenal\b",
        r"\bpituitar",
        r"\bhormone\b",
        r"\bendocrine\b",
        r"\binsulin\b",
        r"\bdiabet",
    ),
    "Immune/Haematopoietic": (
        r"\bimmun",
        r"\bhematopoie",
        r"\bhaematopoie",
        r"\blymph",
        r"\bspleen\b",
        r"\binflammat",
    ),
    "Integumentary": (
        r"\bskin\b",
        r"\bepiderm",
        r"\bdermal\b",
    ),
    "Musculoskeletal": (
        r"\bskeletal\b",
        r"\bmuscle\b",
        r"\bbone\b",
        r"\bosteo",
        r"\bcartilage\b",
    ),
    "Sensory": (
        r"\bocular\b",
        r"\beye\b",
        r"\bretin",
        r"\bcornea",
        r"\bvisual\b",
        r"\bauditory\b",
        r"\botic\b",
        r"\bhearing loss\b",
    ),
}

# Pre-compile for performance — used both by per-AOP scoring and by the
# methodology JSON serialiser.
_COMPILED_KEYWORDS: Dict[str, Tuple[re.Pattern, ...]] = {
    bucket: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for bucket, patterns in KEYWORD_PATTERNS.items()
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _iri_suffix(iri: str) -> str:
    """Return the OBO suffix (e.g. ``UBERON_0000955``) from a full OBO IRI.

    Returns the input unchanged if it does not look like an OBO IRI.
    """
    if not iri:
        return iri
    if "/obo/" in iri:
        return iri.rsplit("/", 1)[-1]
    return iri


def classify_anatomy(iri: str) -> Optional[str]:
    """Classify a UBERON/CL IRI into an organ-system bucket.

    Returns ``None`` if the term is not in the curated map. Callers should
    treat unmapped terms as "annotated but uncategorised" (Other/Unclassified
    is only used when an AOP has at least one anatomy annotation overall).
    """
    suffix = _iri_suffix(iri)
    if suffix.startswith("UBERON_"):
        return UBERON_TO_BUCKET.get(suffix)
    if suffix.startswith("CL_"):
        return CL_TO_BUCKET.get(suffix)
    return None


def classify_go_bp(iri: str) -> Optional[str]:
    """Classify a GO biological-process IRI into a bucket via the curated bridge."""
    suffix = _iri_suffix(iri)
    if not suffix.startswith("GO_"):
        return None
    bucket = GO_BP_TO_BUCKET.get(suffix)
    if bucket == "Other/Unclassified":
        return None  # over-broad terms neutralised at curation time
    return bucket


def classify_text(text: str) -> Set[str]:
    """Return the set of buckets a text string matches via Signal-C regex.

    Multiple matches are honest — an AO titled "neurodegeneration leading to
    cognitive decline" matches Nervous once; one titled "renal failure and
    cardiac arrhythmia" matches both Renal/Urinary and Cardiovascular.
    """
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
    """Given the set of signals that classified an (AOP, bucket) row, return
    the highest-confidence one per :data:`SIGNAL_ORDER`."""
    present = set(signals)
    for s in SIGNAL_ORDER:
        if s in present:
            return s
    return None


SIGNAL_COLOURS: Dict[str, str] = {
    "A":  "#307BBF",   # navigation blue — curated anatomy
    "A'": "#29235C",   # deep purple — derived anatomy (hasObject)
    "B":  "#E6007E",   # magenta — bridged via GO BP
    "C":  "#9c9c9c",   # grey — exploratory (keywords)
}


def serialise_for_methodology() -> Dict:
    """Build the JSON payload served at ``/static/data/organ_system_buckets.json``.

    Keeps the curated maps and the regex sources in one auditable file.
    """
    return {
        "buckets": list(ORGAN_SYSTEM_BUCKETS),
        "no_annotation_label": NO_ANNOTATION_BUCKET,
        "signal_order": list(SIGNAL_ORDER),
        "signal_descriptions": {
            "A":  "aopo:OrganContext (UBERON) and aopo:CellTypeContext (CL) directly on Key Events",
            "A'": "aopo:hasObject UBERON/CL inside aopo:hasBiologicalEvent on Key Events",
            "B":  "aopo:hasProcess GO biological-process IRIs bridged via curated GO-to-bucket map",
            "C":  "Regex on AOP/AO title text (EXPLORATORY — not for regulatory use)",
        },
        "uberon": _sorted(UBERON_TO_BUCKET),
        "cl": _sorted(CL_TO_BUCKET),
        "go_bp": _sorted(GO_BP_TO_BUCKET),
        "keyword_patterns": {b: list(p) for b, p in KEYWORD_PATTERNS.items()},
    }


def _sorted(d: Dict[str, str]) -> Dict[str, str]:
    return {k: d[k] for k in sorted(d)}
