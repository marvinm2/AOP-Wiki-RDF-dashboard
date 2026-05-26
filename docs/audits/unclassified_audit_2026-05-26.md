# Unclassified-AOP audit — 2026-05-26

- **Snapshot:** `2026-04-01`
- **Scope:** `all` (every KE counts toward Signals A/A'/B; Signal C runs on AOP titles regardless)
- **Total unclassified:** **161** AOPs (`Organ System == "No annotation"`)

Each AOP below has been categorised by the *primary* reason it didn't classify. The categorisation is heuristic — an AOP with FMA-not-in-cache AND title-matches-Signal-C will be reported under FMA, because that's the more actionable tooling fix.

## Category counts

| Count | Category |
|------:|----------|
| 59 | **genuinely_unannotated** — Genuinely unannotated (no IRIs on any KE, title doesn't match Signal C) |
| 50 | **uberon_missing_from_cache** — Has UBERON IRIs absent from the cache — likely UBERON version skew |
| 19 | **cl_unmapped** — CL (cell type) IRIs that don't classify to any organ bucket |
| 12 | **exotic_ontology** — Annotations from ontologies the classifier doesn't currently recognise |
| 8 | **go_no_ro_axiom** — GO biological-process IRIs without an RO:0002296 axiom to a UBERON anchor |
| 7 | **fma_missing_from_cache** — Has FMA IRIs absent from the cache — likely older FMA IDs not in Ubergraph |
| 6 | **phenotype_no_anatomy_axiom** — HP/MP terms with no anatomy axiom — typically behavioural phenotypes by design |

## genuinely_unannotated  (59 AOPs)

_Genuinely unannotated (no IRIs on any KE, title doesn't match Signal C)_

- [`621`](https://identifiers.org/aop/621) — NOX2 Driven Oxidative Stress-Induced α-Synuclein Radical Formation and Tau Nitration Leading to Neurodegeneration
  - Title would match Signal C: Nervous
- [`227`](https://identifiers.org/aop/227) — NSAID induced PTGS1 inactivation to gastric ulcer
- [`236`](https://identifiers.org/aop/236) — Serotonin 1A Receptor Agonism leading to Anti-depressant Activity via Ca Channel Inhibition
- [`5`](https://identifiers.org/aop/5) — pentachlorophenol early events
- [`461`](https://identifiers.org/aop/461) — Deposition of ionising energy leads to population decline via impaired oogenesis and spermatogenesis
  - Title would match Signal C: Reproductive

_…and 54 more — see the JSON sidecar for the complete list._

## uberon_missing_from_cache  (50 AOPs)

_Has UBERON IRIs absent from the cache — likely UBERON version skew_

- [`479`](https://identifiers.org/aop/479) — Mitochondrial complexes inhibition leading to left ventricular function decrease via increased myocardial oxidative stress
  - IRIs: CL_0000255, GO_0005739, GO_0008219, MP_0003674, UBERON_0000062
- [`598`](https://identifiers.org/aop/598) — Excessive reactive oxygen species leading to growth inhibition via protein oxidation and reduced cell proliferation
  - IRIs: CHEBI_15422, CHEBI_26523, CL_0000000, GO_0005623, GO_0005739, GO_0006754, … (+9 more)
- [`360`](https://identifiers.org/aop/360) — Chitin synthase 1 inhibition leading to mortality
  - IRIs: CL_0000658, GO_0004100, GO_0018990, GO_0042335, UBERON_0000483, UBERON_0001002, … (+1 more)
- [`157`](https://identifiers.org/aop/157) — Deiodinase 1 inhibition leading to increased mortality via reduced posterior swim bladder inflation 
  - IRIs: CHEBI_28774, GO_0003824, GO_0048798, MP_0005473, NBO_0000371, PCO_0000001, … (+5 more)
- [`156`](https://identifiers.org/aop/156) — Deiodinase 2 inhibition leading to increased mortality via reduced anterior swim bladder inflation
  - IRIs: CHEBI_28774, GO_0003824, GO_0048798, MP_0005473, NBO_0000371, PCO_0000001, … (+5 more)

_…and 45 more — see the JSON sidecar for the complete list._

## cl_unmapped  (19 AOPs)

_CL (cell type) IRIs that don't classify to any organ bucket_

- [`141`](https://identifiers.org/aop/141) — Alkylation of DNA leading to cancer 2
  - IRIs: CHEBI_16991, CL_0000255, GO_0006281, GO_0006305, D009154, D009369
- [`151`](https://identifiers.org/aop/151) — AhR activation leading to preeclampsia
  - IRIs: CL_0000255, CL_0000566, GO_0004874, GO_0010467, GO_0017162, GO_0046983, … (+5 more)
- [`435`](https://identifiers.org/aop/435) — Deposition of ionising energy leads to population decline via pollen abnormal
  - IRIs: CHEBI_16991, CL_0000255, GO_0005694, GO_0051305, PCO_0000008, RBO_00015021
- [`567`](https://identifiers.org/aop/567) — Binding to plastoquinone B site leading to decreased population growth rate via photosystem II inhibition
  - IRIs: CHEBI_15422, CL_0000000, GO_0006754, PCO_0000001, PCO_0000008
- [`389`](https://identifiers.org/aop/389) —  Oxygen-evolving complex damage leading to population decline via inhibition of photosynthesis
  - IRIs: CHEBI_15422, CL_0000000, GO_0006754, PCO_0000001, PCO_0000008

_…and 14 more — see the JSON sidecar for the complete list._

## exotic_ontology  (12 AOPs)

_Annotations from ontologies the classifier doesn't currently recognise_

- [`184`](https://identifiers.org/aop/184) — Nosema infection causes abnormal role change that leads to colony loss/failure
  - IRIs: D012380, D056631
- [`292`](https://identifiers.org/aop/292) — Inhibition of tyrosinase leads to decreased population in fish
  - IRIs: PCO_0000001, PCO_0000008
- [`339`](https://identifiers.org/aop/339) — DNA methyltransferase inhibition leading to population decline (4)
  - IRIs: PCO_0000001, PCO_0000008
- [`179`](https://identifiers.org/aop/179) — Varroa mite infestation increases virus levels leading to colony loss/failure
  - IRIs: D019562, D056631
- [`341`](https://identifiers.org/aop/341) — DNA methyltransferase inhibition leading to transgenerational effects (2)
  - IRIs: PCO_0000001, PCO_0000008

_…and 7 more — see the JSON sidecar for the complete list._

## go_no_ro_axiom  (8 AOPs)

_GO biological-process IRIs without an RO:0002296 axiom to a UBERON anchor_

- [`358`](https://identifiers.org/aop/358) — Chitinase inhibition leading to mortality
  - IRIs: GO_0018990, D009026
- [`180`](https://identifiers.org/aop/180) — Varroa mite infestation contributes to abnormal foraging leading to colony loss/failure
  - IRIs: GO_0060756, D056631
- [`359`](https://identifiers.org/aop/359) — Chitobiase inhibition leading to mortality
  - IRIs: GO_0018990, D009026
- [`397`](https://identifiers.org/aop/397) — Bulky DNA adducts leading to mutations
  - IRIs: CHEBI_16991, GO_0006281, D009154
- [`466`](https://identifiers.org/aop/466) — Doda decarboxylase leading to mortality
  - IRIs: CHEBI_17029, GO_0006031, D009026

_…and 3 more — see the JSON sidecar for the complete list._

## fma_missing_from_cache  (7 AOPs)

_Has FMA IRIs absent from the cache — likely older FMA IDs not in Ubergraph_

- [`298`](https://identifiers.org/aop/298) — Increase in reactive oxygen species (ROS) leading to human treatment-resistant gastric cancer
  - IRIs: CHEBI_26523, CL_0000000, CL_0000066, FMA_66768, GO_0001837, GO_0061355, … (+6 more)
- [`216`](https://identifiers.org/aop/216) — Deposition of energy leading to population decline via DNA strand breaks and follicular atresia
  - IRIs: Thesaurus.owl#C25830, FMA_74412, PCO_0000001, PCO_0000008, RBO_00015021, VT_1000294
- [`305`](https://identifiers.org/aop/305) — 5α-reductase inhibition leading to short anogenital distance (AGD) in male (mammalian) offspring
  - IRIs: CHEBI_79436, CL_0000255, FMA_264621, GO_0004882, GO_0010468, GO_0030521, … (+3 more)
- [`296`](https://identifiers.org/aop/296) — Oxidative DNA damage leading to chromosomal aberrations and mutations
  - IRIs: Thesaurus.owl#C25830, CHEBI_16991, CHEBI_26523, CL_0000255, FMA_74412, GO_0005694, … (+5 more)
- [`592`](https://identifiers.org/aop/592) — DBDPE-induced DNA strand breaks and LDH activity inhibition leading to population growth rate decline via energy metabolism disrupt and apoptosis
  - IRIs: Thesaurus.owl#C25830, CHEBI_16991, CL_0000255, FMA_74412, GO_0006281, GO_0009566, … (+3 more)

_…and 2 more — see the JSON sidecar for the complete list._

## phenotype_no_anatomy_axiom  (6 AOPs)

_HP/MP terms with no anatomy axiom — typically behavioural phenotypes by design_

- [`183`](https://identifiers.org/aop/183) — Nosema infection increases energetic demands leading to colony loss/failure
  - IRIs: GO_0060756, MP_0004889, D013312, D056631
- [`82`](https://identifiers.org/aop/82) — Abnormal role change in worker caste contributes to reduced brood care and leads to colony loss/failure
  - IRIs: GO_0002164, GO_0019098, MP_0001386, D056631
- [`189`](https://identifiers.org/aop/189) — Type I iodothyronine deiodinase (DIO1) inhibition leading to altered amphibian metamorphosis
  - IRIs: CHEBI_18258, CHEBI_28774, GO_0003824, GO_0007552, MP_0005473, PR_000006480
- [`217`](https://identifiers.org/aop/217) — Gastric ulcer formation
  - IRIs: GO_1990150, HP_0003002
- [`81`](https://identifiers.org/aop/81) — Increased metabolic stress contributes to abnormal foraging and leads to colony loss/failure
  - IRIs: GO_0060756, MP_0004889, PCO_0000001, D001066, D013312, D056631

_…and 1 more — see the JSON sidecar for the complete list._

## Notes

- **Signal C false-negative (33 AOPs):** these AOPs are listed as unclassified, but their title *would* match the Signal C keyword set. They appear unclassified because they have no Key-Event rows in `_query_aop_signal_rows` (which requires `aopo:has_key_event`), and Signal C is seeded inside that loop. Look for the `Title would match Signal C:` hint in the categorised lists above. Fixing this is a small classifier-side change: seed Signal C from `_query_aop_universe` (which always has the title) instead of from the KE-driven query.

- The `genuinely_unannotated` category is the only one that is **not** a tooling false-negative — those AOPs really do lack any organ/cell/object/process triples on every KE, *and* their title doesn't match the Signal C keyword set.
- For `fma_missing_from_cache`, the fix is regenerating the cache against a newer FMA snapshot or adding a manual FMA→bucket override layer.
- For `uberon_generic_or_non_vertebrate`, candidates for a dedicated 'Other-taxa / non-vertebrate' bucket include swim bladder, eggshell, cuticle, generic 'organ'/'tissue' terms.
- For `phenotype_no_anatomy_axiom`, the policy is intentional: behavioural HP/MP terms have no anatomy axiom and we don't fall back to keyword bridges at the term level.

Full per-AOP breakdown: `unclassified_audit_2026-05-26.json`.