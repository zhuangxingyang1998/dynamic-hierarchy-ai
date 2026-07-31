# Stage 1 Formal Campaign V4 Result

## Decision

Canonical campaign v4 completed and passed its preregistered formal
confirmation gate:

- campaign state: `completed`;
- verified training seeds: `8/8`;
- per-run target: `8000/8000` paired optimizer steps;
- aggregate decision: `formal_confirmation_passed`;
- aggregate `stage2_unblocked`: `true`;
- aggregate SHA256:
  `95F4147F05CD31C1133418AF17A2F3061B37044A265D7FF0BB0BB06EBEE15631`.

This decision authorizes starting Stage 2 work. It does not claim that any
model in Stage 1 learned a dynamic hierarchy. D-true receives the fixed true
tree topology and source references as privileged input; A receives no tree,
and D-sham receives a deterministic wrong source alignment.

## Frozen Identity

- Campaign package:
  `runs/stage1-literal-formal-v4-campaign`
- Campaign-manifest file SHA256:
  `809AC5703E1FFC4C59E7DCE9C129050E9A062B8B1F10C95A7651439B9888D9B8`
- Campaign manifest identity:
  `7a42b0741e339b2caa8636890fc0ad7f35c45e545a1243c9316c4ef52081dcb9`
- Snapshot manifest:
  `9517bc53e9f2a4950001409d0ac001fff1efff2ebf1def2bcc404d7e5ae5ef27`
- Source manifest:
  `47ad080b7559a2bbf91d54eb33af7ea7fb99902b8f237bc39c04169da66ddcf8`
- Candidate prerequisite result SHA256:
  `6F964AE37B7354FB0F8E3B228F4BF3BF0B34C90CE6BC820F312BECAD8B49D615`

All eight results have the same frozen source and snapshot manifests. The
coordinator verified each result before launching the next seed.

## Experimental Scale

- Training seeds: `991501`, `991511`, `991531`, `991541`, `991547`,
  `991567`, `991579`, and `991589`.
- Final-evaluation content seeds: `992501`, `992519`, and `992531`.
- Foundation seed: `992549`.
- Final splits: two in-distribution, one depth extrapolation, and one held-out
  topology extrapolation split.
- Final scale: `10,010` examples per split and evaluation seed.
- Total final examples: `960,960` per model across all training seeds.
- Campaign wall time: `30,216.089s` from the first worker start to the last
  terminal status.
- Runtime warnings: zero in all eight runs.
- DirectML fallback observability: `unknown`; zero Python warnings are not
  interpreted as proof of zero fallback.

## Aggregate Statistics

Accuracy values below are means over the three fixed evaluation seeds and
eight independent training seeds. Lower bounds are the preregistered one-sided
paired Student-t bounds after Bonferroni correction over eight comparisons.
Effect values are percentage-point differences.

| Split | A | D-true | D-sham | Mean D-true - A | Lower bound | Mean D-true - D-sham | Lower bound | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID depth-3 skew | 22.0596% | 100.0000% | 57.8259% | 77.9404 | 75.7506 | 42.1741 | 36.7981 | 3.0 |
| ID depth-3 balanced | 14.3548% | 99.9992% | 17.6856% | 85.6444 | 85.4647 | 82.3135 | 81.1329 | 3.0 |
| Depth-5 skew | 14.5962% | 99.9942% | 27.1728% | 85.3979 | 85.0803 | 72.8213 | 69.1360 | 2.0 |
| Held-out depth-3 branched | 15.8109% | 100.0000% | 38.0798% | 84.1891 | 83.5384 | 61.9202 | 57.5269 | 2.0 |

All eight statistical conditions passed. Both in-distribution splits passed
both effects, and both preregistered extrapolation splits also passed both
effects.

## Integrity Audit

The aggregate contains 38 integrity and eligibility checks for each of eight
training seeds. All `304/304` checks are literally `true`. The six top-level
campaign conditions and all eight statistical conditions are also true.

The frozen aggregator was run independently against the exact eight verified
results with both canonical Python source roots ahead of the editable install.
The recomputed output was byte-identical to the canonical 22,131-byte
aggregate and had the same SHA256.

| Seed | Run | Result SHA256 |
| ---: | --- | --- |
| 991501 | `stage1-formal-v4-01-991501-20260731T003527Z` | `033D7465517CD3B9B0E4D27F0C5579799F551C54DCD34D0996999A2A19D109FA` |
| 991511 | `stage1-formal-v4-02-991511-20260731T013849Z` | `0D840726D46132B0E0C3FB3970D485A830317E2679D857158CDE35B2D96ACE53` |
| 991531 | `stage1-formal-v4-03-991531-20260731T024111Z` | `32643472B8F4A621087700EB1A65E87C8F96FA4743B5CB58EFAE194D2B0655D4` |
| 991541 | `stage1-formal-v4-04-991541-20260731T034253Z` | `92D3508062B6DF5594FEA14C61932911BC5518ED3324E2826259116B388C408A` |
| 991547 | `stage1-formal-v4-05-991547-20260731T044356Z` | `DDA936102A710E25341C1881390E73011FF949F6DA4C99F952D267D62A2C384A` |
| 991567 | `stage1-formal-v4-06-991567-20260731T054538Z` | `AED3206ED57AA2EB88FF044EBED7EF7991C753EE37FD67D9F6194FD91C6F8305` |
| 991579 | `stage1-formal-v4-07-991579-20260731T065040Z` | `BDDD9FC5E7B570BA1F45BE957D69AFD5AF7373E4CFE22981D8B4762394133FAC` |
| 991589 | `stage1-formal-v4-08-991589-20260731T075226Z` | `9EEDE58007CD2C01484F3FB1A5B2DAC960A573C1A370B67AC5FD27C8C7D312F8` |

Each individual result deliberately records `stage2_unblocked=false`; no
single seed can authorize Stage 2. Only the exact eight-result aggregate
records `stage2_unblocked=true`.

## Interpretation Boundary

Confirmed fact: in this synthetic modulo-seven task and frozen experiment,
correct privileged structure produced a large, consistent advantage over both
the ordinary Transformer and the architecture-matched wrong-structure sham.

Supported inference: the gain is not explained solely by D's recursive
architecture, because D-sham used the same architecture, initialization,
optimizer, update count, examples, parameter budget, and compose budget but
performed much worse.

Not established: autonomous boundary discovery, learned `MERGE/STOP`
decisions, general language reasoning, or general real-world hierarchy
learning. D-sham is also only one preregistered wrong-structure intervention.
Those questions belong to Stage 2 and later controls.
