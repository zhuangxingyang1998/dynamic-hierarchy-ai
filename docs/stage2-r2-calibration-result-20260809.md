# Stage 2 R2 Calibration Result

Date: 2026-08-09

Packet: `DH-S2-R2`

Decision: `calibration_inconclusive`

## What Ran

The DirectML calibration used training seed `821101`, FP32, hidden size 64,
120 optimizer updates, four CPU threads, and one AMD Radeon RX 9060 XT worker.
All eight trainable controls received the same 10,080 query rows and update
count. Five fixed policies reused B-query weights during intervention
evaluation, giving the complete 13-control matrix.

Training drew 5,040 base families, of which 4,393 were unique. Evaluation used
420 independent base families and 840 query rows in each of three profiles,
for 1,260 unique evaluation families total. Train/evaluation base-family
overlap was zero. Query-label counts were exactly uniform.

## Task Results

Chance accuracy is `1/7 = 14.29%`.

| Profile | A-Q-param | B-query | B-noQ-router | Best fixed | D-true |
| --- | ---: | ---: | ---: | ---: | ---: |
| ID, 5 leaves | 14.64% | 15.00% | 14.52% | 16.55% | 14.17% |
| Length OOD, 8 leaves | 15.00% | 15.60% | 15.48% | 15.60% | 12.62% |
| Topology OOD, 6 leaves | 15.95% | 15.00% | 14.52% | 15.00% | 16.90% |

These are single-seed calibration measurements without preregistered positive
margins. None supports a performance claim.

## Structural Result

B-query, B-noQ-router, and B-sham all selected immediate `STOP` on 100% of
evaluation rows. Their query-pair traces were identical, contained no merge
edge, and had zero exact-tree rate. B-query therefore did not learn
query-conditioned structure; it collapsed to the explicit F-stop control.

The parameter-matched Transformer differed from B by `0.1144%` parameters.
The declared full-candidate forward-operation estimate differed by `0.0504%`.
This remains an operation estimate rather than an exact DirectML FLOP count.
DirectML fallback status remains unknown because no public fallback counter is
available; zero Python warnings were observed.

## Telemetry Recovery

The original result incorrectly filtered uppercase `MERGE` trace actions using
a lowercase comparison. This affected structure telemetry only. The original
result was preserved, the comparison was repaired and regression-tested, and
evaluation was rerun from the unchanged final checkpoint with zero optimizer
updates. Every task accuracy and cross-entropy value matched the original
result exactly. `F-add` and `F-sub` now each show the expected `0.5` exact-tree
rate, validating the repaired metric.

## Decision

The calibration did not establish that the task models learned the arithmetic
problem, because A, B, D-true, and D-sham all remained near chance. It also
observed an exact all-STOP collapse in B. The result does not show that dynamic
hierarchy is ineffective; it shows that this training setup cannot yet test
the hypothesis.

R3 must first preregister a task-learning feasibility gate. On this benchmark,
which always requires a complete expression result, B should learn only the
query-conditioned MERGE order and should not receive STOP as a legal action.
Useful variable stopping requires a separate benchmark. Multi-seed candidate
training, continuous phase, natural language, and formal claims remain blocked.

Public evidence: [Stage 2 R2 calibration bundle](../evidence/stage2-r2-calibration/README.md).
