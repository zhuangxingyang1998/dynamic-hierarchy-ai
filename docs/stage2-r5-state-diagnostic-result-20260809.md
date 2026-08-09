# Stage 2 R5 State Diagnostic Result

Date: `2026-08-09`

Packet: `DH-S2-R5D-R1`

Status: `diagnostic_complete`

This is a post-hoc, read-only diagnostic. It does not amend the canonical R5.1
`fixed_query_failed` result.

## Question

R5.1 root models missed one fixed ADD validation row and two fixed SUB rows.
Teacher-state models passed when trained to consume literal embeddings. The first
working diagnosis was that a generated merge state might contain a decodable
answer but be unusable by the second composition.

This diagnostic tested that explanation directly by replacing the first state
between the two composer calls.

## Execution Boundary

The diagnostic loaded the retained round-600 checkpoint on CPU and evaluated all
eight fixed-query branches on their 42-row validation batches. It performed zero
optimizer updates and zero backward calls. It did not materialize either reserve
batch. The frozen config, canonical result, one-shot ledger, and final checkpoint
had identical SHA256 values before and after execution.

Hand-replayed ordinary and teacher logits matched the model APIs exactly. CPU
correct-row counts reproduced the DirectML ledger for every branch; maximum
cross-entropy drift was below `1.2e-7`.

## Root And Aux-True Results

| Branch | Learned state | Literal state | Same-value transplant | Wrong-state counterfactual | Intermediate readout |
| --- | ---: | ---: | ---: | ---: | ---: |
| ADD root | 41/42 | 31/42 | 41/42 | 41/42 | 42/42 |
| SUB root | 40/42 | 37/42 | 42/42 | 41/42 | 37/42 |
| ADD aux-true | 41/42 | 31/42 | 41/42 | 41/42 | 42/42 |
| SUB aux-true | 41/42 | 38/42 | 42/42 | 42/42 | 42/42 |

`Literal state` replaces the generated first state with the model's own literal
embedding for the true intermediate value. It made every root/aux-true branch
worse, so the simple claim that root models merely need canonical literal states
is false for these trained weights.

`Same-value transplant` uses another validation row's generated state with the
same exact intermediate value. ADD root repaired its one old error but damaged
one previously correct row, leaving 41/42. SUB root repaired both errors without
damage. Thus same-value states are close to operationally equivalent but not
perfectly interchangeable.

`Wrong-state counterfactual` applies a bijection over all generated states while
changing every assigned intermediate label. Accuracy in the table is measured
against the arithmetic answer implied by the transplanted state's value, not the
original answer. All real-answer accuracies fell to zero, while counterfactual
accuracy stayed at 41/42 or 42/42. This is strong causal evidence that the second
composer actually consumes the generated state's arithmetic semantics.

## Matched Diagnostics

The aux-sham intermediate readout was 0% by construction, yet its generated
states still achieved 42/42 same-value and wrong-state counterfactual accuracy on
SUB, and 42/42 on both transplant tests for ADD. An auxiliary classifier's label
therefore is not a reliable description of the operational state protocol.

Teacher models showed the opposite interface specialization. Their learned-state
paths scored only 4/42 ADD and 8/42 SUB, while true literal states scored 42/42.
Wrong literal states also produced their exact counterfactual answers 42/42.
Teacher training built a literal-embedding consumer; root training built a
generated-state consumer. One interface should not be substituted for the other
without training it.

Cosine or Euclidean nearest-literal retrieval reached only 55%-74% on root and
aux-true states despite near-perfect operational counterfactual behavior.
Geometric proximity to literal embeddings is therefore not an adequate state
quality metric here.

## Revised Interpretation

The diagnostic rejects the strongest version of the previous representation
failure story. R5 root states are not merely decodable; they are operationally
usable, and their semantics causally control the second composition.

The remaining defect is narrower. Generated states representing the same
modulo-seven value are not guaranteed to be operationally interchangeable.
Within-value transplantation can repair errors, damage another row, or remove
both SUB root errors. This is consistent with unwanted instance-specific
variation inside a semantic value class or a downstream decision boundary that
is not invariant to that variation. The diagnostic does not distinguish those
two explanations.

## Decision

Do not train a direct literal-alignment objective merely because teacher models
passed; the post-hoc intervention predicts that naive substitution can hurt.
R6 should instead test operational state congruence: states representing the
same arithmetic value should be interchangeable under every legal outer
composition. A true same-value intervention must be compared with a
compute-matched sham grouping.

R6 remains untrained and requires an independent construction audit.

## Evidence

Published diagnostic result:

- `evidence/stage2-r5-state-diagnostic/result.json`
- SHA256:
  `2A81640994726E8FF5E75CB7FC01ADA59C3CC0DAC38A8E3B4B7B5DFD3D54DAF0`
