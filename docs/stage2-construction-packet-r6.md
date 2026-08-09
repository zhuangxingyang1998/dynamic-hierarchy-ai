# Stage 2 Construction Packet R6 R1

Packet ID: `DH-S2-R6-R1`

Phase: `state_congruence`

Status:

- `NotReadyForConstruction`: independent adversarial audit is still required.
- `NotReadyForCalibration`: no R6 code, smoke, or terminal suite exists.
- `NotReadyForPairedQuery`: this packet remains fixed-query only.
- `NotReadyForRouting`: no router or phase controller is legal here.
- `NotReadyForCandidateClaim`: one seed is diagnostic only.

## End-to-End Result

Determine whether explicitly training generated states of the same modulo-seven
value to be interchangeable fixes the last fixed-query errors, and whether any
gain depends on the true arithmetic equivalence classes rather than extra loss or
compute.

R6 does not align generated states to literal embeddings. It trains and tests an
operational relation:

```text
state x represents value v
state y represents the same value v
therefore every legal outer composition must accept x or y interchangeably
```

## Evidence That Selects This Direction

The frozen `DH-S2-R5D-R1` diagnostic found:

- root/aux generated states retained 41/42 to 42/42 counterfactual arithmetic
  accuracy after every state was replaced by a state of a wrong semantic value;
- same-value swaps repaired both SUB root errors, while an ADD swap repaired one
  row and damaged another;
- literal-embedding substitution reduced root accuracy;
- teacher and root models learned different state interfaces.

Therefore direct literal alignment is not the default R6 intervention. The
remaining causal hypothesis is within-value operational variance.

## Data Contract

Use a new, non-overlapping serialized operator pattern: three literals with
operators `+-`. Queries remain fixed:

- ADD-first: first compose `a+b`, then subtract `c`;
- SUB-first: first compose `b-c`, then add `a`.

Both queries have the same mathematical final answer, which is acceptable because
R6 does not test query-conditioned trees. The different first/outer operations
exercise both generated-state interfaces without reusing R5's `-+` families.

Generate all 343 ordered triples over `0..6`. For every family derive both fixed
queries before splitting. Family hashes include values and operators but exclude
query and all truth.

Freeze a 245/49/49 family split. Let `f` be the final value, `i_add` the
ADD-first intermediate value, and `i_sub` the SUB-first intermediate value. The
proposed deterministic assignment is:

```text
split_code = (f + 2*i_add + 3*i_sub) mod 7
validation: split_code == 0
reserve:    split_code == 1
train:      split_code in {2, 3, 4, 5, 6}
```

The linear map from `(a,b,c)` to `(f,i_add,i_sub)` must first be proved
bijective over modulo seven. The construction packet auditor must then enumerate
the proposal and confirm 245/49/49 counts, exact balance over each of the three
values in every split, and base-family disjointness before implementation. A new
frozen R6 hash salt identifies these families but may not alter the split.

R5 train, validation, reserve, and diagnostic rows are historical and may not be
reclassified as R6 evidence. R6's new operator serialization must have zero base
hash overlap with every R5 family.

## Initialization And Common Budget

- Seed: to be frozen after the split audit; it must not equal an earlier seed.
- Initial state: exact passed R5 Rung 1 state digest
  `9c133c17c9dfcfe8bffbd8b71ea1a7d3ecd724dd037792be2fc6acc9e6b426ce`.
- Do not inherit any R5 fixed-query weights.
- Hidden/feedforward dimensions: `48/96`; dropout `0.0`.
- Optimizer and learning rate: R5 DirectML-compatible AdamW, `0.003`.
- Full-batch updates: `300` per branch unless a preconstruction budget audit
  freezes a different common value before any run.
- Every sibling is cloned from the same Rung 1 state before optimizer creation.

Changing seed, model size, steps, learning rate, or loss weight after seeing an
R6 result is a new packet, not calibration.

## Mandatory Fixed-Query Matrix

Build the following six branches separately for ADD-first and SUB-first:

| Branch | First state consumed by ordinary path | Additional objective |
| --- | --- | --- |
| root | generated | none |
| teacher | true literal embedding | none; privileged diagnostic |
| aux-true | generated | true intermediate classification |
| aux-sham | generated | histogram-matched sham classification |
| congruence-true | generated | same-value transplant final loss |
| congruence-sham | generated | sham-group transplant final loss |

`congruence-true` and `congruence-sham` must have identical parameters, two
second-composition calls, loss counts, batch order, and realized/max compute.

For congruence-true, deterministically rotate generated states within each true
intermediate-value class and apply the original final target to the transplanted
path. Cover every nonidentity rotation uniformly across updates.

For congruence-sham, first assign a stable histogram-preserving deranged class to
every training row, then use the identical rotation schedule within those sham
classes. The original final target remains unchanged. Every sham assignment must
change, and true/sham group sizes, state uses, and compute must match.

Intermediate truth may select the privileged congruence branch's training
partners but may never enter model input or inference. This branch is diagnostic,
not a deployable candidate.

## Evaluation And Causal Gates

Evaluate ordinary answers with no intermediate truth input. In addition, execute
two hard state interventions from the same frozen branch weights:

1. exhaustive same-value closure: replace every target row's first state with
   every validation state having the same true value;
2. exhaustive counterfactual closure: replace every target first state with all
   49 validation states and score against the arithmetic answer implied by the
   source state's true value.

For a query branch to pass validation it must reach:

- ordinary answer accuracy `49/49`;
- all seven predicted classes;
- exhaustive same-value closure accuracy `100%`;
- exhaustive counterfactual closure accuracy `100%`;
- finite cross-entropy and intervention metrics;
- exact replay and state-source receipts.

A branch opens its one-shot reserve only after all validation conditions pass.
Reserve uses the same gates and cannot replay after durable opening.

R6's primary diagnostic requires both congruence-true query branches to pass
validation and reserve. Dispositions are:

- `state_congruence_failed`: either true-congruence query fails;
- `mechanism_non_specific`: true-congruence passes but both matched sham branches
  also pass the same gates;
- `state_congruence_candidate`: both true-congruence queries pass, at least one
  matched sham query fails, all causal receipts pass, and no invariant fails;
- `implementation_invalid` or `calibration_incomplete` retain their R5 meanings.

Root, teacher, and auxiliary outcomes are diagnostic and cannot override the
true-versus-sham congruence disposition.

## Required Controls And Canaries

- exact symbolic solver over all 343 families;
- query-only and every one-/two-literal partial-input shortcut audit must remain
  at the balanced seven-class baseline; the full three-literal exact solver must
  remain perfect;
- base-family overlap audit against all R5 splits;
- exact inherited-state and optimizer-after-clone receipts;
- per-label partner coverage and no-self-map receipts;
- sham histogram, derangement, bijection, and compute equality;
- ordinary, same-value, and all-state counterfactual prediction matrices;
- canonical literal injection as report-only, never a positive gate;
- no paired model, router, STOP, continuous phase, or natural-language task.

## Recovery And Evidence

Reuse R5.1's pre-gate checkpoint plus monotonic one-shot branch ledger semantics.
The packet auditor must add crash tests for interruption between congruence
branches and during exhaustive intervention evaluation.

Publish frozen config, result, ledger, split receipts, intervention matrices, and
SHA256SUMS. Keep checkpoints local recovery artifacts.

## Construction Gates

Before `ReadyForConstruction`:

1. independently verify the 343-family `+-` algebra and proposed balanced split;
2. prove true/sham partner schedules are bijective and compute matched;
3. freeze the new seed, exact split salt, budget, thresholds, and DirectML config;
4. map every gate to focused and terminal tests;
5. confirm R2-R5 source/checkpoint compatibility and read-only boundaries;
6. obtain an independent `ReadyForConstruction` verdict.

No R6 implementation or training is authorized by this R1 packet alone.
