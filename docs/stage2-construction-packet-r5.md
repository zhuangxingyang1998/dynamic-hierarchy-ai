# Stage 2 Construction Packet R5.1

Packet ID: `DH-S2-R5.1`

Status:

- `ConstructionComplete`: the audit-amended arithmetic ladder is implemented.
- `CalibrationComplete`: the single frozen run ended `fixed_query_failed` after
  Rung 1 passed and both fixed-query root branches failed validation.
- `NotReadyForRouting`: no learned merge router is trained in this packet.
- `NotReadyForCandidateClaim`: one diagnostic seed cannot establish an effect.
- `NotReadyForPhaseC`: continuous phase remains blocked.

R5.1 supersedes the rejected R5 draft. The first read-only audit rejected R5
because it used only one fixed query, changed domains between rungs, reinitialized
each rung, lacked matched auxiliary and structure interventions, and left batch
exposure and per-query gates underspecified.

The implemented packet passed focused and full CPU/DirectML validation. Two
independent re-audits returned `ReadyForCalibration`. The only authorized run
then completed 600 rounds without opening Rung 3. Exact results and interpretation
are recorded in `docs/stage2-r5-arithmetic-ladder-result-20260809.md`.

## End-to-End Result

Construct and run one bounded diagnostic ladder that locates the first failure
among:

```text
single shared binary composition
  -> ADD-first and SUB-first fixed-query recursion
  -> one shared paired-query recursion
  -> correct-tree versus wrong/fixed-tree intervention
```

Rung 1 trains one arithmetic state. Every later model is cloned from that exact
passed checkpoint, not newly initialized and not inherited from another later
rung. Rung 3 opens only after both fixed-query root-only branches pass validation
and reserve. Failure stops the main path without changing seed, model, exposure,
or thresholds.

## Causal Question

R4 executed the correct B-oracle tree but failed to learn modulo-seven
composition. R5.1 asks, in order:

1. Can a shared neural binary composer fit all 98 addition/subtraction facts?
2. Can that same learned state be reused recursively under each query in
   isolation?
3. Can one shared model bind both queries without multi-task interference?
4. Does paired success causally depend on the supplied tree?
5. If root-only recursion fails, do teacher state or true intermediate labels
   rescue it while a matched sham does not?

R5.1 does not test boundary discovery. Passing the whole ladder reaches only
`ReadyForRoutingDesign`, not authorization to train a router.

## Ownership And Interfaces

- The exhaustive data owner issues values, operators, query IDs, final labels,
  first-merge labels, hashes, split membership, and exact-solver receipts.
- `LadderModelInput` contains only values, operators, and answer query IDs.
  Passing targets or a complete generated split to a model is a type error.
- The shared composer owns literal/operator/query embeddings, one binary
  composition function, and one seven-class readout.
- Final and intermediate labels are loss targets only. They never enter ordinary
  forward input.
- A teacher-state diagnostic may replace the learned first merge with the
  embedding of its correct value. This is privileged intervention, not a
  candidate mechanism.
- The runtime owns full-batch order, exact exposure counts, branch creation,
  checkpoint cloning, gates, one-shot reserve state, and result accounting.
- The exact solver is a data canary, not a trainable control.

## Common N3 Domain And Split

All fixed-query and paired-query experiments use the same expression `a-b+c`
and the same 294 families for which:

```text
ADD-first: a - (b + c) mod 7
SUB-first: (a - b) + c mod 7
```

produce unequal labels. The 294 families cover every unequal ordered answer
pair exactly seven times.

The base-family hash is canonical SHA256 over values and operators only. It
excludes query, split, final labels, and intermediate labels. Within each ordered
answer-pair stratum, rank by canonical SHA256 over:

```json
{"family_hash":"...","salt":"DH-S2-R5.1|821501|shared-n3"}
```

Then take the first five families for train, sixth for validation, and seventh
for reserve:

| Split | Families per ordered pair | Families | Paired rows |
| --- | ---: | ---: | ---: |
| train | 5 | 210 | 420 |
| validation | 1 | 42 | 84 |
| reserve | 1 | 42 | 84 |

The canonical family order is ordered label-pair lexicographic order, then the
salted SHA256 rank. Paired batches contain all ADD-first rows in canonical family
order, followed by all SUB-first rows in the same family order. This permits
contiguous hard-path execution on DirectML without indexed-gradient CPU fallback.
A fixed-query branch receives the corresponding 210/42/42 contiguous rows. Both
queries follow their family into the same split.

The runtime must report exact label/query balance and three two-literal lookup
canaries `(a,b)`, `(a,c)`, and `(b,c)` for every held-out split. Any canary above
`0.50` is an implementation-invalid shortcut, not a training result.

## Rung 1: Atomic Fit And B-Core Bridge

The full domain contains every `(left, operator, right)` over `0..6` and `+/-`,
exactly 98 rows, with a constant query. Every row appears once in the full batch
and the batch is repeated for exactly 300 updates, so every atomic fact receives
300 exposures.

Train the pure shared composer with root loss. It must achieve all 98 rows
correct, cross-entropy `<=0.05`, all seven predicted classes, and finite metrics.

The learned pure state is then copied into an execution bridge built from the
existing `Stage2MergeClassifier`:

- literal embeddings map to token IDs `8..14`;
- ADD/SUB embeddings map to the existing operator token IDs;
- query embedding, composer, and classifier are copied exactly;
- position projection is zeroed;
- with one active root, terminal attention is semantically inert;
- router and STOP modules are not used.

Pure and bridged root logits must match within `1e-5` on all 98 rows. This bridge
does not claim the whole R4 architecture is equivalent; it proves that the R5
arithmetic state executes through the existing B composition/readout path.

Rung 1 failure yields `representation_fit_failed` and creates no later models.

## Rung 2: Two Fixed Queries And Diagnostics

After Rung 1 passes, clone its exact final state into eight models before any
Rung 2 optimizer is created:

| Query | root-only | teacher-state | aux-true | aux-sham |
| --- | --- | --- | --- | --- |
| ADD-first | yes | yes | yes | yes |
| SUB-first | yes | yes | yes | yes |

Each model receives the matching 210-row fixed-query full batch for exactly 300
updates. Thus every training family/query receives 300 exposures. No shuffle,
minibatch, remainder, early stopping, or cross-branch weight sharing is allowed.

- root-only: final-answer cross-entropy only.
- teacher-state: final loss only, but the first composed state is replaced by the
  literal embedding of the exact first-merge value before the second merge.
- aux-true: final loss plus first-merge cross-entropy against the true value.
- aux-sham: identical extra readout, loss weight, and compute, but targets are a
  deterministic histogram-preserving derangement of the true target vector.
  Within each query group independently, labels are stably sorted, rotated by
  that group's maximum class count, and restored to row order; construction
  fails unless every row changes.

ADD-first first-merge truth is `(b+c) mod 7`; SUB-first truth is `(a-b) mod 7`.
Root/teacher/aux-true/aux-sham have identical architecture, parameters, inherited
state, input order, and final loss. Aux-true and aux-sham are loss/compute matched.
Teacher-state remains privileged and unmatched by design.

Each branch independently validates, opens reserve once only on validation pass,
and passes only with zero final-answer errors, cross-entropy `<=0.10`, all seven
classes, finite metrics, and a passing reserve. Intermediate accuracy is
report-only. Both root-only query branches must pass before Rung 3 opens.

Interpretation is constrained:

- teacher pass after root failure is consistent with recurrent state-generation
  or credit-transfer failure;
- aux-true pass while root and aux-sham fail is consistent with useful semantic
  intermediate supervision;
- neither observation proves a unique cause in one seed.

## Rung 3: Paired Queries And Structure Intervention

Only after both fixed-query root branches pass, clone the Rung 1 final state into
three fresh paired models: root-only, aux-true, and aux-sham. Do not inherit Rung
2 weights. Each model receives the 420-row paired full batch for 300 updates, so
each family/query row again receives 300 exposures.

Validation and reserve root gates require:

- ADD-first accuracy exactly `1.0`;
- SUB-first accuracy exactly `1.0`;
- paired-family both-correct rate exactly `1.0`;
- aggregate cross-entropy `<=0.10`;
- all seven classes and finite metrics.

For every passing paired model, evaluate without weight changes:

- correct query-specific oracle order;
- opposite-query tree while the answer head receives the correct query;
- fixed-left tree;
- fixed-right tree.

For root-only to reach `ReadyForRoutingDesign`:

- opposite-tree accuracy must be `<=0.10`;
- best fixed-tree accuracy must be `<=0.50`;
- correct-tree accuracy must exceed opposite-tree accuracy by at least `0.40`;
- bridge logits through the existing B execution path must remain within `1e-5`.

If paired answers pass but intervention does not, disposition is
`structure_decorative`; learned routing remains blocked.

For `paired-root`, passing validation answers admits the one-shot reserve even
when the validation structure intervention fails. Answer and structure gates are
then aggregated separately across validation and reserve, so a causal-structure
failure cannot be misreported as an answer failure.

## Initialization, Optimizer, And Exposure Freeze

- Revision: `stage2-r5.1`
- Phase: `arithmetic_ladder`
- Run kind: `calibration_only`
- Seed: `821501`
- Backend: DirectML FP32, `deterministic=false`
- CPU threads: `4`
- Hidden/feedforward dimensions: `48/96`; dropout `0.0`
- Optimizer: DirectML-compatible AdamW
- Learning rate: `0.003`
- Betas: `(0.9, 0.999)`; epsilon `1e-8`; weight decay `0.0`
- Full-batch updates: exactly `300` per opened branch at every rung
- Auxiliary loss weight: `1.0`
- Checkpoint interval: `25` complete multi-branch updates
- Cooperative yield: `1` ms after each complete multi-branch update
- Time budget: `30` minutes

Rung 1 initializes from seed `821501 + 101`. Every later model is cloned from the
exact passed Rung 1 state before optimizer creation. Root/diagnostic siblings must
have byte-identical inherited state digests. There is no early stopping.

## Gate And Disposition Vocabulary

Legal completed dispositions are:

- `representation_fit_failed`: Rung 1 or B-core bridge fails;
- `fixed_query_failed`: either fixed-query root branch fails validation/reserve;
- `paired_query_failed`: fixed queries pass but paired root answer gate fails;
- `structure_decorative`: paired root answers pass but intervention gate fails;
- `ladder_pass`: paired root answer and intervention gates pass;
- `implementation_invalid`: invariant, exact solver, bridge, ledger, backend, or
  test contract fails;
- `calibration_incomplete`: cooperative STOP, resource, or time limit ends an
  opened training stage.

Auxiliary and teacher outcomes are diagnostic fields and do not override the
root-path disposition.

## Recovery And One-Shot Reserve

Every rung/branch has an atomic ledger state:

```text
unopened -> validation_failed
unopened -> reserve_opened -> complete
```

Once `reserve_opened` is durable, interruption before `complete` permanently
forbids replay. Completed evidence is reused byte-for-byte on resume. The runtime
must save an exact pre-gate checkpoint before validation so completed evidence
always refers to reconstructible weights.

Checkpoints must retain and verify packet ID, config digest, partition digest,
Rung 1 inherited-state digest, current rung, active branches, full-batch update
cursor, model/optimizer/RNG state, cumulative receipts, and ledger digest.
Recovery is at-least-once for training and fail-closed for opened reserve.

## Required Canaries And Non-Claims

- Exact solver accuracy is `1.0` on every split/query.
- Query-only lookup is exactly `1/7`.
- Input-only lookup on paired rows is exactly `0.5`.
- Every held-out two-literal lookup canary is `<=0.50`.
- Family hashes exclude query and all truth; row hashes add query only.
- Aux-sham has the same label histogram within every query group, loss count,
  and readout compute as aux-true.
- Rung 2 and Rung 3 share the exact same 210/42/42 family partition.

One seed is diagnostic only. Even `ladder_pass` does not establish learned
hierarchy, repair R4, or authorize a formal claim. A future routing design must
still freeze its control matrix and formal multi-seed rule separately.

## Write Scope

Allowed production anchors:

- new `stage2_ladder_config.py`, `stage2_ladder_data.py`,
  `stage2_ladder_model.py`, and `stage2_ladder_runtime.py` modules;
- new `scripts/run_stage2_ladder.py`;
- R5.1 smoke and frozen calibration configs;
- focused CPU and DirectML tests;
- packet, result, evidence, README, protocol, and development log.

R2/R3/R4 source, configs, checkpoints, results, and evidence remain read-only.

## Construction And Terminal Gates

1. Build the common salted 294-family split and all canaries.
2. Build the composer, teacher/aux losses, B-core bridge, and interventions.
3. Build inherited checkpoint cloning, exact exposure accounting, branch gates,
   reserve ledger, and recovery.
4. Pass focused CPU tests and DirectML forward/backward/checkpoint tests.
5. Pass CPU and DirectML smoke, R2-R4 checkpoint compatibility, and both full
   project suites.
6. Obtain a fresh independent `ReadyForCalibration` audit.
7. Run exactly one frozen DirectML calibration.
8. Stop at the disposition, publish exact evidence, update docs, commit, and
   push. Do not tune or launch another seed.

Terminal acceptance requires machine-readable proof of every count, exposure,
gate, canary, inheritance digest, bridge comparison, intervention, ledger state,
and explicit non-claim above. No focused PASS or smoke alone completes R5.1.
