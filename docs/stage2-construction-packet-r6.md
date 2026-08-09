# Stage 2 Construction Packet R6 R7

Packet ID: `DH-S2-R6-R7`

Phase: `state_congruence`

Status:

- `ReadyForConstruction`: R7 authorizes only the bounded acceptance repairs
  required by the R6 implementation audit.
- `ReadyForCalibration`: the final independent R7 audit authorizes exactly one
  frozen canonical run under `DH-S2-R6-R7`.
- `NotReadyForPairedQuery`: this packet remains fixed-query only.
- `NotReadyForRouting`: no router, STOP action, or phase controller is legal.
- `NotReadyForCandidateClaim`: one seed can produce a diagnostic signal only.

Independent R2 design verdict: `ReadyForConstruction`, with no remaining design
findings. The first implementation audit returned `NotReadyForCalibration` on
run lifecycle, evidence binding, reserve isolation, measured-budget receipts,
and missing recovery tests. R3 addressed those findings without changing the
task, families, branch objectives, optimizer, or calibration budget.

## R2 Implementation Audit Disposition

The first construction audit confirmed the split, canaries, all 34 partner maps,
R5 inheritance, loss normalization, producer gradients, and DirectML fallback
boundary. It rejected calibration because a pre-existing ledger could be read
without first restoring its model checkpoint, `implementation_invalid` could
exit successfully, smoke could reach reserve code, training receipts were partly
declarative, and completed evidence was not fully immutable or replay-tested.

R3 therefore requires model-state-bound cohort evidence, measured operation and
source-use counters, strict run-directory lifecycle checks, nonzero invalid
execution, smoke reserve prohibition, and terminal recovery tests.

## R3 Implementation Audit Disposition

The R3 re-audit independently confirmed data/maps, row-level pair evidence,
DirectML gate/resume coverage, state summaries, and reserve prohibition. It
still returned `NotReadyForCalibration`: saved dispositions were not re-derived
from metrics on recovery, checkpoints were not bound to one run instance,
illegal reserve/execution combinations could survive, and operation counts were
reported by high-level `forward` code instead of measured at `_compose` and
`_logits`.

R4 freezes a complete ledger state table, exact disposition re-derivation,
validation replay for pre-gate ledger extension, per-run immutable identity,
run-local checkpoint paths and SHA256 receipts, optimizer-state binding, and
real call-site operation tracing. It changes no scientific task or budget.

## R4 Implementation Audit Disposition

The R4 audit confirmed run identity, optimizer/model binding, foreign/hash path
rejection, data/maps, and the DirectML boundary. It still returned
`NotReadyForCalibration`: no round-0 checkpoint existed, checkpoint-file and
latest-receipt publication could strand a replay at the same filename, illegal
open reserve state was normalized before validation, gate decisions trusted
cached booleans, first/outer compose counts were split from one total, and the
trainer could create a manifest without explicit fresh authorization.

R5 adds recoverable round-0 publication, unique atomically renamed checkpoint
files, orphan-safe replay, raw cohort gate derivation, pre-normalization open
state validation, separate first/outer call regions, and explicit fresh-manifest
authorization. It changes no scientific task or budget.

## R5 Implementation Audit Disposition

The R5 audit confirmed round-0 publication, orphan-safe checkpoint replay,
run-instance identity, real call-site operation counting, the exhaustive split,
all partner maps, and the DirectML fallback boundary. It still returned
`NotReadyForCalibration`: the run identity did not bind a frozen source
snapshot, an interrupted open reserve was rewritten before loading and verifying
the latest checkpoint, and the gate still trusted cached replay fields and
cross-entropy summaries that could not be reconstructed from frozen rows.

R6 freezes the complete Python source set and project metadata before round 0,
binds its digest through every persistent artifact, and rejects resume when the
current/imported source differs. An open reserve remains pending and byte-stable
through construction; only a verified latest checkpoint, exact state binding,
full invariant receipt, and validation replay may convert it to stranded. Gate
derivation now reconstructs pair identities, family/row IDs, source and target
values, counterfactual labels, masks, counts, accuracy, and cross-entropy from
the frozen batch plus per-row NLL evidence. It changes no task, family, model,
optimizer, branch objective, or calibration budget.

## R6 Implementation Audit Disposition

The R6 audit confirmed the 35-file source freeze and normal source binding,
frozen `245/49/49` data, 306-step DirectML config, R5 inheritance, operation and
training receipts, and absent canonical run. It still returned
`NotReadyForCalibration`: completed reserve evidence could extend an unopened
checkpoint without exact model-output replay, crashes between frozen config,
source snapshot, run-instance, and round 0 could strand the canonical directory,
and constructor failures before assignment could write status without the
already available source/run identity.

R7 adds read-only exact replay for a reserve cohort that is already marked
complete, without training, model selection, reopening, or a second research
decision. It defines strict initialization-only recovery for frozen-config,
verified-snapshot, run-instance, and orphan-round-0 boundaries, including safe
cleanup of exact app-owned snapshot temporary directories. Failure/status
receipts recover every source/run identity that was fully verified before a
constructor error. It changes no scientific task or budget.

## R1 Audit Disposition

The independent R1 audit returned `NotReadyForConstruction`. R1's modulo-seven
algebra and 245/49/49 counts were correct, but its sham could collapse to a
renaming of the true groups or inject mathematically false targets. It also
failed to freeze the exchange gradient, an extra-compute control, the R6 model
interface, partial-input canary semantics, cohort reserve gating, and the exact
calibration configuration.

R2 removes the false-label sham. It compares same-value exchange against two
legal matched controls:

- `self-duplicate`: repeat the target's own generated state and target;
- `mixed-counterfactual`: use a different-value generated state and recompute
  the mathematically correct answer for that source state in the target outer
  context.

## End-to-End Result

Determine whether end-to-end training on same-value state exchange closes the
remaining fixed-query errors, and whether any closure is specific relative to
duplicate compute and general valid state recombination.

The claim is deliberately narrow:

```text
within two fixed producer/consumer interfaces,
generated states with the same modulo-seven value should be interchangeable
```

R6 does not claim every legal operator, operand slot, query, or recursive depth.
The two interfaces are:

- ADD-first: `(a+b)-c`, generated state is the left operand of outer SUB;
- SUB-first: `a+(b-c)`, generated state is the right operand of outer ADD.

The two queries have the same final value `a+b-c`. R6 uses them only to exercise
these two interfaces; it is not a query-conditioned-structure experiment.

## Data And Identity Contract

Generate all 343 ordered triples `(a,b,c)` over `0..6` with the serialized
operator pattern `+-`. Derive both fixed-query rows after the base family exists.

Canonical unsalted base identity must use the same domain as R5:

```text
base_family_id = SHA256(canonical_json({values:[a,b,c], operators:[ADD,SUB]}))
query_row_id   = SHA256(canonical_json({base_family_id, query_id}))
```

The base identity contains no stage name, seed, query, split, label, or truth.
Partition/rank receipts are separate. Cross-stage overlap is audited with this
unsalted identity, so changing a salt cannot conceal reuse. The `+-` identities
must have zero overlap with every R5 `-+` split.

For each family define:

```text
f     = (a+b-c) mod 7
i_add = (a+b) mod 7
i_sub = (b-c) mod 7
```

The inverse is frozen as:

```text
a = (f-i_sub) mod 7
c = (i_add-f) mod 7
b = (i_add+i_sub-f) mod 7
```

Thus `(a,b,c) -> (f,i_add,i_sub)` is bijective. Freeze this deterministic split:

```text
split_code = (f + 2*i_add + 3*i_sub) mod 7
validation: split_code == 0
reserve:    split_code == 1
train:      split_code in {2,3,4,5,6}
```

Required enumeration receipts:

- 343 unique families and 343 unique `(f,i_add,i_sub)` triples;
- train/validation/reserve counts `245/49/49`;
- each split has uniform marginals for `f`, `i_add`, and `i_sub`:
  `35` per class in train and `7` per class in validation/reserve;
- family and row IDs are disjoint across splits;
- partition digest is over sorted unsalted base IDs and the literal split rule.

Canonical partition serialization is frozen as:

```text
canonical_json({
  packet: "DH-S2-R6-R7",
  split_rule: "(f+2*i_add+3*i_sub)%7",
  splits: {train: sorted_ids, validation: sorted_ids, reserve: sorted_ids}
})
```

Required SHA256 receipts are:

- partition: `8425fa0161ac6682d4644e7350bc4d80d41fe498de03b8313a64364218f5fa52`;
- train IDs: `f019acf6bcad4e9d007cc1301b7ee3082d6d176ebbfde88863a269d2522addc1`;
- validation IDs: `c07618d26bd3011701f02c8e9bcc23cdb8e0b7995870a3097c374922ef53d20e`;
- reserve IDs: `7d75bf9c3f8601157669f7411d6d0048110a75a3ecff1f9e3c19760f8a9addfb`.

No model receives `split_code`, split identity, intermediate truth, family IDs,
or labels as input.

## Shortcut Threat Model

Canaries are lookup rules fitted on the train split and then frozen. Ties choose
the lowest label. All relevant keys are present in train.

For validation and reserve, separately and for both query rows, require:

- query-only lookup: exactly `7/49`;
- every one-literal-plus-query lookup: exactly `7/49`;
- every two-literal-plus-query lookup: exactly `0/49`;
- full three-literal symbolic solver: exactly `49/49`.

An evaluation-only lookup fitted on validation or reserve labels can recover
`49/49` from any two literals because the split is a linear holdout plane. This
must be reported as a leakage canary, never used for model selection or training,
and is why reserve remains one-shot. The construction claim is only that no
train-fitted partial lookup predicts held-out labels above chance.

## Frozen Initialization And Configuration

Calibration constants:

- revision/phase: `stage2-r6` / `state_congruence`;
- run kind: `calibration_only`;
- seed: `821601`;
- device: `directml`; `deterministic=false`; CPU threads `4`;
- hidden/feedforward/dropout: `48/96/0.0`;
- optimizer: `DirectMLCompatibleAdamWCore`, learning rate `0.003`, betas
  `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.0`;
- intervention weight `1.0`, normalized as specified below;
- training updates: `306` for every branch;
- checkpoint interval: `17` updates; yield `1 ms`; time budget `30 minutes`;
- CPU/RAM guard values remain R5's `92/75 percent`, `1.5/2.5 GiB`, and
  `3/3` pressure/recovery samples;
- ordinary, same-value, non-self same-value, all-state, and wrong-state accuracy
  thresholds: exactly `1.0`;
- ordinary and intervention cross-entropy maximum: `0.10`;
- required ordinary predicted classes: `7`.

Initial state is read from the retained R5 checkpoint only:

- path: `runs/stage2-r5-ladder-directml-821501/checkpoints/r5-00000600-final.pt`;
- checkpoint SHA256:
  `18327E373F937D353297811DB60C7180B9B3823FE49B4E7CDB09EE27D6EFD489`;
- checkpoint schema/packet/global round: `1` / `DH-S2-R5.1` / `600`;
- frozen R5 config SHA256:
  `4B64023623B3DE1AC23D06E718ADA1C9BB639085CF95688A4EAC1FED03D5DCA7`;
- checkpoint config digest:
  `159adaaa5bbc6854ac862f071f6709a3a722140a9f711ed5195bb1eaa17d391a`;
- checkpoint partition digest:
  `1701144f08fe7b7ee72b30b210c4922a14a3a4da69694ebb092db0c2cbace2d1`;
- state key: `rung1_state`;
- required state digest:
  `9c133c17c9dfcfe8bffbd8b71ea1a7d3ecd724dd037792be2fc6acc9e6b426ce`.

Do not inherit R5 fixed-query models or any optimizer state. Construct every R6
branch separately from a cloned CPU copy of `rung1_state`, then create a fresh
empty optimizer. Receipts must prove equal state digests, distinct model and
parameter objects, empty optimizer state before the first update, equal parameter
counts, and no parameter-storage sharing.

Any change to these constants after seeing R6 output is a new packet.

## R6 Model Interface

Implement R6 in new modules. Do not broaden R5's hard-coded `-+` model semantics.
The R6 model must retain state-dict key and tensor-shape compatibility with
`ArithmeticComposerModel` so the R5 Rung 1 state loads exactly, but its forward
contract accepts only three-literal `+-` rows.

The model exposes two internal operations:

1. produce the query-specific first state from the row;
2. compose an explicitly supplied intermediate state in the target row's outer
   context and read the answer.

The ordinary path supplies the row's own generated first state. The intervention
path supplies a source row's generated first state. Unselected or literal states
cannot reach the answer unless a named diagnostic explicitly requests them.

## Mandatory Branch Matrix

Train these five branches separately for ADD-first and SUB-first, for ten models:

| Branch | State used by second path | Second-path target | Purpose |
| --- | --- | --- | --- |
| root | none | none | ordinary task ceiling |
| teacher | true literal state | ordinary target | privileged interface ceiling |
| self-duplicate | target's own generated state | ordinary target | matched extra-compute control |
| congruence-true | another same-value generated state | ordinary target | tested mechanism |
| mixed-counterfactual | different-value generated state | recomputed target | valid recombination control |

`self-duplicate`, `congruence-true`, and `mixed-counterfactual` are the matched
set. They must share one implementation and differ only in source indices and
second-path targets. They have identical parameters, batch order, two second
compositions, two readouts, two CE terms, normalized loss, backward graph shape,
optimizer updates, and operation receipts.

Root and teacher are diagnostic ceilings and are not compute-matched claims.
Auxiliary readout branches are not repeated in R6 because R5 already established
that readout labels do not define operational state semantics.

Root training is exactly one generated first-state path, one outer composition,
one readout, and `CE(ordinary_logits, ordinary_target)`. Teacher training is
exactly one literal-consumer path: the true intermediate literal embedding goes
directly into the outer composition, followed by one readout and the same single
CE. Teacher does not execute or optimize a generated ordinary/intervention path;
a generated state may be produced only during named report-only evaluation.

## Frozen Partner Maps

Every train query has 245 rows and exactly 35 rows per true intermediate value.
Within each value class, sort rows by unsalted base-family ID. A source map is
always `target_index -> source_index`.

For schedule index `k in 1..34`:

- self: source is the target itself;
- true: source is rank `(target_rank+k) mod 35` in the same value class;
- mixed:
  - `value_offset = 1 + ((k-1) mod 6)`;
  - `rank_offset = floor((k-1)/6)`;
  - source value is `(target_value+value_offset) mod 7`;
  - source rank is `(target_rank+rank_offset) mod 35` in that value class.

Each map must be a 245-row bijection. True maps have no self-map and preserve
every intermediate value. Mixed maps have no self-map and change every
intermediate value. At training step `s in 0..305`, use
`k = 1 + (s mod 34)`, so every map executes exactly nine times. Mapping digests,
per-row source-use counts, value transitions, and cycle counts are evidence.

Every branch uses the same fixed full batch of 245 rows on every update. Rows are
sorted by unsalted base-family ID. There is no shuffle, minibatch, sampling,
curriculum, adaptive schedule, or early stop. All 306 rounds execute regardless
of training metrics. Within each round, branch updates are sequential in this
exact order:

```text
fixed-add-root
fixed-add-teacher
fixed-add-self-duplicate
fixed-add-congruence-true
fixed-add-mixed-counterfactual
fixed-sub-root
fixed-sub-teacher
fixed-sub-self-duplicate
fixed-sub-congruence-true
fixed-sub-mixed-counterfactual
```

Inherited integer meanings are frozen as `ADD_OP=0`, `SUB_OP=1`,
`ADD_FIRST=0`, and `SUB_FIRST=1`. They must index the same R5 operator/query
embedding rows; remapping or introducing new IDs is forbidden.

## Gradient And Loss Contract

The intervention source state is not detached. The second-path loss propagates
through the target outer composer and through the selected source state's first
composer. Because every partner map is bijective, every generated state is used
once by an ordinary target and once as an intervention source per matched update.

For each matched branch:

```text
L_ordinary     = CE(ordinary_logits, ordinary_target)
L_intervention = CE(intervention_logits, intervention_target)
L_total        = (L_ordinary + 1.0 * L_intervention) / 2.0
```

Self-duplicate executes the second composition and readout twice even though its
two targets and states are identical. No cached logits may replace that call.
True uses the ordinary target because the source value is equal. Mixed uses:

```text
ADD-first: y_cf = (v_source - c_target) mod 7
SUB-first: y_cf = (a_target + v_source) mod 7
```

Tests must prove both formulas against the exact solver. False targets are an
implementation error. Report ordinary/intervention loss separately, gradient
presence and finite norms for producer/composer/readout parameter groups, source
use counts, and forward/backward elapsed time. Exact operation counters must show
matched equality; wall time is descriptive.

Operation counts are measured only by instrumentation at the actual `_compose`
and `_logits` call sites while a training forward is active. `forward`, branch
mode, or the runtime may not declare those counts. `first_states` and
`outer_logits` establish distinct active call regions, so a missing first compose
cannot be hidden by an extra outer compose. CE terms are counted where the
corresponding loss tensors are actually created. Tests deliberately inject both
an extra readout and a balanced first-to-outer compose drift and require the
budget gate to fail.

## Evaluation Contract

Evaluation uses one frozen branch at a time with no gradients and no intermediate
truth in the ordinary path. Compute all source first states once, then insert
them into every target outer context.

For each branch and split report:

- ordinary: 49 targets;
- same-value closure: `49*7 = 343` source/target pairs;
- non-self same-value closure: `343-49 = 294` pairs;
- all-state counterfactual closure: `49*49 = 2401` pairs;
- wrong-state counterfactual closure: `2401-343 = 2058` pairs.

Pair direction is always target outer context plus source intermediate state.
Counterfactual labels use the source true intermediate value and the target's
outer literal exactly as in the training formulas. Record pair IDs, source and
target family IDs, source value, target answer, counterfactual answer,
predictions, class counts, accuracy, CE, and matrix digests.

Teacher closures use source literal states. All other branches use their own
generated source states. Literal-state injection into generated-state branches
is report-only and cannot satisfy a gate.

A branch's semantic accuracy gate requires:

- ordinary `49/49` and all seven predicted classes;
- same-value `343/343`;
- non-self same-value `294/294`;
- all-state counterfactual `2401/2401`;
- wrong-state counterfactual `2058/2058`;
- finite metrics, exact replay, denominators, pair direction, and source receipts.

The confidence gate separately requires each of these five explicitly named
cross-entropies to be at most `0.10`, using no averaging across matrices:

- ordinary CE over 49 rows;
- same-value CE over 343 pairs;
- non-self same-value CE over 294 pairs;
- all-state counterfactual CE over 2401 pairs;
- wrong-state counterfactual CE over 2058 pairs.

The full gate is semantic accuracy plus confidence. Congruence-true must pass the
full gate to reach reserve. Root, self-duplicate, and mixed-counterfactual block
a mechanism signal as soon as their semantic accuracy gate passes, even if any
of their CE values exceeds `0.10`. A confidence-only difference can never be
reported as state-congruence evidence.

Persisted `semantic_accuracy_passed`, `confidence_passed`, and
`full_gate_passed` values are caches, not authorities. Every decision and resume
recomputes denominators, correct counts, predicted-class and class-histogram
receipts, five CEs, pair direction, exact replay, pair lengths, and intervention
matrix digest from the raw cohort fields, then requires the cached booleans to
match. Reserve uses the same derivation and never trusts its saved summaries.

## Causal Decision And Reserve Gate

Evaluate all ten validation branches in memory, then atomically persist one
complete validation cohort. Diagnostic branches never consume reserve.

Validation disposition order:

1. any invariant failure -> `implementation_invalid`;
2. either corresponding root branch passes its semantic accuracy gate ->
   `task_ceiling`, regardless of congruence-true outcome;
3. either congruence-true branch fails its full gate ->
   `state_congruence_failed`;
4. either corresponding self-duplicate branch passes its semantic accuracy gate
   -> `control_sufficient`;
5. either corresponding mixed-counterfactual branch passes its semantic accuracy
   gate ->
   `valid_augmentation_non_specific`;
6. otherwise the true pair alone is eligible for reserve.

This strict rule requires the tested mechanism to close both fixed interfaces
while every corresponding ordinary, duplicate-compute, and valid-recombination
control still fails at least one preregistered gate.

Reserve is one cohort, not ten branch-level openings:

1. atomically write `true_reserve_state="reserve_opened"` before constructing
   either reserve batch;
2. evaluate both congruence-true query branches;
3. atomically write both results and `true_reserve_state="complete"`;
4. both must pass the full gate for disposition `state_congruence_signal`;
   otherwise disposition is `state_congruence_failed`.

Only `run_kind="calibration_only"` with reserve state exactly `unopened` may
perform step 1. Smoke may evaluate validation and may report validation
eligibility, but it must terminate with reserve unmaterialized, research
disposition `null`, and no reserve claim. Reserve state `complete` is read-only;
`reserve_opened` becomes stranded on recovery; every other state/operation
combination is `implementation_invalid`.

The complete legal state table is fail-closed:

- before validation: validation/reserve are both `unopened`; execution,
  research, validation disposition, and all cohort evidence are `null`;
- validation persisted before decision: validation is `complete`, reserve is
  `unopened`, and execution/research/validation disposition remain `null`;
- validation failure/ceiling/control: reserve is `not_opened`, execution is
  `completed`, and research exactly equals the re-derived validation decision;
- implementation failure: reserve is `not_opened`, execution is
  `implementation_invalid`, and research is `null`;
- eligible smoke: reserve is `not_opened`, execution is `completed`, research is
  `null`, and the smoke prohibition receipt is present;
- eligible calibration: reserve moves atomically from `unopened` to
  `reserve_opened`, then to `complete`; a recovered open state becomes
  `reserve_stranded` with the same execution disposition and research `null`;
- completed reserve: execution is `completed` and research is re-derived only
  from the two persisted true-branch reserve metrics.

Every other `run_kind x validation x reserve x execution x research` combination
is rejected before training or result reconstruction. Training additionally
requires the exact initial unopened state on every call.

`state_congruence_signal` is a one-seed diagnostic result, not a candidate or
learned-hierarchy claim. It authorizes only the design of the next fixed-query or
paired-query gate.

If execution stops after reserve is durably opened but before complete evidence
is written, construction must leave the ledger byte-stable and mark the state
as pending. Only after resume has loaded the SHA-bound latest checkpoint,
reconstructed its full model/optimizer/training binding, validated the complete
invariant schema and raw validation evidence, and exactly replayed validation
may it atomically change the persistent state to `reserve_stranded`. It then
publishes execution disposition `reserve_stranded`, research disposition
`null`, and exits nonzero. It may not evaluate, reopen, or rematerialize an
incomplete reserve or produce any R6 research disposition. A reserve already
persisted as complete may be materialized only for deterministic, read-only
integrity replay after checkpoint restoration; replay cannot train, select a
model, alter the ledger, or issue a second research decision. The canonical
calibration run directory is
`runs/stage2-r6-congruence-directml-821601`; calibration-only execution must
reject every other run directory. Once this packet/config identity has a
`reserve_stranded` ledger, it is permanently ineligible for a fresh run or a new
output root. Retrying requires a new packet and config identity.

Validation interventions may be recomputed before reserve opening. A completed
validation cohort must be byte/digest stable on resume.

## Recovery And Evidence

Training recovery is at-least-once. Partner schedule is derived solely from the
persisted stage step, so replay preserves the frozen map sequence. Checkpoints
contain config/partition/source/state digests, model and optimizer states,
cumulative receipts, partner counts, RNG state, ledger snapshot/digest, and
elapsed time.

Every completed validation or reserve cohort is bound to the exact ten branch
model-state digests, stage/global round, cumulative training digest, measured
operation/source-use/value-transition receipts, partner schedule, and update
sequence digest that produced it. Optimizer-state digests for all ten branches
are part of the same binding. A restored checkpoint must reproduce this
binding before a completed ledger can be interpreted. A stale or foreign ledger
is an implementation error, never reusable evidence.

Each run owns an immutable `run-instance.json` containing a random 128-bit ID,
canonical run path, packet, config digest, partition digest, and source-snapshot
digest. Before this manifest and round 0 are published, the runner copies
`pyproject.toml`, the R6 runner, and every Python module in
`src/dynamic_hierarchy` into `snapshot/`, records every SHA256 and one aggregate
digest, and verifies the actual imported package/runner paths. Resume recomputes
both frozen-copy and current-source hashes and rejects any path or byte drift.
The source digest and run-manifest digest are present directly in the ledger,
every checkpoint, latest-checkpoint receipt, status, and result. Checkpoint
paths are run-relative and must resolve directly inside that run's
`checkpoints` directory. The latest receipt binds the exact checkpoint SHA256;
absolute, escaping, foreign-run, source-mismatched, or hash-mismatched paths are
rejected.

Manifest creation requires an explicit fresh authorization issued by the runner
inside the mutex after lifecycle validation. The trainer defaults to load-only.
Even with authorization, source-snapshot creation accepts only the just-written
frozen config, and run-instance creation accepts only that config plus the
verified snapshot. Status, failure, result, attempt, ledger, checkpoint, STOP,
or any unknown evidence forbids creation.

Run-directory lifecycle checks occur while holding the named mutex. A fresh run
requires an empty directory and no `--resume`. Any nonempty directory requires
`--resume`; after initialization has published a run instance, scientific state
requires a valid latest checkpoint. An existing `result.json` is immutable and
rejects both fresh and resume execution. A completed ledger without a result may
reconstruct the result only after loading its bound checkpoint and performing
the specified read-only integrity replays; it may not train or issue a second
cohort decision.

Before round 0, resume has one narrower initialization-only state machine. A
directory containing only the exact frozen config may rebuild the source
snapshot and run instance; a complete verified snapshot may rebuild only the
run instance; a run instance without a latest receipt may publish
`initial-recovery`. Checkpoints without a run instance, a run instance without
a complete snapshot, or any status/failure/ledger/STOP/unknown evidence before
run-instance creation fail closed. Exact `.snapshot.<uuid>.tmp` directories are
app-owned incomplete copies and may be removed before rebuilding; no other path
may be deleted or ignored.

Before status publication or any optimizer update, a fresh run publishes a
verified source snapshot, run instance, and round-0 checkpoint. Every checkpoint
first writes to a unique temporary name, atomically renames to a unique final
`.pt`, computes SHA256, and only then atomically replaces `latest.json`. A crash
before the first latest receipt is recoverable only when frozen config, source
snapshot, run manifest, and optional orphan checkpoint files are the entire run
evidence. A round-0 orphan and a later orphan are both ignored safely; replay
from the last verified state writes another unique filename and cannot collide.

On recovery, validation disposition and every terminal research disposition are
re-derived from persisted metrics and current recomputed invariants. If a
pre-gate checkpoint is extended by a completed validation ledger, validation is
also replayed from the restored model and must reproduce the exact cohort digest
without changing the ledger. A mismatch is an implementation failure.

If that checkpoint is also extended by a completed reserve ledger, both true
branches are evaluated again from the restored model over the same frozen 49-row
reserve. The complete row-level reserve digest, including predictions and NLL,
must match without changing the ledger. Coherent edits to reserve predictions,
NLL, summaries, matrix digest, reserve digest, and research disposition still
fail this replay.

The gate first rebuilds all 2,401 pair identities and labels from the frozen
evaluation batch, then rebuilds every subset, histogram, correct count,
accuracy, error pair, and mean cross-entropy from predictions and per-row NLL.
Duplicate IDs, foreign family/row IDs, altered source/target values, altered
counterfactual labels, negative NLL/CE, inconsistent accuracy, or cached-boolean
contradictions are implementation errors. It then validates measured receipts
before any disposition: every branch has
exactly the configured number of updates and `245*steps` examples; calibration
has exactly 306 updates, every partner map executes nine times per matched
branch, every source row is used 306 times, observed value transitions match the
actual maps, measured first/outer/readout/CE counts match the branch path, and
the full branch-update sequence is exactly 306 repetitions of the frozen order.
Any mismatch sets execution disposition `implementation_invalid`, research
disposition `null`, leaves reserve unopened, writes failure evidence, and exits
nonzero.

Required published evidence after an authorized calibration:

- frozen config and source manifest, result, validation/reserve ledger,
  partition and canary receipt;
- partner-map and compute receipt;
- intervention matrix digests and row-level error receipts;
- runtime warnings and DirectML fallback-observability boundary;
- SHA256SUMS.

Checkpoints remain local recovery artifacts. The inherited R5 checkpoint and all
R2-R5 evidence are immutable.

## Write Scope

Allowed only after `ReadyForConstruction`:

- `src/dynamic_hierarchy/stage2_congruence_config.py`;
- `src/dynamic_hierarchy/stage2_congruence_data.py`;
- `src/dynamic_hierarchy/stage2_congruence_model.py`;
- `src/dynamic_hierarchy/stage2_congruence_runtime.py`;
- `scripts/run_stage2_congruence.py`;
- `tests/test_stage2_congruence.py` and focused DirectML additions;
- `configs/stage2-r6-smoke-cpu.json`;
- `configs/stage2-r6-smoke-directml.json`;
- `configs/stage2-r6-congruence-directml.json`;
- R6 audit/result/evidence and project status documentation.

Forbidden:

- edits to R2-R5 source semantics, configs, runs, checkpoints, or evidence;
- paired-query, router, STOP, continuous-phase, natural-language, or extra-seed
  work;
- calibration before a separate `ReadyForCalibration` audit.

## Construction Acceptance Map

Focused tests must cover:

1. exhaustive algebra, split counts/marginals, IDs, overlap, solver, and exact
   canary values;
2. all 34 true/mixed maps, bijection, no self-map, value preservation/change,
   nine-cycle coverage, and checkpoint reconstruction;
3. exact R5 checkpoint/state binding, state-dict compatibility, independent
   clones, empty fresh optimizer state, and mutation isolation;
4. both R6 fixed interfaces, ordinary/intervention replay, no-detach producer
   gradients, normalized losses, counterfactual targets, and matched operation
   counts;
5. all evaluation denominators, non-self/wrong subsets, pair direction, finite
   metrics, and exact gate order;
6. source snapshot/path/hash binding, atomic model-state-bound validation,
   stale-ledger rejection, reserve unopened on every noneligible and smoke
   outcome, reserve-open crash fail-closed only after verified restore,
   validation replay stability, and completed result immutability;
7. measured full-batch, operation, source-use, value-transition, 34-map, and
   branch-order receipts through checkpoint reconstruction;
8. duplicate/foreign pair evidence, source/target/label mutation, negative NLL,
   contradictory accuracy, coherent completed-reserve tampering, every
   pre-round-0 publication boundary, failure identity retention, round-0 orphan,
   unknown-evidence manifest rejection, CPU and DirectML smoke/gate/resume with
   no R5 mutation, and explicit fallback boundary.

Terminal construction acceptance requires focused CPU/DirectML tests, both full
project suites, compileall, both pip checks, project text validation,
`git diff --check`, R3/R4/R5 checkpoint compatibility, and an independent
`ReadyForCalibration` verdict. Smoke output stays outside canonical evidence.

## R7 Construction Gate

Before calibration:

1. implement every R2 through R6 implementation-audit repair above;
2. verify every frozen constant, mathematical count, state binding, and measured
   budget receipt;
3. pass focused CPU/DirectML and full-project terminal validation;
4. independently audit this exact R7 packet and implementation;
5. record `ReadyForCalibration` explicitly.

Before the independent verdict below, R7 under `ReadyForConstruction` authorized
repair implementation and construction validation only.

## R7 Calibration Authorization

The final independent read-only audit returned `ReadyForCalibration` after the
exact-current CPU and DirectML full suites and all static checks passed. It
authorizes exactly one run of `configs/stage2-r6-congruence-directml.json` in
`runs/stage2-r6-congruence-directml-821601`. The authorization is diagnostic
only and does not extend to another seed, retry under a new output root, learned
routing, phase, paired-query training, or a candidate claim.

## R7 Calibration Result

The single authorized DirectML run completed all `306` rounds under seed
`821601`. All frozen accounting, source, run-instance, checkpoint, model,
optimizer, operation, partner-map, source-use, transition, and branch-order
receipts passed. No Python runtime warnings were observed; DirectML fallback
observability remains `unknown`.

The research disposition is `task_ceiling`, not a positive congruence result.
The unregularized fixed SUB root already passed every semantic and confidence
gate, including `2401/2401` all-state and `2058/2058` wrong-state
counterfactuals. Both congruence-true branches passed, but both legal
mixed-counterfactual controls passed as well. Fixed ADD root remained at
`2394/2401` all-state accuracy.

The terminal reserve state is `not_opened`; no reserve rows were materialized or
evaluated. This packet authorizes no retry, extra seed, learned routing, phase,
or paired-query work. The next design must create a harder fixed-query task in
which root, self-duplicate, and mixed-counterfactual controls fail before a true
congruence branch can support a mechanism-specific claim.
