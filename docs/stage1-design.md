# Revised Stage 1 Design

## Scientific Boundary

Revised Stage 1 compares three classifiers on identical examples:

- **A** is an ordinary Transformer with no tree input.
- **D-true** is a privileged-structure diagnostic. It receives only tree
  topology and source-token indices through `StructureOnlyBatch`.
- **D-sham** has exactly D-true's architecture, initial parameters, parameter
  count, optimizer, learning rate, update count, and examples. Its leaf and
  operator source alignments are deterministically permuted.

D-true is not called an upper bound. It is a diagnostic intervention whose
benefit must exceed both A and the architecture-matched sham. None of these
models learns `MERGE` or `STOP`. Campaign v4 unblocked starting Stage 2, but
Stage 2 itself remains unstarted.

All output heads have exactly seven legal classes. Labels are integers
`0..6`, and expression arithmetic is modulo the prime seven.

## Leak-Resistant Balance

Every revised training and evaluation batch is exactly balanced over the seven
classes. Generation order is fixed:

1. Choose an ordered binary-tree shape, leaf variables, and add/subtract
   operators without a target class.
2. Recompute the root's linear coefficients modulo seven.
3. Reject only an all-zero coefficient structure. A configured hard attempt
   limit fails closed.
4. After all accepted structures exist, create a uniformly permuted balanced
   target schedule.
5. Select a random variable with nonzero coefficient, sample all other
   bindings uniformly, and solve the selected binding with its modular inverse.
6. Randomize binding order and serialize the ordinary problem input.

Binding lookup is the zero-merge special case: the queried variable is set to
the balanced target directly. Target class, attempt count, and rejection
metadata never enter token input or `StructureOnlyBatch`.

## Curriculum

The checkpointed, config-driven stages are:

1. `binding_lookup` with zero merges.
2. `depth_1`.
3. `depth_2`.
4. `depth_3`.
5. `mixed_consolidation`.

Each optimizer step uses one homogeneous profile. Mixed consolidation
deterministically alternates homogeneous depth/topology batches, because the
privileged diagnostic batches one shared compose call at each postorder merge
index. The saved curriculum position includes stage, stage-local step, and next
profile. Resume verifies it against the saved global step under the existing
at-least-once recovery semantics.

## Topologies And Isolation

The generator supports:

- `skew`: exact-height chains with random left/right orientation.
- `balanced`: complete balanced binary trees.
- `branched`: random ordered trees with two non-leaf root branches, exact
  requested height, and one more merge than the height.

Each ordered shape has a canonical parenthesized representation and SHA256
shape ID. Training uses skew and balanced shapes. Named evaluation includes
in-distribution skew/balanced splits, a held-out depth-5 skew split, and a
held-out branched shape/topology split. Declared and observed shape IDs are
audited. Every example also has a content hash; final evaluation reports
training/evaluation and pairwise evaluation overlap counts.

## Fairness And Diagnostics

A, D-true, and D-sham see the same effective batch in the same order. They use
the same optimizer algorithm, learning rate, effective batch size, gradient
accumulation divisor, and update count. D-true and D-sham begin with identical
weights. Parameter counts and differences are recorded.

The structure-only API contains node IDs, child IDs, root ID, and source-token
indices only. It has no values, target, target-selection metadata, operators,
variables, or intermediate arithmetic annotations.

Telemetry records these separately:

- node count per sample;
- maximum tree depth per sample;
- combined merge-node count per sample;
- actual calls to the shared compose module per batch.

## Evaluation And Gates

Heartbeat evaluation is deliberately small and candidate-only. Final evaluation
uses named splits, fixed content seeds, exactly balanced labels, per-class label
and prediction counts, cross-entropy, accuracy, majority baseline, paired
A-only/D-true-only/both/neither outcomes, and mean/min/max/range across seeds.

A per-run candidate gate requires:

- D-true to beat A by the configured margin on every in-distribution split and
  every required evaluation seed;
- D-true to beat D-sham by the configured margin on the same splits;
- at least one depth or topology extrapolation split to satisfy both margins
  across every required evaluation seed;
- A and D-true to exceed the exactly balanced majority baseline where required;
- content-disjointness and shape-partition audits to pass.

Even a passing single run cannot unblock Stage 2. Formal confirmation preserves
eight independent training seeds and at least 10,000 examples per named
split/content-seed. Stage 2 requires an aggregate confirmation across all eight
training seeds. Smaller development runs are explicitly candidate-only.

## Runtime And Durability

The low-impact runner remains unchanged in principle: two CPU threads,
BelowNormal process priority, cooperative yield, hysteretic CPU/RAM pausing,
atomic status/checkpoint replacement, exclusive launch lock, per-run worker
mutex, and frozen source snapshots.

Snapshots include all authored Markdown documents, configs, scripts, package
sources, tests, locks, and root project files. They exclude virtual
environments, caches, generated data, and prior runs.

Checkpoint recovery is periodic at-least-once, not exact-once. A crash can
replay steps after the last complete checkpoint. The curriculum position at
that checkpoint is restored exactly and checked against its global step.

DirectML fallback observability remains `unknown`; absence of Python warnings
does not establish absence of fallback.

## Literal Arithmetic Revision

`operand_mode` is part of the frozen configuration. `bound_variable` retains
the variable-binding generator as a harder diagnostic axis and cannot pass the
structural gate. `literal` places leaf values directly in token IDs `8..14`,
while labels remain classes `0..6`.

For literal data, shape and operators are generated before the target. Each
leaf has its own linear coefficient modulo seven. After a balanced target is
chosen, one nonzero-coefficient leaf is selected uniformly and solved with a
modular inverse. Neither target nor solving metadata enters the tokens or
`StructureOnlyBatch`.

The fixed literal stages are `literal_c0`, `literal_c1`, `literal_depth_2`,
`literal_depth_3`, and `literal_rehearsal`. Every stage after C0 rehearses both
C0 and C1 with homogeneous batches. Each completed stage boundary evaluates
fixed balanced C0/C1 sets and checkpoints those results. The final foundation
condition is A and D-true C0 `>=0.99` and C1 `>=0.98`, each on at least 700
examples. D-sham is not a foundation condition.

## Active Evaluation Exclusion

Final evaluation uses one shared accepted-content set across every named split
and evaluation seed. Candidate examples are rejected before model execution
when their content hash appears in training or in that shared set. Per-class
quotas preserve exact balance even when rejection rates differ by label.

Generation reports candidate count, accepted count, training-content
exclusions, prior-evaluation exclusions, label-quota rejections, structural
attempts, and structural rejections. The configurable hard candidate limit is
`batch_size * max_evaluation_generation_attempts_per_example`. Reaching it
raises an error and produces no partial evaluation batch. The post-hoc overlap
audit remains as verification, not enforcement.

Heartbeat and literal stage-boundary/foundation evaluations add their accepted
content hashes to a checkpointed pre-final set. Final evaluation receives that
set separately from training hashes and from hashes accepted earlier during the
same final evaluation. All three domains are rejected before model execution.

Every successful final evaluation then appends its accepted hashes to
`historical_final_evaluation_content_hashes`. Checkpoint schema 3 persists this
fourth exclusion domain. Resume restores it before any subsequent final
evaluation, which rejects it separately and records
`evaluation_overlap_with_historical_final`. Schema-2 checkpoints remain
loadable only when they contain no unrecoverable historical final evaluation;
otherwise resume fails closed.

Formal final evaluation is conditional on exact target completion. STOP,
time-budget, and training-failure paths save checkpoint/status/result evidence
without invoking final generation. Before the first formal holdout read, the
worker writes an atomic one-time final-attempt marker. Completed final state can
reconstruct the terminal result without holdout reuse; an ambiguous started
attempt without complete final evidence fails closed.

## Run Eligibility

`completed` is reserved for exactly `target_steps_reached` at
`global_step == optimizer_steps`. Every model must record exactly that many
optimizer updates and `optimizer_steps * effective_batch_size` examples, and
the curriculum must be complete. STOP and time-budget exits are `incomplete`;
runtime exceptions are `failed`. Final evaluation may still be diagnostic for
nonformal work, but a formal incomplete run never touches the final holdout.
Its candidate gate is forcibly false and it is ineligible for confirmation
aggregation.

Formal launch authorization requires a completed candidate result pinned by
canonical config SHA256, source-manifest hash, snapshot-manifest hash, and exact
result-file SHA256. Operand mode, training completion, evaluation scale/seeds,
and all candidate/foundation/learning gates are then revalidated. Blank pins in
the current formal config are an intentional launch block.

Identity pins are insufficient on their own.
`validated_experiment_spec_digest` canonicalizes a fully validated config,
retains scientific fields by default, and removes only an explicit allowlist of
selected training seed, final-evaluation scale/seed, prerequisite identity,
device, and resource/runtime controls. The immutable candidate keeps this
original identity digest. A second canonical
`validated_experiment_compatibility_spec_digest` also permits the newly
declared eight-seed confirmation set and fresh foundation seed. It still binds
curriculum, optimizer steps, effective batch construction, operand mode,
models, learning rate, data, topologies, gate policy/thresholds, foundation
scale/thresholds, and confirmation statistics. Candidate, formal config,
worker, sequential runner, and aggregator must agree on the compatibility
digest.

Every schema boolean is validated by identity. Only actual JSON booleans pass;
truthy strings and integers fail closed.

Historical schema-2 results locate the observed step at
`checkpoint_recovery.current_step` and the target at
`config.optimizer_steps`. They can be described accurately as target-complete,
but they lack explicit aggregation eligibility. New candidate authorization and
formal aggregation require schema 3, top-level `global_step` and `target_steps`,
and `run_eligible_for_aggregation=true`.

## Post-Hoc Baseline Policy

The original `joint_all_required_v1` gate remains the decision rule for the
completed 8,000-step candidate. That result is
`failed-under-original-gate`.

The separately configured `privileged_structure_posthoc_v1` policy keeps all
D-true advantage thresholds. It requires D-true above majority plus margin on
every required split, A nonconstant on every required split, and A above the
unchanged floor on at least one in-distribution split. The policy is post-hoc
because it was designed after observing the old candidate. It requires a new
training seed and new frozen evaluation seeds. That revalidation passed in
`runs/stage1-20260730T152137Z`; the result is a prerequisite for preparation,
not one of the eight formal confirmation runs.

## Legacy Sequential Confirmation

Formal confirmation uses eight fresh training seeds and three fresh final
evaluation seeds, plus a fresh foundation seed. The sequential runner validates
the candidate pins and compatibility digest before creating sequence state. A
project-level coordinator mutex rejects a second runner, and a live-worker scan
rejects a different project Stage 1 worker before launch or resume. It launches
at most one DirectML worker, waits for a terminal result, then requires schema
3, exact 8,000-step completion, aggregation eligibility, formal evaluation
scale, all per-run gates, and overlap/shape audits before advancing. Any failed
or invalid result stops the sequence. Recoverable timeout results are archived
and resumed from checkpoint with a new worker-session budget. User STOP waits
for explicit Resume. Failed exceptions are nonrecoverable except for one
versioned historical bug shape: the exact incomplete-finalization
`RuntimeError` raised when the old worker called the learning gate before final
evaluation. Recovery requires schema 3, an exact frozen formal config and
digests, valid candidate pins and file-backed manifests, a matching pre-target
checkpoint, empty final evaluation, no on-disk final marker, `not_started`, and
fail-closed gate/status fields. Any mismatch remains failed. The accepted
artifact is moved byte-for-byte into `attempt-results`; STOP must first be
cleared by explicit Resume. A pid-less shell is rebuilt only after proving that
no execution evidence exists. The monitor waits for actual worker exit before
advancing. Restarting the same command resumes the recorded run; only eight
verified results reach the independent aggregator.

Seed freshness is enforced at runtime. The eight training, three final-
evaluation, and one foundation seeds must be pairwise disjoint and absent from
discoverable historical nonformal config/run evidence.

Formal provenance is file-backed. Each result embeds the snapshot manifest;
result verification and aggregation recompute every snapshot file digest and
both source/snapshot manifest hashes from the actual run directory. Missing
paths, empty manifests, malformed SHA256 values, file-set drift, or content
changes fail closed.

This original queue is now historical. Seed 1 and seed 2 were captured from
different complete source/snapshot manifests, so they cannot be combined under
the preregistered same-manifest aggregation rule. The legacy state, runs,
results, checkpoints, and STOP evidence remain read-only. Seed 1 is an observed
positive legacy-queue result; seed 2 is untouched-holdout engineering evidence.
Neither is a campaign-v4 statistical unit.

The prepared v2 campaign was never launched. Independent pre-launch review
rejected it because launch receipts could poison seed freshness after restart
and launch/resume selected a mutable worktree launcher.

The prepared v3 campaign was also never launched. It repaired those paths, but
runtime module-path auditing showed `dynamic_hierarchy` still resolved through
the editable worktree because `canonical-snapshot/src` was not first on
`sys.path`.

## Canonical Campaign V4

Campaign v4 freezes exactly one canonical snapshot before creating any run.
The snapshot contains every file selected by `snapshot_sources`, plus a
generated environment receipt and a candidate-identity receipt containing all
candidate pins and verification checks. The snapshot manifest is recomputed
after those receipts are added. An outer campaign manifest binds the full
snapshot hash, source hash, config and compatibility digests, environment and
candidate receipt digests, and all twelve seed assignments.

Every one of the eight run directories is materialized by copying that frozen
snapshot. The coordinator executes from the campaign snapshot, and launch or
resume invokes `start_stage1.ps1` from the materialized run snapshot while
passing the pinned external DirectML Python path. Both canonical Python source
roots precede the editable install, and a startup assertion rejects any project
module loaded from outside those roots. The launcher accepts the
prepared directory only when its initial file set is exactly `snapshot` plus
`campaign-receipt.json`. Source edits made after campaign creation therefore
cannot enter later seeds. Before launch, resume, or result acceptance, all
canonical and materialized file hashes plus the live GPU/driver/runtime identity
are recomputed. A result must independently pass the normal formal verifier and
match both campaign source/snapshot pins. Canonical corruption, missing
receipts, environment drift, or manifest drift fails closed.

Campaign v4 has separate config, state, aggregate output, run prefix, and
coordinator mutex. Its training seeds are `991501`, `991511`, `991531`,
`991541`, `991547`, `991567`, `991579`, and `991589`; final-evaluation seeds
are `992501`, `992519`, and `992531`; the foundation seed is `992549`. Runtime
freshness scans all discoverable config and run evidence before freezing and
requires all twelve values to be mutually distinct and historically unused.

Only eight verified campaign-v4 results are passed to the unchanged
aggregator. Its exact-seed, complete-run, full-manifest, paired-data, gate, and
same-manifest requirements remain intact.

## Confirmation Statistics

Formal per-sample evidence is a correctness bitmask: bit 0 A, bit 1 D-true, and
bit 2 D-sham. Aggregate accuracy is not substituted when these masks are
missing. For each split and training seed, the paired effects are recomputed
from the masks and averaged over the fixed evaluation seeds.

The preregistered independent unit is the training seed (`n=8`). The aggregate
gate uses one-sided paired Student-t lower confidence bounds. Familywise alpha
is `0.05`; Bonferroni correction covers every named split times both effects.
All in-distribution lower bounds must exceed their declared thresholds, and at
least one preregistered extrapolation split must exceed both thresholds. Missing
or inconsistent masks, incomplete runs, or failed content/shape audits block
the statistical gate.

## Recorded V4 Outcome

Campaign v4 completed all eight declared training seeds at exactly 8,000
paired steps. All eight per-run gates, all 304 integrity checks, all eight
Bonferroni-corrected statistical conditions, and the aggregate gate passed.
The canonical aggregate records `formal_confirmation_passed` and
`stage2_unblocked=true`; SHA256 is
`95F4147F05CD31C1133418AF17A2F3061B37044A265D7FF0BB0BB06EBEE15631`.

Across the four named splits, D-true mean accuracy was at least `99.9942%`.
The smallest corrected one-sided lower bound was `75.7506` percentage points
for D-true minus A and `36.7981` points for D-true minus D-sham. These exceed
their registered thresholds by large margins. The result confirms a
privileged fixed-structure intervention, not learned structure. Full evidence
and interpretation are in
`docs/stage1-formal-v4-confirmation-result-20260731.md`.

## D-sham Mapping

D-sham uses `content-keyed-derangement-v1`: a domain-separated SHA256 mapping
of the ordinary input tokens deterministically selects derangements for leaf
and operator source references. It preserves D-true's architecture, parameters,
optimizer updates, topology, and compose calls while avoiding one global cyclic
mapping.

C0 has one leaf and therefore no possible source permutation; D-sham is
identical to D-true and has no discriminative value. C1 has two leaves and one
operator, so only the leaf swap is possible and the operator cannot be
permuted. C1 is consequently a limited foundation diagnostic, not strong
evidence for privileged structure. Deeper multi-leaf/multi-operator splits are
the intended D-true versus D-sham comparisons.
