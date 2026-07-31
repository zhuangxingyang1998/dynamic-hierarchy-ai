# Revised Stage 1 Protocol Addendum

This addendum narrows the implementation after the negative 2026-07-30 pilot.
It does not modify `docs/research-protocol.md`.

## P0 Corrections

1. The prediction domain is exactly the prime field `Z/7Z`, not the token
   vocabulary. Labels and logits therefore have seven legal classes.
2. Exact label balance is algebraic. Structure, leaves, and operators are
   sampled before targets; one binding is solved with a modular inverse.
3. D-true is a privileged-structure diagnostic, not an upper bound.
4. D-sham is an architecture-, initialization-, optimizer-, and
   update-matched negative control with deterministic source misalignment.
5. Shape IDs and content hashes make topology and example overlap auditable.
6. Node count, maximum tree depth, merge count, and compose module calls are
   distinct measurements.

## Confirmation Rule

Smoke and learning-gate runs establish execution and basic non-collapse only.
They are candidate-only regardless of their score.

Formal confirmation requires the configured eight independent training seeds:

`931101, 931117, 931127, 931151, 931163, 931181, 931193, 931211`.

Each training seed is evaluated on every named split with every configured
content seed and at least 10,000 exactly balanced examples per split/content
seed. Every per-run candidate condition must pass, and the eventual
cross-training-seed aggregation must preserve the declared thresholds. A
single training seed can never authorize Stage 2. Training-seed selection is a
validated runtime override from the frozen config's declared seed list; it does
not require editing the source snapshot.

## Stopping Rule

Stage 2 remains blocked when any of these is true:

- D-true does not beat A on all required in-distribution seed results.
- D-true does not beat D-sham on all required in-distribution seed results.
- No extrapolation split satisfies both advantages across all required seeds.
- Required A or D-true results do not exceed the balanced majority baseline.
- Shape or content-overlap audits fail.
- Fewer than eight independent training seeds have completed confirmation.

No long formal run was launched while implementing this addendum.

## Literal Arithmetic Addendum

The bound-variable task is preserved under `operand_mode=bound_variable` as a
separate harder axis and is not eligible for the structural gate. The bounded
revision uses `operand_mode=literal`, direct value tokens `8..14`, and labels
`0..6`. Topology and operators precede target selection; one independent
nonzero leaf coefficient is solved modulo seven.

Literal curriculum boundaries emit first-class fixed C0/C1 evaluations. A
candidate must evaluate at least 700 exactly balanced examples for each task
and satisfy A C0 `>=0.99`, D-true C0 `>=0.99`, A C1 `>=0.98`, and D-true C1
`>=0.98`. D-sham remains architecture- and compute-matched but is intentionally
excluded from this foundation condition. Thresholds must not be weakened after
observing results.

## Evaluation Exclusion Repair

Final evaluation must actively exclude every content hash already observed in
training or accepted by an earlier evaluation split/seed. A shared exclusion
set is mandatory across the whole final evaluation. Exact per-class quotas are
maintained during rejection sampling, all exclusion categories are counted,
and exhaustion of the configured deterministic attempt limit fails closed.
The overlap audit is retained only as an independent terminal check.

Formal STOP, timeout, and training-failure exits do not run final evaluation and
consume no formal holdout hashes. Only exact target completion may start formal
final evaluation. A successful final appends hashes to a checkpointed
historical-final set; any recovery uses that set without regenerating the
holdout. A checkpoint with final data but no recoverable historical hashes is
not resumable.

The 8,000-step structural candidate config is candidate-only. Its stage
allocations are 200 C0, 1,600 C1 with 1:3 C0:C1 rehearsal, 1,800 depth-2,
1,800 depth-3, and 2,600 final rehearsal updates. The fresh post-hoc candidate
at `runs/stage1-20260730T152137Z` completed all 8,000 steps and passed its
declared foundation, learning, overlap, shape, and candidate gates. It
authorizes preparation only; it is not formal confirmation.

## Formal Integrity Addendum

A result is complete only after exactly the configured optimizer steps with
`target_steps_reached`, complete curriculum position, and exact A/D-true/D-sham
update and example counts. STOP and timeout results are incomplete; exceptions
are failed. Incomplete formal runs save training evidence only. A started formal
final is one-time: recovery may reuse completed final state but may not generate
the holdout again.

Candidate authorization must match pinned config, source-manifest,
snapshot-manifest, and result-file digests. An arbitrary historical
`result.json` cannot authorize formal work. The aggregator independently repeats
the completion, config, source, evaluation-scale, candidate-evidence, and
per-sample-data checks for every training seed.

Every formal result embeds a nonempty snapshot manifest. Worker, sequential
checker, and aggregator read the actual snapshot, validate safe relative paths,
recompute every listed file SHA256 and both manifest hashes, and fail closed on
missing, malformed, changed, or unlisted authored files.

Candidate identity retains its immutable
`validated_experiment_spec_digest`. Candidate/formal scientific compatibility
is separately pinned by
`validated_experiment_compatibility_spec_digest`. Both canonical digests bind
scientific fields by default, including curriculum, optimizer steps, operand
mode, batch construction, models, learning rate, data, topology, gate,
foundation scale and thresholds, and confirmation statistics. The
compatibility form additionally permits the preregistered fresh confirmation
training seeds and foundation seed, alongside final-evaluation seed/scale and
runtime controls. Worker, sequential runner, and aggregator recompute it. All
schema booleans require literal JSON boolean values.

Formal evaluation reserves `941101`, `941119`, and `941129`; foundation
evaluation reserves `941149`. None appeared in pre-existing run results or
configs when the formal plan was frozen. It records per-sample three-model
correctness masks.
The sequential runner repeats the freshness scan before creating state. Formal
training, final-evaluation, and foundation seeds must be pairwise disjoint and
absent from historical nonformal config or run evidence.
The cross-training-seed gate uses preregistered one-sided paired Student-t lower
bounds with Bonferroni familywise correction at alpha `0.05`. No per-sample
data may be reconstructed from aggregate accuracy.

D-sham uses the preregistered content-keyed deterministic derangement mapping.
C0 cannot distinguish D-true from D-sham; C1 has only a two-leaf swap and one
unpermutable operator. Structural claims therefore depend on the deeper named
splits and the corrected D-true-over-D-sham effect.

## Post-Hoc Baseline Policy

The completed 8,000-step candidate remains
`failed-under-original-gate`. Its result exposed a scientific mismatch between
the privileged-structure hypothesis and the stronger original requirement that
A exceed majority plus margin on every OOD split. This is not a retroactive
pass.

`docs/stage1-posthoc-baseline-addendum.md` registers a candidate-only policy
that keeps all D-true thresholds unchanged, requires D-true above baseline on
every required split, requires A nonconstant on every required split, and
requires A above the unchanged floor on at least one in-distribution split.
The fresh revalidation passed. The legacy formal config remains blocked; the
new pinned formal-confirmation config is prepared but has not been launched.
