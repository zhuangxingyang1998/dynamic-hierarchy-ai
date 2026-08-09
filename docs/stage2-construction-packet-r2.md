# Stage 2 Construction Packet R2

Packet ID: `DH-S2-R2`

Status:

- `ReadyForConstruction`: generator, model/control interfaces, tests, CPU and
  DirectML smoke runs, and one explicitly non-confirmatory calibration run.
- `NotReadyForCandidateClaim`: positive effect margins, collapse thresholds,
  and intervention thresholds require calibration evidence.
- `NotReadyForFormal`: the immutable multi-seed campaign is a later revision.
- `NotReadyForPhaseC`: continuous phase remains outside this packet.

Execution result on 2026-08-09: construction and one DirectML calibration
completed. The canonical result remains `calibration_inconclusive`: all task
models stayed near chance and B-query selected immediate `STOP` on every
evaluation row. R3 is required before any candidate claim or further training.

R2 replaces the rejected multi-expression R1 draft. That design primarily
tested query-conditioned compute allocation and still admitted a single fixed
parse. R2 makes the correct tree itself query-dependent.

## End-to-End Result

Implement a benchmark in which the identical unparenthesized expression has two
different correct trees because the query changes operator precedence. Train
and evaluate the complete control matrix under one frozen calibration config.
Produce a `calibration_only` result with data, learning, structure-intervention,
collapse, compute, provenance, warning, and backend evidence.

Calibration can reject an implementation or expose a promising effect. It
cannot mark Stage 2 scientifically passed, select a post-hoc success margin, or
launch a multi-seed campaign.

## Scientific Boundary

Stage 1 confirmed the value of externally supplied structure on one modulo-seven
task. R2 asks whether a query-aware router can learn which adjacent pair to
merge when different questions require incompatible trees over the same input.

This is narrower than a general dynamic hierarchy and is not a novelty claim.
Relevant prior work already includes
[H-Net](https://arxiv.org/abs/2507.07955),
[BLT](https://arxiv.org/abs/2412.09871),
[Gumbel Tree-LSTM](https://arxiv.org/abs/1707.02786),
[StructFormer](https://aclanthology.org/2021.acl-long.559/),
[ACT](https://arxiv.org/abs/1603.08983),
[Universal Transformer](https://arxiv.org/abs/1807.03819), and
[Mixture-of-Depths](https://arxiv.org/abs/2404.02258). R2's distinguishing
intervention is the same base input with query-dependent oracle trees, plus a
router that is query-blind in the decisive matched control.

R2 primarily validates learned `MERGE` order. It executes `STOP` and tests early
stopping, but the full-expression task normally requires all leaves to reach one
root. Variable useful stopping is a later benchmark and cannot be claimed here.

## Query-Family Generator

`Stage2PrecedenceFamilyGenerator` is the sole owner of ordinary input, query,
label, oracle tree, split/profile identity, generation counts, and hashes.

One base family is a flat modulo-seven expression with no parentheses:

```text
BOS literal operator literal operator ... operator literal
```

Operators are `ADD` and `SUB`. Every accepted expression contains both. The
same base expression produces exactly two rows:

```text
X + QUERY_ADD_FIRST
X + QUERY_SUB_FIRST
```

The oracle evaluator repeatedly reduces the highest-precedence remaining
operator; equal-precedence ties reduce leftmost first. The two query values swap
the precedence order. Every reduction emits a binary node with child IDs,
source span, and separating operator source index.

Generation order is strict:

1. Choose a declared split profile, leaf count, operator-pattern class, operator
   sequence, and all literal values without a query or target.
2. Derive both oracle trees and labels from the completed base expression.
3. Reject a family when the trees are identical or the two labels are equal.
4. Accept only when the current quota for the ordered unequal label pair is
   open.

There are 42 legal ordered unequal label pairs. Each declared
`(leaf_count, operator_pattern_class)` stratum is generated in complete blocks
of 42 families, giving exact marginal and off-diagonal joint balance without
solving or altering a leaf. A hard candidate-attempt ceiling fails closed.

An operator pattern is precedence-sensitive only when at least one `SUB`
appears before a later `ADD` (the serialized pattern contains `-+`). Mixed
patterns of the form `ADD+ SUB+` are forbidden strata: modulo-seven addition
and subtraction give the same result under both precedence queries for every
literal assignment, so their 42-pair quota can never fill. Exhaustive design
checks over leaf counts 3 through 6 confirmed that every allowed pattern reaches
all 42 unequal ordered pairs and every forbidden monotone pattern reaches zero.

Training strata use leaf counts 4, 5, and 6 and both alternating and
precedence-sensitive run-based classes. Evaluation declares separate
in-distribution strata, longer leaf counts, and held-out precedence-sensitive
operator-pattern classes. Exact serialized patterns and induced oracle shape
IDs are recorded. Training and held-out shape partitions may not overlap where
a split declares topology holdout.

## Identity, Isolation, And Canaries

`base_family_hash` covers the canonical ordinary expression identity but
excludes query, label, and oracle result. `query_row_hash` covers the family
hash plus query identity. Split exclusion occurs on `base_family_hash` before
model execution. A query row is never counted as an independent base family.

Training, heartbeat, calibration evaluation, and future final evaluation use
separate content seeds and disjoint base-family domains. Accepted/rejected
candidate counts, label-pair quotas, pattern IDs, and shape IDs remain auditable.

Mandatory data canaries are:

- exact solver agreement for both queries;
- byte-identical ordinary prefixes within every pair;
- zero train/evaluation base-family overlap;
- uniform query-label contingency;
- exact 42-pair balance in every complete stratum block;
- query-only and input-only classifier probes reported separately;
- paired counterexamples proving left-only, right-only, always-add-first, and
  always-sub-first policies are each wrong on accepted data.

## Restricted Views And Causal Path

Flat controls receive ordinary tokens, semantic positions, masks, and the real
query.

Merge models receive literal leaf source positions and the separating ordinary
operator token for each legal adjacent pair. They receive no oracle edge, shape
ID, label, alternative-query label, target schedule, intermediate value,
attempt count, or rejection metadata.

For B-query:

- the router may read left state, right state, separating operator embedding,
  query state, and normalized step index;
- the shared composer may read left state, right state, and separating operator
  embedding, but not the query;
- the answer head may read the terminal hard-path root and the real query, but
  not raw input tokens, soft action probabilities, or unselected candidate
  states.

B-noQ-router is identical, with the real query replaced by one learned constant
only at the router. Its composer and answer head are unchanged. This isolates
query access to structure selection rather than query-conditioned composition.

D-true receives a query-specific `StructureOnlyBatch` containing only oracle
topology and ordinary source indices. D-sham receives a deterministic wrong
alignment with the same topology, compose count, architecture, and
initialization. Passing complete generator truth is a runtime type error.

## MERGE And STOP Semantics

At each step, every adjacent active pair is a legal merge candidate. Adjacency
is recomputed after the selected pair is replaced by its parent. The parent
keeps ordered child references, the union source span, and the operator source
index used at that reduction.

The router scores all current merge candidates plus one global `STOP`. Ties use
the lowest stable action index. Training uses deterministic straight-through
softmax: hard argmax in the forward pass and softmax gradients in the backward
pass. Evaluation uses the same hard forward path. Gumbel noise is outside R2.

Unselected candidates may participate in the estimator's backward surrogate,
but their states, probabilities, and summaries must not reach the forward answer
head. A hard-path substitution test must prove that replacing unselected
candidates leaves forward logits byte-identical.

Early `STOP` sends the current active-node sequence to a query-aware terminal
readout and freezes the sample. The maximum merge budget is `leaf_count - 1`;
reaching one root terminates normally. Reaching the budget with multiple nodes
is an invalid state and fails closed. Zero-merge output is legal and measurable.
No ponder penalty is used in R2.

DirectML must use comparison-and-broadcast hard masks rather than
`one_hot`/scatter. CPU and DirectML use the same equations and result schema.
DirectML determinism and zero fallback remain unclaimed.

## Mandatory Control Matrix

All trainable controls consume the same family order, examples, update count,
optimizer algorithm, learning rate, and evaluation families.

| ID | Mechanism | Required interpretation |
| --- | --- | --- |
| A-Q-param | Ordinary query-aware Transformer, parameter matched to B | Standard sequence baseline |
| A-Q-flop | Ordinary query-aware Transformer sized/repeated to B's measured realized FLOPs | Realized-compute baseline |
| A-recur | Shared-block query-aware recurrence/halting without merge structure | Adaptive/recurrent compute control |
| B-query | Learned query-conditioned adjacent `MERGE/STOP` | Candidate mechanism |
| B-noQ-router | B with query hidden only from router | Input-only structure control |
| B-sham | B with deterministic content-keyed query permutation at router | Query intervention sham |
| F-stop | B modules with immediate `STOP` | No-merge control |
| F-left | B modules with deterministic leftmost merges | Fixed orientation control |
| F-right | B modules with deterministic rightmost merges | Fixed orientation control |
| F-add | B modules always prioritizing `ADD` | Fixed input policy |
| F-sub | B modules always prioritizing `SUB` | Fixed input policy |
| D-true | Query-specific oracle tree | Privileged structural diagnostic |
| D-sham | Query-specific wrong alignment | Architecture-matched structural sham |

Fixed controls reuse B's embedding, composer, and answer head interface. Where
trainable weights are shared for an intervention evaluation, the result must say
so; where a control is trained separately, its initialization and optimizer
receipt must be recorded. The best mandatory fixed control is the comparator,
not a cherry-picked subset.

A-Q-param and A-Q-flop may be two configurations because exact parameter and
FLOP matching need not be simultaneously possible. The result reports both.

## Complete Budget Accounting

For every model, record parameter count, optimizer-state bytes, input padding,
executed recurrent steps, candidate scores, candidate compositions, selected
compositions, unselected compositions used by the surrogate, router overhead,
forward and backward wall time, examples/second, and peak allocated tensor
estimate. DirectML timing synchronizes an updated parameter before the timer and
after the final optimizer write.

Chosen merge count alone is never called compute matching. A hierarchy gain
cannot be attributed to structure unless A-Q-flop, A-recur, B-noQ-router,
B-sham, and all fixed policies are present.

## Trace, Intervention, And Collapse Evidence

Preserve complete traces for a bounded fixed diagnostic family set and aggregate
the rest. Record selected action, legal actions, hard/soft score summary, source
spans, operator source index, child IDs, stop step, and final node count.

Required aggregates include:

- oracle span/edge precision, recall, F1, and exact-tree rate;
- same-family trace Jaccard distance between the two queries;
- answer accuracy and paired correctness by base family;
- performance after replacing B's trace with every fixed policy and B-sham;
- router gradient norm and parameter delta;
- immediate-stop, early-stop, full-reduction, forced-invalid, always-left,
  always-right, query-identical, and constant-action rates;
- mean/min/max actions by leaf count, pattern class, query, and oracle depth.

Different traces alone do not pass any scientific gate. The learned trace must
cause a reproducible task effect over matched alternatives.

## Construction Order

1. Implement config, generator, exact evaluator, family hashing, exclusions,
   restricted views, and canaries.
2. Implement one-step compose, deterministic straight-through selection, hard
   forward isolation, provenance, and STOP/budget semantics.
3. Implement recursive B, B-noQ-router, B-sham, and all fixed policies through
   one interface.
4. Implement A-Q-param, A-Q-flop, A-recur, D-true, D-sham, paired training,
   evaluation, checkpoints, and complete budget accounting.
5. Run focused CPU tests, focused DirectML tests, then the project terminal test
   suites. No training calibration starts until both backend contracts pass.
6. Run one frozen DirectML calibration and produce `calibration_only` evidence.

## Construction Gates

Before calibration:

- every generator invariant, canary, family-exclusion, and type restriction
  passes;
- the two queries produce different oracle trees and labels for every accepted
  family;
- hard forward isolation is proven;
- router gradients and parameter updates are finite and nonzero on CPU and
  DirectML;
- all fixed policies execute from the same merge interface;
- STOP, adjacency, tie breaking, invalid budget, and provenance are tested;
- all controls report parameters and complete compute categories;
- checkpoint resume preserves dataset position, model/control state, optimizer,
  trace counters, and at-least-once recovery disclosure.

## Frozen Calibration

The first calibration uses one DirectML worker, training seed `821101`, no
Gumbel noise, no ponder penalty, FP32, hidden size 64, four attention heads where
applicable, and a maximum of 600 optimizer steps. Every evaluation stratum uses
at least ten complete 42-family blocks. All controls are evaluated on identical
families. Runtime is capped at 30 minutes and may stop earlier on a fail-closed
error. These numbers measure feasibility and variance; they are not positive
effect thresholds.

The resource profile may use four CPU threads, a short cooperative yield, and
one DirectML worker. It retains hysteretic CPU/RAM pausing, a per-run lock,
atomic status/checkpoint replacement, explicit stop/resume, and preserved
stdout/stderr/warnings. Existing Codex, MCP, Blender, and unrelated project
processes are not duplicate workers and must not be terminated.

Calibration output is one of:

- `implementation_invalid`;
- `calibration_negative`;
- `calibration_promising`;
- `calibration_inconclusive`.

It cannot output `candidate_passed` or `formal_confirmation_passed`.

## SpecificationMissing For R3

Calibration must supply the evidence needed to freeze:

- candidate effect margins against A-Q, A-recur, B-noQ-router, B-sham, and the
  best fixed policy;
- learned-trace intervention margin and query-sensitivity threshold;
- collapse thresholds and realized-compute matching tolerance;
- candidate seed count, step budget, and allowed hyperparameter-search budget;
- formal eight-seed effect definitions, family-level aggregation,
  multiplicity family, one-sided confidence procedure, and non-inferiority
  margin if D proximity is discussed;
- at least 10,000 independent base families per required formal split/content
  seed, rather than 10,000 query rows;
- immutable snapshot, seed freshness, campaign, and holdout-attempt procedure.

Until R3 freezes these values, no positive Stage 2 claim or formal campaign is
authorized.

## Phase-C Boundary

Continuous phase requires a stable, noncollapsed B result and a separate
packet. C must differ from B only in the ordinary gate versus circular phase
mechanism and must be matched on data, parameters, optimizer, realized compute,
and all eight formal seeds. `C > B` is required for a phase claim.

## Change Order And Stop Conditions

Return `ChangeOrderRequired` when:

- any model except D receives an oracle edge, intermediate result, alternate-
  query label, rejection fact, or shape ID;
- a same-family pair no longer has identical ordinary input;
- balancing solves or mutates a leaf after choosing a desired label;
- a fixed policy ceases to have accepted counterexamples;
- unselected candidates affect forward logits;
- router gradients vanish or become nonfinite on either backend;
- DirectML needs a semantically different model rather than an equivalent
  operator rewrite;
- compute accounting omits candidate/unselected/backward/padding/router cost;
- D-true loses the structural signal, making the benchmark suspect;
- a repair changes data, architecture, loss, and budget simultaneously;
- a positive margin is selected after reading a candidate or formal holdout.

Every implementation, command, backend warning, resource-policy change, and
result decision is written to `docs/development-log.md`. README status changes
only after calibration evidence exists and must retain the candidate-only
boundary.
