# Literal 8,000-Step Candidate Result

## Immutable Decision

`runs/stage1-20260730T135229Z` is a completed candidate run and is
`failed-under-original-gate`. This document is a read-only interpretation of
its `result.json`; the run, snapshot, and artifacts remain unchanged.

The result file SHA256 is
`95F2173F1F4E5FF3C51092217D75A3EF19AD229B1A877F219A8E4CCAC8624CF7`.
Its schema is version 2. Completion is evidenced by:

- `state = completed`;
- `reason = target_steps_reached`;
- `checkpoint_recovery.current_step = 8000`;
- `config.optimizer_steps = 8000`;
- `metrics.curriculum_position.complete = true`;
- every model has 8,000 optimizer updates and 448,000 examples.

Schema 2 has no top-level `global_step`, `target_steps`, or
`run_eligible_for_aggregation`. The checker may locate its historical
completion evidence, but candidate authorization and formal aggregation require
schema 3 plus explicit `run_eligible_for_aggregation=true`.

## Observed Result

The fixed foundation gate passed at step 8,000 on 700 examples per C0/C1 task.
Content and shape audits passed, and the result recorded zero warnings.

| Split | A mean | D-true mean | D-sham mean | Original baseline condition |
| --- | ---: | ---: | ---: | --- |
| ID depth-3 skew | 0.210714 | 1.000000 | 0.520000 | pass |
| ID depth-3 balanced | 0.156429 | 1.000000 | 0.153571 | fail |
| Depth-5 skew | 0.155000 | 1.000000 | 0.200000 | fail |
| Held-out depth-3 branched | 0.147857 | 1.000000 | 0.289286 | fail |

Labels are exactly balanced, so the majority baseline is `1/7 = 0.142857`.
The original required floor is `0.172857`. Every registered D-true-over-A and
D-true-over-D-sham advantage passed, but three
`A_and_D_true_above_majority` conditions failed. Therefore
`candidate_pass=false` under the frozen original gate.

A predicted 5 to 7 distinct classes on every split/seed. It exceeded the
original baseline floor on both ID-skew evaluation seeds, but not on both seeds
of any other split. D-true reached 1.0 everywhere. This is strong evidence for
the privileged-structure intervention in this candidate, not evidence that A
learned the same deep or topology-general algorithm.

## Scientific Judgment

The original gate is not mathematically contradictory: A and D-true can both
exceed baseline while D-true retains an advantage. It is, however, mismatched
to the narrow structural-induction hypothesis when it requires A to exceed the
same floor on every OOD split. Failure of the unprivileged model on OOD data is
part of the effect the experiment is intended to measure; making that failure
an automatic veto conflates an A non-collapse check with the structural-effect
test.

The post-hoc proposal keeps every D-true advantage threshold unchanged and
requires:

- D-true above the existing majority-plus-margin floor on every required split;
- nonconstant A predictions on every required split/seed;
- A above the unchanged floor on at least one in-distribution split across all
  evaluation seeds;
- all existing foundation, overlap, shape, D-true-over-A, D-true-over-D-sham,
  and extrapolation conditions.

This proposal does not reclassify the observed run. It must be tested with the
new config, training seed `82421`, and frozen evaluation seeds `92041`,
`92051`, and foundation seed `92063`. No formal config may adopt the proposal
until that independent candidate passes. One candidate still cannot unblock
Stage 2.
