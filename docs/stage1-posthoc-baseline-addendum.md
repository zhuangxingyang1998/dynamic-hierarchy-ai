# Post-Hoc Baseline Gate Addendum

## Status

This addendum was written after inspection of the completed 8,000-step result.
It is post-hoc and does not alter the original protocol, the original result,
or the decision recorded for `runs/stage1-20260730T135229Z`.

The old run remains `failed-under-original-gate`.

## Rationale

The registered structural question is whether correct privileged composition
produces a reproducible advantage over the unprivileged A model and the
compute-matched D-sham control. Requiring A to exceed majority plus margin on
every OOD split adds a different claim: that A itself generalizes to every
registered depth/topology shift. That is a useful stronger target, but it is
not necessary to identify a privileged-structure effect.

The original policy remains `joint_all_required_v1`. The proposed policy is
`privileged_structure_posthoc_v1`.

## Proposed Candidate Rule

All numeric thresholds remain unchanged. Across every configured evaluation
seed:

1. D-true must exceed A and D-sham by the existing category-specific margins
   on all in-distribution splits and on at least one extrapolation split.
2. D-true must exceed the exactly balanced majority baseline by the existing
   `minimum_above_majority` margin on every split marked
   `required_above_majority`.
3. A must predict more than one legal class on every required split.
4. A must exceed majority plus the unchanged margin on at least one
   in-distribution split.
5. The fixed literal foundation, content-disjointness, and shape-validity gates
   must pass.

This separates A's learning/non-collapse sanity check from the causal
structure intervention. It does not weaken any D-true effect threshold.

## Required Revalidation

The proposal is candidate-only until a fresh run completes under
`configs/stage1-revised-literal-posthoc-revalidation-directml.json`.
That config preserves the validated 8,000-step curriculum and uses:

- new training seed `82421`, drawn from a newly declared eight-seed set;
- new frozen final evaluation seeds `92041` and `92051`;
- new frozen foundation evaluation seed `92063`;
- 700 exactly balanced examples per split/seed;
- schema 3 completion and explicit aggregation eligibility.

The revalidation completed at `runs/stage1-20260730T152137Z` and passed the
declared candidate gate. The result remains candidate-only.

Formal preparation generated
`configs/stage1-revised-literal-formal-confirmation-directml.json` with eight
fresh training seeds, fresh final/foundation evaluation seeds, 10,010 examples
per split/seed, paired sample data, and corrected cross-seed statistics.
Candidate identity retains the original validated digest. A separate
compatibility digest permits only preregistered seed/scale changes while
binding every scientific model/training/gate/foundation threshold field.
No formal run was launched.
