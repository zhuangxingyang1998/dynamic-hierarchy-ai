# Stage 1 Canonical Formal Campaign V3

## Decision

The original formal sequence and the prepared-but-never-launched v2 campaign
are retired from formal statistics. Their state and evidence remain read-only:

- `runs/stage1-literal-formal-sequence.json`
- `runs/stage1-formal-01-931101-20260730T172025Z`
- `runs/stage1-formal-02-931117-20260730T193056Z`
- `runs/stage1-literal-formal-v2-campaign`
- `runs/stage1-literal-formal-v2-sequence.json`

Independent pre-launch review rejected v2 because a launch receipt could
self-collide with seed-freshness scanning after coordinator restart, while
launch and resume still selected `start_stage1.ps1` from the mutable worktree.
No v2 training seed was launched. V3 fixes both control-plane defects and uses
an entirely fresh seed registry.

Seed 1 is an observed positive result from the old queue: schema 3, completed
`8000/8000`, aggregation-eligible, and formal-final complete. It is not a v3
confirmation seed. Seed 2 stopped at `7255/8000`; its final evaluation is empty
and formal-final marker is `not_started`. It is evidence for checkpoint,
soft-STOP, and untouched-holdout recovery behavior, not statistical evidence.

The scientific reason for retirement is cross-seed provenance mismatch. Seed 1
records source/snapshot hashes
`dca072b6fe5f18a91d09885a7912c00119e3f126126f722d8bd3fb0443212a5a` /
`5dae735118518b46bd9aaafefec882481851be332d9380cc37603f9b37835e7b`.
Seed 2 records
`b2999ea78c7817847ba5f08f2849863e090a57a672f1341ab9e1e02f027905e5` /
`00945ec2f9ed56b8b9612a456411255069e5d81a7559c75f65c28b41d9625f8d`.
The preregistered aggregator requires one complete source/snapshot manifest
across all training seeds. That requirement is unchanged.

## V3 Identity

The versioned config is
`configs/stage1-revised-literal-formal-confirmation-v3-directml.json`.
It preserves the passing candidate's 8,000-step literal curriculum, optimizer,
model, learning rate, data/topology, post-hoc gate, foundation gate, and paired
Bonferroni-corrected confirmation statistics. Formal evaluation remains 10,010
examples per split and evaluation seed.

Fresh seed registry:

- training: `971401`, `971419`, `971429`, `971437`, `971449`, `971467`,
  `971483`, `971491`
- final evaluation: `981401`, `981419`, `981437`
- foundation: `981451`

The twelve values are pairwise distinct and must remain absent from every
discoverable historical config/run record. The runtime scan fails closed on
overlap or unreadable evidence.

## Canonical Package

Before any run, `--prepare-only` creates
`runs/stage1-literal-formal-v3-campaign`. It contains:

- `canonical-snapshot/`, copied once from the complete current snapshot
  enumeration;
- `canonical-snapshot/campaign/environment-receipt.json`;
- `canonical-snapshot/campaign/candidate-identity.json`;
- `campaign-manifest.json`.

The snapshot manifest is regenerated after the two receipts are inserted. The
outer manifest binds its own canonical digest, full source/snapshot manifest
hashes, formal config and scientific-spec digests, candidate pins/result
identity, environment receipt, and all seed assignments. The separate atomic
state is `runs/stage1-literal-formal-v3-sequence.json`; aggregate output is
`runs/stage1-literal-formal-v3-confirmation.json`; run names begin with
`stage1-formal-v3-`.

Every run is materialized from the same `canonical-snapshot/`. Current worktree
changes made after freezing cannot enter later seeds: the coordinator is
executed from the campaign's canonical snapshot, and launch/resume uses the
materialized run's own canonical `start_stage1.ps1`. The external DirectML
Python path is passed explicitly and recorded in each launch receipt. Canonical
or materialized corruption, missing receipts, extra campaign-root entries,
digest drift, or a result source/snapshot mismatch fails closed.

The environment receipt binds the DirectML Python executable and digest,
dependency locks and installed versions, computer identity, GPU adapter,
PNP device identity, and driver version. The canonical coordinator recomputes
that identity before every seed and fails closed on drift.

Launch receipts, v3 state, aggregate output, the canonical campaign root, and
state-owned v3 run roots are explicitly excluded from the historical seed scan.
Other configs and run evidence, including all v2 artifacts, remain in scope.

## Execution

Freeze only, with no training:

```powershell
.\.venv-directml\Scripts\python.exe `
  scripts\run_stage1_confirmation_sequence_v3.py --prepare-only
```

Launch or resume the serial campaign from its frozen coordinator:

```powershell
$env:DYNAMIC_HIERARCHY_PROJECT_ROOT = (Resolve-Path ".").Path
$coordinator = ".\runs\stage1-literal-formal-v3-campaign\" +
  "canonical-snapshot\scripts\run_stage1_confirmation_sequence_v3.py"
& .\.venv-directml\Scripts\python.exe $coordinator
```

The coordinator launches at most one DirectML worker. Exact user STOP remains
paused until the existing control script removes STOP. Recoverable timeout
results resume from checkpoint. Failed worker exceptions fail closed. Only
eight verified v3 results are passed to the unchanged aggregator; old queue
paths are never candidates for that list.
