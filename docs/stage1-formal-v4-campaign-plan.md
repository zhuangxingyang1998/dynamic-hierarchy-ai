# Stage 1 Canonical Formal Campaign V4

## Decision

The original formal sequence and the prepared-but-never-launched v2/v3
campaigns are retired from formal statistics. Their state and evidence remain
read-only:

- `runs/stage1-literal-formal-sequence.json`
- `runs/stage1-formal-01-931101-20260730T172025Z`
- `runs/stage1-formal-02-931117-20260730T193056Z`
- `runs/stage1-literal-formal-v2-campaign`
- `runs/stage1-literal-formal-v2-sequence.json`
- `runs/stage1-literal-formal-v3-campaign`
- `runs/stage1-literal-formal-v3-sequence.json`

Independent pre-launch review rejected v2 because a launch receipt could
self-collide with seed-freshness scanning after coordinator restart, while
launch and resume still selected `start_stage1.ps1` from the mutable worktree.
No v2 training seed was launched. V3 fixed both reported defects, but a second
pre-launch review and runtime module-path audit proved that its canonical
coordinator still loaded `dynamic_hierarchy` through the mutable editable
worktree install because only the snapshot root, not `canonical-snapshot/src`,
was placed first on `sys.path`. No v3 training seed was launched. V4 puts both
canonical source roots first and asserts every loaded project module remains
inside them.

Seed 1 is an observed positive result from the old queue: schema 3, completed
`8000/8000`, aggregation-eligible, and formal-final complete. It is not a v4
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

## V4 Identity

The versioned config is
`configs/stage1-revised-literal-formal-confirmation-v4-directml.json`.
It preserves the passing candidate's 8,000-step literal curriculum, optimizer,
model, learning rate, data/topology, post-hoc gate, foundation gate, and paired
Bonferroni-corrected confirmation statistics. Formal evaluation remains 10,010
examples per split and evaluation seed.

Fresh seed registry:

- training: `991501`, `991511`, `991531`, `991541`, `991547`, `991567`,
  `991579`, `991589`
- final evaluation: `992501`, `992519`, `992531`
- foundation: `992549`

The twelve values are pairwise distinct and must remain absent from every
discoverable historical config/run record. The runtime scan fails closed on
overlap or unreadable evidence.

## Canonical Package

Before any run, `--prepare-only` creates
`runs/stage1-literal-formal-v4-campaign`. It contains:

- `canonical-snapshot/`, copied once from the complete current snapshot
  enumeration;
- `canonical-snapshot/campaign/environment-receipt.json`;
- `canonical-snapshot/campaign/candidate-identity.json`;
- `campaign-manifest.json`.

The snapshot manifest is regenerated after the two receipts are inserted. The
outer manifest binds its own canonical digest, full source/snapshot manifest
hashes, formal config and scientific-spec digests, candidate pins/result
identity, environment receipt, and all seed assignments. The separate atomic
state is `runs/stage1-literal-formal-v4-sequence.json`; aggregate output is
`runs/stage1-literal-formal-v4-confirmation.json`; run names begin with
`stage1-formal-v4-`.

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

Launch receipts, v4 state, aggregate output, the canonical campaign root, and
state-owned v4 run roots are explicitly excluded from the historical seed scan.
Other configs and run evidence, including all v2/v3 artifacts, remain in scope.

## Execution

Freeze only, with no training:

```powershell
.\.venv-directml\Scripts\python.exe `
  scripts\run_stage1_confirmation_sequence_v4.py --prepare-only
```

Launch or resume the serial campaign from its frozen coordinator:

```powershell
$env:DYNAMIC_HIERARCHY_PROJECT_ROOT = (Resolve-Path ".").Path
$coordinator = ".\runs\stage1-literal-formal-v4-campaign\" +
  "canonical-snapshot\scripts\run_stage1_confirmation_sequence_v4.py"
& .\.venv-directml\Scripts\python.exe $coordinator
```

The coordinator launches at most one DirectML worker. Exact user STOP remains
paused until the existing control script removes STOP. Recoverable timeout
results resume from checkpoint. Failed worker exceptions fail closed. Only
eight verified v4 results are passed to the unchanged aggregator; old queue
paths are never candidates for that list.

## Recorded Outcome

The campaign ran from `2026-07-31T00:35:29.598042Z` through
`2026-07-31T08:59:05.687389Z`. All eight seeds reached exactly `8000/8000`,
completed formal final evaluation, passed their per-run gates, and were marked
`verified`. No runtime warning was recorded.

The unchanged frozen aggregator accepted exactly those eight results and wrote
`runs/stage1-literal-formal-v4-confirmation.json`. Its decision is
`formal_confirmation_passed`, `stage2_unblocked` is `true`, and its SHA256 is
`95F4147F05CD31C1133418AF17A2F3061B37044A265D7FF0BB0BB06EBEE15631`.
An independent frozen-source recomputation produced byte-identical output.
See `docs/stage1-formal-v4-confirmation-result-20260731.md`.
