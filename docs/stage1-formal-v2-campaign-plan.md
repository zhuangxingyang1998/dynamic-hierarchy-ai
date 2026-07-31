# Stage 1 Canonical Formal Campaign V2

## Decision

The original formal sequence is retired from formal statistics. Its state and
all run evidence remain read-only:

- `runs/stage1-literal-formal-sequence.json`
- `runs/stage1-formal-01-931101-20260730T172025Z`
- `runs/stage1-formal-02-931117-20260730T193056Z`

Seed 1 is an observed positive result from the old queue: schema 3, completed
`8000/8000`, aggregation-eligible, and formal-final complete. It is not a v2
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

## V2 Identity

The versioned config is
`configs/stage1-revised-literal-formal-confirmation-v2-directml.json`.
It preserves the passing candidate's 8,000-step literal curriculum, optimizer,
model, learning rate, data/topology, post-hoc gate, foundation gate, and paired
Bonferroni-corrected confirmation statistics. Formal evaluation remains 10,010
examples per split and evaluation seed.

Fresh seed registry:

- training: `951301`, `951307`, `951319`, `951331`, `951343`, `951367`,
  `951373`, `951389`
- final evaluation: `961201`, `961213`, `961231`
- foundation: `961243`

The twelve values are pairwise distinct and must remain absent from every
discoverable historical config/run record. The runtime scan fails closed on
overlap or unreadable evidence.

## Canonical Package

Before any run, `--prepare-only` creates
`runs/stage1-literal-formal-v2-campaign`. It contains:

- `canonical-snapshot/`, copied once from the complete current snapshot
  enumeration;
- `canonical-snapshot/campaign/environment-receipt.json`;
- `canonical-snapshot/campaign/candidate-identity.json`;
- `campaign-manifest.json`.

The snapshot manifest is regenerated after the two receipts are inserted. The
outer manifest binds its own canonical digest, full source/snapshot manifest
hashes, formal config and scientific-spec digests, candidate pins/result
identity, environment receipt, and all seed assignments. The separate atomic
state is `runs/stage1-literal-formal-v2-sequence.json`; aggregate output is
`runs/stage1-literal-formal-v2-confirmation.json`; run names begin with
`stage1-formal-v2-`.

Every run is materialized from the same `canonical-snapshot/`. Current worktree
changes made after freezing cannot enter later seeds. Canonical or materialized
corruption, missing receipts, extra campaign-root entries, digest drift, or a
result source/snapshot mismatch fails closed.

## Execution

Freeze only, with no training:

```powershell
.\.venv-directml\Scripts\python.exe `
  scripts\run_stage1_confirmation_sequence_v2.py --prepare-only
```

Launch or resume the serial campaign:

```powershell
.\.venv-directml\Scripts\python.exe `
  scripts\run_stage1_confirmation_sequence_v2.py
```

The coordinator launches at most one DirectML worker. Exact user STOP remains
paused until the existing control script removes STOP. Recoverable timeout
results resume from checkpoint. Failed worker exceptions fail closed. Only
eight verified v2 results are passed to the unchanged aggregator; old queue
paths are never candidates for that list.
