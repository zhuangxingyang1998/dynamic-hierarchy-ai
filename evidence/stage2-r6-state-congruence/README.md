# Stage 2 R6 State Congruence Evidence

This bundle preserves the single authorized `DH-S2-R6-R7` DirectML diagnostic.

## Files

- `frozen-config.json`: the exact frozen calibration configuration.
- `run-instance.json`: the immutable random run identity and source binding.
- `source-manifest.json`: the 35-file frozen source manifest.
- `r6-evaluation-ledger.json`: row-level validation evidence and all receipts.
- `result.json`: terminal execution and research dispositions.
- `status.json`: final machine-readable run status.
- `SHA256SUMS.txt`: hashes for all six machine-readable JSON evidence files.

## Result

The run completed all 306 rounds and all implementation/accounting invariants
passed. It ended as `task_ceiling`: fixed SUB root already achieved `2401/2401`
all-state accuracy without congruence training, and both mixed-counterfactual
controls also achieved `2401/2401`.

Congruence-true branches passed, but this bundle does not support a
congruence-specific benefit. The true reserve remained `not_opened`; no reserve
rows were materialized or evaluated. No learned routing, paired queries, STOP,
or continuous phase were tested.

The 22 model checkpoints remain local and unpublished. See
`docs/stage2-r6-state-congruence-result-20260810.md` for the interpretation and
next experimental gate.
