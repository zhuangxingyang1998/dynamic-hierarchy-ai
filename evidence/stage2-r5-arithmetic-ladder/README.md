# Stage 2 R5.1 Arithmetic Ladder Evidence

This bundle preserves the single authorized Stage 2 R5.1 DirectML diagnostic.

## Files

- `frozen-config.json`: the preregistered R5.1 configuration.
- `result.json`: the completed canonical result.
- `r5-evaluation-ledger.json`: the persistent validation/reserve record.
- `SHA256SUMS.txt`: hashes for the three published evidence files.

## Interpretation

The canonical disposition is `fixed_query_failed`. Rung 1 fitted all 98 binary
addition/subtraction facts. In Rung 2, fixed ADD root scored 41/42 and fixed SUB
root scored 40/42 on validation, so neither root branch opened reserve and the
paired-query rung was not created.

Both teacher-state branches scored 42/42 on validation and reserve when the
correct first-merge value was supplied. This is consistent with a problem in the
self-generated recursive state, but one seed does not identify a unique cause.
Learned routing, extra seeds, and continuous phase were not started. See
`docs/stage2-r5-arithmetic-ladder-result-20260809.md` for the full interpretation.

The local final checkpoint is a recovery artifact and is not published. Its
recorded SHA256 is
`18327E373F937D353297811DB60C7180B9B3823FE49B4E7CDB09EE27D6EFD489`.
