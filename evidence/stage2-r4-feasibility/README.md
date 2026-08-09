# Stage 2 R4 Feasibility Evidence

This bundle preserves the single authorized Stage 2 R4 DirectML feasibility
calibration.

## Files

- `frozen-config.json`: the preregistered R4 configuration.
- `result.json`: the completed canonical result.
- `r4-evaluation-ledger.json`: the persistent validation/reserve opening record.
- `SHA256SUMS.txt`: hashes for the three published evidence files.

## Interpretation

The canonical disposition is `feasibility_failed`. Training used 1,680 unique
families with the same 25,200 total family exposures as R3, but A-Q-param,
B-oracle, and D-true remained near seven-class chance during training and failed
held-out validation. B-oracle still executed every oracle tree exactly and fully.

The validation failure left the 336-family final reserve unopened. This is not a
learned-routing result. Routing, extra seeds, and continuous phase were not
started. See `docs/stage2-r4-feasibility-result-20260809.md` for the complete
interpretation.

The local final checkpoint is a recovery artifact and is not published. Its
recorded SHA256 is
`D3DAD19620C594CED11E760275D03B2BCD85917197FC7F1DD7ED8D7B5023AC60`.
