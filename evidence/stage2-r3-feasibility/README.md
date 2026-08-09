# Stage 2 R3 Feasibility Evidence

This bundle preserves the only authorized Stage 2 R3 DirectML feasibility
calibration.

## Files

- `frozen-config.json`: the preregistered feasibility configuration.
- `result.json`: the completed canonical result.
- `SHA256SUMS.txt`: hashes for both published files.

## Interpretation

The canonical disposition is `feasibility_failed`. B-oracle followed the
source-only oracle tree exactly and always fully reduced the expression, but
B-oracle and D-true failed all four required held-out accuracy/cross-entropy
gates. The models fit the repeatedly exposed fixed training pools and did not
generalize to disjoint family blocks.

This is not a learned-routing result. R3 routing, extra seeds, and continuous
phase were not started. See
`docs/stage2-r3-feasibility-result-20260809.md` for the complete interpretation.

The local final checkpoint is a recovery artifact and is not published. Its
recorded SHA256 is
`8F625813562743D2E4577E2925C49BE0D873E0F5D86A973F6E36402CF0A7416A`.
