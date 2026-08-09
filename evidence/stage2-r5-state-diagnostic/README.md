# Stage 2 R5 State Diagnostic Evidence

This bundle preserves the bounded post-hoc diagnostic `DH-S2-R5D-R1`.

## Files

- `result.json`: all eight fixed-branch geometry, replay, transplant, row, and
  preservation receipts.
- `SHA256SUMS.txt`: the exact result hash.

## Boundary

The diagnostic used only the R5.1 validation split. It performed zero backward
calls and optimizer updates, never materialized reserve, and left the canonical
R5.1 run artifacts byte-identical.

It found that generated root states causally control counterfactual arithmetic,
while same-value states are not perfectly interchangeable. Literal embedding
substitution made root accuracy worse, so R6 targets operational congruence
rather than direct literal alignment.

This does not amend R5.1, establish learned hierarchy, or authorize R6 training.
See `docs/stage2-r5-state-diagnostic-result-20260809.md`.
