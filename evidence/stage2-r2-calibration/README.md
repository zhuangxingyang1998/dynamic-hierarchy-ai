# Stage 2 R2 Calibration Evidence

This bundle preserves the first Stage 2 R2 DirectML calibration and its
evaluation-only telemetry repair.

## Files

- `frozen-config.json`: the calibration configuration.
- `result-invalid-structure-telemetry.json`: original completed result. Task
  metrics are valid; structure metrics are invalid because of a case-mismatched
  trace filter.
- `result-reevaluated.json`: canonical calibration result. It reused the
  unchanged final checkpoint, performed no optimizer updates, and recomputed
  structure telemetry after the tested repair.

## SHA256

| File | SHA256 |
| --- | --- |
| `frozen-config.json` | `5C66570ED1A8E4F568FFB9CAAC38B5926C969BA6AD52A3E33920DECCFED4CD36` |
| `result-invalid-structure-telemetry.json` | `2F5A6D315835A85F3AF32D4E0E8295EBCACB880CBF94DB04FDC5DDC6D16E96ED` |
| `result-reevaluated.json` | `03B9A949B2F8486B6D9F7DC40EFAFC7FFAEC22BE5E96AE8C0B13523D66911E6E` |

The local final checkpoint is a recovery artifact and is not published. Its
recorded SHA256 is
`B65E0727E4A094D45A15B005F1232C1F1A0D97F69E78A6C227C78B22D0AF321E`.

## Interpretation

The canonical disposition is `calibration_inconclusive`. All task models
remained near chance, and B-query selected immediate STOP on every evaluation
row. This bundle does not establish learned hierarchy, a positive candidate
effect, novelty, formal confirmation, or readiness for continuous phase.
