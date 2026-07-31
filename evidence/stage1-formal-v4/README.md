# Stage 1 Formal V4 Public Evidence

This directory is the public reproducibility bundle for the completed
eight-training-seed Stage 1 campaign.

## Contents

- `aggregate.json` is byte-identical to the canonical aggregate. Its SHA256 is
  `95F4147F05CD31C1133418AF17A2F3061B37044A265D7FF0BB0BB06EBEE15631`.
- `campaign-manifest.json` is byte-identical to the frozen campaign manifest.
- `results/*.json` contains the eight complete per-seed result records,
  including per-sample correctness masks.
- `publication-index.json` records every original result SHA256 and every
  published-copy SHA256.

The original result files contain the local project root in a provenance path.
The published copies change only absolute paths under that root into portable
repository-relative POSIX paths. Scientific fields, masks, metrics, manifests,
and gate evidence are unchanged. The exact transformation is implemented by
`scripts/export_stage1_public_evidence.py`.

Periodic training checkpoints are intentionally omitted. They are recovery
artifacts rather than required statistical evidence and total several
gigabytes. The aggregate, per-seed result records, source/configuration, and
complete analysis code are included.
