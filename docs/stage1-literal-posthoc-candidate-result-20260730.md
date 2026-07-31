# Literal Post-Hoc Candidate Result

## Decision

`runs/stage1-20260730T152137Z/result.json` is a completed, passing,
candidate-only result. It authorizes formal-confirmation preparation but does
not unblock Stage 2.

Verified immutable evidence:

- result SHA256:
  `6F964AE37B7354FB0F8E3B228F4BF3BF0B34C90CE6BC820F312BECAD8B49D615`;
- schema 3, `completed`, `target_steps_reached`, `8000/8000`;
- `run_eligible_for_aggregation=true`;
- candidate config digest:
  `6dc56311c9240c23cee7b275646a8abb959e3a0d2f5223fb91c3b4b52673e735`;
- source manifest:
  `beba1ff1590f8d5335bcd4860247455a38564075c91b4fef7d74b9950ca1ebfc`;
- snapshot manifest:
  `24ab7b89da10e913f32d7630893cffd19aff20ad85302c0b04acfc18cdb3ca6e`;
- recorded experiment-spec digest:
  `3bbd419242e354584c220e8436de0d9691b9c570f81688fc3fdccb20b85efa93`.

Foundation, content-disjointness, shape validity, learning, and candidate gates
all passed with zero runtime warnings.

| Split | A mean | D-true mean | D-sham mean |
| --- | ---: | ---: | ---: |
| ID depth-3 skew | 0.247857 | 1.000000 | 0.642143 |
| ID depth-3 balanced | 0.141429 | 1.000000 | 0.178571 |
| Depth-5 skew | 0.154286 | 1.000000 | 0.284286 |
| Held-out depth-3 branched | 0.166429 | 1.000000 | 0.386429 |

This confirms the post-hoc candidate gate for one fresh training seed. It is
not an eight-seed confidence statement and is not formal confirmation.

## Formal Compatibility

The candidate's original validated digest includes its declared training and
foundation evaluation seeds. Formal confirmation must use fresh values, so the
immutable original digest is retained as an identity pin while a versioned
compatibility digest excludes only selected/declarative training seeds and
final/foundation evaluation seed values and scales.

Candidate and prepared formal config both recompute compatibility digest:

`7303d6c121b170e7f9b2ed7ba043af7ab19dc175478860aaa2e3d87be81c7025`.

Curriculum, optimizer steps, batch construction, operand mode, models,
learning rate, data, topology sets, evaluation splits, gate policy and
thresholds, foundation scale/thresholds, minimum eight-seed requirement, and
confirmation statistics remain bound.

The prepared config is
`configs/stage1-revised-literal-formal-confirmation-directml.json`. Worker
candidate-prerequisite validation passes every check. No formal run has been
started.
