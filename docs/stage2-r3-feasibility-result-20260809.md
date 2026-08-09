# Stage 2 R3 Feasibility Result

Date: `2026-08-09`

Packet: `DH-S2-R3`

Disposition: `feasibility_failed`

## Plain-Language Result

R3 asked a question that must be answered before training a dynamic router:

> If the model is handed the correct merge order, can it learn to solve unseen
> expressions at all?

The answer from this calibration is no. The three models fit the small fixed
training pools, but none generalized to different base expressions. B-oracle
executed the supplied tree exactly, so the failure cannot be attributed to
early STOP or to selecting the wrong merge path.

This does not show that dynamic hierarchy is ineffective. Learned routing was
not trained in this run. It shows that the current data curriculum is not yet a
valid foundation for testing learned routing.

## Frozen Run

- Run: `runs/stage2-r3-feasibility-directml-821301`
- Seed: `821301`
- Backend: DirectML FP32 on AMD Radeon RX 9060 XT
- Updates: `600/600`
- Elapsed time: `662.703s`
- Runtime warnings: `0`
- DirectML fallback observability: `unknown`
- Final reason: `target_steps_reached`

The run alternated two fixed balanced pools:

| Pool | Pattern | Base families | Query rows |
| --- | --- | ---: | ---: |
| n=3 | `-+` | 42 | 84 |
| n=4 | `-+-` | 42 | 84 |

There were 84 unique training families, 25,200 total family exposures, and
25,116 repeated exposures. Evaluation used 84 different families with zero
train/evaluation overlap.

## Results

Training values below are cumulative over the 600 updates. Because the same
fixed pools were reused, they measure fit to those pools rather than independent
sample generalization.

| Control | Parameters | Training accuracy | Training mean loss |
| --- | ---: | ---: | ---: |
| A-Q-param | 109,383 | 92.79% | 0.1993 |
| B-oracle | 92,489 | 98.02% | 0.0706 |
| D-true | 75,527 | 98.15% | 0.0697 |

Held-out results:

| Profile | Control | Accuracy | Cross-entropy | Seven predicted classes |
| --- | --- | ---: | ---: | --- |
| n=3 `-+` | A-Q-param | 14.29% | 7.3259 | yes |
| n=3 `-+` | B-oracle | 2.38% | 7.7687 | yes |
| n=3 `-+` | D-true | 10.71% | 5.7169 | yes |
| n=4 `-+-` | A-Q-param | 11.90% | 6.3887 | yes |
| n=4 `-+-` | B-oracle | 14.29% | 6.4323 | yes |
| n=4 `-+-` | D-true | 14.29% | 4.9530 | yes |

Seven-class chance accuracy is 14.29%. The preregistered gate required both
B-oracle and D-true to achieve accuracy at least 50%, cross-entropy at most
1.50, all seven predicted classes, and finite metrics on both profiles. All four
required control/profile rows failed.

## Structural Checks

B-oracle used `selection_path="forced_selected_only"`. On both held-out
profiles it recorded:

- exact-tree rate: `1.0`;
- edge F1: `1.0`;
- full-reduction rate: `1.0`;
- early-stop rate: `0.0`;
- STOP scores: `0`.

The oracle router had zero gradient and zero parameter delta, as intended. It
received only source topology through `StructureOnlyBatch`, not labels or
intermediate arithmetic values.

## Interpretation Boundary

Confirmed:

- The R3 no-STOP and selected-only oracle execution path works on CPU and
  DirectML smoke tests and followed the intended trees in the calibration.
- The fixed training pools can be fit by all three controls.
- The fitted models did not generalize to the held-out family blocks.
- The frozen feasibility gate failed, so routing is not authorized.

Not established:

- that learned query-conditioned routing works or fails;
- that dynamic hierarchy improves a Transformer;
- that the task is impossible with a broader curriculum;
- that continuous phase is useful;
- that these results transfer to natural language or large models.

The most plausible current diagnosis is severe overfitting caused by repeatedly
training on only 84 independent base families. It is an evidence-supported
interpretation, not a proof of the model's internal strategy.

## Next Decision

No extra R3 seed, routing run, or Phase C run is allowed. A future R4 packet
should broaden independent training-family coverage while preserving a frozen
validation split and untouched final reserve. It should first require B-oracle
and D-true to generalize to unseen families. Learned routing becomes meaningful
only after that prerequisite passes.

## Evidence

- Public result:
  `evidence/stage2-r3-feasibility/result.json`
- Frozen config:
  `evidence/stage2-r3-feasibility/frozen-config.json`
- Result SHA256:
  `A57C93DAAAA1E76FAD2E3F989A1BE43D6EE8BA18AE4FAEDDCEB4F2E1612B697B`
- Frozen config SHA256:
  `AD6323C4CEE1C39EEE63EC5AC1DF6D30EF04480C3F196463ADBD0ADE4F56885E`
- Local final checkpoint SHA256:
  `8F625813562743D2E4577E2925C49BE0D873E0F5D86A973F6E36402CF0A7416A`

The checkpoint is retained locally as a recovery artifact and is not published.
