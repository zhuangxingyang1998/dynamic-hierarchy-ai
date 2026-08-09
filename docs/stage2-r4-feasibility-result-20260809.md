# Stage 2 R4 Feasibility Result

Date: `2026-08-09`

Canonical disposition: `feasibility_failed`

Reserve state: `unopened`

## Question

R3 trained repeatedly on only 84 unique base families. It fitted that small pool
but failed on unseen families. R4 asked a narrower question: does substantially
broader family coverage, with the same total number of family exposures, make
the existing A-Q-param, exact B-oracle, or D-true models learn a transferable
modulo-seven composition rule?

R4 did not train a learned router. It is a prerequisite experiment for routing.

## Frozen Experiment

The legal n=3 `-+` and n=4 `-+-` domains were partitioned before training:

| Split | Independent base families | Query rows |
| --- | ---: | ---: |
| train | 1,680 | 3,360 per complete schedule cycle |
| validation | 336 | 672 |
| final reserve | 336 | 672 if opened |

Training used 40 balanced blocks for 600 optimizer updates. The schedule made 15
complete cycles, so every training family appeared exactly 15 times. This kept
the 25,200 family exposures used by R3 while increasing unique training families
from 84 to 1,680. Train/validation overlap was zero.

The frozen validation gate required B-oracle and D-true to reach at least 50%
accuracy, at most 1.50 cross-entropy, finite metrics, and all seven predicted
classes on both validation profiles. Any failure had to leave the reserve
unopened.

## Execution

The single authorized DirectML run completed `600/600` updates in `801.5`
seconds on `privateuseone:0` / AMD Radeon RX 9060 XT.

| Model | Training accuracy | Mean training loss |
| --- | ---: | ---: |
| A-Q-param | 15.21% | 1.9462 |
| B-oracle | 15.21% | 1.9425 |
| D-true | 15.15% | 1.9444 |

For reference, seven-class random accuracy is 14.29% and uniform cross-entropy
is approximately 1.9459. Unlike R3, the expanded R4 pool was not fitted during
the frozen 600-step budget.

| Validation profile | A-Q-param | B-oracle | D-true |
| --- | ---: | ---: | ---: |
| n=3 `-+` accuracy | 9.52% | 5.95% | 3.57% |
| n=3 cross-entropy | 1.9660 | 2.0127 | 2.0198 |
| n=4 `-+-` accuracy | 12.24% | 7.82% | 7.48% |
| n=4 cross-entropy | 1.9549 | 1.9632 | 1.9721 |

All four required B-oracle/D-true gate cells failed. B-oracle nevertheless had
`exact_tree_rate=1.0`, `edge_f1=1.0`, `full_reduction_rate=1.0`, no STOP actions,
and selected-only composition on both profiles. The failure therefore occurred
after correct oracle structure selection, not because B-oracle executed the
wrong tree or stopped early.

The evaluation ledger recorded `validation_failed`. It did not generate or
evaluate any reserve family, and `reserve_opened=false` is preserved in the
canonical result.

## Interpretation

Confirmed by this run:

- Increasing unique training-family coverage by 20 times did not establish
  learning under the unchanged 600-step architecture and training budget.
- R3's small-pool memorization was not the only obstacle.
- Correct oracle structure alone did not make the current B-oracle or D-true
  implementation learn the transferable arithmetic rule in R4.

A reasonable working diagnosis is that the next bottleneck lies in the current
arithmetic representation, credit assignment through composition, curriculum,
or optimization budget. This run does not identify which one is responsible.

This result does not prove that dynamic hierarchy is ineffective, that the
architecture can never learn the task, or that a larger training budget would
necessarily fail. It also says nothing about learned query-conditioned routing,
because routing was deliberately not trained.

## Next Gate

Do not start routing or continuous phase next. First preregister an arithmetic
causal ladder that separately tests:

1. one n=2 addition or subtraction modulo seven;
2. n=3 with one fixed query and an externally supplied composition order;
3. paired-query n=3 with the existing exact oracle structure;
4. a diagnostic intermediate-node loss, kept separate from the candidate claim;
5. an exact symbolic solver/table baseline and matched train/held-out partitions.

Only after B-oracle and D-true reliably generalize at the required rung should
the project resume learned routing. This ladder is a proposed next packet, not
an authorized run.

## Evidence

Published evidence is under `evidence/stage2-r4-feasibility/`:

- result SHA256: `561A35F62608A23751E42249586A2B1EA8503AFC651E46A959AEDA851BD2869D`
- frozen config SHA256: `92788FFD78635D90C142135AE10DFE9735AA17C85C5AEEE714AC67A4A240A693`
- evaluation ledger SHA256: `C887BD2C376345E12426FBDAC813AD1D116C98DB8788C416AD1A0B61A5C5425D`
- local unpublished final-checkpoint SHA256: `D3DAD19620C594CED11E760275D03B2BCD85917197FC7F1DD7ED8D7B5023AC60`

The result's 81-file source manifest matched the working source after the run.
Zero Python warnings were observed. DirectML exposes no public fallback counter,
so fallback observability remains `unknown`.
