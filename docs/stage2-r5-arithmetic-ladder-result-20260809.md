# Stage 2 R5.1 Arithmetic Ladder Result

Date: `2026-08-09`

Canonical disposition: `fixed_query_failed`

Last opened rung: fixed-query recursion

Paired-query rung: not created

## Question

R4 executed the supplied oracle tree correctly but did not learn transferable
modulo-seven arithmetic. R5.1 separated that failure into three preregistered
rungs:

1. fit every binary addition and subtraction fact;
2. reuse the same arithmetic state for a fixed ADD-first or SUB-first recursion;
3. only if both fixed queries pass, bind both queries in one paired model and
   intervene on its tree.

This is a one-seed diagnostic. It does not train a router.

## Frozen Experiment

Rung 1 used all 98 ordered binary facts. Rungs 2 and 3 shared one exhaustive
294-family `-+` domain, salted before training into 210 train, 42 validation, and
42 one-shot reserve families. Fixed ADD, fixed SUB, and paired-query rows use the
same family membership.

Every opened branch inherited the exact passed Rung 1 state before its optimizer
was created. Each opened rung used 300 full-batch updates. Root-only, teacher
state, true intermediate supervision, and per-query histogram-matched sham
supervision were trained as separate branches.

Two independent read-only reviews initially rejected construction on recovery,
sham-matching, and answer-versus-structure gate defects. After those defects and
their regression tests were repaired, both reviews returned
`ReadyForCalibration`. The calibration was then run exactly once.

## Execution

The frozen DirectML run completed 600 total rounds in `107.672s` on
`privateuseone:0` / AMD Radeon RX 9060 XT. It emitted zero Python warnings.
DirectML exposes no public fallback counter, so fallback observability remains
`unknown`.

Rung 1 passed all gates:

| Branch | Fit accuracy | Cross-entropy | B-core bridge |
| --- | ---: | ---: | ---: |
| binary root | 98/98 (100%) | 0.0000263 | <= `1e-5` |

The eight Rung 2 branches then completed 300 updates each:

| Query and branch | Validation | CE | Intermediate readout | Reserve |
| --- | ---: | ---: | ---: | --- |
| ADD root | 41/42 (97.62%) | 0.0976 | 100% | closed |
| ADD teacher state | 42/42 (100%) | 0.0000 | 90.48% | 42/42 (100%) |
| ADD aux-true | 41/42 (97.62%) | 0.0495 | 100% | closed |
| ADD aux-sham | 41/42 (97.62%) | 0.0420 | 0% | closed |
| SUB root | 40/42 (95.24%) | 0.2323 | 88.10% | closed |
| SUB teacher state | 42/42 (100%) | 0.0000050 | 95.24% | 42/42 (100%) |
| SUB aux-true | 41/42 (97.62%) | 0.0601 | 100% | closed |
| SUB aux-sham | 42/42 (100%) | 0.0015 | 0% | 42/42 (100%) |

The fixed root gate required zero errors on both queries. ADD root missed one
validation family and SUB root missed two, so both failed before reserve. The
runtime therefore emitted `fixed_query_failed` and did not create or train any
paired-query branch.

## Interpretation

Confirmed by this run:

- The shared composer can fit all 98 binary modulo-seven facts under this budget.
- The exact Rung 1 state was inherited by every fixed-query branch.
- Root-only recursive reuse did not reach the preregistered zero-error gate on
  either fixed query.
- Supplying the exact first-merge value as the next state made both queries pass
  validation and reserve perfectly.
- Correct intermediate classification alone did not rescue either aux-true
  branch: both decoded the intermediate value perfectly but still missed one
  final answer.

The strongest working diagnosis is narrower than R4's: the model can learn
binary facts, and the downstream computation works when given a canonical value
state, but its self-generated first-merge state is not yet reliably usable by the
second composition. This is consistent with a state-representation or recursive
credit-assignment problem. It is not a unique causal proof.

The SUB aux-sham branch passed while SUB aux-true did not. In one seed this is a
diagnostic warning, not evidence that false supervision is helpful. It prevents
claiming a clean auxiliary-supervision effect and argues against tuning from
this run.

## Decision

Do not start learned routing, continuous phase, extra seeds, or the paired-query
rung. The next construction packet should isolate whether a learned merge state
is operationally equivalent to the canonical literal state. Suitable controls
include direct state alignment, a matched sham alignment, and intervention on
the state consumed by the second merge. Those are proposed directions, not an
authorized run.

## Evidence

Published evidence is under `evidence/stage2-r5-arithmetic-ladder/`:

- result SHA256: `3A56C05A3A8566A3E1E0AFEE1628329B7DCC83DD0C37EC336D67053F103C3B1B`
- frozen config SHA256: `4B64023623B3DE1AC23D06E718ADA1C9BB639085CF95688A4EAC1FED03D5DCA7`
- evaluation ledger SHA256: `8847E7BD0F1098A930B9C8DD725D2AA0DCBD197B91E1FED90286132AE7961ACD`
- local unpublished final-checkpoint SHA256: `18327E373F937D353297811DB60C7180B9B3823FE49B4E7CDB09EE27D6EFD489`

The local checkpoint remains a recovery artifact and is not published.
