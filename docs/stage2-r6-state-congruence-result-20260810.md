# Stage 2 R6 State Congruence Result

Date: `2026-08-10`

Packet: `DH-S2-R6-R7`

Execution disposition: `completed`

Research disposition: `task_ceiling`

## Question

The R5 diagnostic showed that generated intermediate states causally control the
next arithmetic composition, but same-value state instances were not perfectly
interchangeable. R6 asked whether training those states to behave consistently
under every legal outer composition would remove that remaining defect.

This was a fixed-query diagnostic. It did not train a router, discover a tree,
use `STOP`, test paired queries, or test continuous phase.

## Frozen Execution

The single authorized run used DirectML seed `821601` and ten isolated branches.
Each branch completed `306` full-batch updates over the exhaustive 245-family
training split. Validation used 49 disjoint families and evaluated:

- 49 ordinary rows;
- 343 same-value state substitutions;
- 294 nonself same-value substitutions;
- 2,401 all-state counterfactual substitutions;
- 2,058 wrong-state counterfactual substitutions.

The run completed on `privateuseone:0` / AMD Radeon RX 9060 XT in `328.5`
recorded seconds. All measured update, operation, partner-map, source-use,
value-transition, branch-order, source-snapshot, run-instance, model, optimizer,
checkpoint, and inherited-artifact invariants passed. No Python runtime warnings
were observed. DirectML has no public fallback counter, so fallback observability
remains `unknown`.

## Results

The gate required exact accuracy on every semantic cohort, all seven predicted
classes, finite metrics, and cross-entropy no greater than `0.1`.

| Branch | Ordinary | All state | Wrong state | All-state CE | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed ADD root | 49/49 | 2394/2401 | 2052/2058 | 0.011113 | fail |
| fixed ADD self-duplicate | 49/49 | 2394/2401 | 2052/2058 | 0.011113 | fail |
| fixed ADD teacher | 49/49 | 2401/2401 | 2058/2058 | 0.000000 | pass |
| fixed ADD congruence-true | 49/49 | 2401/2401 | 2058/2058 | 0.000398 | pass |
| fixed ADD mixed-counterfactual | 49/49 | 2401/2401 | 2058/2058 | 0.000418 | pass |
| fixed SUB root | 49/49 | 2401/2401 | 2058/2058 | 0.003357 | pass |
| fixed SUB self-duplicate | 49/49 | 2401/2401 | 2058/2058 | 0.003357 | pass |
| fixed SUB teacher | 49/49 | 2401/2401 | 2058/2058 | 0.000006 | pass |
| fixed SUB congruence-true | 49/49 | 2401/2401 | 2058/2058 | 0.000095 | pass |
| fixed SUB mixed-counterfactual | 49/49 | 2401/2401 | 2058/2058 | 0.000096 | pass |

The decisive result is not that congruence-true passed. Fixed SUB root passed
without the proposed congruence objective, and the legal mixed-counterfactual
controls passed despite grouping different intermediate values. The benchmark
therefore cannot distinguish the intended mechanism from easier, nonspecific
solutions.

ADD and SUB were also asymmetric: ADD root retained seven all-state errors while
SUB root had none. A pass on one fixed query cannot be generalized to a common
state protocol.

## Decision

The preregistered root-control rule assigns `task_ceiling`. This is a valid
completed experiment and a negative diagnostic result, not an implementation
failure. It provides no evidence that state-congruence training caused the
observed success.

The true reserve remained `not_opened`; no reserve row was materialized or
evaluated. Do not add seeds, retry under another output root, or proceed to
learned routing, paired-query training, or continuous phase from this result.

The next experiment must first build a more discriminating fixed-query task.
Under the same parameter and compute budget, root, self-duplicate, and legal
mixed-counterfactual controls should fail while a true same-value constraint can
uniquely improve held-out operational interchangeability. Only then would
additional seeds or routing experiments be informative.

## Evidence

Public evidence is under `evidence/stage2-r6-state-congruence/`. The canonical
local run is `runs/stage2-r6-congruence-directml-821601`.

Key SHA256 values:

- source snapshot digest:
  `51e8dd5ec7f4917121e392c8d9b4ebf756bb687f420e24ac00f261fb12cb8e10`
- frozen config:
  `DB15FCE8A71DCE3F3C4B0EC2D9A3E6362B544A68E4C1A55ED62C9D57CD779D0A`
- evaluation ledger:
  `077FAC337F80C266408FEABCFABF1E85F143051A027E94F7F4D567CCC1D7570D`
- result:
  `03C767F4E7FD4820F097DCEF0A4B712484BEB35791C51429BF24317B2C722348`
- final local checkpoint:
  `2A568D03143E1D39BDED8756FACFC627B11275CAE5D440A495D4BC3E091040C8`

Checkpoints remain local and are not part of the public evidence bundle.
