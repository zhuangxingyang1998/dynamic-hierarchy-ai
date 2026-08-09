# Stage 2 Construction Packet R4

Packet ID: `DH-S2-R4`

Status:

- `Executed`: construction, validation, and the single authorized calibration are complete.
- `FeasibilityFailed`: all four required validation gate cells failed.
- `ReserveUnopened`: the final reserve was neither generated nor evaluated.
- `NotReadyForRouting`: learned query-conditioned routing remains blocked.
- `NotReadyForCandidateClaim`: one feasibility seed cannot establish a candidate effect.
- `NotReadyForFormal`: multi-seed confirmation is outside this packet.
- `NotReadyForPhaseC`: continuous phase remains blocked.

## End-to-End Result

Construct and run one bounded DirectML feasibility calibration that changes the
R3 experiment primarily by replacing its two 42-family fixed pools with an
exhaustive, balanced train/validation/reserve partition of the legal n=3 and n=4
domains.

The run must answer whether A-Q-param, the exact selected-path B-oracle, and
D-true can generalize after broader family coverage. Learned routing is not
trained. A failed validation gate leaves the final reserve unopened and stops
the packet. A validation pass authorizes exactly one evaluation of the frozen
reserve from the same final checkpoint.

## Causal Question And Retained Components

R3 showed exact oracle-tree execution but severe held-out failure after training
on only 84 unique base families. The evidence is consistent with overfitting,
but does not prove that data coverage is the only cause.

R4 retains without architectural change:

- the modulo-seven labels and paired precedence queries;
- A-Q-param, B-oracle, and D-true model specifications;
- B-oracle `forced_selected_only` source-only routing;
- no learned STOP and full reduction for B-oracle;
- AdamW-compatible optimization, learning rate `0.001`, and 600 updates;
- the R3 feasibility thresholds: accuracy `>=0.50`, cross-entropy `<=1.50`, all
  seven predicted classes, and finite metrics;
- DirectML FP32, four CPU threads, and the existing resource guard.

The intended independent variable is unique family coverage and the split
protocol. R4 does not claim perfect isolation from order effects because the
larger fixed schedule necessarily changes which family appears at each update.

## Assembly And Ownership

```text
legal family domain
  -> label-pair strata
  -> frozen disjoint train/validation/reserve blocks
  -> 40-block deterministic training schedule
  -> A-Q-param / B-oracle / D-true optimization
  -> validation gate
  -> [only on pass] one reserve evaluation
  -> disposition, checkpoint, result, and public evidence
```

Fact ownership remains unchanged:

- `Stage2PrecedenceFamilyGenerator` owns base families, labels, hashes, and
  source-only oracle structures.
- `Stage2OrdinaryBatch` owns model-visible tokens, queries, positions, and
  targets used by the loss. It contains no generator rejection metadata or
  intermediate arithmetic values.
- `StructureOnlyBatch` is the sole privileged structure interface.
- `Stage2MergeClassifier` owns hard adjacent composition and trace receipts.
- `Stage2Trainer` owns deterministic split construction, schedule selection,
  optimization, gates, recovery, and result accounting.
- `scripts/run_stage2.py` owns bounded execution and atomic status/result files.

## Frozen Family Partition

Only families whose two query labels differ are legal. The complete legal
domain sizes are already exhaustively known:

| Profile | Legal families | Families per ordered unequal label pair |
| --- | ---: | ---: |
| n=3, `-+` | 294 | 7 |
| n=4, `-+-` | 2,058 | 49 |

Each block contains exactly one family from each of the 42 unequal ordered label
pairs, hence 42 base families and 84 paired query rows.

| Split | n=3 blocks/families | n=4 blocks/families | Total families |
| --- | ---: | ---: | ---: |
| train | 5 / 210 | 35 / 1,470 | 1,680 |
| validation | 1 / 42 | 7 / 294 | 336 |
| final reserve | 1 / 42 | 7 / 294 | 336 |

The three splits exhaust all 2,352 legal families exactly once. Construction is
deterministic from the R4 seed. Every later split excludes all earlier hashes.
The runtime must fail closed on overlap, duplicate accepted families, incorrect
stratum counts, or an incomplete domain partition after the reserve is opened.

Validation and reserve are semantically distinct. Validation may determine
whether the packet continues. The reserve cannot affect training, checkpoint
selection, thresholds, or validation disposition and is generated/evaluated
only after validation passes.

## Frozen Training Schedule

The 40 train blocks are constructed once and retained through checkpoint
recovery. A seed-derived deterministic permutation fixes their order. Training
uses one 84-row block per optimizer update and cycles that same 40-block schedule
for 15 complete epochs, totaling 600 updates.

Therefore:

- every one of the 1,680 train families receives exactly 15 exposures;
- total family exposures remain 25,200, equal to R3;
- per-step batch shape and model compute remain equal to R3;
- unique training-family coverage grows from 84 to 1,680;
- schedule keys, hashes, block counts, exposure histogram, and reconstruction
  digest must be reported and checkpointed.

## Evaluation And Gates

Evaluation has two sequential gates.

### Validation gate

B-oracle and D-true must each satisfy every threshold on both
`r4_validation_n3` and `r4_validation_n4`:

- accuracy `>=0.50`;
- cross-entropy `<=1.50`;
- all seven answer classes have positive prediction counts;
- all recorded values are finite.

If any validation row fails, disposition is `feasibility_failed`,
`reserve_opened=false`, and no reserve family may be generated or evaluated.

### Reserve gate

Only after validation passes, evaluate `r4_reserve_n3` and `r4_reserve_n4` once
from the unchanged final weights. The same thresholds apply to B-oracle and
D-true. A complete R4 feasibility pass requires both validation and reserve
gates to pass.

A-Q-param is report-only because its parameter count is not exactly matched to
B-oracle. Training-pool fit is diagnostic and never substitutes for either
held-out gate.

Legal completed dispositions are:

- `feasibility_pass`: validation and reserve both pass;
- `feasibility_failed`: validation or opened reserve fails;
- `implementation_invalid`: an invariant, backend, evidence, or test path fails;
- `calibration_incomplete`: STOP, time, or resource conditions end training early.

## Frozen Calibration

- Revision: `stage2-r4`
- Phase: `feasibility`
- Seed: `821401`
- Optimizer updates: exactly `600`
- Backend: DirectML FP32, `deterministic=false`
- CPU threads: `4`
- Time budget: `30` minutes
- Checkpoint interval: `25` updates
- Cooperative yield: `2` ms per complete multi-model update
- Models: unchanged R3 feasibility specifications
- Controls: exactly `A-Q-param`, `B-oracle`, `D-true`

No threshold, family allocation, seed, model size, learning rate, or training
budget may change after inspecting validation or reserve results.

## Recovery And Evidence

Recovery remains at-least-once and may replay at most one checkpoint interval.
Checkpoints must reconstruct and compare:

- every fixed train block and query-row hash;
- the exact 40-block schedule;
- global step and next schedule position;
- unique and repeated exposure accounting;
- models, optimizers, RNG state, and cumulative receipts;
- validation/reserve opening state and accepted hashes if evaluation completed.

Result evidence must separately report training, validation, and reserve family
counts/digests, whether reserve was opened, split overlap, block balance,
exposure uniformity, structure metrics, runtime warnings, and DirectML fallback
observability. Zero Python warnings still do not prove zero fallback.

## Write Scope

Allowed production anchors:

- `src/dynamic_hierarchy/stage2_config.py`
- `src/dynamic_hierarchy/stage2_runtime.py`
- `scripts/run_stage2.py`
- R4 JSON configs
- focused Stage 2 CPU/DirectML tests
- R4 packet/result/evidence and existing project documentation

`stage2_data.py` and model architectures should remain unchanged unless a
construction-time invariant proves the existing balanced generator cannot
produce the frozen partition. Such a change requires a packet amendment before
implementation.

R2 and R3 config serialization, model construction, fixed pools, checkpoints,
smokes, and dispositions must remain backward compatible.

## Construction Order

1. Add the R4 revision/profile-block contracts without changing R2/R3 JSON.
2. Build 40 disjoint balanced train blocks and a deterministic schedule.
3. Add split-aware validation-first evaluation and reserve withholding.
4. Add result/checkpoint receipts and fail-closed overlap/domain checks.
5. Add CPU and DirectML smoke configs plus the frozen calibration config.
6. Pass focused tests, R2/R3 regression smoke, and dual-backend project suites.
7. Run exactly one fresh R4 calibration.
8. Preserve evidence, update README/protocol/log, commit, and push.

## Terminal Acceptance And Stop Conditions

Focused contracts must prove:

- R2/R3 serialized configs omit R4-only profile block fields;
- R4 has exactly 40 train blocks and 1,680 unique train families;
- each block has one family per unequal label pair and 84 paired rows;
- every train family receives the same exposure count after a complete schedule;
- checkpoint resume selects the same next block and reproduces optimizer loss;
- validation and train hashes are disjoint;
- reserve evaluation is unreachable on validation failure;
- a synthetic passing validation opens reserve exactly once;
- malformed or missing gate metrics fail closed;
- B-oracle remains exact, selected-only, no-STOP, and full-reduction;
- CPU and DirectML smoke complete with valid dispositions.

Terminal acceptance is one fresh run of the frozen R4 DirectML config from the
verified source state. If validation fails, stop without opening reserve. If an
opened reserve fails, stop without tuning. Do not start learned routing, extra
seeds, Phase C, natural-language work, or a formal campaign under this packet.

## Execution Record

The frozen seed `821401` run completed `600/600` updates with 1,680 unique train
families, 25,200 family exposures, 15 exact schedule cycles, and zero
train/validation overlap. All train-family exposure counts were exactly 15.

A-Q-param, B-oracle, and D-true remained near seven-class chance during
training. B-oracle and D-true then failed the accuracy and cross-entropy gates on
both validation profiles. B-oracle's selected-only oracle execution remained
exact, no-STOP, and full-reduction, so structural execution did not cause the
gate failure.

The persistent evaluation ledger recorded `validation_failed` before any
reserve construction. The canonical result therefore has
`reserve_opened=false` and zero reserve families, as required. The packet stops
here with disposition `feasibility_failed`; it does not authorize routing or
additional training. See `stage2-r4-feasibility-result-20260809.md` and
`../evidence/stage2-r4-feasibility/` for interpretation and exact evidence.
