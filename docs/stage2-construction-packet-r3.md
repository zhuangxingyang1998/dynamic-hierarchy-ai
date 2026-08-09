# Stage 2 Construction Packet R3

Packet ID: `DH-S2-R3`

Status:

- `Executed`: the frozen feasibility slice completed at 600/600 updates.
- `FeasibilityFailed`: B-oracle and D-true failed every required held-out gate.
- `NotReadyForRouting`: the no-STOP routing contract remains frozen but is not authorized to run.
- `NotReadyForCandidateClaim`: a single calibration seed can only pass or fail the feasibility gate.
- `NotReadyForFormal`: multi-seed routing confirmation remains outside this packet.
- `NotReadyForPhaseC`: continuous phase remains blocked until discrete routing passes.

R3 responds to the R2 calibration result. R2 left the basic task near chance and
allowed B to choose immediate `STOP` even though every benchmark row required a
complete expression value. R3 separates task learnability from route discovery
and removes `STOP` from B's learned action set on this benchmark.

## End-to-End Result

Construct and run one bounded DirectML feasibility calibration over paired
three-leaf and four-leaf precedence-query families. The run must prove whether
the exact B composer/readout can learn the task when supplied the oracle merge
order. It must also report an ordinary Transformer and the existing D-true
diagnostic on the identical family rows.

Only a preregistered feasibility pass authorizes a separate routing calibration.
A feasibility failure stops the packet after evidence and documentation are
written; it does not authorize post-hoc threshold changes, extra seeds, Phase C,
or a formal campaign.

## Scientific Questions

The feasibility slice asks:

1. Can a conventional query-aware model learn the paired modulo-seven task?
2. Can the exact B composer and answer head learn it when the correct merge order
   is forced without labels or intermediate arithmetic values?
3. Can the privileged D-true diagnostic learn the same rows?

The later routing slice asks:

4. With composition feasibility established, can query access to B's router
   improve both answer accuracy and tree accuracy over query-blind, sham, fixed,
   flat, and compute-only controls?

R3 does not test useful variable stopping. That requires a future task where a
correct answer can genuinely use different numbers of reductions.

## Assembly And Ownership

The causal chain is:

```text
frozen config
  -> fixed train-family pools
  -> paired query rows and source-only oracle trees
  -> selected phase control matrix
  -> hard full-reduction forward paths
  -> held-out family evaluation
  -> preregistered gate
  -> checkpoint, status, result, and public evidence
```

Fact ownership remains exclusive:

- `Stage2PrecedenceFamilyGenerator` owns base families, paired labels, hashes,
  and source-only oracle trees.
- `Stage2OrdinaryBatch` owns the model-visible input. It never contains labels
  beyond the training target, tree truth, intermediate arithmetic, rejection
  metadata, or generator internals.
- `StructureOnlyBatch` is the only privileged structure interface.
- `Stage2MergeClassifier` owns hard adjacent merge execution and trace receipts.
- `Stage2Trainer` owns model/control construction, optimization, split exclusion,
  gate calculation, compute receipts, and recovery state.
- `scripts/run_stage2.py` owns bounded process execution and atomic result/status
  publication.

## Data Contract

The same unparenthesized base expression `X` produces two rows: `ADD-first` and
`SUB-first`. The query changes the oracle tree and label. Base-family hashes
exclude profile name, query, label, and structure.

The feasibility profiles are frozen as:

| Role | Leaves | Operators | Shape partition |
| --- | ---: | --- | --- |
| train | 3 | `-+` | train |
| train | 4 | `-+-` | train |
| held-out evaluation | 3 | `-+` | heldout |
| held-out evaluation | 4 | `-+-` | heldout |

Each profile uses one exact 42-family block: one family for every unequal ordered
label pair. Each family produces both query rows, so each block has 84 rows and
each query has exactly six examples of every output class.

R3 generates each training block once, records its hashes, and reuses that fixed
pool across optimizer steps. Evaluation blocks are generated from independent
seeds while excluding every training and earlier evaluation family. This is
required because the complete three-leaf domain has only 343 base inputs; fresh
random generation on every update would eventually consume the held-out domain.

Reuse is an explicit calibration property, not a claim of independent examples.
Results must report unique training families, repeated optimizer exposures, and
zero train/evaluation family overlap.

## Model And Control Contract

### Feasibility phase

The trainable and evaluated controls are exactly:

- `A-Q-param`: ordinary query-aware Transformer.
- `B-oracle`: the exact R3 B composer, terminal readout, and classifier with the
  query-specific merge order supplied through `StructureOnlyBatch`.
- `D-true`: the existing privileged source-only tree diagnostic.

`B-oracle` receives no labels or intermediate values. Its forced action uses a
direct hard candidate selection; its router is not trained and cannot influence
the forward or backward composition path. This isolates composer/readout
learnability from routing.

### Routing phase

If and only if feasibility passes, the later routing matrix is:

- `A-Q-param`, `A-Q-flop`, and `A-recur`.
- `B-query`, `B-noQ-router`, `B-sham`, and `B-oracle`.
- `F-stop`, `F-left`, `F-right`, `F-add`, and `F-sub` interventions using
  `B-query` weights.
- `D-true` and `D-sham`.

### No-STOP candidate semantics

For R3 learned B models, every nonterminal step scores only currently adjacent
merge candidates. The hard argmax merge is executed, active adjacency is
recomputed, and execution continues until one root remains. Every row therefore
has exactly `leaf_count - 1` selected merges, `stop_scores = 0`,
`early_stop_rate = 0`, and `full_reduction_rate = 1`.

The `stop_router` module is absent from R3 B models, not merely masked. R2 keeps
its original STOP-capable architecture for checkpoint and result reproducibility.
`F-stop` remains an evaluation-only R3 intervention that executes zero merges;
it is not a legal learned action.

Unselected candidate states remain excluded from the hard forward answer path.
Straight-through probabilities may carry gradients only for learned B routing.
Fixed and oracle policies select their candidate directly.

## Frozen Feasibility Calibration

- Revision: `stage2-r3`
- Phase: `feasibility`
- Training seed: `821301`
- Optimizer updates: at most `600`
- Training schedule: alternate the fixed n=3 and n=4 blocks
- Families per profile: `42`
- Evaluation blocks per profile: `1`
- Backend: DirectML, FP32, `deterministic=false`
- CPU threads: `4`
- Time budget: at most `30` minutes
- Checkpoint interval: `25` updates
- Cooperative yield: `2` ms after every complete multi-model update

The feasibility gate passes only when every held-out n=3 and n=4 profile meets
all of these conditions for both `B-oracle` and `D-true`:

- accuracy `>= 0.50`;
- cross-entropy `<= 1.50`;
- all seven answer classes appear in predictions;
- all values are finite.

`A-Q-param` is reported but is not a gate condition. These thresholds are frozen
before the R3 calibration and must not be changed in response to its outcome.

The legal dispositions are:

- `feasibility_pass`: every required row passes;
- `feasibility_failed`: the run completes but at least one row fails;
- `implementation_invalid`: an invariant, test, backend, or evidence path fails;
- `calibration_incomplete`: STOP, resource, or time budget ends training early.

## Future Routing Gate

The routing thresholds are frozen now so they cannot be selected after looking
at the later result. On every required IID profile, B-query must:

- reach answer accuracy `>= 0.50` and full reduction rate `1.0`;
- exceed B-noQ-router and B-sham accuracy by at least `0.10`;
- exceed the best fixed intervention by at least `0.05`;
- reach exact-tree rate `>= 0.60`;
- keep same-family query-identical trace rate `<= 0.25`.

It must also beat A-Q-param and A-recur by at least `0.05` on at least one frozen
OOD profile. A later packet must freeze routing steps, OOD profiles, seeds, and
multiplicity handling before that calibration can run.

## Compute And Recovery

Compute receipts include parameter counts, every scored and composed candidate,
selected compositions, recurrence, padding, synchronized wall time, optimizer
state bytes, and the existing lower-bound memory estimate. R3 operation estimates
must exclude the removed B stop-router work. They remain estimates, not exact
DirectML FLOP counters.

Checkpoints remain at-least-once: a crash may replay up to one checkpoint interval.
They must include the fixed training pools or sufficient generator state and
hash evidence to reconstruct the identical pools, plus all models, optimizers,
RNG state, cumulative receipts, and evaluation state.

DirectML warning and fallback reporting remains unchanged. No Python warnings do
not prove zero fallback because no public fallback counter is available.

## Construction Order

1. Add revision/phase contracts and fail-closed validation while retaining R2.
2. Add fixed training-pool generation and checkpoint recovery.
3. Add R3 B without a stop router and source-only oracle hard selection.
4. Make model/control construction phase-specific.
5. Add gate calculation, dispositions, and complete R3 receipts.
6. Add R3 CPU and DirectML smoke configs plus the frozen calibration config.
7. Run focused tests, dual-backend smoke, then the one authorized calibration.
8. Preserve evidence and update README, research protocol, and development log.

## Acceptance And Stop Conditions

Focused acceptance must prove:

- R2 still constructs STOP-capable B and preserves its exact control matrix.
- R3 B has no stop-router parameters and always reaches one root.
- B-oracle follows each row's source-only oracle merge order exactly.
- learned B retains nonzero router gradients with hard forward selection.
- fixed train pools are reproducible, paired, balanced, reused, and disjoint from
  evaluation families after checkpoint resume.
- gate calculation fails closed on missing, nonfinite, or incomplete metrics.
- both Stage 2 CPU and DirectML smoke runs complete.

Terminal acceptance is one fresh run of the frozen R3 feasibility config from the
verified source state. Do not start the routing calibration when feasibility
fails. Do not start extra seeds, raise the budget, tune thresholds on held-out
results, introduce Phase C, or claim learned hierarchy under this packet.

## Execution Record

The frozen DirectML run `stage2-r3-feasibility-directml-821301` completed all
`600/600` updates in `662.703s` with disposition `feasibility_failed`. It used 84
unique fixed training families over 25,200 family exposures and evaluated 84
different families with zero overlap.

B-oracle followed the source-only oracle path exactly on both evaluation
profiles: edge F1, exact-tree rate, and full-reduction rate were all `1.0`; STOP
scores and early-stop rate were `0`. Despite this, held-out B-oracle accuracy was
`2.38%` for n=3 and `14.29%` for n=4. D-true reached `10.71%` and `14.29%`.
Every required accuracy/cross-entropy gate failed.

The models fit the repeated fixed pools during training but did not generalize
to held-out families. This is a task-feasibility failure, not a learned-routing
result. Per the frozen stop rule, no routing calibration, extra seed, or Phase C
run was started. The canonical report is
`docs/stage2-r3-feasibility-result-20260809.md`.
