# Stage 2 R5 State Diagnostic Packet R1

Packet ID: `DH-S2-R5D-R1`

Status:

- `ReadyForConstruction`: the bounded read-only diagnostic is frozen below.
- `NotTraining`: no parameter, optimizer, scheduler, or random seed update is legal.
- `NotReserveEvaluation`: no reserve batch may be materialized or scored.
- `PostHocDiagnosticOnly`: results may narrow R6 design but cannot amend R5.1.

## End-to-End Result

Load the retained R5.1 final checkpoint on CPU, reproduce every fixed-query
validation result, expose the first generated merge state, and causally replace
that state before the second merge. Emit one machine-readable report that says
whether information is merely decodable from the state or operationally usable
by the next composition. Use that report to freeze an R6 construction packet.

## Assembly And Ownership

```text
R5.1 frozen config + final checkpoint + common family split
  -> fail-closed checkpoint verifier
  -> fixed ADD/SUB validation batches only
  -> exact first-state replay
  -> state geometry and causal interventions
  -> post-hoc evidence report
  -> R6 design packet
```

- `ArithmeticLadderData` remains the sole family/split owner.
- `ArithmeticComposerModel` remains the sole arithmetic state/composition owner.
- The diagnostic may read model state dictionaries but must not construct an
  optimizer or call backward.
- The original R5.1 run directory is immutable. Hash its frozen config, result,
  ledger, and final checkpoint before and after the diagnostic.
- Diagnostic output is written outside the run directory.

## Input Freeze

- Canonical run: `runs/stage2-r5-ladder-directml-821501`
- Required disposition: `fixed_query_failed`
- Required global round: `600`
- Required checkpoint SHA256:
  `18327E373F937D353297811DB60C7180B9B3823FE49B4E7CDB09EE27D6EFD489`
- Required partition digest:
  `1701144f08fe7b7ee72b30b210c4922a14a3a4da69694ebb092db0c2cbace2d1`
- Legal split: `validation`
- Legal rows: 42 fixed ADD plus 42 fixed SUB for each matching branch
- Legal branches: root, teacher, aux-true, and aux-sham for each fixed query
- Paired models must be absent.

The diagnostic must fail if any frozen receipt differs. It must not call
`ArithmeticLadderData.batch(..., "reserve")` under any condition.

## Replay And Interventions

For every legal branch, reproduce the model's first state and second composition
with the model's own embeddings, composer, and readout. The hand replay must
match the ordinary `forward` logits within `1e-6`. Replacing the first state with
the true literal embedding must match the existing teacher API within `1e-6`.

Record these answer paths without changing weights:

- `learned`: the generated first state.
- `true_canonical`: the literal embedding of the exact intermediate value.
- `readout_canonical`: the literal embedding selected by the model's own
  intermediate readout.
- `nearest_canonical`: the cosine-nearest literal embedding.
- `same_label_transplant`: another row's generated state with the same exact
  intermediate label, using a deterministic within-label rotation with no
  fixed points.
- `sham_state_transplant`: a bijection over all validation states whose assigned
  labels are a deterministic histogram-preserving derangement of the true
  intermediate labels.
- `sham_canonical`: the literal embedding of that same deranged label vector.

For sham paths, report accuracy against both the real answer and the arithmetic
counterfactual implied by the substituted intermediate value. A state path is
not called causal merely because its ordinary accuracy changes.

## State Geometry

For each branch report:

- intermediate readout accuracy;
- cosine-nearest and Euclidean-nearest literal retrieval accuracy;
- mean cosine similarity and Euclidean distance to the true literal embedding;
- mean true-versus-best-wrong cosine margin;
- per-label row counts;
- row-level receipts for every ordinary error and whether each intervention
  repairs, preserves, or changes it.

Geometry is descriptive. No threshold is preregistered and no probe is fitted.

## Terminal Gates

The diagnostic is `diagnostic_complete` only if:

1. all eight fixed branches and exactly 42 validation rows per branch are present;
2. the canonical runtime path reproduces the ledger's integer correct-row count
   exactly and
   cross-entropy within `1e-4` across the CPU/DirectML boundary;
3. ordinary and teacher replay differences are both at most `1e-6`;
4. same-label transplant has no self-map and sham transplant is bijective,
   histogram-preserving, and changes every label;
5. all metrics are finite and every required row receipt is emitted;
6. no gradients or optimizer updates occur;
7. both reserve batches remain unmaterialized;
8. all four canonical run-artifact hashes are unchanged.

Any failed invariant yields `implementation_invalid`; partial diagnostics cannot
be used to design R6.

## Write Scope

Allowed:

- one new state-diagnostic module and runner;
- focused CPU tests;
- this packet, one diagnostic result/evidence bundle, the R6 construction packet,
  README/protocol/development-log updates.

Forbidden:

- edits to R2-R5 training semantics or frozen evidence;
- reserve access, training, tuning, extra seeds, paired-query construction,
  learned routing, or continuous phase;
- overwriting any file inside the canonical R5.1 run directory.
