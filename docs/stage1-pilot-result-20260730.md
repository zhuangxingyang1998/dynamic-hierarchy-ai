# Stage 1 Pilot Result - 2026-07-30

## Decision

The formal Stage 1 pilot completed normally, but it did not establish a
true-structure advantage. Both A and D collapsed to the same constant output on
a larger post-hoc evaluation. Under the research protocol's stopping rule,
learned `MERGE/STOP` controller work remains blocked until a revised diagnostic
task produces a reproducible D-over-A signal.

This is a negative result for this pilot configuration. It is not evidence that
recursive composition or adaptive hierarchy cannot work in general.

## Frozen run

- Run: `runs/stage1-20260730T094624Z`
- Frozen snapshot:
  `d8c0639e5dc054f34a10f0fbe7ba532d2704d00ac69ab37fdd27b8ed9cabc103`
- Device: DirectML `privateuseone:0`, AMD Radeon RX 9060 XT
- Final reason: `time_budget_reached`
- Wall-clock budget used: `7200.25` seconds
- Final paired optimizer step: `20160`
- Effective batch size per model and step: `16`
- Examples seen by each model: `322560`
- Final checkpoint: `checkpoint-00020160.pt`
- Runtime warnings: none observed
- DirectML fallback observability: unknown

Windows per-process counters sampled during the run showed worker PID `12792`
using the GPU `Compute 0` engine at approximately 20-25 percent, with about
153 MB dedicated and 48 MB shared GPU memory. This confirms observed GPU
activity, but it does not provide a complete DirectML fallback audit.

## Recorded training metrics

| Model | Parameters | Cumulative accuracy | Mean loss |
| --- | ---: | ---: | ---: |
| A, ordinary Transformer | 79,616 | 17.8122% | 2.062862 |
| D, true-structure diagnostic | 79,232 | 17.7672% | 2.066335 |

These cumulative training metrics do not show a D advantage.

## Larger post-hoc evaluation

The run's built-in final evaluation used only 16 examples per scale, which is
too small for interpretation. A read-only CPU evaluation loaded the final
checkpoint from the frozen snapshot and evaluated 4,096 deterministic examples
at each scale. Neither models, optimizer state, checkpoints, nor run metadata
were changed.

Both models predicted output class zero for every one of the 12,288 examples.
Consequently, A and D had identical accuracy at every scale.

| Scale | Expression depth | Uniform chance | Majority-label baseline | A accuracy | D accuracy | A CE | D CE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 12.5000% | 17.3096% | 17.3096% | 17.3096% | 2.060472 | 2.063445 |
| 2 | 6 | 12.5000% | 13.3789% | 11.5234% | 11.5234% | 2.121843 | 2.113758 |
| 4 | 12 | 12.5000% | 13.0127% | 12.1582% | 12.1582% | 2.111828 | 2.103075 |

At scale 1, the constant prediction exactly matches the evaluation set's
majority-label baseline. At scales 2 and 4 it is below the majority-label
baseline. There were zero examples on which only one of A or D was correct.
The pilot therefore provides no length-extrapolation evidence for either model.

## Learnability diagnostic

A separate in-memory CPU diagnostic repeatedly trained fresh models on one
fixed batch of 64 examples using the same optimizer core and learning rate.

| Model | Initial accuracy | First checked perfect fit | Final loss |
| --- | ---: | ---: | ---: |
| A | 10.9375% | 125 updates | 0.041288 |
| D | 0.0000% | 25 updates | 0.452562 |

This verifies that gradients, parameter updates, and both model paths can
memorize a fixed batch. It does not demonstrate rule learning or
generalization. The contrast with the formal run localizes the observed failure
to learning across newly generated examples rather than a completely inert
training implementation.

## Scientific boundaries

- This is one non-deterministic DirectML seed.
- The generator produces randomly oriented skew chains, not general trees.
- The frozen run predates the active-tree structure-only API hardening.
  Independent review found no actual label, binding-value annotation, or
  intermediate-value read in the frozen D implementation.
- D still had to infer serialized variable bindings from ordinary input tokens.
  True tree structure removed expression parsing uncertainty but did not supply
  binding values or arithmetic results.
- Fixed-batch memorization is only an implementation diagnostic.
- The result does not distinguish insufficient curriculum, task imbalance,
  optimizer dynamics, model capacity, or an unsuitable D architecture.

## Required next gate

Do not start Stage 2 learned hierarchy yet. First revise Stage 1 so that the
oracle-structure diagnostic has a fair opportunity to demonstrate general rule
learning:

1. Add a curriculum with binding lookup, one operation, and progressively
   deeper composition as separate measured stages.
2. Use larger evaluation sets and multiple seeds; report per-class prediction
   distributions and paired A/D outcomes.
3. Control or stratify output-label imbalance without exposing labels to either
   model.
4. Add balanced and genuinely branched grammars plus held-out topology splits.
5. Require D to beat A reproducibly on in-distribution data and retain an
   advantage under depth extrapolation before implementing learned
   `MERGE/STOP`.
