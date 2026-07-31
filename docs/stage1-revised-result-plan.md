# Revised Stage 1 Result Plan

## Execution Tiers

| Tier | Config | Purpose | Scientific status |
| --- | --- | --- | --- |
| CPU smoke | `configs/stage1-revised-smoke-cpu.json` | Import, forward/backward, curriculum, checkpoint, final-result schema | Execution only |
| DirectML smoke | `configs/stage1-revised-smoke-directml.json` | DirectML parity and device-resident optimizer state | Execution only |
| DirectML learning gate | `configs/stage1-revised-learning-directml.json` | Detect constant-output collapse before expensive work | Candidate-only |
| DirectML confirmation | `configs/stage1-revised-formal-directml.json` | One member of the eight-training-seed confirmation matrix | Candidate-only until aggregated |
| Literal candidate | `configs/stage1-revised-literal-candidate-directml.json` | Fixed-boundary C0/C1 foundation plus structural candidate gate | Candidate-only |
| Longer literal structural candidate | `configs/stage1-revised-literal-structural-candidate-directml.json` | 8,000-step C0/C1/depth curriculum with active final-eval exclusion | Completed; failed under original gate |
| Post-hoc gate revalidation | `configs/stage1-revised-literal-posthoc-revalidation-directml.json` | Fresh-seed test of the explicitly amended A sanity rule | Passed candidate-only |
| Legacy literal formal plan | `configs/stage1-revised-literal-formal-directml.json` | Older 2,000-step plan | Remains incompatible and blocked |
| Legacy literal formal queue | `configs/stage1-revised-literal-formal-confirmation-directml.json` | Original eight-seed queue | Abandoned for formal statistics; artifacts read-only |
| Canonical literal formal campaign v2 | `configs/stage1-revised-literal-formal-confirmation-v2-directml.json` | Superseded pre-launch package | Rejected by independent review; never launched |
| Canonical literal formal campaign v3 | `configs/stage1-revised-literal-formal-confirmation-v3-directml.json` | Superseded pre-launch package | Rejected by module-path audit; never launched |
| Canonical literal formal campaign v4 | `configs/stage1-revised-literal-formal-confirmation-v4-directml.json` | Frozen imports/control plane, eight fresh training seeds, 10,010 examples per split/seed | Completed; formal confirmation passed and Stage 2 is unblocked |

## Formal Matrix

The formal config preserves:

- eight independent training seeds;
- three fixed evaluation content seeds;
- 10,010 examples for every named split/content-seed;
- two in-distribution splits;
- one depth-OOD split;
- one held-out branched shape/topology split.

Campaign v4 first creates and verifies one canonical snapshot without creating
a run or launching a worker:

```powershell
.\.venv-directml\Scripts\python.exe `
  scripts\run_stage1_confirmation_sequence_v4.py --prepare-only
```

The launch/resume command is:

```powershell
$env:DYNAMIC_HIERARCHY_PROJECT_ROOT = (Resolve-Path ".").Path
$coordinator = ".\runs\stage1-literal-formal-v4-campaign\" +
  "canonical-snapshot\scripts\run_stage1_confirmation_sequence_v4.py"
& .\.venv-directml\Scripts\python.exe $coordinator
```

All eight runs are materialized from that same canonical snapshot. Re-running
the command resumes the separate v4 state file and monitors or resumes the
current run before considering another seed. A project-level coordinator mutex
rejects a second runner, and a live-worker scan rejects a different Stage 1
worker before launch or resume. It never launches two workers.
Recoverable timeout results are preserved and resumed from checkpoint. User
STOP pauses until explicit Resume. Failed exceptions are nonrecoverable in v4.
The monitor waits for worker exit, and pid-less prepared directories are
launched only when their canonical receipt is valid and no other launch evidence
exists. Any canonical corruption, integrity failure, candidate prerequisite
failure, nonrecoverable result, or per-run gate failure stops the sequence. The
aggregator still requires the exact declared seed set, identical configs except
for training seed, the same source and snapshot manifests, complete paired data,
and every single-run gate.

## Required Result Fields

Every run result must contain:

- config and frozen source/snapshot manifests;
- a nonempty embedded snapshot manifest whose file and manifest hashes
  independently recompute from the actual run snapshot;
- A, D-true, and D-sham parameter counts and update counts;
- complete curriculum position and per-stage generation accounting;
- structural rejection and acceptance rates;
- per-split/per-seed label and prediction counts, cross-entropy, accuracy, and
  paired outcomes;
- shape IDs and content-overlap audit;
- candidate gate conditions and thresholds;
- `stage2_unblocked=false` for a single run;
- resource policy, runtime warnings, and DirectML fallback observability.

Literal results additionally contain `operand_mode`,
`stage_boundary_evaluations`, and `foundation_gate`. The formal literal config
retains eight training seeds and 10,010 examples per split/content seed, but it
must not be launched unless the literal candidate has passed both its fixed
foundation gate and its per-run structural candidate gate.

The original 8,000-step result remains `failed-under-original-gate`. The fresh
post-hoc candidate at `runs/stage1-20260730T152137Z` passed and is documented
in `docs/stage1-literal-posthoc-candidate-result-20260730.md`. It is the sole
pinned prerequisite for the prepared formal config.

Formal runs additionally require:

- exact target-step completion and `run_eligible_for_aggregation=true`;
- result schema 3 with top-level `global_step` and `target_steps`;
- pinned candidate config/source/snapshot/result/original-spec digests;
- matching candidate/formal compatibility-spec digest;
- fresh evaluation seeds `992501`, `992519`, and `992531`;
- fresh foundation seed `992549`;
- runtime evidence that all 12 formal seeds are pairwise disjoint and absent
  from historical nonformal configs/runs;
- an atomic completed one-time formal-final attempt record;
- complete per-sample correctness masks for all three models;
- zero overlap with training, heartbeat, foundation, prior split, and prior
  evaluation-seed content;
- zero overlap with every final-evaluation content hash persisted earlier in
  the same run;
- a validated experiment-spec digest matching the pinned candidate and the
  current formal config;
- a cross-training-seed one-sided paired-t lower bound for each registered
  D-true effect, Bonferroni-corrected over eight split/effect comparisons.

Formal campaign-v4 training seeds are `991501`, `991511`, `991531`, `991541`,
`991547`, `991567`, `991579`, and `991589`. They are disjoint from all
campaign-v4 final/foundation values and discoverable historical evidence. The
versioned config contains all candidate pins and preserves the passing
candidate's scientific compatibility digest.

The old queue state at `runs/stage1-literal-formal-sequence.json` and both
seed-1/seed-2 run trees are immutable historical evidence. Their full
source/snapshot manifests differ, so the unchanged aggregator correctly refuses
to treat them as one confirmation campaign. Seed 1 is an observed positive old
queue result. Seed 2 never accessed formal holdout and is retained only as
engineering/recovery evidence. Neither may be included in campaign-v4
aggregation.

No campaign-v2 or campaign-v3 training launch occurred. V4 contains the repaired
canonical coordinator imports, launcher, receipt exclusions, and live
environment identity gate.

## Recorded Formal Result

Campaign v4 completed the exact eight-seed matrix. Every run reached
`8000/8000`, completed its one-time final evaluation, passed foundation and
structural gates, emitted no runtime warning, and was marked `verified`.

The aggregate passed all 304 per-run checks, six top-level conditions, and
eight Bonferroni-corrected statistical conditions. It records
`decision=formal_confirmation_passed` and `stage2_unblocked=true`. The
canonical aggregate SHA256 is
`95F4147F05CD31C1133418AF17A2F3061B37044A265D7FF0BB0BB06EBEE15631`;
an independent frozen-source rerun produced exactly the same bytes.

This closes the Stage 1 prerequisite and permits Stage 2 implementation. It
does not itself establish learned `MERGE/STOP`, autonomous segmentation, or a
general dynamic hierarchy. The full decision record is
`docs/stage1-formal-v4-confirmation-result-20260731.md`.
