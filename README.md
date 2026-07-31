# dynamic-hierarchy-ai

This is a Windows CPU/DirectML research harness for controlled symbolic reasoning. Stage 0 provides the ordinary Transformer baseline. Revised Stage 1 compares A, privileged-structure D-true, and architecture-matched D-sham; it does not claim that a dynamic hierarchy has been learned.

## Status

**Implemented facts**

- Deterministic synthetic tasks for repeated-symbol continuation with a per-sample random, nonempty query prefix, and unambiguous variable binding with unique variables per sample.
- A small Transformer classifier with learned token embeddings and continuous, semantic position features.
- Configuration-driven CPU and DirectML training, JSON run records, warmup-aware synchronized performance metrics, environment diagnostics, and focused tests.
- A candidate-only hierarchy controller interface that produces `MERGE`/`STOP` scores without changing the baseline computation or claiming recursive structure was learned.
- A revised Stage 1 curriculum covering binding lookup, expression depths 1/2/3, and mixed consolidation.
- Prime-modulus algebraic balancing, skew/balanced/branched shapes, canonical shape IDs, and content-overlap auditing.
- Privileged D-true and source-permuted D-sham diagnostics that receive no targets, binding values, rejection metadata, or intermediate arithmetic values.
- Canonical campaign v4 formally confirmed a fixed true-structure diagnostic advantage across eight independent training seeds; its exact aggregate unblocks starting Stage 2.

**Candidate hypotheses, not results**

- Contextual interaction plus continuous position features may help distinguish repeated values in different roles.
- Learned dynamic hierarchy, shared recursive merging, and phase-controlled decisions may improve extrapolation.
- No experiment in this repository establishes learned dynamic hierarchy or autonomous boundary discovery.
- Learned `MERGE/STOP` control is authorized by the Stage 1 aggregate but remains unstarted Stage 2 work.

**Backend boundary**

This Windows 10 machine does not use ROCm. AMD's Radeon Windows limitations state that ROCm-backed ML training is unsupported, and no OS or driver upgrade was performed. DirectML is a separate Microsoft-provided DirectX 12 compatibility backend, not AMD ROCm. Microsoft documents PyTorch with DirectML for training and inference, while the [DirectML repository](https://github.com/microsoft/DirectML) now states that the project is in maintenance mode. This project therefore pins the old pre-release `torch-directml 0.2.5.dev240914` and its PyTorch 2.4.1 dependency in an isolated environment.

The audited GPU is an AMD Radeon RX 9060 XT with `8,539,602,944` bytes (`7.95 GiB`, recorded as 8GB) of VRAM. The legacy 32-bit `MemorySize` reading of roughly 4GB is truncated and is not used.

## Quick start (CPU)

The project requires Python 3.12 because the current CPU PyTorch installation route used here supports it. Once it is installed side-by-side:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/train_stage0.py --config configs/smoke.json
.\.venv\Scripts\python.exe scripts/check_project_text.py
```

## Quick start (DirectML)

DirectML must stay isolated from the CPU environment because it resolves an older PyTorch and a different NumPy version. The CPU environment uses Torch `2.13.0+cpu` / NumPy `2.4.4`; the DirectML runtime reports Torch `2.4.1+cpu` / NumPy `2.5.1` with `torch-directml 0.2.5.dev240914`. The paired configs and optimizer algorithm are the same, but this is not a completely identical software-environment comparison.

```powershell
py -3.12 -m venv .venv-directml
.\.venv-directml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-directml\Scripts\python.exe -m pip install -r requirements-directml.lock
.\.venv-directml\Scripts\python.exe -m pip install --no-deps -e .
.\.venv-directml\Scripts\python.exe scripts/check_directml.py
.\.venv-directml\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-directml\Scripts\python.exe scripts/train_stage0.py --config configs/smoke-directml.json
```

## Benchmark

Each CPU/DirectML pair is identical except for `device`. Both use the same `DirectMLCompatibleAdamWCore`, a dense AdamW core subset rather than the complete `torch.optim.AdamW` API. Performance configs explicitly set `deterministic=false` on both backends; the strict deterministic CPU smoke remains unchanged. Warmup steps execute real forward, backward, and optimizer updates but are excluded from throughput. An updated `classifier.weight` is synchronized immediately before timing and after the final measured optimizer write; loss and gradient scalar reads happen afterward.

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_stage0.py --config configs/benchmark-sustained-short-cpu.json --repeats 3
.\.venv-directml\Scripts\python.exe scripts/benchmark_stage0.py --config configs/benchmark-sustained-short-directml.json --repeats 3
.\.venv\Scripts\python.exe scripts/benchmark_stage0.py --config configs/benchmark-throughput-long-cpu.json --repeats 3
.\.venv-directml\Scripts\python.exe scripts/benchmark_stage0.py --config configs/benchmark-throughput-long-directml.json --repeats 3
```

Current machine results, measured serially on 2026-07-30:

| Workload | Backend | Parameters | Measured / warmup | Median steps/s | Min-max steps/s | Median examples/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sustained-short | CPU | 4,830,000 | 100 / 10 | 28.140 | 27.731-28.318 | 225.122 |
| sustained-short | DirectML | 4,830,000 | 100 / 10 | 22.049 | 20.447-22.739 | 176.391 |
| throughput-long | CPU | 4,871,040 | 40 / 5 | 7.047 | 6.964-7.540 | 112.755 |
| throughput-long | DirectML | 4,871,040 | 40 / 5 | 17.478 | 16.682-18.547 | 279.651 |

For sustained-short, DirectML/CPU median throughput was `0.784x`; for throughput-long it was `2.480x`. The full min-max spread divided by the median was approximately `2.1%` and `10.4%` for CPU and DirectML sustained-short, and `8.2%` and `10.7%` for CPU and DirectML throughput-long. Each JSON retains all three samples plus median, min, max, and nearest-rank approximate p95.

The throughput-long workload uses batch 16, vocabulary 128, repeat length 32, binding pairs 16, and evaluation scale 1 only. It is a backend throughput test, not the protocol's length-extrapolation experiment. The earlier `benchmark-cpu.json` and `benchmark-directml.json` 10-step results are retained as initial short runs only and are not the final comparison.

All runs completed backward and produced finite nonzero gradient evidence. The known warning-producing unsupported `lerp` optimizer path and fused Transformer eval fastpath are avoided with the same core optimizer and standard Transformer path on both backends. This does not prove that DirectML performed no silent CPU fallback: there is no public DirectML fallback counter available here. DirectML JSON therefore records `fallback_observability.status = "unknown"` and distinguishes “no Python warnings observed” from “no fallback.”

## Future AMD GPU boundary

Windows 11 is only an official PyTorch environment entry point, not authorization for this project to train on an AMD GPU: AMD's current [Radeon Windows limitations](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html) state "No ML training support." Future AMD GPU training should prefer AMD's official Linux route and must re-check the then-current [compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html) and installation instructions before implementation. Do not reuse this CPU environment for any future GPU work.

## Revised Stage 1

The negative original pilot is preserved in
`runs/stage1-20260730T094624Z` and
[docs/stage1-pilot-result-20260730.md](docs/stage1-pilot-result-20260730.md).
Its frozen artifacts are not modified.

The revision uses exactly seven output classes and exactly balanced labels.
Structures, leaves, and operators are generated before target assignment. A
random nonzero-coefficient binding is solved in the prime field modulo seven.
Training uses skew and balanced shapes; evaluation includes named IID,
depth-OOD, and held-out branched shape/topology splits.

The revised generator now has an explicit `operand_mode`. `bound_variable`
preserves the original lookup-and-binding task as a separate harder diagnostic
axis. `literal` uses value tokens `8..14`; topology and operators are sampled
before the uniformly selected target, then one nonzero-coefficient leaf is
solved modulo seven. Only `literal` is eligible for the structural gate.

Literal runs record fixed C0 and C1 evaluations at curriculum boundaries.
The foundation gate requires A and D-true to reach C0 `>=0.99` and C1
`>=0.98` on at least 700 fixed exactly balanced examples per task. D-sham is
still compute-matched and reported, but it is not a C0/C1 foundation condition.
The passing post-hoc candidate and prepared formal plan are documented in
`docs/stage1-literal-posthoc-candidate-result-20260730.md`. The older
`configs/stage1-revised-literal-formal-directml.json` remains incompatible.

The next bounded structural candidate is
`configs/stage1-revised-literal-structural-candidate-directml.json`. It contains
8,000 paired updates with a 1:3 C0:C1 rehearsal ratio in the C1 stage, larger
depth-2/depth-3 allocations, and a 2,600-step final rehearsal. It remains
candidate-only. Its completed result is
`failed-under-original-gate`; see
`docs/stage1-literal-8000-result-20260730.md`.

The original gate required A and D-true above majority plus margin on every
required split. The explicit post-hoc amendment separates A non-collapse from
the privileged-structure effect without changing any D threshold. It was
revalidated with
`configs/stage1-revised-literal-posthoc-revalidation-directml.json`, fresh
training seed `82421`, and fresh frozen evaluation seeds. The new candidate
passed; the old result remains failed under its original gate.

Final evaluation now rejects content during generation rather than relying on a
post-hoc overlap audit. Every accepted hash must be absent from training and
from all earlier accepted evaluation splits/seeds. Results record separate
training-content, prior-evaluation, and label-quota rejection counts. Exhausting
the configured attempt bound fails the run without returning partial data.

Heartbeat and stage-boundary/foundation evaluation hashes are checkpointed as
pre-final exclusions. Final evaluation must also reject those hashes. Only a
run with `reason=target_steps_reached`, `global_step=optimizer_steps`, complete
per-model update/example counts, and `run_eligible_for_aggregation=true` can be
reported as `completed`. STOP and time-budget exits are `incomplete`; exceptions
are `failed`, and both force the candidate gate closed.

Formal holdout evaluation begins only after exact target completion. STOP,
time-budget, and training-failure exits write checkpoint/status/result evidence
without generating formal final examples. A formal-final attempt is atomically
marked before holdout access and can be recovered from completed final state
without evaluating again; an ambiguous partial attempt fails closed.

Legacy schema-2 results store the observed step at
`checkpoint_recovery.current_step` and the target at
`config.optimizer_steps`. The checker recognizes those historical completion
facts but does not infer the missing aggregation-eligibility field. Candidate
authorization and formal aggregation require schema 3 and explicit
`run_eligible_for_aggregation=true`.

Completed final-evaluation hashes are also cumulative run state. Checkpoint
schema 3 stores `historical_final_evaluation_content_hashes`; resume restores
them, later final evaluation actively rejects them, and the overlap audit
reports their count, digest, and per-dataset overlap. A legacy checkpoint that
contains final evaluation but cannot recover its hashes is rejected rather than
silently reusing content.

Formal candidate authorization is pinned to exact candidate config, source
manifest, snapshot manifest, result-file, and validated experiment-spec
digests. The prepared formal-confirmation config pins the passing candidate;
the older formal config retains empty pins and remains unusable.

Formal result checking and aggregation read each run's actual snapshot,
recompute every listed file SHA256 and both manifest hashes, and require a
nonempty embedded snapshot manifest. Missing, malformed, unlisted, or changed
authored files fail closed.

Authorization retains the candidate's immutable
`candidate_prerequisite_experiment_spec_digest`. Because formal confirmation
must use fresh training/final/foundation seed values, a second versioned
compatibility digest excludes only those seed declarations, final evaluation
scale, prerequisite identity, device, and resource/runtime controls. Worker
and aggregator independently recompute both. Curriculum, updates, batch,
models, learning rate, data, topology, gate/foundation thresholds, and
confirmation statistics remain bound.

Schema boolean evidence is accepted only when its JSON value is literally
`true` or `false` as required. Strings such as `"false"` and `"true"` never
coerce into passing evidence.

Formal campaign v4 uses fresh evaluation seeds `992501`, `992519`, and
`992531`, with foundation seed `992549`. It emits one correctness bitmask per sample for A,
D-true, and D-sham. Confirmation uses the independent training seed as its
statistical unit, computes one-sided paired Student-t lower bounds for D-true
minus A and D-true minus D-sham, and applies Bonferroni correction across all
split/effect comparisons.

D-true and D-sham have identical architecture, initial parameters, optimizer,
learning rate, examples, update count, and compose count. D-sham receives a
deterministic wrong source alignment. A candidate gate requires D-true to beat
both A and D-sham. A single run always records `stage2_unblocked=false`.

Campaign v4 completed all eight declared training seeds. All `304/304`
per-run integrity checks and all eight Bonferroni-corrected statistical
conditions passed. The canonical aggregate records
`decision=formal_confirmation_passed` and `stage2_unblocked=true`; its SHA256
is `95F4147F05CD31C1133418AF17A2F3061B37044A265D7FF0BB0BB06EBEE15631`.
An independent frozen-source recomputation was byte-identical. This confirms
the value of supplied true structure in the registered synthetic experiment;
it does not show that Stage 1 learned structure. See
[the formal v4 result](docs/stage1-formal-v4-confirmation-result-20260731.md).

The repository includes a
[public evidence bundle](evidence/stage1-formal-v4/README.md) with the exact
aggregate, frozen campaign manifest, all eight complete result records, and a
source-to-publication hash index. Multi-gigabyte periodic checkpoints remain
local because they are recovery artifacts rather than statistical inputs.

Run the bounded smoke explicitly:

```powershell
.\scripts\start_stage1.ps1 -Config configs/stage1-revised-smoke-directml.json
```

The launcher has no implicit config. Formal confirmation must be selected
explicitly and must not begin until both smoke runs and the short learning gate
pass. The formal plan preserves eight independent training seeds and 10,010
examples per named split/content-seed. Smaller runs are candidate-only.

Freeze and verify the canonical campaign before any launch:

```powershell
.\.venv-directml\Scripts\python.exe `
  scripts\run_stage1_confirmation_sequence_v4.py --prepare-only
```

This creates one versioned, immutable campaign package containing the complete
snapshot file set, environment receipt, candidate identity, and source/snapshot
manifest pins. It does not create a run or start a worker. Launch or resume the
serial campaign from the frozen coordinator with:

```powershell
$env:DYNAMIC_HIERARCHY_PROJECT_ROOT = (Resolve-Path ".").Path
$coordinator = ".\runs\stage1-literal-formal-v4-campaign\" +
  "canonical-snapshot\scripts\run_stage1_confirmation_sequence_v4.py"
& .\.venv-directml\Scripts\python.exe $coordinator
```

The v4 runner uses training seeds `991501`, `991511`, `991531`, `991541`,
`991547`, `991567`, `991579`, and `991589`. Each run is materialized from the
same canonical campaign snapshot. The coordinator itself is run from that
snapshot, and each launch/resume uses the materialized run's canonical launcher,
so later worktree changes cannot enter a run. The coordinator puts both the
canonical snapshot root and its `src` directory first on `sys.path`, then
asserts every loaded project module resides under those frozen roots.
Before every launch, resume, and result acceptance, the coordinator recomputes
the campaign, run snapshot, source manifest, receipt pins, and live environment
identity, including GPU/driver and DirectML Python. It waits for and
verifies each terminal result before launching the next; any failure stops the
sequence. A project-level coordinator mutex and live-worker scan prevent a
second campaign runner or another project Stage 1 worker from overlapping it.
Running the same command again resumes from its separate atomic v4 state. Only
after all eight results pass does it invoke the unchanged confirmation
aggregator.

Recoverable timeout results are archived under the same run and resumed from
checkpoint with a fresh per-worker-session runtime budget. A user STOP leaves
the v4 sequence in `paused_recoverable`; use Resume to clear STOP, then run the
same canonical command. Failed worker exceptions are nonrecoverable in campaign
v4. A
pid-less partial launch is accepted only when the materialized campaign
snapshot and receipt are intact and no other launch evidence exists. The
monitor waits for actual worker exit before considering the next seed.

Before campaign creation, the runner scans existing configs and run evidence.
All eight training seeds, three final-evaluation seeds, and the foundation seed
must be pairwise disjoint and absent from that history.

The original queue and the prepared-but-never-launched v2/v3 package/state
artifacts are retained read-only but excluded from campaign-v4 statistics.
Their complete source/snapshot manifests differ across seeds, so the unchanged
same-manifest aggregation rule cannot combine them. Seed 1 remains an observed
positive legacy result; seed 2 remains engineering evidence that its formal
holdout was untouched. See `docs/stage1-formal-v4-campaign-plan.md`.

Status and safe control commands are:

```powershell
.\scripts\control_stage1.ps1 -RunDir '<run-directory>' -Action Status
.\scripts\control_stage1.ps1 -RunDir '<run-directory>' -Action Pause
.\scripts\control_stage1.ps1 -RunDir '<run-directory>' -Action Resume
.\scripts\control_stage1.ps1 -RunDir '<run-directory>' -Action Stop
```

Checkpoint resume always uses the original snapshot:

```powershell
.\scripts\start_stage1.ps1 -ResumeRun '<run-directory>'
```

Checkpoint recovery remains periodic at-least-once. The curriculum position at
the saved checkpoint is restored exactly, but updates after that checkpoint can
be replayed after a crash. See [docs/stage1-design.md](docs/stage1-design.md),
[the protocol addendum](docs/stage1-revised-protocol-addendum.md), and
[the result plan](docs/stage1-revised-result-plan.md). The post-hoc baseline
amendment is recorded separately in
[docs/stage1-posthoc-baseline-addendum.md](docs/stage1-posthoc-baseline-addendum.md).
The completed formal decision is recorded in
[docs/stage1-formal-v4-confirmation-result-20260731.md](docs/stage1-formal-v4-confirmation-result-20260731.md).

## Reproducibility scope

Each run JSON records its config, per-task/per-scale counts, parameter count, synchronized training time, warmup and measured step counts, throughput, backward evidence, runtime warnings, fallback observability, Python/Torch/NumPy/platform values, installed dependency versions, backend/device name, determinism status, and a SHA256 source manifest. The manifest deliberately includes only authored configs, scripts, package Python files, tests, `pyproject.toml`, [requirements-cpu.lock](requirements-cpu.lock), and [requirements-directml.lock](requirements-directml.lock); it excludes caches, editable-install metadata, both virtual environments, and generated data/runs. CPU smoke enables strict PyTorch deterministic algorithms. Performance benchmarks disable them on both backends. DirectML rejects strict determinism and does not claim deterministic or bit-identical results across backends, hardware, operating systems, libraries, or PyTorch releases.

## Layout

- `src/dynamic_hierarchy/`: generators, A/D models, hierarchy-interface placeholder, and Stage 0/1 runtimes.
- `configs/`: CPU/DirectML smoke, learning-gate, formal-plan, and paired benchmark configs.
- `scripts/`: environment, training, repeated benchmark, snapshot, worker, and control entry points.
- `tests/`: focused Stage 0/1 CPU and optional DirectML checks.
- `evidence/`: publication-safe formal aggregates and per-seed result records.
- `docs/research-protocol.md`: byte-for-byte copy of the supplied research protocol.
- `docs/development-log.md`: commands, outcomes, and known blocks.
