# dynamic-hierarchy-ai

## 中文通俗导读

这项研究想回答一个问题：

> AI 能不能不依赖人工规定的“词、句、段落、章节”，而是根据当前内容和问题，自己把信息组合成临时的推理层级？

现在已经确认的是：在本项目的合成推理任务中，如果把正确结构提供给模型，它会获得巨大而稳定的优势。

现在还没有确认的是：模型能不能自己发现这个结构。Stage 2 R2 的候选模型退化成了立即停止；R3 直接提供正确合并顺序后只记住了小训练池；R4 把独立训练 family 扩大 20 倍后，三种模型又都没有学会基础模七运算。因此问题目前卡在“组合运算为什么没有泛化”，还没有进入自主结构发现的有效检验。

### 最初想法如何变成实验

最初的直觉包含三部分：

1. 相同的数或词出现在不同位置时，可以因为上下文和交互历史不同而具有不同状态。
2. 信息不仅会在当前尺度内连续变化，也可能组合成更高尺度的信息。
3. 层级不应被永久写成“词元、段落、章节、大纲”这样的固定枚举，而应根据任务动态形成。

浮点数的整数位和小数位曾被用作这个想法的类比，但项目没有把浮点数本身当成特殊的新数学对象。机器浮点数的集合有限，也不能保证每次交互都产生从未出现过的数。

工程上，我们改用结构化状态表达这个直觉：

```text
节点状态 = 内容 + 来源位置 + 上下文 + 结构关系 + 连续进度
```

其中，连续进度或“相位”仍然只是假设。它以后可以帮助控制器判断是否继续处理，但尚未被实验验证。

### 希望模型最终怎样工作

所有输入最初都是叶节点。模型反复使用同一套共享函数，根据当前问题选择操作：

- `MERGE`：把相关节点组合成一个更高尺度的节点。
- `STOP`：当前表示已经足以回答问题，停止继续抽象。

长期还可以研究 `LINK`、`EXPAND` 和 `SPLIT`，但在最基础的 `MERGE/STOP` 没有证明有效以前，不引入这些额外自由度。

理想过程不是固定的：

```text
词元 -> 句子 -> 段落 -> 章节 -> 大纲
```

而是由内容和问题决定的：

```text
原始节点 -> 局部组合 -> 更大组合 -> 回答当前问题所需的结构
```

同一份输入面对不同问题，可以形成不同结构。每个高层节点还保留到原始节点的引用，避免摘要后永久丢失细节。

### 为什么先做 Stage 1

直接让模型学习动态结构会同时引入很多变量。如果实验失败，我们无法判断是“结构没有用”，还是“模型没有学会寻找结构”。

所以 Stage 1 先回答一个更基础的问题：

> 假如正确结构已经知道，它究竟能不能帮助推理？

我们让三个模型处理完全相同的模七加减表达式。输出只有 `0..6` 七个类别，训练和测试标签严格平衡。

| 模型 | 得到的信息 | 作用 |
| --- | --- | --- |
| A | 普通符号序列 | 普通 Transformer 基线 |
| D-true | 序列加正确树结构和正确来源位置 | 检验正确结构是否有用 |
| D-sham | 与 D-true 相同的递归架构，但来源位置被确定性打乱 | 排除“只是架构或计算量更多” |

例如，面对下面的表达式：

```text
((3 + 5) - 2) mod 7
```

A 需要仅从序列中处理组合关系。D-true 还会获得类似下面的正确组合骨架：

```text
      -
     / \
    +   2
   / \
  3   5
```

D-true 只得到树拓扑和指向普通输入的来源位置，不会得到答案、中间运算结果或其他解题捷径。D-sham 使用相同的递归计算模块和相同计算次数，但这些来源位置被打乱。

### 我们走过的过程

#### Stage 0：建立实验基础

项目先实现了小型 Transformer、可复现的合成任务、CPU/DirectML 训练、检查点、运行记录和测试。这个阶段主要证明实验基础设施能够工作，不验证动态层级。

#### 第一版 Stage 1：失败

最初试验中，A 和早期 D 都退化为几乎恒定的输出，没有表现出结构优势。我们把它保留为负结果，没有把“能记住固定批次”误写成“学会了规则”。

这次失败暴露了标签不平衡、任务难度混杂、测试规模不足和树形不丰富等问题。

#### 修订版 Stage 1：重新控制变量

我们随后加入了：

- 七个类别的严格平衡；
- 从简单到较深表达式的课程；
- 倾斜树、平衡树和留出的分支树；
- 训练内容与正式测试内容的主动隔离；
- D-sham 架构匹配对照；
- 八个独立训练种子和预先规定的统计门槛；
- 冻结代码、配置、环境和结果哈希的完整性检查。

#### 正式 Stage 1：通过

八个训练种子都完成了 `8000/8000` 步。每个模型在全部训练种子上的正式测试总量为 `960,960` 个样本。

| 测试 | A | D-true | D-sham |
| --- | ---: | ---: | ---: |
| 普通深度 3 倾斜树 | 22.06% | 100.00% | 57.83% |
| 普通深度 3 平衡树 | 14.35% | 100.00% | 17.69% |
| 未训练深度 5 倾斜树 | 14.60% | 99.99% | 27.17% |
| 留出的深度 3 分支树 | 15.81% | 100.00% | 38.08% |

八项预先规定的统计条件全部通过，八个运行的完整性检查为 `304/304` 通过。独立重新聚合得到与正式结果逐字节相同的文件和 SHA256。

### 结果应该怎样理解

#### 已确认的事实

- 在这个合成模七任务和冻结实验中，正确结构产生了巨大且稳定的优势。
- 优势不能只用递归架构或额外组合计算来解释，因为架构匹配的 D-sham 明显更差。
- 这个优势在普通测试、更深表达式和未见过的树形上都出现了。
- Stage 1 的正式门槛已经通过，因此可以开始 Stage 2。

#### 有证据支持的推断

- 为推理提供正确的组合关系可能比把所有符号主要作为平坦序列处理更有效。
- 动态结构值得继续研究，而不是在第一步就因为“结构本身无效”而终止。

#### 尚未证明

- 模型能自己发现正确边界或树结构。
- 模型已经学会 `MERGE/STOP`。
- 连续相位比普通门控更好。
- 结果可以直接推广到自然语言、大模型或现实任务。
- 这种方法可以取代 Transformer。

D-true 获得的是外部提供的正确结构。因此，它接近 100% 不代表未来的自主模型也会接近 100%。当前实验测量的是“正确结构的价值”，不是“学习结构的难度”或预期收益大小。

### 当前进度图

| 研究问题 | 状态 |
| --- | --- |
| 正确层级结构是否有用？ | 已正式确认 |
| 优势是否只是递归架构或更多计算造成的？ | D-sham 对照基本排除了这一解释 |
| 模型能否自己发现合并边界？ | 尚未证明；R4 可行性门失败，路由实验仍暂停 |
| 连续相位能否帮助决定合并时机？ | 尚未验证 |
| 自主结构能否外推到更长推理链和新树形？ | 未知 |
| 能否改善真实语言模型？ | 未知 |

### Stage 2 R2：已经实现，但第一次校准没有学会

Stage 2 R2 不再向候选模型提供正确结构。它给同一条无括号表达式提出两种问题：`ADD` 优先还是 `SUB` 优先。问题改变正确的运算顺序、树和答案，因此模型不能只背一棵固定树。

R2 同时运行普通 Transformer、参数匹配和运算量估算匹配基线、无合并结构的共享递归、查询盲路由、查询置换 sham、五种固定策略、候选动态路由、D-true 和 D-sham，共 13 个控制。硬路由的前向路径只能使用被选中的节点，未选候选只参与梯度估计。

2026-08-09 的 DirectML 校准使用种子 `821101`，完成 120 次更新；每个模型看过 10,080 个 query 行。评估包含三个 profile，每个 profile 有 420 个独立 base family、840 个 query 行，训练与评估 family 重叠为零。

| 评估 | 普通 A | B-query | B-noQ | 最佳固定策略 | D-true |
| --- | ---: | ---: | ---: | ---: | ---: |
| 长度 5，交替运算符 | 14.64% | 15.00% | 14.52% | 16.55% | 14.17% |
| 长度 8 外推 | 15.00% | 15.60% | 15.48% | 15.60% | 12.62% |
| 长度 6，留出运算形状 | 15.95% | 15.00% | 14.52% | 15.00% | 16.90% |

七分类随机水平是 14.29%。更关键的是，B-query 在三个评估中都对 100% 样本立即执行 `STOP`，轨迹与 F-stop 完全相同，正确树率为 0。A、A-recur、D-true 和 D-sham 也都仍接近随机水平。因此这不是“动态结构无效”的证据，而是一次明确的校准失败：任务学习前提尚未建立，候选路由又走进了最便宜的停止解。

R3 因此没有直接增加种子或引入连续相位。它先设置学习可行性门，并在必须归约到根的 benchmark 中移除候选 B 的 `STOP` 动作，只检查普通模型、正确结构诊断和获得正确合并顺序的 B 是否能推广到未见表达式。

### Stage 2 R3：结构链正确，但可行性门失败

R3 使用两个固定训练池：长度 3 的 `-+` 和长度 4 的 `-+-`，各含 42 个 base family 和 84 个双查询行。三个模型训练 600 步，累计训练准确率分别为 A 92.79%、B-oracle 98.02%、D-true 98.15%。这说明优化器和模型能够拟合训练池。

但在 84 个完全未见的 held-out family 上，结果接近七分类随机水平：

| 评估 | 普通 A | B-oracle | D-true |
| --- | ---: | ---: | ---: |
| 长度 3，`-+` | 14.29% | 2.38% | 10.71% |
| 长度 4，`-+-` | 11.90% | 14.29% | 14.29% |

B-oracle 在两个评估中的正确树率、边 F1 和完整归约率都是 100%，`STOP` 次数为零。这排除了“又提前停止了”和“oracle 执行错树”这两种解释。训练与评估 family 重叠也为零。最符合证据的解释是：模型严重拟合了反复出现的 84 个训练 family，却没有学会可迁移的模七运算规则。

预注册门槛要求 B-oracle 和 D-true 在两个评估上都达到至少 50% 准确率且交叉熵不超过 1.50；四个必需格子全部失败，正式 disposition 为 `feasibility_failed`。按照冻结协议，项目没有启动查询条件路由、额外种子或连续相位训练。

R4 已按这个要求扩大独立训练 family，并冻结验证集与最终保留集。结果如下。

### Stage 2 R4：扩大数据仍未建立基础学习

R4 把独立训练 family 从 84 个增加到 1,680 个，同时保持总 family exposure 为 25,200。40 个平衡数据块循环 15 次，每个训练 family 恰好出现 15 次。验证集另含 336 个未见 family，训练与验证重叠为零；最终还预留了 336 个 family，只有验证通过才允许打开。

| 模型 | 训练准确率 | n=3 验证 | n=4 验证 |
| --- | ---: | ---: | ---: |
| 普通 A | 15.21% | 9.52% | 12.24% |
| B-oracle | 15.21% | 5.95% | 7.82% |
| D-true | 15.15% | 3.57% | 7.48% |

七分类随机水平是 14.29%。这一次不是“训练池拟合很好、未见样本失败”，而是三种模型在冻结的 600 步预算内连扩大后的训练分布也没有学会。B-oracle 的正确树率、边 F1 和完整归约率仍都是 100%，所以正确结构确实被执行了，但当前神经表示和训练方法没有把它转化为可迁移的模七计算能力。

R4 四个必需验证格子全部失败，正式 disposition 为 `feasibility_failed`。验证账本在失败后封闭，336 个 reserve family 没有生成或评估。最稳妥的下一步不是立刻训练路由，而是把任务拆成“单步运算 -> 固定查询组合 -> 双查询 oracle 组合”的算术因果阶梯，分别检查表示能力、组合过程的信用分配和训练预算。

### 对当前主流推理的意义

当前结果还不是可以直接加入大语言模型的改进，也没有证明现有主流推理路线错误。

它的研究价值在于确认了一项必要前提：在一个受控任务中，正确的层级组织确实比平坦输入和错误结构更有价值。如果 Stage 1 没有通过，就没有充分理由继续投入自主结构学习；现在这条研究路线获得了进入下一实验阶段的依据。

Stage 2 R2、R3 和 R4 的负校准继续缩小了问题范围，但没有改变对主流推理的结论。R4 表明问题不只是小数据池记忆：在当前模型、课程和 600 步预算下，正确结构模型也没有学会扩大训练分布中的模七组合规则。只有后续模型先通过分级算术学习门，再稳定学会 query-dependent 合并并胜过查询盲、固定、sham 和计算匹配控制，才值得测试长上下文、自然语言和真实推理任务。

### 继续阅读

- 严格的研究约束和长期假设：[research-protocol.md](docs/research-protocol.md)
- Stage 1 的实验设计：[stage1-design.md](docs/stage1-design.md)
- Stage 1 正式结果：[stage1-formal-v4-confirmation-result-20260731.md](docs/stage1-formal-v4-confirmation-result-20260731.md)
- 可公开复核的证据包：[evidence/stage1-formal-v4/README.md](evidence/stage1-formal-v4/README.md)
- Stage 2 R2 施工包：[stage2-construction-packet-r2.md](docs/stage2-construction-packet-r2.md)
- Stage 2 R2 校准结果：[stage2-r2-calibration-result-20260809.md](docs/stage2-r2-calibration-result-20260809.md)
- Stage 2 R2 公开证据：[evidence/stage2-r2-calibration/README.md](evidence/stage2-r2-calibration/README.md)
- Stage 2 R3 施工包：[stage2-construction-packet-r3.md](docs/stage2-construction-packet-r3.md)
- Stage 2 R3 可行性结果：[stage2-r3-feasibility-result-20260809.md](docs/stage2-r3-feasibility-result-20260809.md)
- Stage 2 R3 公开证据：[evidence/stage2-r3-feasibility/README.md](evidence/stage2-r3-feasibility/README.md)
- Stage 2 R4 施工包：[stage2-construction-packet-r4.md](docs/stage2-construction-packet-r4.md)
- Stage 2 R4 可行性结果：[stage2-r4-feasibility-result-20260809.md](docs/stage2-r4-feasibility-result-20260809.md)
- Stage 2 R4 公开证据：[evidence/stage2-r4-feasibility/README.md](evidence/stage2-r4-feasibility/README.md)

最简洁而准确的当前结论是：

> 我们已经证明“正确结构值得学习”；但 R2 退化为立即停止，R3 记住了小训练池，R4 扩大数据后仍没有学会基础组合运算，所以“模型能够自己学会结构”仍未开始得到有效验证。

## Technical Reference

This is a Windows CPU/DirectML research harness for controlled symbolic reasoning. Stage 0 provides the ordinary Transformer baseline. Revised Stage 1 compares A, privileged-structure D-true, and architecture-matched D-sham. Stage 2 R2 implements learned hard routing and matched interventions, but collapsed to immediate STOP. Stage 2 R3 verified oracle full reduction but overfit a small fixed family pool. Stage 2 R4 expanded unique training coverage 20-fold under equal total exposure; all three feasibility models remained near chance and failed validation. None of these results establishes learned hierarchy.

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
- Stage 2 R2 implements paired precedence queries, hard adjacent merge routing, adaptive recurrence, all 13 mandatory controls, family-level isolation, checkpoint recovery, and complete common compute receipts on CPU and DirectML.
- The first 120-step DirectML calibration completed but B-query collapsed to immediate STOP on every evaluation row; all task models remained near chance, so the result is calibration-inconclusive.
- Stage 2 R3 preserves R2 compatibility, removes learned STOP for full-expression reduction, adds a source-only selected-path B-oracle and frozen feasibility gate, and completed its 600-step DirectML calibration.
- R3's oracle structure execution was exact, but B-oracle and D-true failed every held-out feasibility gate after fitting the repeated training pools. Routing, extra seeds, and continuous phase remain blocked.
- Stage 2 R4 adds exhaustive balanced train/validation/reserve partitions, uniform 40-block scheduling, persistent one-shot reserve gating, reconstruction receipts, and backward-compatible R2/R3 serialization.
- The single R4 DirectML run completed 600 steps with 1,680 unique training families. All models remained near chance, all required validation gates failed, and the reserve remained unopened.

**Candidate hypotheses, not results**

- Contextual interaction plus continuous position features may help distinguish repeated values in different roles.
- Learned dynamic hierarchy, shared recursive merging, and phase-controlled decisions may improve extrapolation.
- No experiment in this repository establishes learned dynamic hierarchy or autonomous boundary discovery.
- Stage 2 R2 does not establish learned `MERGE/STOP`; its first calibration observed an all-STOP collapse.
- Stage 2 R3 does not test learned routing; its `B-oracle` receives the correct source-only merge order, and the failed gate concerns held-out task generalization.
- Stage 2 R4 also does not test learned routing. Its failure narrows the next question to arithmetic representation, composition credit assignment, curriculum, and budget without deciding among them.

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
.\.venv\Scripts\python.exe scripts/run_stage2.py --config configs/stage2-smoke-cpu.json
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
.\.venv-directml\Scripts\python.exe scripts/run_stage2.py --config configs/stage2-smoke-directml.json
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

- `src/dynamic_hierarchy/`: generators, A/D/B models, hierarchy controls, and Stage 0/1/2 runtimes.
- `configs/`: CPU/DirectML smoke, learning-gate, formal-plan, and paired Stage 2 configs.
- `scripts/`: environment, training, repeated benchmark, snapshot, worker, and control entry points.
- `tests/`: focused Stage 0/1/2 CPU and optional DirectML checks.
- `evidence/`: publication-safe formal aggregates and per-seed result records.
- `docs/research-protocol.md`: byte-for-byte copy of the supplied research protocol.
- `docs/development-log.md`: commands, outcomes, and known blocks.
