# pegainfer vs forge — 对比分析

**日期:** 2026-05-24
**对比对象仓库:** `github.com/xiaguan/pegainfer`(约 339 stars,9 名贡献者,默认分支 `main`)
**方法:** 通读公开仓库源码 + workspace `Cargo.toml` + `build.rs` 文件;并与 `feat/phase1-mvp` 分支上的 forge 交叉比对。

## 1. pegainfer 概况

- **规模**:约 84 K 行 Rust 代码(约为 forge 18 K 的 5 倍),约 19 K 行自写的 CUDA C++/CUH,约 10 K 行 Python 用于 Triton / TileLang AOT 内核生成。
- **Workspace**:30 个 crate,其中包括一棵 14 crate 的 `pegainfer-comm/*` 子树,用于在 RDMA 上实现 EP all-to-all。crate 划分以模型为边界——每个架构拥有自己的 `scheduler`、`executor`、`kernel_plan`、`batch_decode`、`prefill`、`unified_forward`(参见 `pegainfer-qwen3-4b/src/`)。
- **支持模型**:Qwen3-4B/8B、Qwen3.5(混合 linear + full attn)、DeepSeek-V2-Lite、DeepSeek-V4-Flash(MoE + MLA + FP8/FP4 稀疏注意力)、Kimi-K2。每个均通过 feature gate 控制。
- **多 GPU**:通过 cudarc 调用 NCCL,另有基于 IB Verbs + GDRCopy 的自研 PPLX 风格 EP all-to-all。DSv4 跨 `cuda:0..7` 做 MP8,Kimi 走 DP/EP。
- **运行时量化**:Marlin W4/INT4(Kimi),TileLang 生成的 FP8/FP4(DSv4)。
- **活跃度**:每日有提交,约 9 名贡献者。forge:122 次提交,单一作者。

## 2. CUDA 集成:pegainfer 并不是纯 Rust

尽管 README 打着 "Pure Rust + CUDA" 的标语,实际情况是:

- `pegainfer-kernels/build.rs` 每次构建都会调用 **`nvcc`** 编译 `pegainfer-kernels/csrc/` 下 28 个以上的 `.cu` 文件,通过 `PEGAINFER_NVCC_JOBS` 控制的并行作业池调度。许多文件直接来自 vLLM(`kimi_k2/vllm_marlin/*`)。
- 构建时**需要 Python + Triton**(`PEGAINFER_TRITON_PYTHON`)对 `tools/triton/{gated_delta_rule_chunkwise_kernels, flash_attention_prefill_hd256_kernel}.py` 进行 AOT 编译生成 PTX,然后把 PTX 链接进二进制。
- `--features deepseek-v4` 还额外需要 **Python + TileLang**(`PEGAINFER_TILELANG_PYTHON`),通过 `tools/tilelang/deepseek_v4/generate.py` 生成 FP8/FP4 CUDA 代码。
- 内嵌了 `pegainfer-kernels/third_party/flashinfer/` 的 C++ 头文件(FlashInfer paged-attention 内核套件)。
- **`bindgen 0.72.1`** 作为 workspace 依赖被 `pegainfer-cupti/build.rs` 以及 `cuda-sys` / `cudart-sys` / `libibverbs-sys` / `gdrapi-sys` 使用(EP all-to-all FFI)。
- 在 `pegainfer-comm/crates/pegainfer-comm-torch-lib/` 中有一处 **LibTorch cxx 桥接**,用于 benchmark 互操作;feature gate 控制(`hw-cuda`,默认关闭)。推理运行时不链接它。
- HTTP 前端是 **vLLM 的**:`vllm-engine-core-client`、`vllm-server`、`vllm-text`、`vllm-tokenizer` 都从 `github.com/vllm-project/vllm@65b7a812` 拉取。Pegainfer 并不拥有自己的 OpenAI 接口层——它说的是 vLLM 的 ZeroMQ MQ 协议。

**forge 相比之下** 在运行时构建/依赖层面真正接近纯 Rust:

- `cudarc 0.17.8`(CUDA Driver API 绑定),没有 `bindgen`,没有 `cuda-sys`。
- 所有自定义内核均在**后端初始化时通过 NVRTC** 编译——`cargo build` 阶段没有 `nvcc` 步骤。
- `forge-flash/build.rs` 中只有一处 `cc::Build`,用于内嵌的 FlashAttention v2(首次构建约 30 分钟 nvcc 编译,之后有缓存)。
- 没有 Python。没有 PyTorch。没有 vLLM。没有 NCCL。
- 自有的、基于 axum 的 OpenAI 兼容 HTTP 层。

代价正如预期:forge `cargo build` 在只装了 CUDA Toolkit 的新机器上就能跑;pegainfer 需要 CUDA Toolkit + `uv` + Python + `pip install torch+triton` + 可选 TileLang + `CUDA_HOME` / `PEGAINFER_TRITON_PYTHON` 等环境变量。代价是 forge 这边内核覆盖面要小得多。

## 3. 逐项判定

🟢 = forge 领先 · 🟡 = 基本持平 · 🔴 = forge 落后

| 维度 | 判定 | 备注 |
|---|---|---|
| Workspace 分层 | 🟡 | forge 按关注点划分(core/runtime/scheduler/kvcache/backend),pegainfer 按模型 crate + 共享内核划分 |
| CUDA 集成纯度 | 🟢 | 见第 2 节 |
| 构建复杂度 | 🟢 | `cargo build` vs nvcc + Python + uv + 环境变量 |
| 自定义内核覆盖 | 🔴 | pegainfer 约 20 个内核族,forge 约 7 个 |
| KV cache 设计 | 🟡 | 两边都是设备端 paged。pegainfer 的 `KvPool::padding_permit`(`pegainfer-core/src/kv_pool.rs:64`)正是 forge Task 5 所需要的 bucket-padding 技巧 |
| 注意力内核覆盖 | 🔴 | pegainfer:FlashInfer paged、hd128/hd256 prefill、gated-delta-rule linear、DSv4 sparse + indexer、Kimi MLA。forge:FA2 + paged + naive,F32/F16 |
| 模型覆盖 | 🔴 | 1 个(Llama) vs 4 个架构,含 MoE/MLA/linear *(写于 2026-05-24;见末尾 Amendment 2026-06-26)* |
| 运行时量化 | 🔴 | Marlin W4 / FP8 / FP4 vs 无(仅 GGUF 加载期量化) |
| 多 GPU / TP / EP | 🔴 | 基于 IB Verbs+GDRCopy 的 14 crate `pegainfer-comm` vs 无 |
| 调度器 | 🟡 | forge 有 chunked prefill 准入门;pegainfer 有 CUDA-graph bucket 规划 |
| FSM 约束解码 | 🟢 | forge:通过 `regex-automata` DFA 实现 JSON Schema + regex。pegainfer:仅 temp/top-k/top-p(18 行 sampler) |
| 投机解码 | 🟢 | forge 有 n-gram;pegainfer 没有 |
| HTTP 服务 / 可观测性 | 🔴 | pegainfer 复用 vLLM 前端 + nvtx + fastrace + jemalloc + `bench_serving` 二进制 |
| 测试纪律 | 🔴 | pegainfer:每个模型对 HF 输出做 parity 测试 + criterion benches + vllm-baseline crate。forge:约 18 个测试文件,无 HF parity |
| 文档 | 🟡 | 两边都有维护;形态不同 |
| 活跃度 | 🔴 | 每日 / 9 名贡献者 vs 单一作者 |

## 4. forge 可以借鉴的点

- **以模型为边界的 executor crate**(`pegainfer-qwen3-4b/src/{scheduler,executor,kernel_plan}.rs`)——加入 Qwen3 时就采用这种划分,不要塞进 `forge-model-llama`。 *(2026-06-26 撤回,见末尾 Amendment)*
- **`KvPool::padding_permit`**(`pegainfer-core/src/kv_pool.rs:64`)——让 CUDA Graph 捕获在变化 batch size 下也能复用的具体模式。**这正是阻塞 forge Task 5 的点**(graph 捕获所需的持久化 buffer)。 *(2026-06-26 撤回,见末尾 Amendment)*
- **`tikv-jemallocator` 全局分配器**(`pegainfer-server/src/main.rs:11`)——直接落地的吞吐提升。
- **AOT Triton 内核模式**(`pegainfer-kernels/build.rs` + `tools/triton/`)——如果 forge 哪天想做 linear attention,这是最干净的先例。但要注意它会把 Python 引入构建链。
- **`kernel-call-trace` feature flag**——可发布的诊断能力,以 feature 门控。

## 5. pegainfer 可以从 forge 借鉴的点

- FSM 约束解码(`forge-runtime/src/constraints/fsm.rs`)——pegainfer 的 18 行 sampler 是显而易见的生产可用性短板。
- N-gram 投机解码(`forge-runtime/src/speculative/`)——与模型正交,几乎是免费的吞吐提升。
- Chunked prefill 准入门(`forge-scheduler/src/continuous.rs` → `rejected_seq_ids`)——pegainfer 的调度器没有限制单步 prefill 成本。
- 带 F32/F16 双路径的 `Backend` trait——pegainfer 仅支持 BF16,使得 CPU 上测试不可能。

## 6. 结论

**Pegainfer 在覆盖广度上领先约 5 倍**(4 个模型族含 MoE/MLA/linear、多 GPU EP、运行时量化、来自真实团队的每日提交)。
**forge 在运行时特性打磨上领先**(FSM 约束解码、n-gram 投机解码、chunked prefill、F32/F16 双精度对等),**构建链条也确实更干净**(`cudarc` + NVRTC + 一次 cc 编译的 FA2,无 Python)。

**杠杆率最高的差距**:加入第二个模型架构——Qwen3 是显而易见的选择——同时引入 pegainfer 的按模型 crate 划分。基于 forge 已有的 `Backend` trait + `PagedKvCache`,一名工程师估计需要 2–3 周。多 GPU TP/EP 的工作量大得多(pegainfer 用了 14 个 crate + IB Verbs/GDRCopy 绑定),应当等到双 DGX-Spark 互联就绪后再做。

**ROADMAP.md 不需要改变方向**,但 Phase A 的优先级应重排:完成 CUDA Graphs(进行中)→ **加入 Qwen3 + 按模型 crate 拆分** → W4A16 量化 → 多 GPU。

---

## Amendment (2026-06-26)

本文写于 2026-05-24,5 周后回看,两条结论因事实更新而失效。原文保留作 snapshot,修订集中在此。

### A1. 模型覆盖(第 3 节表 / 第 6 节"杠杆率最高的差距")

原文判定"forge 1 个(Llama)"是错的——写文档时没读 [forge-models/forge-transformer/src/registry.rs](../../forge-models/forge-transformer/src/registry.rs)。实际 forge 已支持 4 个 dense decoder 架构:`LlamaForCausalLM` / `Qwen2ForCausalLM` / `Qwen3ForCausalLM` / `MistralForCausalLM`,共享一份参数化 `TransformerModel`,Qwen2 的 QKV bias 与 Qwen3 的 per-head QK-norm 都通过权重存在性检测启用(见 [docs/architecture-coverage.md](../architecture-coverage.md))。

**真正的模型覆盖差距**:不是"forge 缺 Qwen3",而是"forge 缺 *异构* 架构"——MoE(Mixtral / Qwen3-MoE / DeepSeek-V2-Lite)、MLA(DSv2/V3)、hybrid linear+full attention(Qwen3.5)。dense decoder 这一档已经追平。

**对原文第 4 节"以模型为边界的 executor crate"的影响**:撤回。pega 的极致 per-model 拆分是为 DSv4 MoE / MLA / FP8 这种结构性异构服务的;在 forge 当前只跑 dense decoder + GQA 的情况下,拆 per-model crate 是 over-engineering。等到加 MoE 才是真分叉契机。

### A2. CUDA Graphs / `KvPool::padding_permit`(第 4 节第 2 条 / 第 6 节优先级)

原文把 CUDA Graphs 列入 Phase A 优先级,并把 pega 的 `KvPool::padding_permit` 当作 forge Task 5 的关键 unblock。

**ROADMAP.md 已把 GB10 上的 CUDA Graphs 列为 Non-goal**:`cuGraphLaunch` 在 GB10 上 +3 ms/step(eager 模式靠 host-GPU 提交/执行重叠摊销 launch cost;graph 一次性提交反而失这种重叠)。这是 GB10 驱动行为,不是 forge bug。证据见 memory `forge-decode-host-bound`。

**对策**:`padding_permit` 模式只在 Hopper / Blackwell 上线后才值得借鉴;"finish CUDA Graphs (in progress)" 的建议作废。

### A3. 修订后的 Phase A 优先级

1. ~~CUDA Graphs~~(GB10 上 Non-goal)→ 改攻 **MoE 架构**(Mixtral 最自然,因为 pega 的 Qwen3.5 / DSv4 / Kimi-K2 都是 MoE,且 MoE 是逼出 per-model crate 边界的真契机)
2. **W4A16 / Marlin 量化**(forge 当前 Q8_0 单流胜出 1.54×,W4A16 是 batch serving 主流)
3. **多 GPU TP/EP**(等双 DGX-Spark 链路就绪)

**未变化的判定**:第 2 节 "CUDA 集成纯度" + 第 5 节 "pegainfer 可以从 forge 借鉴" + 多 GPU 工作量评估,事实上都仍然成立。
