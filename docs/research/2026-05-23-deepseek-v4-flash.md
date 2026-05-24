# DeepSeek-V4-Flash — Architecture Survey & Forge Porting Gap

**Date:** 2026-05-23
**Status:** Research only. No code changes proposed. Treat as long-term north star.

## TL;DR

DeepSeek-V4-Flash is the cost-tier model in the V4 family released 2026-04-24 under MIT. **284B total / 13B active MoE, 1M context, FP8+FP4 native, 43 layers.** Architecturally it is not "another DeepSeek with bigger numbers" — it ships **five mechanisms forge does not have any analogue for**: hybrid Compressed-Sparse / Heavily-Compressed Attention (CSA + HCA), a FP4 lightning indexer, Manifold-Constrained Hyper-Connections (mHC) replacing residuals, hash/Engram layers, and FP4 expert weights. Realistic to load and run = **net-new model implementation + net-new backend kernels**, ~2 months of focused work after the prerequisite Tier 1–3 ports below.

For forge's near-term scope (single GB10 → 2× DGX-Spark next week), V4-Flash is **out of scope this quarter**. This doc exists so the design choices we make for MoE / MLA / FP8 in Tier 2–3 don't paint us into a corner that blocks V4 later.

## Sources

- HF blog: <https://huggingface.co/blog/deepseekv4>
- Model card: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- Tech report PDF: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>
- API preview note (2026-04-24): <https://api-docs.deepseek.com/news/news260424>
- Inference README: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/README.md>

## 1. Model facts (verbatim from `config.json`)

```json
{
  "architectures": ["DeepseekV4ForCausalLM"],
  "num_hidden_layers": 43,
  "hidden_size": 4096,
  "num_attention_heads": 64,
  "num_key_value_heads": 1,
  "head_dim": 512,
  "qk_rope_head_dim": 64,
  "q_lora_rank": 1024,
  "o_lora_rank": 1024,
  "o_groups": 8,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 6,
  "moe_intermediate_size": 2048,
  "scoring_func": "sqrtsoftplus",
  "topk_method": "noaux_tc",
  "routed_scaling_factor": 1.5,
  "norm_topk_prob": true,
  "swiglu_limit": 10.0,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "num_hash_layers": 3,
  "num_nextn_predict_layers": 1,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "hc_eps": 1e-06,
  "sliding_window": 128,
  "compress_rope_theta": 160000,
  "rope_theta": 10000,
  "rope_scaling": {
    "type": "yarn", "factor": 16,
    "beta_fast": 32, "beta_slow": 1,
    "original_max_position_embeddings": 65536
  },
  "max_position_embeddings": 1048576,
  "vocab_size": 129280,
  "torch_dtype": "bfloat16",
  "quantization_config": {
    "quant_method": "fp8", "fmt": "e4m3",
    "scale_fmt": "ue8m0", "weight_block_size": [128, 128],
    "activation_scheme": "dynamic"
  },
  "expert_dtype": "fp4",
  "compress_ratios": [0,0, 4,128,4,128, ...(alternating)..., 4,128, 4,0]
}
```

Storage: **~158B params** in safetensors with mixed `BF16 / F8_E4M3 / F8_E8M0 / F4 / I8 / I64 / F32` tensor types.

## 2. What is new vs DeepSeek-V3 / V3.2

| Mechanism | V3 / V3.2 | V4-Flash | Why it matters for forge |
|---|---|---|---|
| **Attention** | Multi-head Latent Attention (MLA) — 1 KV head, latent dim 512, dense over full seq | **CSA + HCA hybrid** per `compress_ratios`. Layer 0/1 dense; layers 2–42 alternate CSA (4× compress, top-k=512 sparse) and HCA (128× compress, dense over compressed). Per-layer sliding-window=128 recency branch. Layer 43 (MTP) dense. | KV cache layout is per-layer. Existing `forge-kvcache` paged design assumes uniform layer shape; V4 needs **layer-typed** cache. |
| **Sparse selector** | n/a | **Lightning indexer**: 64-head, dim 128, **FP4** ReLU-scored dot product, picks top-512 compressed blocks per query | New FP4 dot-product kernel; new top-k selection kernel; reroute attention path through indexer output. |
| **Residual** | Standard `x + sublayer(x)` | **mHC** (Manifold-Constrained Hyper-Connections): 4 lanes, Sinkhorn-projected (20 iters, eps 1e-6) | Replaces every residual add. Affects autograd (training) and forward (inference). For inference: implement the projection forward only. |
| **Knowledge layers** | n/a | **3 hash layers** (likely "Engram" — static-knowledge lookup running parallel to MoE) | Net-new layer type. Unclear if skippable for inference. |
| **MoE routing** | `topk_method: greedy`, sigmoid scoring, aux loss | `topk_method: noaux_tc` (no auxiliary loss, top-correction), **`scoring_func: sqrtsoftplus`**, `routed_scaling_factor: 1.5` | Routing kernel must match; getting this wrong silently degrades quality. |
| **Experts** | 256 routed + 1 shared (same as V3), `top-8` | 256 routed + 1 shared, **top-6**, `moe_intermediate_size: 2048` | Marginally lighter per-token; same MoE infrastructure once Tier 3 is built. |
| **Expert weight dtype** | FP8 (V3 already had FP8) | **FP4** (e2m1-ish) | FP4 GEMM is new even relative to a hypothetical V3 port. Tooling: DeepGEMM FP4, or CUTLASS 3.7 FP4 SM100. |
| **MTP** | Optional in V3 | 1 MTP layer (`num_nextn_predict_layers: 1`) | Slots into speculative decoding; replaces forge's n-gram speculator for V4 if we want self-speculation. |
| **SwiGLU** | Standard | **`swiglu_limit: 10.0`** (output clamp) | One-line forward change. |
| **RoPE** | YaRN, factor 40, original 4K (V3) | YaRN factor 16, original **64K → 1M**, `rope_theta=10000` for main, **`compress_rope_theta=160000`** for compressed branch | Two RoPE bases per layer (main + compressed). |
| **Vocab** | 129280 | 129280 (same) | DeepSeek tokenizer; not yet wired in `forge-loader`. |
| **Quantization** | FP8 (e4m3), 128×128 weight blocks, dynamic activation scale | Same FP8 for non-experts; **FP4 for experts**; KV mostly FP8, RoPE dims stay BF16; lightning indexer FP4 | Existing forge backends are F32/F16/BF16. **No FP8 support, no FP4 support.** |

The DeepSeek tech report claims V4-Pro uses **27% of single-token inference FLOPs and 10% of KV cache** vs V3.2 at 1M-token context. V4-Flash is similar.

## 3. Memory envelope on 2× DGX-Spark (2× GB10, ~256 GB unified)

- Static weights: ~158B params; FP8 non-experts + FP4 experts ≈ **110–120 GB** (rough; experts dominate).
- KV cache at 128k context (typical chat): with 128× compression on most layers + sliding-window-128 fast path, ≈ **single-digit GB**.
- KV cache at full 1M context: ≈ **20–40 GB** (10% of comparable GQA-bf16 baseline per DeepSeek's claim).
- **Fits across two GB10 with PP**. Single-box: weights alone leave little room for prefill activations; likely needs PP across the two boxes.

Inference reference impl uses `torchrun --nproc-per-node 4` with `MP=4` (tensor-parallel default). For a 2-node × 1-GPU layout, PP fits the topology better (one NIC hop per stage boundary, not per layer).

## 4. Gap to forge — what would have to be built

Layered against the existing trait surface from CLAUDE.md (Backend / Tensor / Model / KvCache / Scheduler):

**`Backend` (forge-core/src/backend.rs)**
- `matmul_fp8`: e4m3 weights × bf16 activations, per-block (128×128) scales — large kernel job. DeepGEMM is the reference.
- `matmul_fp4`: MoE experts. Needs FP4 (e2m1) packed format, group scales. CUTLASS 3.7+ Blackwell FP4 path is candidate.
- `topk` + `moe_dispatch`: scatter tokens to selected experts, gather back. Single-GPU version is straightforward; multi-GPU EP (DeepEP) is much larger.
- `sparse_attention_with_index`: take indexer top-k → gather compressed K/V blocks → sparse SDPA.
- `dense_attention_compressed`: dense SDPA over 128×-compressed sequence.
- `sliding_window_attention`: standard, just causal SDPA with window mask.
- Sinkhorn projection primitive for mHC (20 iters of row+col normalization).

**`Tensor`**
- New dtypes: `Fp8E4m3`, `Fp4E2m1`, paired block-scale tensors (`F8E8M0` for FP8 scales).
- Packing helpers (FP4 = 2 weights/byte).

**`Model` (`forge-models/forge-model-deepseek-v4`, new crate)**
- `DeepSeekV4Attention`: branches per `compress_ratios[layer]` (0 dense, 4 CSA, 128 HCA). Calls indexer for CSA layers.
- `DeepSeekV4MoE`: 256-expert MoE + 1 shared expert, top-6, noaux_tc + sqrtsoftplus.
- `LightningIndexer`: 64-head FP4 dot product, top-512 select.
- `ManifoldHyperConnect`: replaces residual; 4 lanes + Sinkhorn.
- `HashLayers` (Engram): 3 layers, mechanism TBD from tech report.
- `MTPHead`: extra next-token prediction layer.

**`KvCache` (forge-kvcache)**
- Per-layer typed cache: a CSA layer stores `[blocks_compressed × dim_compressed]`, an HCA layer stores `[blocks_heavy × dim_heavy]`, dense layers store latent KV like V2/V3 MLA. The current paged cache assumes one shape across all layers — needs a layer-id-indexed table of pool descriptors.

**`forge-loader`**
- Mixed-dtype safetensors: `BF16 / F8_E4M3 / F8_E8M0 / F4 / I8 / I64`. Existing loader is F16/BF16/F32 + GGUF Q4_K_M/Q8_0.
- DeepSeek tokenizer (BPE, vocab 129280).
- Optionally a `convert.py`-style step that materializes the `MP=N` shard layout the reference inference expects (or implement our own sharding).

**Scheduler**
- 1M-token context vs current `max_seq_len`: bump `--max-seq-len` ceiling, but admission control still has to account for per-layer KV (CSA layers grow slower than HCA, so the cache headroom changes).
- MTP-aware speculative decoding: forge currently does n-gram speculative; can keep it but MTP is the upstream-blessed path.

## 5. Realistic porting ladder (forge perspective)

| Tier | Model | Adds infra | Estimate | Unlocks |
|---|---|---|---|---|
| 0 | Llama-3.x (done) | — | — | dense Llama-family |
| 1 | **Qwen2.5-7B / 14B** | per-projection bias toggle, minor RoPE freq | 1–2 days | Qwen2 / 2.5 dense models |
| 2 | **DeepSeek-V2-Lite (16B-A2.4B)** | MoE routing kernel, **MLA**, MoE-aware KV layout, DeepSeek tokenizer | 1.5–2 weeks | Moonshot Moonlight, Qwen2.5-MoE share infra |
| 3 | **DeepSeek-V3 (671B-A37B)** | **FP8 matmul** + 128×128 block scales, full-scale MoE, FP8 KV | 3–4 weeks (kernel work dominates) | All V3.x variants, V4 prerequisite |
| 4 | **DeepSeek-V4-Flash** | **FP4 experts**, CSA + HCA + lightning indexer, **mHC**, hash layers, MTP, layer-typed KV | 6–8 weeks on top of Tier 3 | V4 family + future DS releases |

Skipping tiers risks designing the Backend/KvCache traits in a way that closes Tier 4 off. Specific shape constraints to keep open while doing Tier 2/3:

- KvCache pool descriptor: make per-layer pluggable from day one of Tier 2 (so HCA-style heavy compression is just another layer-type later).
- Backend dtype enum: budget for FP8 and FP4 in the enum and the Tensor's typed-slice accessors during Tier 3, even if only FP8 is implemented then.
- Residual ops in the model code: route every `x + sublayer(x)` through a small `combine(x, sublayer_out, layer_idx)` helper, default = add. Tier 4 swaps that helper to mHC without touching every layer.

## 6. Open questions before any Tier-4 work begins

1. **Hash / Engram layers (`num_hash_layers: 3`)**: tech report describes them. Are they skippable for inference quality (e.g., fall through to identity)? Or load-bearing? — read tech report PDF, section TBD.
2. **mHC inference cost**: 20 Sinkhorn iterations per residual is non-trivial. Reference impl probably folds it into a fused kernel; we'd need our own.
3. **FP4 path on GB10 (sm_121)**: Blackwell has hardware FP4 support, but CUTLASS FP4 kernels target sm_100. Does sm_120 cubin run on sm_121? (Related to today's `forge-flash` build issue — same arch question, separate codepath.)
4. **MTP vs n-gram speculative**: forge already has n-gram. Is the perf gap from going to MTP worth Tier 4's MTP-head implementation, or do we ship V4-Flash with the n-gram speculator first?
5. **DeepGEMM vs hand-rolled FP8 matmul**: vendoring DeepGEMM is faster to working state, but is a large C++/CUTLASS dependency similar in weight to `forge-flash`. Decide before Tier 3.

## 7. What is **not** in this doc (and should be elsewhere later)

- V4-Pro (1.6T-A49B) — same arch, different scale; numbers above already mostly translate.
- V4 multimodal — V4 family includes multimodal variants per Clore.ai writeup; vision encoder pipeline is its own work.
- Performance comparison vs vLLM/SGLang on V4-Flash — empirical, requires running the model, comes after Tier 4.
- Detailed mHC math — read tech report; not reproduced here.

---

**Next action when this becomes hot:** read the V4 tech report PDF end-to-end, then write a Tier-4-only design doc (`docs/plans/202X-XX-XX-deepseek-v4-flash-design.md`) that pins down the open questions above. Until Tier 3 lands, this doc is only for keeping Tier 2/3 trait designs forward-compatible.
