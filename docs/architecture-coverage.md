# Architecture Coverage

> Last verified: 2026-06-26 against `forge-transformer::registry::SUPPORTED_ARCHITECTURES`.

Forge serves any model whose `config.json` declares one of the architectures
below. They share a single parameterized implementation in
[`forge-models/forge-transformer`](../forge-models/forge-transformer/src/) —
there is no per-model crate. Architecture-specific features are detected from
the safetensors weight set at load time, not from a config flag or a model-type
match.

## Supported `config.architectures`

| Architecture string | Examples | Optional features picked up | Verification |
| --- | --- | --- | --- |
| `LlamaForCausalLM` | Llama 2, Llama 3, TinyLlama, CodeLlama | — | registered ✓ · unit-tested ✓ · benchmarked ✓ (TinyLlama-1.1B, Llama 3) |
| `Qwen2ForCausalLM` | Qwen2-7B, Qwen2-14B, Qwen2.5-* | QKV bias | registered ✓ · unit-tested ✓ (`test_qwen_features.rs`) · benchmarked ✓ (Qwen2.5-0.5B) |
| `Qwen3ForCausalLM` | Qwen3-0.6B / 4B / 8B / 14B | Per-head QK-norm | registered ✓ · unit-tested ✓ (`test_qwen_features.rs`) · benchmarked ✓ (Qwen3-0.6B / 4B / 8B / 14B) |
| `MistralForCausalLM` | Mistral-7B, codestral | — | registered ✓ · unit-tested via shared Llama path ✓ · **not yet benchmarked** |

"unit-tested" = the loader path + attention wiring is exercised by
`cargo test -p forge-transformer`. "benchmarked" = at least one run of
`scripts/bench_concurrent.py` against the architecture is recorded in
`docs/benchmarks/`. Mistral is *registered* and shares the Llama path
byte-for-byte (no special weights), so it should load and serve, but there is
no recorded end-to-end run yet — treat it as supported-but-unverified until
one lands.

The dispatch lives in [`forge-transformer/src/registry.rs`](../forge-models/forge-transformer/src/registry.rs);
unknown architectures error out.

## How optional features are detected

In [`forge-transformer/src/loader.rs`](../forge-models/forge-transformer/src/loader.rs)
the loader probes the safetensors set for specific tensor names per layer:

- **Qwen2 QKV bias** — `loader.contains("model.layers.{i}.self_attn.q_proj.bias")`.
  If the q_proj bias exists, all three of q/k/v_proj biases are loaded and
  concatenated into a single `[q + 2*kv]` vector matching the fused `wqkv`
  column layout. Llama models simply lack this key and fall through.
- **Qwen3 per-head QK-norm** — `loader.contains("model.layers.{i}.self_attn.q_norm.weight")`.
  When present, two `[head_dim]` RMSNorm weights are attached to
  `LlamaAttention` via `with_qk_norm(...)` and applied to Q and K *before* RoPE
  in `forward()`. See `LlamaAttention::maybe_q_norm` /
  `LlamaAttention::maybe_k_norm`.

Detection is by *presence*, not by config — a real load failure on a tensor
that exists must propagate as an error rather than silently fall back to the
no-feature path (which would corrupt Qwen attention).

## Why no per-model crate (yet)

The four architectures above are structurally the same dense decoder
transformer + GQA + RoPE; their differences are uniform (a bias add, a
RMSNorm). One parameterized impl + presence-based feature loading covers them
without code duplication.

A per-model crate boundary becomes worth the split when the **forward pass
shape** changes — concrete candidates:

- **MoE** (Mixtral, Qwen3-MoE, DeepSeek-V2-Lite) — router + expert dispatch +
  potentially expert-parallel scheduling.
- **MLA** (DeepSeek-V2/V3) — compressed KV projection changes the attention
  signature and KV-cache layout.
- **Hybrid linear + full attention** (Qwen3.5 family) — different per-layer
  attention types in the same model.

Until then, adding a new dense decoder-transformer variant should follow the
existing pattern: add the architecture string to `SUPPORTED_ARCHITECTURES`,
add presence-based loading for any new optional weight, and attach the new
feature via a builder on `LlamaAttention` (or a new sibling layer struct).
See `LlamaAttention::with_qk_norm` for the established pattern.

## Tokenizer / chat template / `<think>`

Architecture-level support stops at logits. Tokenizer choice and chat
templating live in `forge-server` (the HTTP layer); Qwen3's `<think>` traces
are produced by the model as ordinary tokens and surfaced unchanged through
the streaming SSE channel. See `forge-server/src/api/` for the API layer.
