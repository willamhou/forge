//! Spike: can a full multi-layer `LlamaModel::forward_into` be captured into
//! a CUDA Graph and replayed correctly?
//!
//! The big unknown for CUDA-Graphs-end-to-end: `forward_into` issues
//! `memcpy_htod` calls inside the captured region (embedding indices, RoPE
//! cos/sin, paged-attention block_tables/kv_lens, KV-append slot_mapping).
//! Pageable-host async copies may not be capturable, and even if they are,
//! the captured node bakes the host source pointer — stale on replay.
//!
//! This spike runs the experiment on a real model and reports exactly what
//! happens, so we know whether to (a) celebrate, (b) move host uploads out
//! of the captured region, or (c) switch to pinned host buffers.
//!
//! Usage:
//!   cargo run --release -p forge-server --example graph_decode_spike -- \
//!     --model-path /path/to/model
//!
//! Decision gates printed at exit: GREEN / YELLOW / RED.

use std::path::PathBuf;

use clap::Parser;
use forge_backend_cuda::{CudaBackend, CudaGraphCache};
use forge_core::{Backend, KvCache, Model, ModelInput, SeqMetadata, Tensor};
use forge_kvcache::paged_cache::PagedKvCache;
use forge_loader::{LlamaConfig, SafeTensorsLoader};
use forge_model_llama::{load_llama_model, LlamaPersistentBuffers};

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long, default_value = "0")]
    device: usize,
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg_text = std::fs::read_to_string(cli.model_path.join("config.json"))?;
    let llama_config: LlamaConfig = serde_json::from_str(&cfg_text)?;
    let config = llama_config.to_model_config();
    let vocab = config.vocab_size;

    let backend = CudaBackend::new(cli.device)?;
    let loader = SafeTensorsLoader::new(&cli.model_path)?;
    let model = load_llama_model(&loader, config.clone(), &backend)?;
    println!("== graph_decode spike: {} layers ==", config.num_hidden_layers);

    // Prefill one sequence so the KV cache + paged attention have content.
    let mut kv = PagedKvCache::new(
        backend.clone(),
        256,
        16,
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        config.dtype,
    )?;
    let prompt: Vec<u32> = vec![1, 450, 7483, 310, 3444, 338]; // arbitrary valid ids
    kv.allocate(1, prompt.len())?;
    let prefill = ModelInput {
        token_ids: vec![prompt.clone()],
        positions: vec![(0..prompt.len() as u32).collect()],
        seq_metadata: vec![SeqMetadata {
            seq_id: 1,
            prompt_len: prompt.len(),
            generated_len: 0,
            is_prefill: true,
        }],
    };
    let prefill_out = model.forward(&prefill, &mut kv)?;
    backend.synchronize()?;
    let prefill_logits = backend.copy_to_host_f32(&prefill_out.logits)?;
    let last = &prefill_logits[(prompt.len() - 1) * vocab..prompt.len() * vocab];
    let (next_tok, _) = argmax(last);
    println!("[prefill] next token = {next_tok}");

    // Decode input: 1 token (batch=1) at position prompt.len().
    let decode = ModelInput {
        token_ids: vec![vec![next_tok as u32]],
        positions: vec![vec![prompt.len() as u32]],
        seq_metadata: vec![SeqMetadata {
            seq_id: 1,
            prompt_len: prompt.len(),
            generated_len: 0,
            is_prefill: false,
        }],
    };

    let mut buffers = LlamaPersistentBuffers::new_uniform(&backend, &config, 1)?;

    // ── Baseline: uncaptured forward_into ────────────────────────────
    // Note: forward_into appends to the KV cache (advances seq 1 to
    // prompt.len()+1). We capture the FOLLOWING decode step; for the spike
    // we only care whether capture succeeds + the immediate post-capture
    // launch is sane, not multi-step KV correctness (that's the slot_mapping
    // gap we expect to find).
    model.forward_into(&decode, &mut kv, &mut buffers)?;
    backend.synchronize()?;
    let baseline = backend.copy_to_host_f32(&buffers.logits)?;
    let (b_arg, b_val) = argmax(&baseline);
    println!("[baseline forward_into] argmax={b_arg} ({b_val:.3})");

    // ── Attempt capture ──────────────────────────────────────────────
    let mut cache = CudaGraphCache::new(backend.ctx(), backend.stream());
    let model_ref = &model;
    let decode_ref = &decode;

    println!("[capture] attempting run_or_capture over forward_into ...");
    let cap_result = cache.run_or_capture(1, || {
        model_ref.forward_into(decode_ref, &mut kv, &mut buffers)
    });

    match cap_result {
        Ok(()) => {
            backend.synchronize()?;
            let captured = backend.copy_to_host_f32(&buffers.logits)?;
            let (c_arg, c_val) = argmax(&captured);
            println!("[capture] SUCCEEDED — captured-launch argmax={c_arg} ({c_val:.3})");
            println!("[capture] has(1)={}", cache.has(1));

            // ── Replay ────────────────────────────────────────────────
            match cache.replay(1) {
                Ok(()) => {
                    backend.synchronize()?;
                    let replayed = backend.copy_to_host_f32(&buffers.logits)?;
                    let (r_arg, r_val) = argmax(&replayed);
                    let max_diff = captured
                        .iter()
                        .zip(&replayed)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    println!("[replay] argmax={r_arg} ({r_val:.3}) max|Δ vs captured|={max_diff:.4}");
                    if r_arg == c_arg && max_diff < 1e-3 {
                        println!("\nGREEN — capture + replay both work, results stable.");

                        // ── Latency measurement: eager vs captured replay ──
                        // Both measure per-decode-step GPU cost. Eager pays
                        // ~220 kernel launches/step; replay pays 1 (the graph).
                        let warmup = 5usize;
                        let iters = 100usize;

                        // Captured replay FIRST: one graph launch per step. Must
                        // run before the eager loop below — eager appends tokens,
                        // which grows the kv_lens/block_tables scratches, frees the
                        // device buffers the captured graph baked, and would make
                        // replay read freed memory (the version-bump invalidation
                        // case the production cache must handle by recapturing).
                        for _ in 0..warmup {
                            cache.replay(1)?;
                        }
                        backend.synchronize()?;
                        let t1 = std::time::Instant::now();
                        for _ in 0..iters {
                            cache.replay(1)?;
                        }
                        backend.synchronize()?;
                        let replay_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

                        // Replay per-call-sync (still before any eager grow).
                        let t2 = std::time::Instant::now();
                        for _ in 0..iters {
                            cache.replay(1)?;
                            backend.synchronize()?;
                        }
                        let replay_sync_ms = t2.elapsed().as_secs_f64() * 1e3 / iters as f64;

                        // Eager: full forward_into each step (grows the cache).
                        for _ in 0..warmup {
                            model_ref.forward_into(decode_ref, &mut kv, &mut buffers)?;
                        }
                        backend.synchronize()?;
                        let t0 = std::time::Instant::now();
                        for _ in 0..iters {
                            model_ref.forward_into(decode_ref, &mut kv, &mut buffers)?;
                        }
                        backend.synchronize()?;
                        let eager_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

                        println!("\n── pipelined latency (sync at end, {iters} iters, {} layers) ──", config.num_hidden_layers);
                        println!("  eager forward_into : {eager_ms:.3} ms/step ({:.1} tok/s)", 1e3 / eager_ms);
                        println!("  captured replay    : {replay_ms:.3} ms/step ({:.1} tok/s)", 1e3 / replay_ms);
                        println!("  speedup            : {:.2}x  (saved {:.3} ms/step)",
                                 eager_ms / replay_ms, eager_ms - replay_ms);

                        // Per-call-sync latency: closer to what the engine sees,
                        // since it syncs every token (copy_to_host logits → sample).
                        // Launch overhead can't pipeline across the sync barrier.
                        // (replay_sync_ms was measured above, before the eager grow.)
                        let t3 = std::time::Instant::now();
                        for _ in 0..iters {
                            model_ref.forward_into(decode_ref, &mut kv, &mut buffers)?;
                            backend.synchronize()?;
                        }
                        let eager_sync_ms = t3.elapsed().as_secs_f64() * 1e3 / iters as f64;
                        println!("\n── per-call-sync latency (engine-realistic) ──");
                        println!("  eager forward_into : {eager_sync_ms:.3} ms/step ({:.1} tok/s)", 1e3 / eager_sync_ms);
                        println!("  captured replay    : {replay_sync_ms:.3} ms/step ({:.1} tok/s)", 1e3 / replay_sync_ms);
                        println!("  speedup            : {:.2}x  (saved {:.3} ms/step)",
                                 eager_sync_ms / replay_sync_ms, eager_sync_ms - replay_sync_ms);
                    } else {
                        println!("\nYELLOW — capture + replay run, but replay result differs \
                                  from captured (argmax {c_arg}->{r_arg}, Δ={max_diff:.4}). \
                                  Likely the stale-host-pointer / slot_mapping issue: replay \
                                  re-reads baked host data. Need host uploads out of capture \
                                  region OR pinned persistent host buffers.");
                    }
                }
                Err(e) => {
                    println!("\nYELLOW — capture worked but replay errored: {e}");
                }
            }
        }
        Err(e) => {
            println!("[capture] FAILED: {e}");
            println!("\nRED — cannot capture forward_into as-is. Most likely a \
                      memcpy_htod of pageable host memory inside the capture region \
                      (embedding indices / rope cos-sin / block_tables / slot_mapping). \
                      Fix: move host->device uploads OUT of the captured region (upload \
                      to persistent device buffers before graph.launch), capturing only \
                      the pure-kernel compute. See Task 1 spike Stage E.");
        }
    }

    Ok(())
}
