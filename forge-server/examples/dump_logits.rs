//! Reference-parity harness: load a Llama model, run a raw prefill forward
//! over a fixed token-ID sequence, and print the top-K next-token logits of
//! the LAST position. Bypasses chat templating + tokenization so the result
//! can be compared apples-to-apples against a HuggingFace transformers run on
//! the same token IDs.
//!
//! Usage:
//!   cargo run --release -p forge-server --example dump_logits -- \
//!     --model-path /path/to/model --tokens 1,450,7483,310,3444,338 [--topk 10]
//!
//! Prints (to stdout, machine-parseable):
//!   ARGMAX <token_id>
//!   TOPK <id>:<logit> <id>:<logit> ...
//!   LOGIT <id> <value>   (one line per requested --probe id, if any)

use std::path::PathBuf;

use clap::Parser;
use forge_core::{Backend, KvCache, Model, ModelInput, SeqMetadata};
use forge_kvcache::naive::NaiveKvCache;
use forge_loader::{LlamaConfig, SafeTensorsLoader};
use forge_model_llama::load_llama_model;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    model_path: PathBuf,
    /// Comma-separated input token IDs (the prefill prompt).
    #[arg(long)]
    tokens: String,
    #[arg(long, default_value = "10")]
    topk: usize,
    /// CUDA device ordinal.
    #[arg(long, default_value = "0")]
    device: usize,
    /// Backend: cuda or cpu.
    #[arg(long, default_value = "cuda")]
    backend: String,
    /// Optional comma-separated token IDs to print exact logits for.
    #[arg(long, default_value = "")]
    probe: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let token_ids: Vec<u32> = cli
        .tokens
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<_, _>>()?;

    let config_text = std::fs::read_to_string(cli.model_path.join("config.json"))?;
    let llama_config: LlamaConfig = serde_json::from_str(&config_text)?;
    let model_config = llama_config.to_model_config();
    let vocab = model_config.vocab_size;

    match cli.backend.as_str() {
        "cpu" => {
            let backend = forge_backend_cpu::CpuBackend::new();
            run(&backend, &cli, model_config, &token_ids, vocab)
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            let backend = forge_backend_cuda::CudaBackend::new(cli.device)?;
            run(&backend, &cli, model_config, &token_ids, vocab)
        }
        other => anyhow::bail!("unknown backend {other}"),
    }
}

fn run<B: Backend + Clone>(
    backend: &B,
    cli: &Cli,
    model_config: forge_core::ModelConfig,
    token_ids: &[u32],
    vocab: usize,
) -> anyhow::Result<()> {
    let loader = SafeTensorsLoader::new(&cli.model_path)?;
    let model = load_llama_model(&loader, model_config.clone(), backend)?;

    let mut kv = NaiveKvCache::new(backend.clone(), model_config.num_hidden_layers, 1);
    kv.allocate(0, token_ids.len())?;

    let input = ModelInput {
        token_ids: vec![token_ids.to_vec()],
        positions: vec![(0..token_ids.len() as u32).collect()],
        seq_metadata: vec![SeqMetadata {
            seq_id: 0,
            prompt_len: token_ids.len(),
            generated_len: 0,
            is_prefill: true,
        }],
    };

    let out = model.forward(&input, &mut kv)?;
    backend.synchronize()?;
    let logits_host = backend.copy_to_host_f32(&out.logits)?;

    // logits shape: [num_tokens, vocab]; take the last row.
    let t = token_ids.len();
    let last = &logits_host[(t - 1) * vocab..t * vocab];

    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_unstable_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());

    println!("ARGMAX {}", idx[0]);
    let topk: Vec<String> = idx
        .iter()
        .take(cli.topk)
        .map(|&i| format!("{}:{:.4}", i, last[i]))
        .collect();
    println!("TOPK {}", topk.join(" "));

    if !cli.probe.is_empty() {
        for s in cli.probe.split(',') {
            let id: usize = s.trim().parse()?;
            println!("LOGIT {} {:.6}", id, last[id]);
        }
    }
    Ok(())
}
