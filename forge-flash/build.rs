/// Build script for forge-flash.
///
/// Compiles vendored FlashAttention v2 CUDA sources using `cc::Build`.
/// Requires CUDA toolkit with nvcc for SM80 (Ampere) + SM90 (Hopper).
fn main() {
    println!("cargo:rerun-if-changed=csrc/");

    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Collect forward kernel .cu files (both regular and split-KV variants).
    let fwd_files: Vec<_> = std::fs::read_dir("csrc/flash_attn/src")
        .expect("csrc/flash_attn/src must exist")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "cu")
                && p.file_name()
                    .unwrap()
                    .to_str()
                    .is_some_and(|n| n.starts_with("flash_fwd"))
        })
        .collect();

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file("csrc/flash_api_forge.cu")
        .include("csrc/flash_attn/src")
        .include("csrc/flash_attn")
        .include("csrc/cutlass/include")
        .include(format!("{cuda_home}/include"))
        .define("FORGE_NO_PYTORCH", None)
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("-diag-suppress=177")
        .flag("-diag-suppress=549");

    for f in &fwd_files {
        build.file(f);
    }

    build.compile("flash_attn");

    // Link CUDA runtime.
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
