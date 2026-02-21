/// Build script for forge-backend-cuda.
///
/// When the `flash-attn` feature is enabled, this compiles the FlashAttention
/// C wrapper (`forge-kernels/csrc/flash_attn_wrapper.cu`) and links it as a
/// static library. The wrapper calls through to the FlashAttention library
/// (expected at `FLASH_ATTN_LIB_DIR` or system library paths).
///
/// When `flash-attn` is disabled, this is a no-op.
fn main() {
    #[cfg(feature = "flash-attn")]
    build_flash_attn();
}

#[cfg(feature = "flash-attn")]
fn build_flash_attn() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_dir = std::path::Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("forge-kernels")
        .join("csrc");

    let wrapper_cu = kernels_dir.join("flash_attn_wrapper.cu");
    let wrapper_h = kernels_dir.join("flash_attn_wrapper.h");

    if !wrapper_cu.exists() {
        panic!(
            "flash_attn_wrapper.cu not found at {}",
            wrapper_cu.display()
        );
    }

    // Find CUDA toolkit for nvcc and includes
    let cuda_home = find_cuda_home();

    // Compile the wrapper .cu file into a static library
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file(&wrapper_cu)
        .include(kernels_dir)
        .include(format!("{cuda_home}/include"))
        .flag("-std=c++17")
        .flag("-O2");

    // If FlashAttention headers are available, include them
    if let Ok(fa_include) = std::env::var("FLASH_ATTN_INCLUDE_DIR") {
        build.include(&fa_include);
        build.define("FLASH_ATTN_AVAILABLE", None);
    }

    build.compile("flash_attn_wrapper");

    // Link against CUDA runtime
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=cudart");

    // If a pre-built FlashAttention .so is provided, link it
    if let Ok(fa_lib_dir) = std::env::var("FLASH_ATTN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={fa_lib_dir}");
        println!("cargo:rustc-link-lib=flash_attn");
    }

    // Re-run if sources change
    println!("cargo:rerun-if-changed={}", wrapper_cu.display());
    println!("cargo:rerun-if-changed={}", wrapper_h.display());
    println!("cargo:rerun-if-env-changed=FLASH_ATTN_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=FLASH_ATTN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}

#[cfg(feature = "flash-attn")]
fn find_cuda_home() -> String {
    if let Ok(home) = std::env::var("CUDA_HOME") {
        return home;
    }
    if let Ok(path) = std::env::var("CUDA_PATH") {
        return path;
    }
    if std::path::Path::new("/usr/local/cuda/include/cuda.h").exists() {
        return "/usr/local/cuda".to_string();
    }
    panic!("CUDA toolkit not found. Set CUDA_HOME or CUDA_PATH environment variable.");
}
