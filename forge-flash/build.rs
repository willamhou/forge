/// Build script for forge-flash.
///
/// Compiles vendored FlashAttention v2 CUDA sources.
/// Requires CUDA toolkit with nvcc for SM80+SM90.
fn main() {
    // TODO: Task 3 will add cc::Build compilation here
    println!("cargo:rerun-if-changed=csrc/");
}
