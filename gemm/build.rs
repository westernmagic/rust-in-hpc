use std::env;

fn main() {
    println!("cargo:rustc-link-search={}/lib/intel64", env::var("MKLROOT").unwrap());
    println!("cargo:rustc-link-lib={}", "mkl_intel_lp64");
    println!("cargo:rustc-link-lib={}", "mkl_sequential");
    println!("cargo:rustc-link-lib={}", "mkl_core");
}
