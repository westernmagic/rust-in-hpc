extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_path = String::from(env!("CUDA_PATH"));
    let lib = cuda_path.clone() + "/lib";
    let include = cuda_path.clone() + "/include";
    let header = "cublasXt.h";

    println!("cargo:rustc-link-search={}", lib);
    println!("cargo:rustc-link-lib=cublas");
    // println!("cargo:rerun-if-changed={}", include.clone() + "/" + header);

    let mut builder = bindgen::Builder::default();
    builder = builder.clang_arg(format!("-I{}", include));
    let bindings = builder
        .header(include.clone() + "/" + header)
        .whitelist_function("cublasXt.*")
        .blacklist_type("cuFloatComplex")
        .blacklist_type("cuDoubleComplex")
        .rustified_enum(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
