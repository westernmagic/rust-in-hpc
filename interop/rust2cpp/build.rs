use std::env;
use std::path::PathBuf;

fn main() {

    /*
    // For simple cases, you can compile using cc
    cc::Build::new()
        .cpp(true)
        .file("src/zaxpy.cpp")
        .compile("libzaxpy.a");
    */

    let lib = cmake::build("src");
    println!("cargo:rerun-if-changed=src/zaxpy.hpp");
    println!("cargo:rustc-link-search=native={}/lib", lib.display());
    println!("cargo:rustc-link-lib=zaxpy");
    // TODO: this should not be needed - something is off with the C++ detection
    println!("cargo:rustc-link-lib=stdc++");

    let bindings = bindgen::Builder::default()
        .header("src/zaxpy.hpp")
        // TODO: this should not be needed - something is off with the C++ detection
        .clang_arg("-xc++")
        .clang_arg("-std=c++17")
        .whitelist_function("zaxpy")
        .blacklist_type("std::complex.*")
        .size_t_is_usize(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Could not write bindings!");
}
