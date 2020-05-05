use std::env;
use std::path::PathBuf;

fn main() {

    /*
    cc::Build::new()
        .cpp(true)
        .file("src/zaxpy.cpp")
        .compile("libzaxpy.a");
    */

    let lib = cmake::build("src");
    println!("cargo:rerun-if-changed=src/zaxpy.hpp");
    println!("cargo:rustc-link-search=native={}/lib", lib.display());
    println!("cargo:rustc-link-lib=zaxpy");

    let bindings = bindgen::Builder::default()
        .header("src/zaxpy.hpp")
        .whitelist_function("zaxpy")
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
