// use std::env;
// use std::path::PathBuf;

fn main() {
/*
    // cpp
    let lib = cmake::Config::new("src")
        .very_verbose(true)
        .build();

    println!("cargo:rerun-if-changed=src/cpp/v0_base.hpp");
    println!("cargo:rerun-if-changed=src/cpp/v0_base.cpp");
    println!("cargo:rerun-if-changed=src/cpp/v1_inline.hpp");
    println!("cargo:rerun-if-changed=src/cpp/v1_inline.cpp");
    println!("cargo:rustc-link-search=native={}/lib", lib.display());
    println!("cargo:rustc-link-lib=static=stencil_cpp");
    println!("cargo:rustc-link-lib=stdc++");

    /*
    let bindings = bindgen::Builder::default()
        .header("src/cpp/base.hpp")
        .clang_arg("-xc++")
        .clang_arg("-std=c++17")
        .whitelist_function("diffuse_cpp")
        .size_t_is_usize(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Could not write bindings!");
    */

    // f
    let lib = cmake::Config::new("src")
        .very_verbose(true)
        .build();

    println!("cargo:rerun-if-changed=src/f/CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/f/v0_base.f");
    println!("cargo:rerun-if-changed=src/f/v1_inline.f");
    println!("cargo:rerun-if-changed=src/f/v2_openmp.f");
    println!("cargo:rerun-if-changed=src/f/v3_openmp.f");
    println!("cargo:rustc-link-search=native={}/lib", lib.display());
    println!("cargo:rustc-link-lib=static=stencil_f");
    println!("cargo:rustc-link-lib=gfortran");
    println!("cargo:rustc-link-lib=omp");
*/
}
