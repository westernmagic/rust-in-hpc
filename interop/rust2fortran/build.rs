fn main() {
    let lib = cmake::build("src");
    println!("cargo:rerun-if-changed=src/zaxpy.f");
    println!("cargo:rustc-link-search=native={}/lib", lib.display());
    println!("cargo:rustc-link-lib=zaxpy");
    println!("cargo:rustc-link-lib=gfortran");
}
