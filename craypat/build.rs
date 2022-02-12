use bindgen::{Builder, CargoCallbacks};

use std::env;
use std::path::PathBuf;

fn main() {
	let craypat_root = PathBuf::from(env::var("CRAYPAT_ROOT").unwrap());
	let header = craypat_root.join("include").join("pat_api.h");

	println!("cargo:rerun-if-changed={}", header.to_str().unwrap());
	println!("cargo:rustc-link-lib=_pat_base");

	let bindings = Builder::default()
		.header(header.to_str().unwrap())
		.whitelist_var("PAT_.*")
		.whitelist_function("PAT_.*")
		.blacklist_function("PAT_omp_.*")
		.blacklist_function("PAT_acc_.*")
		.parse_callbacks(Box::new(CargoCallbacks))
		.rustfmt_bindings(true)
		.generate()
		.expect("Unable to generate bindings");

	let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
	bindings
		.write_to_file(out_dir.join("bindings.rs"))
		.expect("Couldn't write bindings!");
}
