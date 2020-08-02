use anyhow::Result;
use itertools::iproduct;

use std::env;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<()> {
	// TODO: hack to rebuild every time to pick up cmake changes
	// println!("cargo:rerun-if-changed=./*");
	
	let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
	let out_dir = PathBuf::from(env::var("OUT_DIR")?);
	let build = Command::new("bash")
		.args(&["scripts/build.sh", out_dir.to_str().unwrap(), "RelWithDebInfo"])
		.current_dir(dir.join("lib"))
		.output()?;

	io::stdout().write_all(&build.stdout)?;
	io::stdout().write_all(&build.stderr)?;
	assert!(build.status.success());

	let mut functions = String::from(r#"
		use lazy_static::lazy_static;
		use std::collections::HashMap;
		use crate::{Compiler, Language, Function};

		use std::str::FromStr;
		
		lazy_static! {
			pub static ref FUNCTIONS: HashMap<(Compiler, Language, &'static str), Function> = {
				let mut map = HashMap::new();
	"#);
	let mut bindings = String::from(r#"extern "C" {"#);

	for (compiler, lang) in iproduct!(
		&["gnu", "intel", "cray", "pgi"],
		&["cpp", "f"]
	).chain(iproduct!(
		&["rustc"],
		&["rs"]
	)) {
		let lib_dir = out_dir.join(compiler).join(lang);
		println!("cargo:rustc-link-search={}", lib_dir.to_str().unwrap());
		println!("cargo:rustc-link-lib=stencil_{}_{}", compiler, lang);

		let lib_path = lib_dir.join(format!("libstencil_{}_{}.so", compiler, lang));

		for func in get_functions(&lib_path)? {
			let version = func.strip_prefix(&format!("diffuse_{}_{}_", compiler, lang)).unwrap();
			functions += &format!(
				r#"
					map.insert((FromStr::from_str("{}").unwrap(), FromStr::from_str("{}").unwrap(), "{}"), {} as Function);
				"#,
				compiler,
				lang,
				version,
				func
			);

			bindings += &format!(
				r#"
					pub fn {} (
						in_field: *mut f32,
						out_field: *mut f32,
						nx: usize,
						ny: usize,
						nz: usize,
						num_halo: usize,
						alpha: f32,
						num_iter: usize
					);
				"#,
				func
			);
				
		}
	}

	functions += r#"
				map
			};
		}
	"#;

	bindings += "}";

	// functions += &bindings;
	bindings += &functions;

	let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
	let mut file = File::create(out_dir.join("functions.rs"))?;
	file.write_all(bindings.as_bytes())?;

	Ok(())
}

// from cargo-binutils
// https://github.com/rust-embedded/cargo-binutils/blob/master/src/rustc.rs
fn sysroot() -> Result<PathBuf> {
	let rustc = env::var("RUSTC").unwrap_or("rustc".to_string());
	let result = Command::new(rustc)
		.args(&["--print", "sysroot"])
		.output()?;
	assert!(result.status.success());
	Ok(PathBuf::from(String::from_utf8(result.stdout)?.trim()))
}

fn nm_path() -> Result<PathBuf> {
	Ok(
		sysroot()?
			.join("lib")
			.join("rustlib")
			.join(rustc_version::version_meta()?.host)
			.join("bin")
			.join("llvm-nm")
	)
}

fn get_functions(lib: &Path) -> Result<Vec<String>> {
	let nm = nm_path()?;
	let result = Command::new(nm)
		.args(&["--just-symbol-name", lib.to_str().unwrap()])
		.output()?;
	assert!(result.status.success());
	Ok(
		String::from_utf8(result.stdout)?
			.lines()
			.filter(|name| name.starts_with("diffuse_") && !name.contains(".") && !name.contains("$") && !name.ends_with("_"))
			.map(|s| s.to_owned())
			.collect()
	)
}
