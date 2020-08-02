#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use anyhow::Result;
use structopt::StructOpt;
use ndarray::prelude::*;
use ndarray_npy::*;
use std::fs::File;
use std::time::Duration;
use stencil::*;

#[derive(StructOpt)]
struct Args {
    #[structopt(long = "--nx", default_value = "128")]
    nx: usize,
    #[structopt(long = "--ny", default_value = "128")]
    ny: usize,
    #[structopt(long = "--nz", default_value = "64")]
    nz: usize,
    #[structopt(long = "--num_iter", default_value = "1024")]
    num_iter: usize,
    #[structopt(long = "--alpha", default_value = "0.03125")]
    alpha: f32,
	#[structopt(long = "--all")]
	all: bool,
	#[structopt(required_unless("all"))]
	compiler: Option<Compiler>,
	#[structopt(required_unless("all"))]
	language: Option<Language>,
	#[structopt(required_unless("all"))]
	version: Option<String>,
}

const num_halo: usize = 2;

#[paw::main]
pub fn main(args: Args) -> Result<()> {
	#[cfg(craypat)]
	craypat::record(false);

    let nx = args.nx;
    let ny = args.ny;
    let nz = args.nz;
    let num_iter = args.num_iter;
    let alpha = args.alpha;
	let compiler = args.compiler;
	let language = args.language;
    let version = args.version;

    let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
    in_field.slice_mut(s![
        (num_halo + nx / 4)..(num_halo + 3 * nx / 4),
        (num_halo + ny / 4)..(num_halo + 3 * ny / 4),
        (nz / 4)..(3 * nz / 4)
    ]).fill(1.0f32);
    let mut in_field = in_field;
    let mut out_field = in_field.clone();

	if args.all {
		run_all(&in_field, &out_field, alpha, num_iter)?;
	} else {
		run(compiler.unwrap(), language.unwrap(), &version.unwrap(), &mut in_field, &mut out_field, alpha, num_iter)?;
	}

    Ok(())
}

fn run_all(
	in_field: &Array3<f32>,
	out_field: &Array3<f32>,
	alpha: f32,
	num_iter: usize
) -> Result<()> {
	for (compiler, language, version) in FUNCTIONS.keys() {
		run(*compiler, *language, version, &mut in_field.clone(), &mut out_field.clone(), alpha, num_iter)?;
	}

	Ok(())
}

fn run(
	compiler: Compiler,
	language: Language,
	version: &str,
	in_field: &mut Array3<f32>,
	out_field: &mut Array3<f32>,
	alpha: f32,
	num_iter: usize
) -> Result<Duration> {
	let nx = in_field.shape()[0] - 2 * num_halo;
	let ny = in_field.shape()[1] - 2 * num_halo;
	let nz = in_field.shape()[2];

	let in_field_filename = format!(
		"{}_{}_{}_in_field.npy",
		compiler.to_string().to_lowercase(),
		language.to_string().to_lowercase(),
		version
	);
    let in_field_file = File::create(in_field_filename)?;
    in_field.write_npy(in_field_file)?;

	let runtime = diffuse(
		compiler,
		language,
		&version,
		in_field,
		out_field,
		num_halo,
		alpha,
		num_iter
	)?;

	println!(
		"{} {} {} {} {} {} {} {}",
		compiler.to_string().to_lowercase(),
		language.to_string().to_lowercase(),
		version,
		nx,
		ny,
		nz,
		num_iter,
		runtime.as_secs_f64()
	);

	let out_field_filename = format!(
		"{}_{}_{}_out_field.npy",
		compiler.to_string().to_lowercase(),
		language.to_string().to_lowercase(),
		version
	);
    let out_field_file = File::create(out_field_filename)?;
    out_field.write_npy(out_field_file)?;

	Ok(runtime)
}
