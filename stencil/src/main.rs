#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use anyhow::Result;
use structopt::StructOpt;
use ndarray::prelude::*;
use ndarray_npy::*;
use std::fs::File;
use stencil::*;

#[derive(StructOpt)]
struct Args {
    #[structopt(long, env, default_value = "128")]
    nx: usize,
    #[structopt(long, env, default_value = "128")]
    ny: usize,
    #[structopt(long, env, default_value = "64")]
    nz: usize,
    #[structopt(long, env, default_value = "1024")]
    num_iter: usize,
    #[structopt(long, env, default_value = "0.03125")]
    alpha: f32,
	#[structopt(long)]
	all: bool,
	#[structopt(long)]
	list: bool,
	#[structopt(required_unless_one(&["all", "list"]))]
	compiler: Option<Compiler>,
	#[structopt(required_unless_one(&["all", "list"]))]
	language: Option<Language>,
	#[structopt(required_unless_one(&["all", "list"]))]
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
	
	let nthreads_omp = std::env::var("OMP_NUM_THREADS").map(|n| usize::from_str_radix(&n, 10).expect("Cannot parse integer"));
	let nthreads_rayon = std::env::var("RAYON_NUM_THREADS").map(|n| usize::from_str_radix(&n ,10).expect("Cannot parse integer"));
	let nthreads = match (nthreads_omp, nthreads_rayon) {
		(Ok(nthreads_omp), Ok(nthreads_rayon)) => {
			if nthreads_omp == nthreads_rayon {
				nthreads_omp
			} else {
				panic!("Different thread counts for OpenMP and Rayon")
			}
		},
		(Ok(nthreads_omp), Err(_)) => { nthreads_omp },
		(Err(_), Ok(nthreads_rayon)) => { nthreads_rayon },
		(Err(_), Err(_)) => { panic!("No thread counts given for OpenMP and Rayon"); }
	};

	std::env::set_var("OMP_NUM_THREADS", format!("{}", nthreads));
	std::env::set_var("RAYON_NUM_THREADS", format!("{}", nthreads));

	if args.list {
		let mut keys: Vec<_> = FUNCTIONS.keys().map(|k| (format!("{}", k.0).to_lowercase(), format!("{}", k.1).to_lowercase(), k.2)).collect();
		keys.sort_unstable();
		for (compiler, language, version) in keys {
			println!("{:6} {:8} {}", compiler, language, version);
		}
		return Ok(())
	}

    let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
    in_field.slice_mut(s![
        (num_halo + nx / 4)..(num_halo + 3 * nx / 4),
        (num_halo + ny / 4)..(num_halo + 3 * ny / 4),
        (nz / 4)..(3 * nz / 4)
    ]).fill(1.0f32);
    let mut in_field = in_field;
    let mut out_field = in_field.clone();

	if args.all {
		run_all(&in_field, &out_field, alpha, num_iter, nthreads)?;
	} else {
		run(compiler.unwrap(), language.unwrap(), &version.unwrap(), &mut in_field, &mut out_field, alpha, num_iter, nthreads)?;
	}

    Ok(())
}

fn run_all(
	in_field: &Array3<f32>,
	out_field: &Array3<f32>,
	alpha: f32,
	num_iter: usize,
	nthreads: usize
) -> Result<()> {
	for (compiler, language, version) in FUNCTIONS.keys() {
		run(*compiler, *language, version, &mut in_field.clone(), &mut out_field.clone(), alpha, num_iter, nthreads)?;
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
	num_iter: usize,
	nthreads: usize
) -> Result<()> {
	let nx = in_field.shape()[0] - 2 * num_halo;
	let ny = in_field.shape()[1] - 2 * num_halo;
	let nz = in_field.shape()[2];

	let in_field_filename = format!(
		"data/{}_{}_{}_{}_{}_{}_{}_{}_in_field.npy",
		compiler.to_string().to_lowercase(),
		language.to_string().to_lowercase(),
		version,
		nx,
		ny,
		nz,
		num_iter,
		nthreads
	);
    let in_field_file = File::create(in_field_filename)?;
    in_field.write_npy(in_field_file)?;

	if let Ok(runtime) = diffuse(
		compiler,
		language,
		&version,
		in_field,
		out_field,
		num_halo,
		alpha,
		num_iter
	) {
		let out_baseline_filename = format!(
			"out_field_{}_{}_{}_{}_base.npy",
			nx,
			ny,
			nz,
			num_iter
		);
		if let Ok(out_baseline) = read_npy::<_, Array3<f32>>(out_baseline_filename) {
			let error = (out_baseline - (out_field as &Array3<f32>)).iter().fold(0.0f32, |m, x| m.max(x.abs()));

			println!(
				"{:6} {:8} {:22} {:5} {:5} {:3} {:5} {:3} {:>15.9} {:>12.9}",
				compiler.to_string().to_lowercase(),
				language.to_string().to_lowercase(),
				version,
				nx,
				ny,
				nz,
				num_iter,
				nthreads,
				runtime.as_secs_f64(),
				error
			);
		} else {
			println!(
				"{:6} {:8} {:22} {:5} {:5} {:3} {:5} {:3} {:>15.9} {:>12.9}",
				compiler.to_string().to_lowercase(),
				language.to_string().to_lowercase(),
				version,
				nx,
				ny,
				nz,
				num_iter,
				nthreads,
				runtime.as_secs_f64(),
				"NA"
			);
		}

		let out_field_filename = format!(
			"data/{}_{}_{}_{}_{}_{}_{}_{}_out_field.npy",
			compiler.to_string().to_lowercase(),
			language.to_string().to_lowercase(),
			version,
			nx,
			ny,
			nz,
			num_iter,
			nthreads
		);
    	let out_field_file = File::create(out_field_filename)?;
    	out_field.write_npy(out_field_file)?;
	} else {
		println!(
			"{:6} {:8} {:22} {:5} {:5} {:3} {:5} {:3} {:>15.9} {:>12.9}",
			compiler.to_string().to_lowercase(),
			language.to_string().to_lowercase(),
			version,
			nx,
			ny,
			nz,
			num_iter,
			nthreads,
			"NA",
			"NA"
		);
	}

	Ok(())
}
