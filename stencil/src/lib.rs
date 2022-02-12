#![feature(test)]

mod functions;
pub use functions::FUNCTIONS;

use anyhow::Result;
use derive_more::Display;
use ndarray::prelude::*;
use structopt::StructOpt;
use thiserror::Error;

use std::hint::black_box;
use std::str::FromStr;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
pub enum Error {
	#[error("function not found")]
	NotFound
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, StructOpt)]
pub enum Compiler {
	Gnu,
	Intel,
	Cray,
	Pgi,
	Rustc,
}

impl FromStr for Compiler {
	type Err = &'static str;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s.to_lowercase().as_str() {
			"gnu"   => Ok(Self::Gnu),
			"intel" => Ok(Self::Intel),
			"cray"  => Ok(Self::Cray),
			"pgi"   => Ok(Self::Pgi),
			"rustc" => Ok(Self::Rustc),
			_       => Err("Bad string"),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, StructOpt)]
pub enum Language {
	Cpp,
	Fortran,
	Rust,
}

impl FromStr for Language {
	type Err = &'static str;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s.to_lowercase().as_str() {
			"cpp"     | "c++" => Ok(Self::Cpp),
			"fortran" | "f"   => Ok(Self::Fortran),
			"rust"    | "rs"  => Ok(Self::Rust),
			_                 => Err("Bad string"),
		}
	}
}

pub type Function = unsafe extern "C" fn(*mut f32, *mut f32, usize, usize, usize, usize, f32, usize) -> ();

pub fn diffuse<'a>(
	compiler: Compiler,
	language: Language,
	version: &str,
	in_field: &'a mut Array3<f32>,
	out_field: &'a mut Array3<f32>,
	num_halo: usize,
	alpha: f32,
	num_iter: usize
) -> Result<Duration> {
	assert_eq!(in_field.shape()[0], out_field.shape()[0]);
	assert_eq!(in_field.shape()[1], out_field.shape()[1]);
	assert_eq!(in_field.shape()[2], out_field.shape()[2]);

	let nx = in_field.shape()[0] - 2 * num_halo;
	let ny = in_field.shape()[1] - 2 * num_halo;
	let nz = in_field.shape()[2];

	let diffuse = FUNCTIONS.get(&(compiler, language, version)).ok_or(Error::NotFound)?;

	unsafe {
		diffuse(
			in_field.as_mut_ptr(),
			out_field.as_mut_ptr(),
			nx,
			ny,
			nz,
			num_halo,
			alpha,
			1
		);
	}

	#[cfg(craypat)]
	craypat::record(true);
	let start = Instant::now();
	unsafe {
		diffuse(
			in_field.as_mut_ptr(),
			out_field.as_mut_ptr(),
			nx,
			ny,
			nz,
			num_halo,
			alpha,
			num_iter
		);
	}
	let end = Instant::now();
	#[cfg(craypat)]
	craypat::record(false);
	black_box(&out_field);

	Ok(end - start)
}
