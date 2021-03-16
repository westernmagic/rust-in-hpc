#![feature(test)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use anyhow::Result;
use structopt::StructOpt;
use ndarray::prelude::*;
use ndarray_npy::*;
use std::time::Instant;
use std::hint::black_box;
use std::fs::File;
use stencil_lib::*;

#[derive(StructOpt)]
struct Args {
    #[structopt(long = "--nx")]
    nx: usize,
    #[structopt(long = "--ny")]
    ny: usize,
    #[structopt(long = "--nz")]
    nz: usize,
    #[structopt(long = "--num_iter")]
    num_iter: usize,
    #[structopt(long = "--alpha", default_value = "0.03125")]
    alpha: f32,
    #[structopt(subcommand)]
    version: Version,
}

#[derive(StructOpt)]
enum Version {
    diffuse_cpp_v0_base,
    diffuse_cpp_v1_inline,

    diffuse_f_v0_base,
    diffuse_f_v1_inline,
    diffuse_f_v2_openmp,
    diffuse_f_v3_openmp,

    diffuse_rs_v0_base,
    diffuse_rs_v0_unchecked,
    diffuse_rs_v1_inline,
    diffuse_rs_v2_fast,
    diffuse_rs_v3_fma,
}

const num_halo: usize = 2;

macro_rules! run {
    ($fn: path, $in_field: ident, $out_field: ident, $num_halo: ident, $alpha: ident, $num_iter: ident) => {
        $fn(&mut $in_field, &mut $out_field, $num_halo, $alpha, 1);
        let start = Instant::now();
        $fn(&mut $in_field, &mut $out_field, $num_halo, $alpha, $num_iter);
        let end = Instant::now();
        let runtime = (end - start).as_secs_f64();
        black_box(&$out_field);

        let nx = $in_field.shape()[0] - 2 * $num_halo;
        let ny = $in_field.shape()[1] - 2 * $num_halo;
        let nz = $in_field.shape()[2];
        println!("{} {} {} {} {} {}", stringify!($fn), nx, ny, nz, $num_iter, runtime);
    }
}

#[paw::main]
fn main(args: Args) -> Result<()> {
    let nx = args.nx;
    let ny = args.ny;
    let nz = args.nz;
    let num_iter = args.num_iter;
    let alpha = args.alpha;
    let version = args.version;

    let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
    in_field.slice_mut(s![
        (num_halo + nx / 4)..(num_halo + 3 * nx / 4),
        (num_halo + ny / 4)..(num_halo + 3 * ny / 4),
        (nz / 4)..(3 * nz / 4)
    ]).fill(1.0f32);
    let mut in_field = in_field;
    let mut out_field = in_field.clone();

    let in_field_file = File::create("in_field.npy")?;
    in_field.write_npy(in_field_file)?;

    match version {
        Version::diffuse_cpp_v0_base => {
            run!(cpp::v0_base::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
        Version::diffuse_cpp_v1_inline => {
            run!(cpp::v1_inline::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },

        Version::diffuse_f_v0_base => {
            run!(f::v0_base::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
        Version::diffuse_f_v1_inline => {
            run!(f::v1_inline::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
        Version::diffuse_f_v2_openmp => {
            run!(f::v2_openmp::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
        Version::diffuse_f_v3_openmp => {
            run!(f::v3_openmp::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },

        Version::diffuse_rs_v0_base => {
            run!(rs::v0_base::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
        Version::diffuse_rs_v0_unchecked => {
            run!(rs::v0_unchecked::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
        Version::diffuse_rs_v1_inline => {
            run!(rs::v1_inline::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
        Version::diffuse_rs_v2_fast => {
            run!(rs::v2_fast::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
        Version::diffuse_rs_v3_fma => {
            run!(rs::v3_fma::diffuse, in_field, out_field, num_halo, alpha, num_iter);
        },
    };

    let out_field_file = File::create("out_field.npy")?;
    out_field.write_npy(out_field_file)?;

    Ok(())
}
