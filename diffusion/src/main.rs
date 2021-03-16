#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]

mod utils;
mod partitioner;
mod diffusion;
mod halo;
use utils::write_field_to_file;
use diffusion::seq2 as apply_diffusion;
use anyhow::Result;
use structopt::StructOpt;
use ndarray::prelude::*;
use std::time::Instant;
use mpi::traits::*;
use fast_float::Fast;

type Value = f32;
// type Value = Fast<f32>;

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
    alpha: Value,
}

const num_halo: usize = 2;

#[paw::main]
fn main(args: Args) -> Result<()> {
    // init
    let nx = args.nx;
    let ny = args.ny;
    let nz = args.nz;
    let num_iter = args.num_iter;
    let alpha = args.alpha;
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    println!("# ranks nx ny nz num_iter time");
    println!("data = np.array( [ \\");

    // setup
    let mut in_field = Array3::<Value>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
    in_field.slice_mut(s![(num_halo + nx / 4)..(num_halo + 3 * nx / 4), (num_halo + ny / 4)..(num_halo + 3 * ny / 4), (nz / 4)..(3 * nz / 4)]).fill(1.0.into());
    /*
    for k in (nz / 4)..(3 * nz / 4) {
        for j in (num_halo + ny / 4)..(num_halo + 3 * ny / 4) {
            for i in (num_halo + nx / 4)..(num_halo + 3 * nx / 4) {
                in_field[[i, j, k]] = 1.0.into();
            }
        }
    }
    */
    let mut out_field = in_field.clone();

    write_field_to_file(&in_field, "./in_field.dat")?;

    // warmup caches
    apply_diffusion(&mut in_field, &mut out_field, alpha, 1);
    let start = Instant::now();
    apply_diffusion(&mut in_field, &mut out_field, alpha, num_iter);
    let end = Instant::now();

    write_field_to_file(&out_field, "./out_field.dat")?;

    let runtime = (end - start).as_secs_f64();

    println!("[{}, {}, {}, {}, {}, {}]", world.size(), nx, ny, nz, num_iter, runtime);

    println!("]");
    Ok(())
}
