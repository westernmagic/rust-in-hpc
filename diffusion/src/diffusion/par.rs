use crate::halo::seq as update_halo;
use ndarray::prelude::*;
use ndarray::parallel::par_azip;
use std::mem::swap;
use crate::num_halo;

#[inline]
pub fn apply_diffusion<'a>(in_field: &'a mut Array3<f32>, out_field: &'a mut Array3<f32>, alpha: f32, num_iter: usize) {
    let nx = in_field.shape()[0] - 2 * num_halo;
    let ny = in_field.shape()[1] - 2 * num_halo;
    let nz = in_field.shape()[2];
    let alpha_20 = -20.0f32 * alpha + 1.0f32;
    let alpha_08 =   8.0f32 * alpha;
    let alpha_02 =  -2.0f32 * alpha;
    let alpha_01 =  -1.0f32 * alpha;

    for iter in 0..num_iter {
        update_halo(in_field, num_halo);

        for k in 0..nz {
            par_azip!((
                out     in out_field.slice_mut(s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    ), k]),
                i0_j0_k in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    ), k]),
                i1mj0_k in in_field.slice(     s![(num_halo - 1)..(nx + num_halo - 1), (num_halo    )..(ny + num_halo    ), k]),
                i1pj0_k in in_field.slice(     s![(num_halo + 1)..(nx + num_halo + 1), (num_halo    )..(ny + num_halo    ), k]),
                i0_j1mk in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo - 1)..(ny + num_halo - 1), k]),
                i0_j1pk in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo + 1)..(ny + num_halo + 1), k]),
                i1mj1mk in in_field.slice(     s![(num_halo - 1)..(nx + num_halo - 1), (num_halo - 1)..(ny + num_halo - 1), k]),
                i1mj1pk in in_field.slice(     s![(num_halo - 1)..(nx + num_halo - 1), (num_halo + 1)..(ny + num_halo + 1), k]),
                i1pj1mk in in_field.slice(     s![(num_halo + 1)..(nx + num_halo + 1), (num_halo - 1)..(ny + num_halo - 1), k]),
                i1pj1pk in in_field.slice(     s![(num_halo + 1)..(nx + num_halo + 1), (num_halo + 1)..(ny + num_halo + 1), k]),
                i2mj0_k in in_field.slice(     s![(num_halo - 2)..(nx + num_halo - 2), (num_halo    )..(ny + num_halo    ), k]),
                i2pj0_k in in_field.slice(     s![(num_halo + 2)..(nx + num_halo + 2), (num_halo    )..(ny + num_halo    ), k]),
                i0_j2mk in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo - 2)..(ny + num_halo - 2), k]),
                i0_j2pk in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo + 2)..(ny + num_halo + 2), k]),
            ) {
                *out =
                      alpha_20 * i0_j0_k
                    + alpha_08 * i1mj0_k
                    + alpha_08 * i1pj0_k
                    + alpha_08 * i0_j1mk
                    + alpha_08 * i0_j1pk
                    + alpha_02 * i1mj1mk
                    + alpha_02 * i1mj1pk
                    + alpha_02 * i1pj1mk
                    + alpha_02 * i1pj1pk
                    + alpha_01 * i2mj0_k
                    + alpha_01 * i2pj0_k
                    + alpha_01 * i0_j2mk
                    + alpha_01 * i0_j2pk
                ;
            });

            swap(in_field, out_field)
        }
    }

    update_halo(out_field, num_halo);
}
