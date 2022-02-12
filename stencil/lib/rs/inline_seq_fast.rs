use ndarray::prelude::*;
use fast_float::Fast;
use std::mem::swap;

#[no_mangle]
pub extern "C" fn diffuse_rustc_rs_inline_seq_fast(
    in_field: *mut f32,
    out_field: *mut f32,
    nx: usize,
    ny: usize,
    nz: usize,
    num_halo: usize,
    alpha: f32,
    num_iter: usize
) {
    assert!(!in_field.is_null());
    assert!(!out_field.is_null());
    assert!(nx > 0);
    assert!(ny > 0);
    assert!(nz > 0);
    assert!(num_halo > 0);
    assert!(!alpha.is_nan());
    assert!(num_iter > 0);

    let mut in_field = unsafe { ArrayViewMut3::<Fast<f32>>::from_shape_ptr((nx + 2 * num_halo, ny + 2 * num_halo, nz).f(), in_field as *mut _) };
    let mut out_field = unsafe { ArrayViewMut3::<Fast<f32>>::from_shape_ptr((nx + 2 * num_halo, ny + 2 * num_halo, nz).f(), out_field as *mut _) };

    let alpha_20 = Fast::<f32>::from(-20.0f32 * alpha + 1.0f32);
    let alpha_08 = Fast::<f32>::from(  8.0f32 * alpha);
    let alpha_02 = Fast::<f32>::from( -2.0f32 * alpha);
    let alpha_01 = Fast::<f32>::from( -1.0f32 * alpha);

    for _iter in 0..num_iter {
        // update_halo(&mut in_field);
        for k in 0..nz {
            for j in num_halo..(ny + num_halo) {
                for i in num_halo..(nx + num_halo) {
                    unsafe {
                        *out_field.uget_mut([i, j, k]) =
                              alpha_20 * *in_field.uget([i, j, k        ])
                            + alpha_08 * *in_field.uget([i - 1, j,     k])
                            + alpha_08 * *in_field.uget([i + 1, j,     k])
                            + alpha_08 * *in_field.uget([i,     j - 1, k])
                            + alpha_08 * *in_field.uget([i,     j + 1, k])
                            + alpha_02 * *in_field.uget([i - 1, j - 1, k])
                            + alpha_02 * *in_field.uget([i - 1, j + 1, k])
                            + alpha_02 * *in_field.uget([i + 1, j - 1, k])
                            + alpha_02 * *in_field.uget([i + 1, j + 1, k])
                            + alpha_01 * *in_field.uget([i - 2, j,     k])
                            + alpha_01 * *in_field.uget([i + 2, j,     k])
                            + alpha_01 * *in_field.uget([i,     j - 2, k])
                            + alpha_01 * *in_field.uget([i,     j + 2, k])
                        ;
                    }
                }
            }
        }

        swap(&mut in_field, &mut out_field);
    }
    // update_halo(&mut out_field);
}
