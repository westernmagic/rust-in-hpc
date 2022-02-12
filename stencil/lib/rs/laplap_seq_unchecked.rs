use ndarray::prelude::*;

#[no_mangle]
pub extern "C" fn diffuse_rustc_rs_laplap_seq_unchecked(
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

    let mut in_field = unsafe { ArrayViewMut3::<f32>::from_shape_ptr((nx + 2 * num_halo, ny + 2 * num_halo, nz).f(), in_field) };
    let mut out_field = unsafe { ArrayViewMut3::<f32>::from_shape_ptr((nx + 2 * num_halo, ny + 2 * num_halo, nz).f(), out_field) };

    let mut tmp1_field = Array2::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo).f());

    for iter in 0..num_iter {
        // update_halo(&mut in_field);
        for k in 0..nz {
            for j in (num_halo - 1)..(ny + num_halo + 1) {
                for i in (num_halo - 1)..(nx + num_halo + 1) {
                    unsafe{
                        *tmp1_field.uget_mut([i, j]) =
                            -4.0f32 * in_field.uget([i,     j,     k])
                            +         in_field.uget([i - 1, j,     k])
                            +         in_field.uget([i + 1, j,     k])
                            +         in_field.uget([i,     j - 1, k])
                            +         in_field.uget([i,     j + 1, k])
                        ;
                    }
                }
            }

            for j in num_halo..(ny + num_halo) {
                for i in num_halo..(nx + num_halo) {
                    let laplap = unsafe {
                        -4.0f32 * tmp1_field.uget([i,     j,   ])
                        +         tmp1_field.uget([i - 1, j,   ])
                        +         tmp1_field.uget([i + 1, j,   ])
                        +         tmp1_field.uget([i,     j - 1])
                        +         tmp1_field.uget([i,     j + 1])
                    };

                    if iter != num_iter - 1 {
                        unsafe {
                            *in_field.uget_mut([i, j, k]) = in_field.uget([i, j, k]) - alpha * laplap;
                        }
                    } else {
                        unsafe {
                            *out_field.uget_mut([i, j, k]) = in_field.uget([i, j, k]) - alpha * laplap;
                        }
                    }
                }
            }
        }
    }
    // update_halo(&mut out_field);
}
