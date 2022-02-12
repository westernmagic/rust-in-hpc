use ndarray::prelude::*;

#[no_mangle]
pub extern "C" fn diffuse_rustc_rs_laplap_seq_zip(
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
			azip!(
				(
					tmp   in tmp1_field.slice_mut(s![(num_halo - 1)..(nx + num_halo + 1), (num_halo - 1)..(ny + num_halo + 1)]),
					i_j_k in in_field.slice(      s![(num_halo - 1)..(nx + num_halo + 1), (num_halo - 1)..(ny + num_halo + 1), k]),
					imj_k in in_field.slice(      s![(num_halo - 2)..(nx + num_halo    ), (num_halo - 1)..(ny + num_halo + 1), k]),
					ipj_k in in_field.slice(      s![(num_halo    )..(nx + num_halo + 2), (num_halo - 1)..(ny + num_halo + 1), k]),
					i_jmk in in_field.slice(      s![(num_halo - 1)..(nx + num_halo + 1), (num_halo - 2)..(ny + num_halo    ), k]),
					i_jpk in in_field.slice(      s![(num_halo - 1)..(nx + num_halo + 1), (num_halo    )..(ny + num_halo + 2), k])
				) {
					*tmp = -4.0f32 * i_j_k + imj_k + ipj_k + i_jmk + i_jpk;
				}
			);

			if iter != num_iter - 1 {
				azip!(
					(
						out   in in_field.slice_mut(s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    ), k]),
						i_j_k in tmp1_field.slice(  s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    )]),
						imj_k in tmp1_field.slice(  s![(num_halo - 1)..(nx + num_halo - 1), (num_halo    )..(ny + num_halo    )]),
						ipj_k in tmp1_field.slice(  s![(num_halo + 1)..(nx + num_halo + 1), (num_halo    )..(ny + num_halo    )]),
						i_jmk in tmp1_field.slice(  s![(num_halo    )..(nx + num_halo    ), (num_halo - 1)..(ny + num_halo - 1)]),
						i_jpk in tmp1_field.slice(  s![(num_halo    )..(nx + num_halo    ), (num_halo + 1)..(ny + num_halo + 1)])
					) {
						let laplap = -4.0f32 * i_j_k + imj_k + ipj_k + i_jmk + i_jpk;
						*out = *out - alpha * laplap;
					}
				);
			} else {
				azip!(
					(
						out   in out_field.slice_mut(s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    ), k]),
						in_   in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    ), k]),
						i_j_k in tmp1_field.slice(   s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    )]),
						imj_k in tmp1_field.slice(   s![(num_halo - 1)..(nx + num_halo - 1), (num_halo    )..(ny + num_halo    )]),
						ipj_k in tmp1_field.slice(   s![(num_halo + 1)..(nx + num_halo + 1), (num_halo    )..(ny + num_halo    )]),
						i_jmk in tmp1_field.slice(   s![(num_halo    )..(nx + num_halo    ), (num_halo - 1)..(ny + num_halo - 1)]),
						i_jpk in tmp1_field.slice(   s![(num_halo    )..(nx + num_halo    ), (num_halo + 1)..(ny + num_halo + 1)])
					) {
						let laplap = -4.0f32 * i_j_k + imj_k + ipj_k + i_jmk + i_jpk;
						*out = in_ - alpha * laplap;
					}
				);
			}
        }
    }
    // update_halo(&mut out_field);
}
