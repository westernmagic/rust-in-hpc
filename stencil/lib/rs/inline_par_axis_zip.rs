use ndarray::prelude::*;
use itertools::izip;
use rayon::prelude::*;
use std::mem::swap;

#[no_mangle]
pub extern "C" fn diffuse_rustc_rs_inline_par_axis_zip(
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

	let alpha_20 = -20.0f32 * alpha + 1.0f32;
	let alpha_08 =   8.0f32 * alpha;
	let alpha_02 =  -2.0f32 * alpha;
	let alpha_01 =  -1.0f32 * alpha;


    for _iter in 0..num_iter {
		izip!(in_field.axis_iter(Axis(2)), out_field.axis_iter_mut(Axis(2))).par_bridge().for_each(|(in_field, mut out_field)| {
			azip!((
			    out     in out_field.slice_mut(s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    )]),
			    i0_j0_k in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo    )..(ny + num_halo    )]),
			    i1mj0_k in in_field.slice(     s![(num_halo - 1)..(nx + num_halo - 1), (num_halo    )..(ny + num_halo    )]),
			    i1pj0_k in in_field.slice(     s![(num_halo + 1)..(nx + num_halo + 1), (num_halo    )..(ny + num_halo    )]),
			    i0_j1mk in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo - 1)..(ny + num_halo - 1)]),
			    i0_j1pk in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo + 1)..(ny + num_halo + 1)]),
			    i1mj1mk in in_field.slice(     s![(num_halo - 1)..(nx + num_halo - 1), (num_halo - 1)..(ny + num_halo - 1)]),
			    i1mj1pk in in_field.slice(     s![(num_halo - 1)..(nx + num_halo - 1), (num_halo + 1)..(ny + num_halo + 1)]),
			    i1pj1mk in in_field.slice(     s![(num_halo + 1)..(nx + num_halo + 1), (num_halo - 1)..(ny + num_halo - 1)]),
			    i1pj1pk in in_field.slice(     s![(num_halo + 1)..(nx + num_halo + 1), (num_halo + 1)..(ny + num_halo + 1)]),
			    i2mj0_k in in_field.slice(     s![(num_halo - 2)..(nx + num_halo - 2), (num_halo    )..(ny + num_halo    )]),
			    i2pj0_k in in_field.slice(     s![(num_halo + 2)..(nx + num_halo + 2), (num_halo    )..(ny + num_halo    )]),
			    i0_j2mk in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo - 2)..(ny + num_halo - 2)]),
			    i0_j2pk in in_field.slice(     s![(num_halo    )..(nx + num_halo    ), (num_halo + 2)..(ny + num_halo + 2)]),
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
		});

        swap(&mut in_field, &mut out_field)
    }
}
