use accel::*;

#[kernel]
unsafe fn diffuse_laplap_accel_kernel1(
	in_field: *mut f32,
	out_field: *mut f32,
	nx: usize,
	ny: usize,
	nz: usize,
	num_halo: usize,
	alpha: f32,
	num_iter: usize,
	iter: usize,
	tmp_field: *mut f32
) {
	let blockDim  = accel_core::block_dim();
	let blockIdx  = accel_core::block_idx();
	let threadIdx = accel_core::thread_idx();
	let i = (blockDim.x * blockIdx.x + threadIdx.x) as usize;
	let j = (blockDim.y * blockIdx.y + threadIdx.y) as usize;
	let k = (blockDim.z * blockIdx.z + threadIdx.z) as usize;

	let index = |i, j, k| { (i + j * (nx + 2 * num_halo) + k * (nx + 2 * num_halo) * (ny + 2 * num_halo)) as isize };

	if (
		0            <= k && k < nz                &&
		num_halo - 1 <= j && j < ny + num_halo + 1 &&
		num_halo - 1 <= i && i < nx + num_halo + 1
	) {
		*tmp_field.offset(index(i, j, k)) =
			-4.0f32 * *in_field.offset(index(i,     j,     k))
			+         *in_field.offset(index(i - 1, j,     k))
			+         *in_field.offset(index(i + 1, j,     k))
			+         *in_field.offset(index(i,     j - 1, k))
			+         *in_field.offset(index(i,     j + 1, k))
		;
	}
}

#[kernel]
unsafe fn diffuse_laplap_accel_kernel2(
	in_field: *mut f32,
	out_field: *mut f32,
	nx: usize,
	ny: usize,
	nz: usize,
	num_halo: usize,
	alpha: f32,
	num_iter: usize,
	iter: usize,
	tmp_field: *mut f32
) {
	let blockDim = accel_core::block_dim();
	let blockIdx = accel_core::block_idx();
	let threadIdx = accel_core::thread_idx();
	let i = (blockDim.x * blockIdx.x + threadIdx.x) as usize;
	let j = (blockDim.y * blockIdx.y + threadIdx.y) as usize;
	let k = (blockDim.z * blockIdx.z + threadIdx.z) as usize;

	let index = |i, j, k| { (i + j * (nx + 2 * num_halo) + k * (nx + 2 * num_halo) * (ny + 2 * num_halo)) as isize };

	if (
		0        <= k && k < nz            &&
		num_halo <= j && j < ny + num_halo &&
		num_halo <= i && i < nx + num_halo
	) {
		let laplap =
			-4.0f32 * *tmp_field.offset(index(i,     j,     k))
			+         *tmp_field.offset(index(i - 1, j,     k))
			+         *tmp_field.offset(index(i + 1, j,     k))
			+         *tmp_field.offset(index(i,     j - 1, k))
			+         *tmp_field.offset(index(i,     j + 1, k))
		;

		if iter != num_iter - 1 {
			*in_field.offset(index(i, j, k)) = *in_field.offset(index(i, j, k)) - alpha * laplap;
		} else {
			*out_field.offset(index(i, j, k)) = *in_field.offset(index(i, j, k)) - alpha * laplap;
		}
	}
}

#[no_mangle]
pub extern "C" fn diffuse_rustc_rs_laplap_accel(
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

	let device = Device::nth(0).unwrap();
	let ctx = device.create_context();

	let size = (nx + 2 * num_halo) * (ny + 2 * num_halo) * nz;
	let in_field_s  = unsafe { std::slice::from_raw_parts_mut(in_field,  size) };
	let out_field_s = unsafe { std::slice::from_raw_parts_mut(out_field, size) };
	let mut in_field_d  = unsafe { DeviceMemory::<f32>::uninitialized(ctx.clone(), size) };
	let mut out_field_d = unsafe { DeviceMemory::<f32>::uninitialized(ctx.clone(), size) };
	let mut tmp_field_d = unsafe { DeviceMemory::<f32>::uninitialized(ctx.clone(), size) };

	in_field_d.copy_from(in_field_s);

	#[allow(non_snake_case)]
	let blockDim = Block::xyz(1, 1, 1);
	#[allow(non_snake_case)]
	let gridDim = Grid::xyz(
		(nx + 2 * num_halo + blockDim.x as usize - 1) / blockDim.x as usize,
		(ny + 2 * num_halo + blockDim.y as usize - 1) / blockDim.y as usize,
		(nz                + blockDim.z as usize - 1) / blockDim.z as usize
	);

	for iter in 0..num_iter {
		diffuse_laplap_accel_kernel1(
			ctx.clone(),
			gridDim,
			blockDim,
			&(
				&in_field_d.as_mut_ptr(),
				&out_field_d.as_mut_ptr(),
				&nx,
				&ny,
				&nz,
				&num_halo,
				&alpha,
				&num_iter,
				&iter,
				&tmp_field_d.as_mut_ptr()
			)
		).unwrap();
		diffuse_laplap_accel_kernel2(
			ctx.clone(),
			gridDim,
			blockDim,
			&(
				&in_field_d.as_mut_ptr(),
				&out_field_d.as_mut_ptr(),
				&nx,
				&ny,
				&nz,
				&num_halo,
				&alpha,
				&num_iter,
				&iter,
				&tmp_field_d.as_mut_ptr()
			)
		).unwrap();
	}

	out_field_s.copy_from(&out_field_d);
}
