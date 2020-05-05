use std::slice;

#[no_mangle]
pub extern "C" fn zaxpy(a: f64, x: *const f64, nx: usize, y: *mut f64, ny: usize) {
    let x = unsafe { slice::from_raw_parts(x, nx) };
    let y = unsafe { slice::from_raw_parts_mut(y, ny) };

    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi += a * (*xi);
    }
}
