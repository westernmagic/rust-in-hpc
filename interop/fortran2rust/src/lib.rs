use std::slice;
use num_complex::Complex64;

#[no_mangle]
pub extern "C" fn zaxpy(a: *const Complex64, x: *const Complex64, nx: usize, y: *mut Complex64, ny: usize) {
    let a = unsafe { *a };
    let x = unsafe { slice::from_raw_parts(x, nx) };
    let y = unsafe { slice::from_raw_parts_mut(y, ny) };

    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi += a * (*xi);
    }
}
