use num_complex::Complex64;

extern "C" {
    pub fn zaxpy_(a: *const Complex64, x: *const Complex64, nx: *const u64, y: *mut Complex64, ny: *const u64);
}
