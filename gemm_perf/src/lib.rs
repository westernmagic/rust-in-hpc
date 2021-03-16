extern crate blas;
extern crate openblas_src;

#[inline]
pub fn rust_dgemm(
    m: u16,
    n: u16,
    k: u16,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64]
) {
    unsafe {
        matrixmultiply::dgemm(m as usize, k as usize, n as usize, alpha, a.as_ptr(), 1, m as isize, b.as_ptr(), 1, k as isize, beta, c.as_mut_ptr(), 1, m as isize)
    }
}

#[inline]
pub fn blas_dgemm(
    m: u16,
    n: u16,
    k: u16,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64]
) {
    unsafe {
        blas::dgemm(b'N', b'N', m as i32, n as i32, k as i32, alpha, a, m as i32, b, k as i32, beta, c, m as i32);
    }
}
