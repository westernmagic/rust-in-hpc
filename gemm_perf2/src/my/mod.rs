use std::arch::x86_64::*;

fn mm(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    // a = m * k
    // b = k * n
    // c = m * n

    unsafe {
        let a0_ = _mm256_broadcast_sd(&a[0 * m]);
        let a1_ = _mm256_broadcast_sd(&a[1 * m]);
        let a2_ = _mm256_broadcast_sd(&a[2 * m]);
        let a3_ = _mm256_broadcast_sd(&a[3 * m]);

        let b_0 = _mm256_loadu_pd(b.as_ptr().offset(0 * (k as isize)));
        let b_1 = _mm256_loadu_pd(b.as_ptr().offset(1 * (k as isize)));
        let b_2 = _mm256_loadu_pd(b.as_ptr().offset(2 * (k as isize)));
        let b_3 = _mm256_loadu_pd(b.as_ptr().offset(3 * (k as isize)));
        
        let 
    }
}
