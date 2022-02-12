fn add_dot(
    k: usize,
    x: &[f64],
    incx: usize,
    y: &[f64],
    gamma: &mut f64
) {
    for p in 0..k {
        *gamma += x[p * incx] * y[p];
    }
}

pub fn mm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize
) {
    for j in (0..n).step_by(4) {
        for i in 0..m {
            add_dot(k, &a[(i + lda * 0)..], lda, &b[(0 + ldb * (j + 0))..], &mut c[i + ldc * (j + 0)]);
            add_dot(k, &a[(i + lda * 0)..], lda, &b[(0 + ldb * (j + 1))..], &mut c[i + ldc * (j + 1)]);
            add_dot(k, &a[(i + lda * 0)..], lda, &b[(0 + ldb * (j + 2))..], &mut c[i + ldc * (j + 2)]);
            add_dot(k, &a[(i + lda * 0)..], lda, &b[(0 + ldb * (j + 3))..], &mut c[i + ldc * (j + 3)]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mm() {
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f64> = (0..).take(m * k).map(|x| x.into()).collect();
        let b: Vec<f64> = (0..).take(k * n).map(|x| x.into()).collect();
        let mut c: Vec<f64> = vec![0.0; m * n];

        mm(m, n, k, &a, m, &b, k, &mut c, m);
    }
}
