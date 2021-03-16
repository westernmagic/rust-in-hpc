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

fn add_dot_1x4(
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize
) {
    add_dot(k, &a[(0 + lda * 0)..], lda, &b[(0 + ldb * 0)..], &mut c[0 + ldc * 0]);
    add_dot(k, &a[(0 + lda * 0)..], lda, &b[(0 + ldb * 1)..], &mut c[0 + ldc * 1]);
    add_dot(k, &a[(0 + lda * 0)..], lda, &b[(0 + ldb * 2)..], &mut c[0 + ldc * 2]);
    add_dot(k, &a[(0 + lda * 0)..], lda, &b[(0 + ldb * 3)..], &mut c[0 + ldc * 3]);
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
            add_dot_1x4(k, &a[(i + lda * 0)..], lda, &b[(0 + ldb * (j + 0))..], ldb, &mut c[(i + ldc * (j + 0))..], ldc);
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
