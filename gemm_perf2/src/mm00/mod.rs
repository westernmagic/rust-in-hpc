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
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i + ldc * j] += a[i + lda * p] * b[p + ldb * j];
            }
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
