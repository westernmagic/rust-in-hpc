fn add_dot_1x4(
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize
) {
    let mut c_00_reg = 0.0;
    let mut c_01_reg = 0.0;
    let mut c_02_reg = 0.0;
    let mut c_03_reg = 0.0;

    for p in 0..k {
        let a_0p_reg = a[0 + lda * p];
        c_00_reg += a_0p_reg * b[p + ldb * 0];
        c_01_reg += a_0p_reg * b[p + ldb * 1];
        c_02_reg += a_0p_reg * b[p + ldb * 2];
        c_03_reg += a_0p_reg * b[p + ldb * 3];
    }

    c[0 + ldc * 0] += c_00_reg;
    c[0 + ldc * 1] += c_01_reg;
    c[0 + ldc * 2] += c_02_reg;
    c[0 + ldc * 3] += c_03_reg;
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
