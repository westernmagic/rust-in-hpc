use ndarray::prelude::*;

fn main() {
    let a = array![
        [1.0, 2.0, 3.0,],
        [4.0, 5.0, 6.0,],
    ];
    let b = array![
        [1.0,  2.0,  3.0,  4.0,],
        [5.0,  6.0,  7.0,  8.0,],
        [9.0, 10.0, 11.0, 12.0,],
    ];
    let c = array![
        [2.0, 6.0, 0.0, 4.0,],
        [7.0, 2.0, 7.0, 2.0,],
    ];
    let r = array![
        [40.0,  50.0,  50.0,  60.0,],
        [90.0, 100.0, 120.0, 130.0,],
    ];

    // even though we are using ndarray with the blas feature enabled, I am not sure if it actually calls MKL/system blas or the blas that it downloads from blas-src...
    let c = 1.0 * a.dot(&b) + 1.0 * c;
    assert_eq!(c, r);
}
