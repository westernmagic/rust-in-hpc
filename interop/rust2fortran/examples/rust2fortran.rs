use rust2fortran::zaxpy_ as zaxpy;
use ndarray::array;

fn main() {
    let a = 10.0;
    let x = array![1.0, 2.0, 3.0];
    let mut y = array![4.0, 5.0, 6.0];

    unsafe {
        zaxpy(&a as *const f64, x.as_ptr(), &x.len() as *const usize as *const u64, y.as_mut_ptr(), &y.len() as *const usize as *const u64);
    };

    println!("{}", y);
}

