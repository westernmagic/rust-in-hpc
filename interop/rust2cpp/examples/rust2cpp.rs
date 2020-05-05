use rust2cpp::zaxpy;
use ndarray::array;

fn main() {
    let a = 10.0;
    let x = array![1.0, 2.0, 3.0];
    let mut y = array![4.0, 5.0, 6.0];

    unsafe {
        zaxpy(a, x.as_ptr(), x.len(), y.as_mut_ptr(), y.len());
    };

    println!("{}", y);
}
