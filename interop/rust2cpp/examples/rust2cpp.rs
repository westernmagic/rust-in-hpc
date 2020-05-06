use rust2cpp::zaxpy;
use ndarray::array;
use num_complex::Complex64;

fn main() {
    let     a = Complex64::new(1.0, 0.0);
    let     x = array![Complex64::new(1.1, 2.2), Complex64::new(3.3,  4.4),  Complex64::new(5.5, 6.6)];
    let mut y = array![Complex64::new(7.7, 8.8), Complex64::new(9.9, 10.10), Complex64::new(11.11, 12.12)];

    unsafe {
        zaxpy(&a as *const Complex64, x.as_ptr(), x.len(), y.as_mut_ptr(), y.len());
    };

    println!("{}", y);
}
