#![feature(abi_ptx)]
#![no_std]

extern crate dummy_utils;
use dummy_utils::{dummy_mul, PAIR};

#[no_mangle]
pub fn dummy_square(x: f64) -> f64 {
    PAIR[0] as f64 * dummy_mul(x, x)
}

#[no_mangle]
pub fn dummy_mul_2(x: f64) -> f64 {
    x * 2.0
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn dummy_math_kernel(x: *mut f64, y: *mut f64) {
    *y.offset(0) = PAIR[1] as f64 + dummy_square(*x.offset(0));
}
