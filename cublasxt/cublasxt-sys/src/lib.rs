#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

use num_complex::{Complex32, Complex64};

pub type cuFloatComplex = Complex32;
pub type cuDoubleComplex = Complex64;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
