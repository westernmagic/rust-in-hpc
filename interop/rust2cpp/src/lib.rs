#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use num_complex::Complex;

type std_complex<T> = Complex<T>;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
