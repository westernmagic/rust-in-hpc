#![cfg(target_os = "cuda")]
#![feature(stdsimd, alloc_error_handler, core_intrinsics)]
#![no_std]

extern crate alloc as core_alloc;
mod alloc;
mod print;
mod panic;

pub use core_alloc::format;
pub use alloc::*;
pub use panic::*;
pub use print::*;
