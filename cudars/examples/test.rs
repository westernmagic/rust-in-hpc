#![feature(asm)]
#![feature(proc_macro_hygiene)]
#![no_std]
use cudars_derive::*;

pub fn t() {
    #[shared] static s: [f64; 4] = blah;
}
