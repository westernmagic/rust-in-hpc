use core::arch::nvptx::*;
use core::panic::PanicInfo;
use crate::{print, println};

#[allow(unreachable_code)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    unsafe {
        print!(
            "CUDA thread ({}, {}, {}) on block ({}, {}, {}) panicked",
            _thread_idx_x(), _thread_idx_y(), _thread_idx_z(),
            _block_idx_x(), _block_idx_y(), _block_idx_z()
        );
    }

    if let Some(s) = info.payload().downcast_ref::<&str>() {
        print!("at '{}'", s);
    }

    if let Some(location) = info.location() {
        print!("{}:{}:{}", location.file(), location.line(), location.column());
    }

    println!();

    unsafe {
        core::intrinsics::breakpoint();
        trap();
        core::hint::unreachable_unchecked();
    }
}
