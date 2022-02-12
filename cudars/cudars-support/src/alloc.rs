use core::arch::nvptx::{malloc, free, trap};
use core_alloc::alloc::{Layout, GlobalAlloc};

pub struct CudaAllocator;

unsafe impl GlobalAlloc for CudaAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        malloc(layout.size()) as *mut u8
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        free(ptr as *mut _);
    }
}

#[alloc_error_handler]
fn alloc_error_handler(_layout: Layout) -> ! {
    unsafe { trap() }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CudaAllocator = CudaAllocator;
