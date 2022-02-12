// We need these nightly features for CUDA
//  - `abi_ptx` for `extern "ptx-kernel"`
//  - `stdsimd` for `_block_dim_x()`, etc.
// CUDA codes needs to be `no_std`, since no standard library is available
#![cfg_attr(target_os = "cuda", feature(abi_ptx, stdsimd))]
#![cfg_attr(target_os = "cuda", no_std)]

// Split the module imports by target
cfg_if::cfg_if! {
    if #[cfg(target_os = "cuda")] {
        use ptx_support::prelude::*;
        use core::arch::nvptx::*;
    } else {
        use rustacuda::prelude::*;
        use rustacuda::launch;
        use rustacuda::function::{BlockSize, GridSize};
        use std::error::Error;
        use std::ffi::CString;
    }
}

// This is the way to define a CUDA kernel (C: `__global__`)
//  - `no_mangle` so that the name is not mangled by the compiler, and we can easily find it in the PTX
//  - `cfg(target_os = "cuda") to conditionally compile only for CUDA
//  - `pub` so that the optimizer doesn't delete the kernel
//  - `unsafe`, since its FFI
//  - `extern "ptx-kernel" which is equivalent to `__global__`
// Ideally, we would hide all this behind a `proc_macro`, which would add all these attributes
#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn sum(x: *const f32, y: *const f32, result: *mut f32) {
    let i = (_block_dim_x() * _block_idx_x() + _thread_idx_x()) as isize;

    *result.offset(i) = *x.offset(i) + *y.offset(i);
}

// Host side wrapper for the kernel
// A `proc_macro` could also generate this
#[cfg(not(target_os = "cuda"))]
fn sum(module: &Module, blocks: GridSize, threads: BlockSize, shared: u32, stream: &Stream, x: &[f32], y: &[f32], result: &mut [f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let n = (blocks.x * threads.x) as usize;
    assert_eq!(n, x.len());
    assert_eq!(n, y.len());

    let mut x = DeviceBuffer::from_slice(&x)?;
    let mut y = DeviceBuffer::from_slice(&y)?;
    let mut result = DeviceBuffer::from_slice(&result)?;

    // Launching the kernel is unsafe, because we are crossing a FFI boundary
    unsafe {
        launch!(
            module.sum<<<blocks, threads, shared, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                result.as_device_ptr()
            )
        )?;
    }
    stream.synchronize()?;

    let mut result_host = vec![0.0f32; n];
    result.copy_to(&mut result_host)?;

    Ok(result_host)
}

#[cfg(not(target_os = "cuda"))]
fn main() -> Result<(), Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    // We must keep a reference to the context, otherwise it will get optimized away
    let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // The build script compiles the CUDA code to a PTX file at `${KERNEL_PTX_PATH}`
    // Load this file *at compile time* into a CString and pass it to the module initializer at
    // runtime for JIT-ing
    let ptx = CString::new(include_str!(env!("KERNEL_PTX_PATH")))?;
    let module = Module::load_from_string(&ptx)?;

    let x = vec![1.0f32, 2.0f32, 3.0f32];
    let y = vec![4.0f32, 5.0f32, 6.0f32];
    let mut result = vec![0.0f32; 3];
    let result_host = sum(&module, 1.into(), 3.into(), 0, &stream, &x, &y, &mut result)?;

    println!("Sums are {}, {}, {}", result_host[0], result_host[1], result_host[2]);

    Ok(())
}
