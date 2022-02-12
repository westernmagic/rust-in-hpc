#![cfg_attr(target_os = "cuda", feature(abi_ptx), feature(stdsimd), feature(asm), feature(proc_macro_hygiene), no_std)]

use cudars_derive::{global, host, device, shared};
use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(target_os = "cuda")] {
        mod dim;

        use dim::*;
        use core::arch::nvptx::*;
        use cudars_support::*;
        use core::convert::TryInto;
    } else {
        use std::ffi::CString;
        use rustacuda::prelude::*;
        use rustacuda::launch;
        use rustacuda::function::{GridSize, BlockSize};
        use rustacuda::memory::DeviceBuffer;
        use lazy_static::lazy_static;
    }
}

cfg_if! {
    if #[cfg(not(target_os = "cuda"))] {
        lazy_static! {
            static ref PTX: CString = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).unwrap();
        }
    }
}

/*
#[global]
fn sum(x: &[f32], y: &[f32], result: &mut [f32]) {
    let i = blockDim.x * blockIdx.x + threadIdx.x;
    result[i] = x[i] + y[i];
}
*/

// device
#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn sum_kernel(x_data: *const f32, x_n: usize, y_data: *const f32, y_n: usize, result_data: *mut f32, result_n: usize) {
    #[shared]
    let mut s: [f32; 3];

    let x = core::slice::from_raw_parts(x_data, x_n);
    let y = core::slice::from_raw_parts(y_data, y_n);
    let result = core::slice::from_raw_parts_mut(result_data, result_n);

    if _thread_idx_x() == 0 {
        for i in 0..3 {
            s[i] = x[i];
        }
    }

    sum(x, y, result)
}

#[device]
fn sum(x: &[f32], y: &[f32], result: &mut [f32]) {
    let blockDim = BlockDim::new();
    let blockIdx = BlockIdx::new();
    let threadIdx = ThreadIdx::new();

    let i = blockDim.x * blockIdx.x + threadIdx.x;
    result[i] = x[i] + y[i];
}

// host
#[host]
fn sum(blocks: GridSize, threads: BlockSize, x: &[f32], y: &[f32], result: &mut [f32]) -> Result<(), Box<dyn std::error::Error>> {
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let module = Module::load_from_string(&PTX)?;

    let mut x_data = DeviceBuffer::from_slice(&x)?;
    let mut y_data = DeviceBuffer::from_slice(&y)?;
    let mut result_data = DeviceBuffer::from_slice(&result)?;

    unsafe {
        launch!(
            module.sum_kernel<<<blocks, threads, 0, stream>>>(
                x_data.as_device_ptr(),
                x.len(),
                y_data.as_device_ptr(),
                y.len(),
                result_data.as_device_ptr(),
                result.len()
            )
        )
    }?;

    result_data.copy_to(result)?;

    Ok(())
}

#[host]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let x = vec![1.0f32, 2.0f32, 3.0f32];
    let y = vec![4.0f32, 5.0f32, 6.0f32];
    let mut result = vec![0.0f32; 3];

    sum(1.into(), 3.into(), &x, &y, &mut result)?;

    println!("Sums are {}, {}, {}", result[0], result[1], result[2]);

    Ok(())
}
