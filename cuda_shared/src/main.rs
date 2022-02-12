#![cfg_attr(target_os = "cuda", feature(abi_ptx, stdsimd, llvm_asm, alloc_error_handler, asm))]
#![cfg_attr(target_os = "cuda", no_std)]

cfg_if::cfg_if! {
    if #[cfg(target_os = "cuda")] {
        use ptx_support::prelude::*;
        use core::arch::nvptx::*;
        extern crate alloc;
        use alloc::{format, alloc::{Layout, GlobalAlloc}};
        use core::panic::PanicInfo;
    } else {
        use rustacuda::prelude::*;
        use rustacuda::launch;
        use rustacuda::function::{BlockSize, GridSize};
        use std::error::Error;
        use std::ffi::CString;
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_os = "cuda")] {
        struct Shared<T> {
            ptr: *mut T,
        }

        impl<T> ::core::ops::Deref for Shared<T> {
            type Target = T;

            fn deref(&self) -> &Self::Target {
                unsafe { &*self.ptr }
            }
        }

        impl<T> ::core::ops::DerefMut for Shared<T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *self.ptr }
            }
        }

        impl<T> Shared<T> {
            const alignment: usize = ::core::mem::align_of::<T>();
            const size: usize = ::core::mem::size_of::<T>();

            pub fn new() -> Self {
                let _tmp: u64;
                let address: u64;
                // const asm: &'static str = format!(".shared .align {} .b8 {}[{}];\n\tmov.u64 $1, {};\n\tcvta.shared.u64 $0, $1;", alignment, "s", size, "s");
                /*
                const asm: &'static str = concat!(
                    ".shared .align ", alignment, " .b8 ", "s", "[", size, "];", "\n\t",
                    "mov.u64 $1, ", "s", ";", "\n\t",
                    "cvta.shared.u64 $0, $1;"
                );
                */
                unsafe {
                    asm! {
                        concat!(
                            ".shared .align {alignment} .b8 s[{size}];", "\n\t",
                            "mov.u64 {dummy}, s;", "\n\t",
                            "cvta.shared.u64 {generic_address}, {dummy};"
                        ),
                        alignment = const Self::alignment,
                        // ident = sym s,
                        size = const Self::size,
                        dummy = out(reg64) _tmp,
                        generic_address = out(reg64) address
                    };
                }
                Self {
                    ptr: address as *mut T,
                }
            }

            pub fn as_ptr(&self) -> *const T {
                self.ptr as *const _
            }

            pub fn as_mut_ptr(&mut self) -> *mut T {
                self.ptr
            }
        }

        struct StaticS {
            ptr: *mut i64,
        }

        impl StaticS {
            pub fn new() -> Self {
                let mut _tmp: u64;
                let mut address: u64;
                unsafe {
                    llvm_asm! {
                    r#"
                        .shared .align 4 .b8 s[512];
                        mov.u64 $1, s;
                        cvta.shared.u64 $0, $1;
                    "#
                    : "=l"(address), "=l"(_tmp)
                    }
                };
                StaticS {
                    ptr: address as *mut i64,
                }
            }

            pub fn as_ptr(&self) -> *const i64 {
                self.ptr
            }

            pub fn as_mut_ptr(&self) -> *mut i64 {
                self.ptr
            }
        }

        pub struct Allocator;

        unsafe impl GlobalAlloc for Allocator {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                malloc(layout.size()) as *mut u8
            }

            unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
                free(ptr as *mut _);
            }
        }

        #[global_allocator]
        static GLOBAL_ALLOCATOR: Allocator = Allocator;

        /*
        #[panic_handler]
        fn panic(_info: &PanicInfo) -> ! {
            unsafe { trap() }
        }
        */

        #[alloc_error_handler]
        fn alloc_error_handler(_layout: Layout) -> ! {
            unsafe { trap() }
        }

        extern "C" {
            fn vprintf(format: *const u8, valist: *const u8) -> i32;
        }

        macro_rules! print {
            ($($arg: tt)*) => {
                let msg = format!($($arg)*);
                unsafe { vprintf(msg.as_ptr(), core::ptr::null_mut()); }
            }
        }
    }
}

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn staticReverse(d: *mut i64, n: i64) {
    // let s = StaticS::new();
    let mut s = Shared::<[i64; 64]>::new();
    let t = _thread_idx_x() as i64;
    let tr = n - t - 1;
    if t == 0 {
        print!("Cuda Original: [");
        for i in 0..64 {
            print!("{}, ", *d.offset(i));
        }
        print!("]\n");
    }
    // *s.as_mut_ptr().offset(t as isize) = *d.offset(t as isize);
    s[t as usize] = *d.offset(t as isize);
    _syncthreads();
    if t == 0 {
        print!("Cuda shared: [");
        for i in 0..64 {
            // print!("{}, ", *s.as_ptr().offset(i));
            print!("{}, ", s[i]);
        }
        print!("]\n");
    }
    // *d.offset(t as isize) = *s.as_ptr().offset(tr as isize);
    *d.offset(t as isize) = s[tr as usize];
    _syncthreads();
    if t == 0 {
        print!("Cuda reversed: [");
        for i in 0..64 {
            print!("{}, ", *d.offset(i));
        }
        print!("]\n");
    }
    _syncthreads();
}

#[cfg(not(target_os = "cuda"))]
fn staticReverse(module: &Module, blocks: GridSize, threads: BlockSize, shared: u32, stream: &Stream, d: &mut [i64]) -> Result<Vec<i64>, Box<dyn Error>> {
    let n = (blocks.x * threads.x) as usize;
    assert_eq!(n, d.len());

    let mut d = DeviceBuffer::from_slice(&d)?;

    unsafe {
        launch!(
            module.staticReverse<<<blocks, threads, shared, stream>>>(
                d.as_device_ptr(),
                n
            )
        )?;
    }
    stream.synchronize()?;
    let mut dd = vec![0i64; n];
    d.copy_to(&mut dd)?;

    Ok(dd)
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

    let mut d: Vec<i64> = Vec::new();
    for i in 0..64 {
        d.push(i);
    }
    println!("Original: {:?}", d);
    let dd = staticReverse(&module, 1.into(), 64.into(), 0, &stream, &mut d)?;

    println!("Reversed: {:?}", dd);

    Ok(())
}
