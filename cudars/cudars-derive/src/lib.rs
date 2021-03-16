#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unused_variables)]
use proc_macro::TokenStream;

mod cuda;
mod global;
mod host;
mod device;
mod shared;
mod constant;

#[proc_macro_attribute]
pub fn cuda(attr: TokenStream, toks: TokenStream) -> TokenStream {
    cuda::cuda_impl(attr, toks)
}

#[proc_macro_attribute]
pub fn global(attr: TokenStream, func: TokenStream) -> TokenStream {
    global::global_impl(attr, func)
}

#[proc_macro_attribute]
pub fn host(attr: TokenStream, func: TokenStream) -> TokenStream {
    host::host_impl(attr, func)
}

#[proc_macro_attribute]
pub fn device(attr: TokenStream, func: TokenStream) -> TokenStream {
    device::device_impl(attr, func)
}

#[proc_macro_attribute]
pub fn shared(attr: TokenStream, var: TokenStream) -> TokenStream {
    shared::shared_impl(attr, var)
}

#[proc_macro_attribute]
pub fn constant(attr: TokenStream, var: TokenStream) -> TokenStream {
    constant::constant_impl(attr, var)
}
