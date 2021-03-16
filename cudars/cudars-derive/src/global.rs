use proc_macro::TokenStream;
use syn::{FnArg, ItemFn, parse_macro_input, parse_quote};
use quote::{format_ident, quote};

pub fn global_impl(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = parse_macro_input!(func as ItemFn);

    let vis = &func.vis;
    let unsafety = &func.sig.unsafety;
    let fn_token = &func.sig.fn_token;
    let ident = &func.sig.ident;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;
    let block = &func.block;

    let kernel_ident = format_ident!("_cudars_kernel_{}", ident);
    let kernel_inputs = "";

    // TODO
    // iterate through arguments
    // match supported types (pointers, references, references to slices, arrays, primitives)
    // add to host_pre
    // add to host_call_args
    // add to host_post
    // add to kernel_inputs
    // add to kernel_pre
    for input in inputs {
        if let syn::FnArg::Typed(arg) = input {
            match &*arg.ty {
                syn::Type::Array(arr) => {},
                syn::Type::Ptr(ptr) => {},
                syn::Type::Reference(reference) => {
                    match &*reference.elem {
                        syn::Type::Slice(slice) => {
                            if reference.mutability.is_some() {
                                // host_pre: let mut x_data = DeviceBuffer::from_slice(&x)?;
                                // host_call_args: x_data.as_device_ptr(), x_len()
                                // kernel_inputs: x_data: *mut T, x_len: usize
                                // kernel_pre: let mut x = ::core::slice::from_raw_parts_mut(x_data, x_len);
                            } else {
                                // host_pre: let mut x_data = DeviceBuffer::from_slice(&x)?;
                                // host_call_args: x_data.as_device_ptr(), x_len()
                                // host_post: x_data.copy_to(x)?;
                                // kernel_inputs: x_data: *const T, x_len: usize
                                // kernel_pre: let x = ::core::slice::from_raw_parts(x_data, x_len);
                            }
                        },
                        _ => {},
                    }
                }
                _ => { panic!("Unsupported argument type") },
            }
        } else {
            panic!("Unsupported argument type");
        }
    };

    let host_inputs = {
        let mut inputs = inputs.clone();
        let blocks: FnArg = parse_quote! { blocks: GridSize };
        let threads: FnArg = parse_quote! { threads: BlockSize };
        inputs.insert(0, blocks);
        inputs.insert(1, threads);
        inputs
    };

    let result = quote! {
        // host
        #[host]
        #vis #unsafety #fn_token #ident(#host_inputs) #output {
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            let module = Module::load_from_string(&PTX)?;

            // host_pre
            unsafe {
                launch!(
                    module.#kernel_ident<<<blocks, threads, 0, stream>>>(
                        // host_call_args
                    )
                )
            }?;
            // host_post
        }

        // device
        #[device]
        #[no_mangle]
        pub unsafe extern "ptx-kernel" #fn_token #kernel_ident(#kernel_inputs) #block

        #[device]
        #vis #unsafety #fn_token #ident(#inputs) #output {
            let blockDim = BlockDim::new();
            let blockIdx = BlockIdx::new();
            let threadIdx = ThreadIdx::new();

            #block
        }
    };
    result.into()
}
