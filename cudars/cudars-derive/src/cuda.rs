use proc_macro::TokenStream;
use quote::quote;

pub fn cuda_impl(_attr: TokenStream, _toks: TokenStream) -> TokenStream {
    let result = quote! {
        #![cfg_attr(target_os = "cuda", feature(abi_ptx), feature(stdsimd), no_std)]
    };
    result.into()
}
