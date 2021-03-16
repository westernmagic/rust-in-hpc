use proc_macro::TokenStream;
use syn::{ItemFn, parse_macro_input};
use quote::quote;

pub fn device_impl(_attr: TokenStream, func: TokenStream) -> TokenStream {
    let func = parse_macro_input!(func as ItemFn);
    let result = quote! {
        #[cfg(target_os = "cuda")]
        #func
    };
    result.into()
}
