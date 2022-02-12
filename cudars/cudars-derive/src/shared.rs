use proc_macro::TokenStream;
use syn::{ItemStatic, LitInt, Pat, Stmt, Token, Type, Ident, parse_macro_input};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use quote::quote;

fn parse_local<'a>(stmt: &'a Stmt) -> (&'a Ident, &'a Box<Type>,) {
    if let Stmt::Local(item) = stmt {
        if let Pat::Type(item) = &item.pat {
            let ty = &item.ty;
            if let Pat::Ident(item) = &*item.pat {
                // let mutability = &item.mutability;
                let ident = &item.ident;
                return (ident, ty,);
            }
        }
    }

    panic!()
}

pub fn shared_impl(_attr: TokenStream, var: TokenStream) -> TokenStream {
    // let item = parse_macro_input!(var as ItemStatic);
    // let ident = &item.ident;
    // let ty = &item.ty;
    let stmt = parse_macro_input!(var as Stmt);
    let (ident, ty,) = parse_local(&stmt);

    let result = quote! {
        struct Shared<T> {
            ptr: *mut T,
        }

        unsafe impl<T> Sync for Shared<T> {}

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
                let mut _tmp: u64;
                let mut address: u64;

                unsafe {
                    asm! {
                        concat!(
                            ".shared .align {alignment} .b8 ", stringify!(#ident), "[{size}];", "\n\t",
                            "mov.u64 {dummy}, ", stringify!(#ident), ";", "\n\t",
                            "cvta.shared.u64 {generic_address}, {dummy};"
                        ),
                        alignment = const Self::alignment,
                        size = const Self::size,
                        dummy = out(reg64) _tmp,
                        generic_address = out(reg64) address
                    };
                }

                Self {
                    ptr: address as *mut T,
                }
            }
        }

        // ::lazy_static::lazy_static! {
        //     static ref #ident: Shared<#ty> = Shared::new();
        // }
        let mut #ident: Shared<#ty> = Shared::new();
    };
    result.into()
}
