# Rust interoperability with other languages

For further tips and other languages see:

 - [The Rust FFI Omnibus](https://jakegoulding.com/rust-ffi-omnibus/)
 - [FFI examples written in Rust](https://github.com/alexcrichton/rust-ffi-examples)
 - [Foreign Function Interface](https://doc.rust-lang.org/nomicon/ffi.html) in the Rustonomicon

# Cave

 - `num_complex::Complex<T>` [must be passed by pointer over FFI boundaries](https://rust-num.github.io/num/num_complex/struct.Complex.html)
