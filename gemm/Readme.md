# Simple `dgemm` in Rust

[`intel-mkl-src`](https://github.com/rust-math/intel-mkl-src) insists on downloading and installing it's own MKL, neccessitating manual passing of the linker flags in `build.rs`.

It would be quite trivial to wrap this around in our own crate, akin to [`intel-mkl-src`](https://github.com/rust-math/intel-mkl-src) or [`mkl_link`](https://github.com/peterhj/libmkl_link)

`zgemm` works good aswell, since Rust's `Comlex<T>` has the same layout as C's `_Complex`.

## Running
```bash
cargo run --release --example blas
cargo run --release --example cublas
cargo run --release --example ndarray
```

## Sources
 - https://github.com/blas-lapack-rs/blas
