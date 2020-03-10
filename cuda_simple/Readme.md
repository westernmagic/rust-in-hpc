# Writing CUDA code in Rust

## Piz Daint toolchain installation

See `installing_rust`.

## Local toolchain installation

```bash
rustup toolchain add nightly
rustup target add nvptx64-nvidia-cuda
cargo install ptx-linker
```

## Existing frameworks

 - [RustaCUDA](https://github.com/bheisler/RustaCUDA)
 - [accel](https://github.com/rust-accel/accel) (semi-maintained)
 - [cuda](https://github.com/japaric-archived/cuda) (unmaintained)


## Pitfalls

 - No `__shared__`
 - No `__const__`
 - No `std` library, only `core`
 - FFI boundary between host and device code

## Ideas

 - `proc_macro` in the spirit of `__global__`, `__host__`, `__device__`
 - `async` kernel launch and executor [accel #65](https://github.com/rust-accel/accel/issues/65)
