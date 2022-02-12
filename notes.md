# Rust programming language in the HPC environment

Rust is a new open-source systems programming language created by Mozilla and a community of volunteers, designed to help developers create fast, secure applications which take full advantage of the powerful features of modern multi-core processors. It combines performance, reliability and productivity at the same time. Several big projects such as Servo (Mozilla’s brand-new browser engine) and Redox (a full Unix-like operating system) are written in this language.

One of the key features of Rust is the ownership model that guarantees the memory-safety and thread-safety. It has many other interesting features, such as a standardised build system, package manager, pattern matching, support for complex numbers, type inference, and efficient C bindings. All this potentially makes Rust a very appealing language for the software development on HPC platforms.

In the first part, the Rust will be evaluated for potential usage at CSCS. In particular, the following questions should be tackled:

 - [x] how to install and run Rust programs with "user access" rights on Piz Daint
 - [/] is MPI wrapper for Rust compatible with Cray’s implementation
 - [ ] how to interface Rust program with numerical libraries, such as MKL, MAGMA, ScaLAPACK, cuBlasXt, etc.
 - [ ] how to write GPU-enabled application in Rust; what are the complications or simplifications comparing to the C/C++/FORTRAN GPU applications
 - [ ] how to debug and profile Rust-based programs
 - [ ] how rich is the functionality of Rust, e.g. the availability of special mathematical functions, support of matrices or multi-dimensional arrays, etc.

In the second part, a performance comparison between Rust and C/C++/FORTRAN will be conducted, by idiomatically implementing a parallel distributed linear algebra algorithm or a scientific mini-app code in the target languages. The performance analysis is not only limited to computational performance, but may include analysis of other factors, such as ease of implementation, number of bugs made, testability, readability, maintainability, etc.


# Sources
 - [Are we learning yet?](https://www.arewelearningyet.com/)
 - [`const_generics`](https://github.com/rust-lang/rust/issues/44580)
 - [Evaluation of performance and productivity metrics of potential programming languages in the HPC environment](https://octarineparrot.com/assets/mrfloya-thesis-ba.pdf)
 - [Rust and MPI](https://stackoverflow.com/questions/22949462/rust-on-grid-computing)
 - [University of Maryland](https://www.glue.umd.edu/hpcc/help/software/rust.html)
 - [Fast Rust in 2018](https://web.archive.org/web/20180124141726/https://adamniederer.com/blog/rust-2018.html)
 - [Rust High Performance](https://www.amazon.com/Rust-High-Performance-performance-applications/dp/178839948X)
 - [Parallel Rust C++](https://parallel-rust-cpp.github.io/introduction.html)
 - [The State of GPGPU in Rust](https://bheisler.github.io/post/state-of-gpgpu-in-rust/)
 - [How to: Run Rust code on your GPU](https://github.com/japaric-archived/nvptx#targets)
 - [Rust 2020: Scientific Rust](https://github.com/willi-kappler/rust_2020)
 - [Rayon](https://github.com/rayon-rs/rayon)
 - https://www.dursi.ca/post/hpc-is-dying-and-mpi-is-killing-it.html
 - [Using Rust as a Complement to C for Embedded Systems Software Development](https://lup.lub.lu.se/student-papers/search/publication/8938297)
 - https://publications.lib.chalmers.se/records/fulltext/219016/219016.pdf
 - [Rust in HPC](https://www.osti.gov/biblio/1485376-rust-hpc)
 - https://github.com/vks/special-fun
 - https://github.com/stainless-steel/special
 - https://github.com/rust-lang/rfcs/issues/785

# Installing Rust on Piz Daint
 1. Install `spack`
    ```bash
    module load daint-mc
    module load EasyBuild-custom
    eb Spack-daint-git-develop.eb
    module use $EASYBUILD_INSTALLPATH/modules/all
    module load Spack
    source /users/msudwoj/easybuild/daint/haswell/software/Spack/daint-git-develop/share/spack/setup-env.sh
    ```
 2. Install `llvm` and `rust`
    ```bash
    module swap PrgEnv-cray PrgEnv-gnu
    spack install llvm -clang -compiler-rt -internal_unwind -libcxx +link_dylib -lld -lldb
    spack install rust@develop+nvptx # for "nightly" = master branch
    spack install rust@1.41.0+nvptx # for stable
    # TODO: how to install ptx-linker automatically
    # TODO: how to set optimization flags for rust
    spack load rust
    cargo install ptx-linker # TODO: with patched rustc-llvm-proxy
    # cargo install mpirun
    ```
 3. Use `rust`
    ```bash
    module swap PrgEnv-cray PrgEnv-gnu
    source ${EBROOTSPACK}/share/spack/setup-env.sh
    module load cudatoolkit
    spack load llvm # TODO: why does spack not generate a module for `module load`?
    spack load rust
    export MPICC=cc
    ```

# GPU programming in Rust
 - https://github.com/bheisler/RustaCUDA
 - https://github.com/rust-accel/accel (semi-maintained)
 - https://github.com/japaric-archived/cuda (unmaintained)

## Problems
 - no `__shared__`
 - no `__const__`
 - not well tested

## Ideas:
 - `proc_macro`s similar to `__global__`, `__host__`, `__device__`
 - `async` kernels
 - `cuda-bindgen`
