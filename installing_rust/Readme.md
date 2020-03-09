# Installing Rust on Piz Daint

## Install `spack`
```bash
module load daint-mc
module load EasyBuild-custom
eb Spack-daint-git-develop.eb
module use ${EASYBUILD_INSTALLPATH}/modules/all
module load Spack
source ${EBROOTSPACK}/share/spack/setup-env.sh
```

## Setup local `spack` repository
Copy `./.spack` to `${HOME}`, be careful not to overwrite you pre-existing spack config

## Install `llvm` and `rust`
```bash
module swap PrgEnv-cray PrgEnv-gnu
spack install llvm -clang -compiler-rt -internal_unwind -libcxx +link_dylib -lld -lldb
spack install rust@develop+nvptx # for master branch (~nightly)
spack install rust@1.41.0+nvptx # for stable
# TODO: how to install ptx-linker automatically
# TODO: how to set optimization flags for rust
spack load rust
patch -u rustc-llvm-proxy/src/path.rs -i llvm-proxy-path.patch
patch -u rust-ptx-linker/Cargo.toml -i ptx-linker.patch
cargo install --path ./rust-ptx-linker
# TODO: optional, currently does not work on Piz Daint (uses `mpiexec`)
# cargo install mpirun
```

## Use `rust`
```bash
# Cray linker fails to link Rust code
module swap PrgEnv-cray PrgEnv-gnu
source ${EBROOTSPACK}/share/spack/setup-env.sh
module load cudatoolkit
# TODO: why does spack not generate a module for `module load`?
spack load llvm
spack load rust
# for `rsmpi` automatic binding generation
export MPICC=cc
```

## Sources
[Spack Package Manager](https://user.cscs.ch/computing/compilation/spack/)
