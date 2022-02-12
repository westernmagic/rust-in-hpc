# Settings for Piz Daint

| `RUSTFLAGS`                          | Usage               |
| ------------------------------------ | ------------------- |
| `-C target-cpu=${CRAY_CPU_TARGET}`   |                     |
| `-C target-cpu=broadwell`            | `daint-mc`          |
| `-C target-cpu=haswell`              | `daint-gpu`         |
| `-C relocation-model=dynamic-no-pic` | `PrgEnv-cray`       |
| `-C save-temps`                      | CrayPat             |
| `-C remark=all`                      | optimization report |

 - module `cce` required for `bindgen`
 - `perftools-lite` does not work with build scripts (`build.rs`): it tries to run the instrumented build script on the login node, and fails
    - use `perftools` manually instead
 - CCE classic and PGI are not supported for linking (flags not compatible with `rustc`, require new linker flavor)
 - `MPICC=cc` required for `rsmpi`
