use ptx_builder::error::Result;
use ptx_builder::prelude::*;

fn main() -> Result<()> {
    println!("cargo:rustc-env=LIBRARY_PATH=/opt/cuda/lib:");
    let builder = Builder::new(".")?;
    CargoAdapter::with_env_var("KERNEL_PTX_PATH").build(builder);
}
