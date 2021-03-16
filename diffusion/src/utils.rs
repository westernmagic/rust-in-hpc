use anyhow::Result;
use ndarray::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::mem::size_of;
use std::convert::TryFrom;
use std::path::Path;
use tofrom_bytes::ToBytes;
use crate::num_halo;

pub fn write_field_to_file<T, P>(field: &Array3<T>, filename: P) -> Result<()> where
    T: ToBytes,
    P: AsRef<Path>
{
    let mut file = BufWriter::new(File::create(filename)?);
    file.write(&u32::try_from(field.ndim())?.to_le_bytes())?;
    file.write(&u32::try_from(8 * size_of::<T>())?.to_le_bytes())?;
    file.write(&u32::try_from(num_halo)?.to_le_bytes())?;
    for &dim in field.shape() {
        file.write(&u32::try_from(dim)?.to_le_bytes())?;
    }

    for k in 0..field.shape()[2] {
        for j in 0..field.shape()[1] {
            for i in 0..field.shape()[0] {
                file.write(field[[i, j, k]].to_le_bytes().as_ref())?;
            }
        }
    }
    /*
    for &element in field.as_slice_memory_order().unwrap().iter() {
        file.write(&element.to_le_bytes())?;
    }
    */

    Ok(())
}
