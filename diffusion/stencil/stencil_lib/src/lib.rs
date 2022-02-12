pub mod cpp;
pub mod f;
pub mod rs;

#[macro_export]
macro_rules! declare {
    ($fn: ident) => {
        extern "C" {
            pub fn $fn(
                in_field: *mut f32,
                out_field: *mut f32,
                nx: usize,
                ny: usize,
                nz: usize,
                num_halo: usize,
                alpha: f32,
                num_iter: usize
            );
        }
    }
}

#[macro_export]
macro_rules! define {
    ($path: path) => {
        pub fn diffuse<'a>(
            in_field: &'a mut ndarray::Array3<f32>,
            out_field: &'a mut ndarray::Array3<f32>,
            num_halo: usize,
            alpha: f32,
            num_iter: usize
        ) {
            assert_eq!(in_field.shape()[0], out_field.shape()[0]);
            assert_eq!(in_field.shape()[1], out_field.shape()[1]);
            assert_eq!(in_field.shape()[2], out_field.shape()[2]);

            let nx = in_field.shape()[0] - 2 * num_halo;
            let ny = in_field.shape()[1] - 2 * num_halo;
            let nz = in_field.shape()[2];

            unsafe {
                $path(
                    in_field.as_mut_ptr(),
                    out_field.as_mut_ptr(),
                    nx,
                    ny,
                    nz,
                    num_halo,
                    alpha,
                    num_iter
                )
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use ndarray::prelude::*;
            use ndarray_npy::*;
            use approx::assert_abs_diff_eq;
            use std::fs::File;
            use std::path::PathBuf;

            #[test]
            fn test() {
                let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                let in_field_file = File::open(path.join("in_field_base.npy")).unwrap();
                let mut in_field = Array3::<f32>::read_npy(in_field_file).unwrap();
                let mut out_field = in_field.clone();

                diffuse(
                    &mut in_field,
                    &mut out_field,
                    2,
                    1.0f32 / 32.0f32,
                    1
                );

                diffuse(
                    &mut in_field,
                    &mut out_field,
                    2,
                    1.0f32 / 32.0f32,
                    1024
                );

                let out_field_file = File::open(path.join("out_field_base.npy")).unwrap();
                let out_field_base = Array3::<f32>::read_npy(out_field_file).unwrap();
                assert_abs_diff_eq!(out_field, out_field_base, epsilon = 2.0e-4_f32);
            }
        }
    }
}
