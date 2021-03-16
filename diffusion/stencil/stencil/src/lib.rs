macro_rules! diffuse {
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
    }
}

pub mod cpp {
    pub mod v0 {
        diffuse!(stencil_lib::cpp::diffuse_cpp_v0_base);
    }
    pub mod v1 {
        diffuse!(stencil_lib::cpp::diffuse_cpp_v1_inline);
    }
}

pub mod f {
    pub mod v0 {
        diffuse!(stencil_lib::f::diffuse_f_v0_base);
    }
    pub mod v1 {
        diffuse!(stencil_lib::f::diffuse_f_v1_inline);
    }
}

pub mod rs {
    pub mod v0 {
        diffuse!(stencil_lib::rs::diffuse_rs_v0_base);
    }
    pub mod v1 {
        diffuse!(stencil_lib::rs::diffuse_rs_v1_inline);
    }
    pub mod v2 {
        diffuse!(stencil_lib::rs::diffuse_rs_v2_fast);
    }
    pub mod v3 {
        diffuse!(stencil_lib::rs::diffuse_rs_v3_fma);
    }
}
