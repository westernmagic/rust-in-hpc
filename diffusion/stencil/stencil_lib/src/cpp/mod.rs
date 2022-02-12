use crate::declare;

declare!(diffuse_cpp_v0_base);
declare!(diffuse_cpp_v1_inline);

pub mod v0_base {
    use crate::define;
    define!(super::diffuse_cpp_v0_base);
}

pub mod v1_inline {
    use crate::define;
    define!(super::diffuse_cpp_v1_inline);
}
