use crate::declare;

declare!(diffuse_f_v0_base);
declare!(diffuse_f_v1_inline);
declare!(diffuse_f_v2_openmp);
declare!(diffuse_f_v3_openmp);

pub mod v0_base {
    use crate::define;
    define!(super::diffuse_f_v0_base);
}

pub mod v1_inline {
    use crate::define;
    define!(super::diffuse_f_v1_inline);
}

pub mod v2_openmp {
    use crate::define;
    define!(super::diffuse_f_v2_openmp);
}

pub mod v3_openmp {
    use crate::define;
    define!(super::diffuse_f_v3_openmp);
}
