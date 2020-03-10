#![allow(dead_code)]

use cublasxt_sys::*;
use std::ptr;
use num_complex::Complex;
use std::convert::{From, Into};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Error {
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let msg = match self {
            Self::NotInitialized  => "The cuBLAS library was not initialized",
            Self::AllocFailed     => "Resource allocation failed inside the cuBLAS library",
            Self::InvalidValue    => "An unsupported value or parameter was passed to the function (a negative vector size, for example)",
            Self::ArchMismatch    => "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision",
            Self::MappingError    => "An access to GPU memory space failed, which is usually caused by a failure to bind a texture",
            Self::ExecutionFailed => "The GPU program failed to execute",
            Self::InternalError   => "An internal cuBLAS operation failed",
            Self::NotSupported    => "The functionnality requested is not supported",
            Self::LicenseError    => "The functionnality requested requires some license and an error was detected when trying to check the current licensing"
        };
        write!(f, "{}", msg)
    }
}

impl std::error::Error for Error {}

impl From<cublasStatus_t> for Error {
    fn from(status: cublasStatus_t) -> Self {
        use cublasStatus_t::*;
        match status {
            CUBLAS_STATUS_NOT_INITIALIZED  => Self::NotInitialized,
            CUBLAS_STATUS_ALLOC_FAILED     => Self::AllocFailed,
            CUBLAS_STATUS_INVALID_VALUE    => Self::InvalidValue,
            CUBLAS_STATUS_ARCH_MISMATCH    => Self::ArchMismatch,
            CUBLAS_STATUS_MAPPING_ERROR    => Self::MappingError,
            CUBLAS_STATUS_EXECUTION_FAILED => Self::ExecutionFailed,
            CUBLAS_STATUS_INTERNAL_ERROR   => Self::InternalError,
            CUBLAS_STATUS_NOT_SUPPORTED    => Self::NotSupported,
            CUBLAS_STATUS_LICENSE_ERROR    => Self::LicenseError,
            _                              => panic!(),
        }
    }
}

pub type Result = std::result::Result<(), Error>;

pub struct Context {
    handle: cublasXtHandle_t
}

impl Context {
    pub fn new() -> std::result::Result<Self, Error> {
        Self::from_devices(&mut [0])
    }

    pub fn from_devices(mut devices: &mut[i32]) -> std::result::Result<Self, Error> {
        let mut handle: cublasXtHandle_t = ptr::null_mut();
        let r = unsafe {
            cublasXtCreate(&mut handle as *mut cublasXtHandle_t)
        };
        assert!(!handle.is_null());
        let context = Self{handle};
        context.device_select(&mut devices)?;
        match r {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(context),
            e                                     => Err(Error::from(e)),
        }
    }

    fn device_select(&self, devices: &mut [i32]) -> std::result::Result<(), Error> {
        let r = unsafe {
            cublasXtDeviceSelect(self.handle, devices.len() as i32, devices.as_mut_ptr())
        };
        match r {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            e                                     => Err(Error::from(e)),
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let r = unsafe {
            cublasXtDestroy(self.handle)
        };
        match r {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => (),
            _                                     => panic!(),
        }
    }
}

pub enum Op {
    N,
    T,
    C,
}

impl Into<cublasOperation_t> for Op {
    fn into(self) -> cublasOperation_t {
        use cublasOperation_t::*;
        match self {
            Self::N => CUBLAS_OP_N,
            Self::T => CUBLAS_OP_T,
            Self::C => CUBLAS_OP_C,
        }
    }
}

pub enum Side {
    L,
    R,
}

impl Into<cublasSideMode_t> for Side {
    fn into(self) -> cublasSideMode_t {
        use cublasSideMode_t::*;
        match self {
            Self::L => CUBLAS_SIDE_LEFT,
            Self::R => CUBLAS_SIDE_RIGHT,
        }
    }
}

pub enum Fill {
    U,
    L,
    F,
}

impl Into<cublasFillMode_t> for Fill {
    fn into(self) -> cublasFillMode_t {
        use cublasFillMode_t::*;
        match self {
            Self::U => CUBLAS_FILL_MODE_UPPER,
            Self::L => CUBLAS_FILL_MODE_LOWER,
            Self::F => CUBLAS_FILL_MODE_FULL,
        }
    }
}

pub enum Diag {
    U,
    N,
}

impl Into<cublasDiagType_t> for Diag {
    fn into(self) -> cublasDiagType_t {
        use cublasDiagType_t::*;
        match self {
            Self::U => CUBLAS_DIAG_UNIT,
            Self::N => CUBLAS_DIAG_NON_UNIT,
        }
    }
}

macro_rules! gemm_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, transa: Op, transb: Op, m: u64, n: u64, k: u64, alpha: $type, a: &[$type], lda: u64, b: &[$type], ldb: u64, beta: $type, c: &mut[$type], ldc: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, transa.into(), transb.into(), m, n, k, &alpha as *const $type, a.as_ptr(), lda, b.as_ptr(), ldb, &beta as *const $type, c.as_mut_ptr(), ldc)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    };
}

macro_rules! symm_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, side: Side, uplo: Fill, m: u64, n: u64, alpha: $type, a: &[$type], lda: u64, b: &[$type], ldb: u64, beta: $type, c: &mut[$type], ldc: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, side.into(), uplo.into(), m, n, &alpha as *const $type, a.as_ptr(), lda, b.as_ptr(), ldb, &beta as *const $type, c.as_mut_ptr(), ldc)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    };
}

macro_rules! syrk_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, uplo: Fill, trans: Op, n: u64, k: u64, alpha: $type, a: &[$type], lda: u64, beta: $type, c: &mut[$type], ldc: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, uplo.into(), trans.into(), n, k, &alpha as *const $type, a.as_ptr(), lda, &beta as *const $type, c.as_mut_ptr(), ldc)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    };
}

macro_rules! syr2k_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, uplo: Fill, trans: Op, n: u64, k: u64, alpha: $type, a: &[$type], lda: u64, b: &[$type], ldb: u64, beta: $type, c: &mut[$type], ldc: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, uplo.into(), trans.into(), n, k, &alpha as *const $type, a.as_ptr(), lda, b.as_ptr(), ldb, &beta as *const $type, c.as_mut_ptr(), ldc)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    };
}

macro_rules! herk_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, uplo: Fill, trans: Op, n: u64, k: u64, alpha: $type, a: &[Complex<$type>], lda: u64, beta: $type, c: &mut[Complex<$type>], ldc: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, uplo.into(), trans.into(), n, k, &alpha as *const $type, a.as_ptr(), lda, &beta as *const $type, c.as_mut_ptr(), ldc)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    };
}

macro_rules! her2k_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, uplo: Fill, trans: Op, n:u64, k: u64, alpha: Complex<$type>, a: &[Complex<$type>], lda: u64, b: &[Complex<$type>], ldb: u64, beta: $type, c: &mut[Complex<$type>], ldc: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, uplo.into(), trans.into(), n, k, &alpha as *const Complex<$type>, a.as_ptr(), lda, b.as_ptr(), ldb, &beta as *const $type, c.as_mut_ptr(), ldc)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    };
}

macro_rules! trsm_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, side: Side, uplo: Fill, trans: Op, diag: Diag, m: u64, n: u64, alpha: $type, a: &[$type], lda: u64, b: &mut[$type], ldb: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, side.into(), uplo.into(), trans.into(), diag.into(), m, n, &alpha as *const $type, a.as_ptr(), lda, b.as_mut_ptr(), ldb)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    };
}

macro_rules! trmm_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, side: Side, uplo: Fill, trans: Op, diag: Diag, m: u64, n:u64, alpha: $type, a: &[$type], lda: u64, b: &[$type], ldb: u64, c: &mut[$type], ldc: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, side.into(), uplo.into(), trans.into(), diag.into(), m, n, &alpha as *const $type, a.as_ptr(), lda, b.as_ptr(), ldb, c.as_mut_ptr(), ldc)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    }
}

macro_rules! spmm_impl {
    ($blas_name: ident, $cublas_name: ident, $type: ty) => {
        #[inline]
        pub fn $blas_name(&self, side: Side, uplo: Fill, m: u64, n: u64, alpha: $type, ap: &[$type], b: &[$type], ldb: u64, beta: $type, c: &mut[$type], ldc: u64) -> Result {
            let r = unsafe {
                $cublas_name(self.handle, side.into(), uplo.into(), m, n, &alpha as *const $type, ap.as_ptr(), b.as_ptr(), ldb, &beta as *const $type, c.as_mut_ptr(), ldc)
            };
            match r {
                cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                e                                     => Err(Error::from(e)),
            }
        }
    }
}

impl Context {
    gemm_impl!(sgemm, cublasXtSgemm, f32);
    gemm_impl!(dgemm, cublasXtDgemm, f64);
    gemm_impl!(cgemm, cublasXtCgemm, Complex<f32>);
    gemm_impl!(zgemm, cublasXtZgemm, Complex<f64>);

    symm_impl!(chemm, cublasXtChemm, Complex<f32>);
    symm_impl!(zhemm, cublasXtZhemm, Complex<f64>);

    symm_impl!(ssymm, cublasXtSsymm, f32);
    symm_impl!(dsymm, cublasXtDsymm, f64);
    symm_impl!(csymm, cublasXtCsymm, Complex<f32>);
    symm_impl!(zsymm, cublasXtZsymm, Complex<f64>);

    syrk_impl!(ssyrk, cublasXtSsyrk, f32);
    syrk_impl!(dsyrk, cublasXtDsyrk, f64);
    syrk_impl!(csyrk, cublasXtCsyrk, Complex<f32>);
    syrk_impl!(zsyrk, cublasXtZsyrk, Complex<f64>);

    syr2k_impl!(ssyr2k, cublasXtSsyr2k, f32);
    syr2k_impl!(dsyr2k, cublasXtDsyr2k, f64);
    syr2k_impl!(csyr2k, cublasXtCsyr2k, Complex<f32>);
    syr2k_impl!(zsyr2k, cublasXtZsyr2k, Complex<f64>);

    syr2k_impl!(ssyrkx, cublasXtSsyrkx, f32);
    syr2k_impl!(dsyrkx, cublasXtDsyrkx, f64);
    syr2k_impl!(csyrkx, cublasXtCsyrkx, Complex<f32>);
    syr2k_impl!(zsyrkx, cublasXtZsyrkx, Complex<f64>);

    herk_impl!(cherk, cublasXtCherk, f32);
    herk_impl!(zherk, cublasXtZherk, f64);

    her2k_impl!(cher2k, cublasXtCher2k, f32);
    her2k_impl!(zher2k, cublasXtZher2k, f64);

    her2k_impl!(cherkx, cublasXtCherkx, f32);
    her2k_impl!(zherkx, cublasXtZherkx, f64);

    trsm_impl!(strsm, cublasXtStrsm, f32);
    trsm_impl!(dtrsm, cublasXtDtrsm, f64);
    trsm_impl!(ctrsm, cublasXtCtrsm, Complex<f32>);
    trsm_impl!(ztrsm, cublasXtZtrsm, Complex<f64>);

    trmm_impl!(strmm, cublasXtStrmm, f32);
    trmm_impl!(dtrmm, cublasXtDtrmm, f64);
    trmm_impl!(ctrmm, cublasXtCtrmm, Complex<f32>);
    trmm_impl!(ztrmm, cublasXtZtrmm, Complex<f64>);

    spmm_impl!(sspmm, cublasXtSspmm, f32);
    spmm_impl!(dspmm, cublasXtDspmm, f64);
    spmm_impl!(cspmm, cublasXtCspmm, Complex<f32>);
    spmm_impl!(zspmm, cublasXtZspmm, Complex<f64>);
}
