#![feature(core_intrinsics)]

use std::cmp::Ordering;
use std::ops::{Add, Sub, Mul, Div, Rem};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};
use std::intrinsics::{fadd_fast, fsub_fast, fmul_fast, fdiv_fast, frem_fast};
use num_traits::{Float, Zero, One};
use derive_more::{Display, LowerExp, UpperExp, FromStr};
use tofrom_bytes::{ToBytes, FromBytes};

#[derive(Clone, Copy, Default, Debug, Display, LowerExp, UpperExp, FromStr)]
#[repr(transparent)]
pub struct Fast<T>(pub T);

pub type FF32 = Fast<f32>;
pub type FF64 = Fast<f64>;

impl<T> Fast<T> {
    #[inline(always)]
    pub fn get(&self) -> &T {
        &self.0
    }
}

impl<T> From<T> for Fast<T> {
    #[inline(always)]
    fn from(x: T) -> Self {
        Self(x)
    }
}

// impl<T> From<Fast<T>> for T

impl<T> PartialEq for Fast<T> where T: PartialEq {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        *self.get() == *rhs.get()
    }
}

impl<T> PartialEq<T> for Fast<T> where T: PartialEq {
    #[inline(always)]
    fn eq(&self, rhs: &T) -> bool {
        *self.get() == *rhs
    }
}

// impl<T> PartialEq<Fast<T>> for T where T: PartialEq

impl<T> PartialOrd for Fast<T> where T: PartialOrd {
    #[inline(always)]
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.get().partial_cmp(rhs.get())
    }
}

impl<T> PartialOrd<T> for Fast<T> where T: PartialOrd {
    #[inline(always)]
    fn partial_cmp(&self, rhs: &T) -> Option<Ordering> {
        self.get().partial_cmp(rhs)
    }
}

// impl<T> PartialOrd<Fast<T>> for T where T: PartialOrd

impl<T> ToBytes for Fast<T> where T: ToBytes + Copy {
    type Output = <T as ToBytes>::Output;

    #[inline(always)]
    fn to_le_bytes(&self) -> Self::Output {
        self.get().to_le_bytes()
    }

    #[inline(always)]
    fn to_be_bytes(&self) -> Self::Output {
        self.get().to_be_bytes()
    }

    #[inline(always)]
    fn to_ne_bytes(&self) -> Self::Output {
        self.get().to_ne_bytes()
    }
}

impl<T> Zero for Fast<T> where T: Zero, Fast<T>: Add<Output = Fast<T>> {
    #[inline(always)]
    fn zero() -> Self {
        T::zero().into()
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.get().is_zero()
    }
}

impl<T> One for Fast<T> where T: One + PartialEq, Fast<T>: Mul<Output = Fast<T>> {
    #[inline(always)]
    fn one() -> Self {
        T::one().into()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.get().is_one()
    }
}

impl<T> FromBytes for Fast<T> where T: FromBytes {
    type Input = <T as FromBytes>::Input;
    type Output = <T as FromBytes>::Output;

    #[inline(always)]
    fn from_le_bytes(bytes: Self::Input) -> Self::Output {
        T::from_le_bytes(bytes).into()
    }

    #[inline(always)]
    fn from_be_bytes(bytes: Self::Input) -> Self::Output {
        T::from_be_bytes(bytes).into()
    }

    #[inline(always)]
    fn from_ne_bytes(bytes: Self::Input) -> Self::Output {
        T::from_ne_bytes(bytes).into()
    }
}

macro_rules! impl_op {
    ($ty: ty, $trait: ident, $method: ident, $op: tt, $intrinsic: path) => {
        impl $trait for Fast<$ty> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                unsafe {
                    Fast($intrinsic(*self.get(), *rhs.get()))
                }
            }
        }

        impl $trait<$ty> for Fast<$ty> {
            type Output = Fast<$ty>;

            #[inline(always)]
            fn $method(self, rhs: $ty) -> Self::Output {
                self $op Fast(rhs)
            }
        }

        impl $trait<Fast<$ty>> for $ty {
            type Output = Fast<$ty>;

            #[inline(always)]
            fn $method(self, rhs: Fast<$ty>) -> Self::Output {
                Fast(self) $op rhs
            }
        }
    }
}

macro_rules! impl_assign_op {
    ($ty: ty, $trait: ident, $method: ident, $op: tt) => {
        impl $trait for Fast<$ty> {
            #[inline(always)]
            fn $method(&mut self, rhs: Fast<$ty>) {
                *self = *self $op rhs;
            }
        }

        impl $trait<$ty> for Fast<$ty> {
            #[inline(always)]
            fn $method(&mut self, rhs: $ty) {
                *self = *self $op rhs;
            }
        }

        impl $trait<Fast<$ty>> for $ty {
            #[inline(always)]
            fn $method(&mut self, rhs: Fast<$ty>) {
                *self = <$ty>::from(*self $op rhs);
            }
        }
    }
}

macro_rules! impl_fast {
    ($ty: ty) => {
        impl From<Fast<$ty>> for $ty {
            #[inline(always)]
            fn from(x: Fast<$ty>) -> Self {
                *x.get()
            }
        }

        impl PartialEq<Fast<$ty>> for $ty {
            #[inline(always)]
            fn eq(&self, rhs: &Fast<$ty>) -> bool {
                *self == *rhs.get()
            }
        }

        impl PartialOrd<Fast<$ty>> for $ty {
            #[inline(always)]
            fn partial_cmp(&self, rhs: &Fast<$ty>) -> Option<Ordering> {
                self.partial_cmp(rhs.get())
            }
        }

        impl_op!($ty, Add, add, +, fadd_fast);
        impl_op!($ty, Sub, sub, -, fsub_fast);
        impl_op!($ty, Mul, mul, *, fmul_fast);
        impl_op!($ty, Div, div, /, fdiv_fast);
        impl_op!($ty, Rem, rem, %, frem_fast);

        impl_assign_op!($ty, AddAssign, add_assign, +);
        impl_assign_op!($ty, SubAssign, sub_assign, -);
        impl_assign_op!($ty, MulAssign, mul_assign, *);
        impl_assign_op!($ty, DivAssign, div_assign, /);
        impl_assign_op!($ty, RemAssign, rem_assign, %);

        // TODO: impl Float
    }
}

impl_fast!(f32);
impl_fast!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_op {
        ($ty: ty, $method: ident, $op: tt) => {
            #[test]
            fn $method() {
                let a: $ty = 42.0;
                let b: $ty = 9.0;

                assert_eq!(a $op b, a $op Fast(b));
                assert_eq!(a $op b, Fast(a) $op b);
                assert_eq!(a $op b, Fast(a) $op Fast(b));
            }
        }
    }

    macro_rules! test_fast {
        ($ty: ty) => {
            use super::*;

            #[test]
            fn eq() {
                let a: $ty = 1.0;

                assert_eq!(Fast(a), Fast(a));
                assert_eq!(a, Fast(a));
                assert_eq!(Fast(a), a);
            }

            #[test]
            fn le() {
                let a: $ty = 1.0;
                let b: $ty = 2.0;

                assert!(Fast(a) < Fast(b));
                assert!(a < Fast(b));
                assert!(Fast(a) < b);
            }

            test_op!($ty, add, +);
            test_op!($ty, sub, -);
            test_op!($ty, mul, *);
            test_op!($ty, div, /);
            test_op!($ty, rem, %);
        }
    }

    mod r#f32 { test_fast!(f32); }
    mod r#f64 { test_fast!(f64); }
}
