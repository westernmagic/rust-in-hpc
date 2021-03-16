pub trait ToBytes {
    type Output: AsRef<[u8]>;

    fn to_le_bytes(&self) -> Self::Output;
    fn to_be_bytes(&self) -> Self::Output;
    fn to_ne_bytes(&self) -> Self::Output;
}

pub trait FromBytes {
    type Input: AsRef<[u8]>;
    type Output;

    fn from_le_bytes(bytes: Self::Input) -> Self::Output;
    fn from_be_bytes(bytes: Self::Input) -> Self::Output;
    fn from_ne_bytes(bytes: Self::Input) -> Self::Output;
}

macro_rules! impl_trait {
    ($ty: ty) => {
        impl ToBytes for $ty {
            type Output = [u8; std::mem::size_of::<$ty>()];

            #[inline(always)]
            fn to_le_bytes(&self) -> Self::Output {
                <$ty>::to_le_bytes(*self)
            }

            #[inline(always)]
            fn to_be_bytes(&self) -> Self::Output {
                <$ty>::to_be_bytes(*self)
            }

            #[inline(always)]
            fn to_ne_bytes(&self) -> Self::Output {
                <$ty>::to_ne_bytes(*self)
            }
        }

        impl FromBytes for $ty {
            type Input = [u8; std::mem::size_of::<$ty>()];
            type Output = $ty;

            #[inline(always)]
            fn from_le_bytes(bytes: Self::Input) -> Self::Output {
                <$ty>::from_le_bytes(bytes)
            }

            #[inline(always)]
            fn from_be_bytes(bytes: Self::Input) -> Self::Output {
                <$ty>::from_be_bytes(bytes)
            }

            #[inline(always)]
            fn from_ne_bytes(bytes: Self::Input) -> Self::Output {
                <$ty>::from_ne_bytes(bytes)
            }
        }
    }
}

impl_trait!(u8);
impl_trait!(u16);
impl_trait!(u32);
impl_trait!(u64);
impl_trait!(u128);
impl_trait!(usize);

impl_trait!(i8);
impl_trait!(i16);
impl_trait!(i32);
impl_trait!(i64);
impl_trait!(i128);
impl_trait!(isize);

impl_trait!(f32);
impl_trait!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test {
        ($ty: ty) => {
            use super::*;

            #[test]
            fn le_bytes() {
                let x = 42 as $ty;
                let bytes = ToBytes::to_le_bytes(&x);
                assert_eq!(x, <$ty as FromBytes>::from_le_bytes(bytes));
            }

            #[test]
            fn be_bytes() {
                let x = 42 as $ty;
                let bytes = ToBytes::to_be_bytes(&x);
                assert_eq!(x, <$ty as FromBytes>::from_be_bytes(bytes));
            }

            #[test]
            fn ne_bytes() {
                let x = 42 as $ty;
                let bytes = ToBytes::to_ne_bytes(&x);
                assert_eq!(x, <$ty as FromBytes>::from_ne_bytes(bytes));
            }
        }
    }

    mod r#u8    { test!(u8);    }
    mod r#u16   { test!(u16);   }
    mod r#u32   { test!(u32);   }
    mod r#u64   { test!(u64);   }
    mod r#u128  { test!(u128);  }
    mod r#usize { test!(usize); }

    mod r#i8    { test!(i8);    }
    mod r#i16   { test!(i16);   }
    mod r#i32   { test!(i32);   }
    mod r#i64   { test!(i64);   }
    mod r#i128  { test!(i128);  }
    mod r#isize { test!(isize); }

    mod r#f32   { test!(f32);   }
    mod r#f64   { test!(f64);   }
}
