#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(specialization)]
#![feature(trivial_bounds)]

use std::fmt;
use std::ops::*;
use std::mem;
use std::mem::MaybeUninit;

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for i32 {
    fn zero() -> Self {0}
}

pub trait One {
    fn one() -> Self;
}

#[repr(transparent)]
pub struct Vector<T, const N: usize>([T; N]);

impl<T, const N: usize> Vector<T, N>
where
    T: Copy
{
    fn from_elem(elem: T) -> Self {
        // Self([elem; N])

        let mut new = MaybeUninit::<Vector<T, N>>::uninit();
        let ptr: *mut T = unsafe { mem::transmute(&mut new) };
        for i in 0..N {
            unsafe { ptr.add(i).write(elem); }
        }
        unsafe { new.assume_init() }
    }
}

#[macro_export]
macro_rules! vector {
    ( $($elem: expr),* $(,)? ) => {
        $crate::Vector([$($elem),*])
    };
    ( $elem: expr; $N: expr ) => {
        $crate::Vector([$elem; $N])
    }
}

impl<T, const N: usize> fmt::Debug for Vector<T, N>
where
    T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector [")?;
        for i in 0..(N - 1) {
            write!(f, "{:?}, ", self[i])?;
        }
        write!(f, "{:?}]", self[N - 1])
    }
}

impl<T, const N: usize> Clone for Vector<T, N>
where
    T: Clone
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T, const N: usize> Copy for Vector<T, N>
where
    T: Copy
{}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Comparison
impl<T, U, const N: usize> PartialEq<Vector<U, N>> for Vector<T, N>
where
    T: PartialEq<U>
{
    fn eq(&self, other: &Vector<U, N>) -> bool {
        for i in 0..N {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}

impl<T, const N: usize> Eq for Vector<T, N>
where
    T: Eq
{}

// Addition / Subtraction
impl<T, const N: usize> Zero for Vector<T, N>
where
    T: Zero,
    T: Copy
{
    fn zero() -> Self {
        // Self([T::zero(); N])
        Self::from_elem(T::zero())
    }
}

impl<T, U, const N: usize> Add<Vector<U, N>> for Vector<T, N>
where
    T: Add<U>,
    T: Copy,
    U: Copy,
    Vector<<T as Add<U>>::Output, N>: Zero
{
    type Output = Vector<<T as Add<U>>::Output, N>;

    fn add(self, rhs: Vector<U, N>) -> Self::Output {
        let mut new = Self::Output::zero();
        for i in 0..N {
            new[i] = self[i] + rhs[i];
        }
        new
    }
}

impl<T, U, const N: usize> AddAssign<Vector<U, N>> for Vector<T, N>
where
    T: AddAssign<U>,
    U: Copy
{
    fn add_assign(&mut self, rhs: Vector<U, N>) {
        for i in 0..N {
            self[i] += rhs[i];
        }
    }
}

impl<T, U, const N: usize> Sub<Vector<U, N>> for Vector<T, N>
where
    T: Sub<U>,
    T: Copy,
    U: Copy,
    Vector<<T as Sub<U>>::Output, N>: Zero
{
    type Output = Vector<<T as Sub<U>>::Output, N>;

    fn sub(self, rhs: Vector<U, N>) -> Self::Output {
        let mut new = Self::Output::zero();
        for i in 0..N {
            new[i] = self[i] - rhs[i];
        }
        new
    }
}

impl<T, U, const N: usize> SubAssign<Vector<U, N>> for Vector<T, N>
where
    T: SubAssign<U>,
    U: Copy
{
    fn sub_assign(&mut self, rhs: Vector<U, N>) {
        for i in 0..N {
            self[i] -= rhs[i];
        }
    }
}

// Multiplication / Division
impl<T, const N: usize> One for Vector<T, N>
where
    T: One,
    T: Copy
{
    fn one() -> Self {
        // Self([T::one(); N])
        Self::from_elem(T::one())
    }
}

impl<T, U, const N: usize> Mul<Vector<U, N>> for Vector<T, N>
where
    T: Mul<U>,
    T: Copy,
    U: Copy,
    Vector<<T as Mul<U>>::Output, N>: One
{
    type Output = Vector<<T as Mul<U>>::Output, N>;

    fn mul(self, rhs: Vector<U, N>) -> Self::Output {
        let mut new = Self::Output::one();
        for i in 0..N {
            new[i] = self[i] * rhs[i];
        }
        new
    }
}

impl<T, U, const N: usize> MulAssign<Vector<U, N>> for Vector<T, N>
where
    T: MulAssign<U>,
    U: Copy
{
    fn mul_assign(&mut self, rhs: Vector<U, N>) {
        for i in 0..N {
            self[i] *= rhs[i];
        }
    }
}

impl<T, U, const N: usize> Div<Vector<U, N>> for Vector<T, N>
where
    T: Div<U>,
    T: Copy,
    U: Copy,
    Vector<<T as Div<U>>::Output, N>: One
{
    type Output = Vector<<T as Div<U>>::Output, N>;

    fn div(self, rhs: Vector<U, N>) -> Self::Output {
        let mut new = Self::Output::one();
        for i in 0..N {
            new[i] = self[i] / rhs[i];
        }
        new
    }
}

impl<T, U, const N: usize> DivAssign<Vector<U, N>> for Vector<T, N>
where
    T: DivAssign<U>,
    U: Copy
{
    fn div_assign(&mut self, rhs: Vector<U, N>) {
        for i in 0..N {
            self[i] /= rhs[i]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector() {
        let a = vector![0, 1, 2, 3];
        let b = Vector([0, 1, 2, 3]);
        assert_eq!(a, b);

        let a = vector![42; 4];
        let b = Vector([42, 42, 42, 42]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_add() {
        let mut a = Vector([0, 1, 2, 3]);
        let b = Vector([4, 5, 6, 7]);
        let c = Vector([4, 6, 8, 10]);
        assert_eq!(a + b, c);

        a += b;
        assert_eq!(a, c);
    }
}
