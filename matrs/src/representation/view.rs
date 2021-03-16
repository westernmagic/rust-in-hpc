use std::ptr::NonNull;
use std::marker::PhantomData;

use super::Representation;
use super::Owned;

pub struct View<'a, T: 'a> {
    data: NonNull<T>,
    r: usize,
    c: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> View<'a, T> {
    pub fn from_raw(ptr: *mut T, rows: usize, cols: usize) -> Self {
        assert!(!ptr.is_null());

        Self {
            data: NonNull::new(ptr).unwrap(),
            r: rows,
            c: cols,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> From<Owned<T>> for View<'a, T> {
    fn from(owned: Owned<T>) -> Self {
        Self {
            data: NonNull::new(owned.as_mut_ptr()).unwrap(),
            r: owned.rows(),
            c: owned.cols(),
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Representation for View<'a, T> {
    type Element = T;

    fn as_mut_ptr(&self) -> *mut Self::Element {
        self.data.as_ptr()
    }

    fn rows(&self) -> usize {
        self.r
    }

    fn cols(&self) -> usize {
        self.c
    }
}
