use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::marker::PhantomData;

use super::Representation;

pub struct Owned<T> {
    data: NonNull<T>,
    r: usize,
    c: usize,
    _marker: PhantomData<T>,
}

impl<T> Owned<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        let layout = Layout::array::<T>(rows * cols).unwrap();
        let ptr = unsafe { alloc(layout) } as *mut T;

        Self {
            data: NonNull::new(ptr).unwrap(),
            r: rows,
            c: cols,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<T>(self.r * self.c).unwrap();
        unsafe { dealloc(self.data.as_ptr() as *mut u8, layout) };
    }
}

impl<T> Representation for Owned<T> {
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

