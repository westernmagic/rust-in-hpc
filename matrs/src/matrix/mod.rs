use crate::representation::{Representation, Owned, View};

pub type Matrix<T> = MatrixBase<Owned<T>>;
pub type MatrixView<'a, T> = MatrixBase<View<'a, T>>;

pub struct MatrixBase<D> {
    data: D,
    rs: usize,
    cs: usize,
}

impl<D> MatrixBase<D> {

}

impl<D> Representation for MatrixBase<D> where D: Representation {
    type Element = D::Element;

    fn as_mut_ptr(&self) -> *mut Self::Element {
        self.data.as_mut_ptr()
    }

    fn rows(&self) -> usize {
        self.data.rows()
    }

    fn cols(&self) -> usize {
        self.data.cols()
    }
}
