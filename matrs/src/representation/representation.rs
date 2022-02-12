pub(crate) trait Representation {
    type Element;

    fn as_ptr(&self) -> *const Self::Element {
        self.as_mut_ptr() as *const _
    }

    fn as_mut_ptr(&self) -> *mut Self::Element;

    fn rows(&self) -> usize;

    fn cols(&self) -> usize;

    fn size(&self) -> usize {
        self.rows() * self.cols()
    }
}
