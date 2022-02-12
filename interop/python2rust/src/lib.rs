use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyComplex;
use ndarray::{ArrayViewD, ArrayViewMutD};
use numpy::PyArrayDyn;
use num_complex::Complex64;

#[pyfunction]
fn zaxpy(a: &PyComplex, x: &PyArrayDyn<Complex64>, y: &PyArrayDyn<Complex64>) -> PyResult<()> {
    let a: Complex64 = a.extract()?;
    let x: ArrayViewD<Complex64> = x.as_array();
    let mut y: ArrayViewMutD<Complex64> = y.as_array_mut();
    y += &(a * &x);

    Ok(())
}

#[pymodule]
fn python2rust(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(zaxpy))?;

    Ok(())
}
