use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use ndarray::{ArrayViewD, ArrayViewMutD};
use numpy::PyArrayDyn;

#[pyfunction]
fn zaxpy(a: f64, x: &PyArrayDyn<f64>, y: &PyArrayDyn<f64>) -> PyResult<()> {
    let x: ArrayViewD<f64> = x.as_array();
    let mut y: ArrayViewMutD<f64> = y.as_array_mut();
    y += &(a * &x);

    Ok(())
}

#[pymodule]
fn python2rust(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(zaxpy))?;

    Ok(())
}
