mod fm;

use fm::{FactorizationMachine, create_factorization_machine};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn rustpy_fm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FactorizationMachine>()?;
    m.add_function(wrap_pyfunction!(create_factorization_machine, m)?)?;
    Ok(())
}
