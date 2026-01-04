//! Python-Bindings für PolyTri
//!
//! Diese Module stellt die Python-API bereit, die exakt der Python-Version entspricht.

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyList, PyTuple};

use crate::polytri::{Boundary, Point, PolyTri};

#[cfg(feature = "python")]
// ============================================================================
// KONVERTIERUNGSFUNKTIONEN
// ============================================================================

/// Konvertiert Python-Array (numpy oder list) zu Vec<Point>
pub fn points_from_python(_py: Python, points: &Bound<'_, PyAny>) -> PyResult<Vec<Point>> {
    // Unterstützt numpy.ndarray
    if let Ok(array) = points.downcast::<PyArray2<f64>>() {
        let array = array.readonly();
        let shape = array.shape();
        if shape[1] != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "points must have shape (N, 2)",
            ));
        }
        let mut result = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            result.push(Point {
                x: *array.get([i, 0]).unwrap(),
                y: *array.get([i, 1]).unwrap(),
            });
        }
        return Ok(result);
    }

    // Unterstützt list von tuples/lists
    if let Ok(list) = points.downcast::<PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            if let Ok(coords) = item.downcast::<PyList>() {
                if coords.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Each point must have 2 coordinates",
                    ));
                }
                let x: f64 = coords.get_item(0)?.extract()?;
                let y: f64 = coords.get_item(1)?.extract()?;
                result.push(Point { x, y });
            } else if let Ok(tuple) = item.downcast::<PyTuple>() {
                if tuple.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Each point must have 2 coordinates",
                    ));
                }
                let x: f64 = tuple.get_item(0)?.extract()?;
                let y: f64 = tuple.get_item(1)?.extract()?;
                result.push(Point { x, y });
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "points must be numpy array or list of tuples/lists",
                ));
            }
        }
        return Ok(result);
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "points must be numpy array or list",
    ))
}

/// Konvertiert Vec<Point> zu numpy.ndarray
pub fn points_to_python(py: Python, points: &[Point]) -> Py<PyArray2<f64>> {
    // Erstelle flaches Array und reshapen es zu 2D
    let mut data = Vec::with_capacity(points.len() * 2);
    for point in points {
        data.push(point.x);
        data.push(point.y);
    }
    // Erstelle 1D-Array
    let array_1d = PyArray1::from_vec_bound(py, data);
    // Reshape zu 2D - verwende PyArrayMethods trait
    let array_2d = array_1d.reshape([points.len(), 2]).unwrap();
    array_2d.unbind().into()
}

/// Konvertiert Python-Boundaries (list of lists) zu Vec<Boundary>
pub fn boundaries_from_python(boundaries: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Boundary>>> {
    if boundaries.is_none() {
        return Ok(None);
    }

    // Prüfe, ob es eine einzelne Liste ist (wird als Liste von Listen behandelt)
    if let Ok(single_list) = boundaries.downcast::<PyList>() {
        // Prüfe, ob es eine Liste von Zahlen ist (einzelne Boundary)
        let mut is_single_boundary = true;
        for item in single_list.iter() {
            if item.extract::<usize>().is_err() {
                is_single_boundary = false;
                break;
            }
        }

        if is_single_boundary && single_list.len() >= 2 {
            // Einzelne Boundary als Liste
            let mut boundary = Vec::with_capacity(single_list.len());
            for item in single_list.iter() {
                boundary.push(item.extract()?);
            }
            return Ok(Some(vec![boundary]));
        }
    }

    let list = boundaries.downcast::<PyList>()?;
    let mut result = Vec::with_capacity(list.len());

    for (idx, item) in list.iter().enumerate() {
        // Unterstütze sowohl PyList als auch numpy-Arrays
        let boundary = if let Ok(boundary_list) = item.downcast::<PyList>() {
            let mut b = Vec::with_capacity(boundary_list.len());
            for item in boundary_list.iter() {
                let index: usize = item.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "boundary {} contains invalid index",
                        idx
                    ))
                })?;
                b.push(index);
            }
            b
        } else if let Ok(array) = item.downcast::<PyArray1<i64>>() {
            // numpy-Array unterstützen
            let array = array.readonly();
            let shape = array.shape();
            let mut b = Vec::with_capacity(shape[0]);
            for i in 0..shape[0] {
                b.push(*array.get([i]).unwrap() as usize);
            }
            b
        } else if let Ok(array) = item.downcast::<PyArray1<usize>>() {
            // numpy-Array mit usize unterstützen
            let array = array.readonly();
            let shape = array.shape();
            let mut b = Vec::with_capacity(shape[0]);
            for i in 0..shape[0] {
                b.push(*array.get([i]).unwrap());
            }
            b
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "boundary {} must be a list or numpy array",
                idx
            )));
        };

        if boundary.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "boundary {} must have at least 2 points",
                idx
            )));
        }

        result.push(boundary);
    }

    Ok(Some(result))
}

/// Konvertiert Vec<Triangle> zu Python list von numpy arrays
pub fn triangles_to_python(py: Python, triangles: &[[usize; 3]]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for triangle in triangles {
        let array = PyArray1::from_vec_bound(py, triangle.to_vec());
        list.append(array)?;
    }
    Ok(list.into())
}

/// Konvertiert Python border (list oder None) zu Option<Vec<usize>>
pub fn border_from_python(border: &Bound<'_, PyAny>) -> PyResult<Option<Vec<usize>>> {
    if border.is_none() {
        return Ok(None);
    }

    let list = border.downcast::<PyList>()?;
    let mut result = Vec::with_capacity(list.len());
    for item in list.iter() {
        result.push(item.extract()?);
    }
    Ok(Some(result))
}

/// Konvertiert Vec<(usize, usize)> zu Python set von tuples
pub fn boundary_edges_to_python(py: Python, edges: &[(usize, usize)]) -> PyResult<PyObject> {
    let set = pyo3::types::PySet::empty_bound(py)?;
    for (i, j) in edges {
        let tuple = PyTuple::new_bound(py, &[i, j]);
        set.add(tuple)?;
    }
    Ok(set.into())
}

// ============================================================================
// PYTHON-KLASSE
// ============================================================================

#[pyclass(name = "PolyTri")]
pub struct PyPolyTri {
    inner: PolyTri,
}

#[pymethods]
impl PyPolyTri {
    #[new]
    #[pyo3(signature = (points, boundaries=None, delaunay=true, holes=true, border=None))]
    fn new(
        py: Python,
        points: &Bound<'_, PyAny>,
        boundaries: Option<&Bound<'_, PyAny>>,
        delaunay: bool,
        holes: bool,
        border: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Konvertiere Python-Input zu Rust-Typen
        let rust_points = points_from_python(py, points)?;

        let rust_boundaries = if let Some(b) = boundaries {
            boundaries_from_python(b)?
        } else {
            None
        };

        let rust_border = if let Some(b) = border {
            border_from_python(b)?
        } else {
            None
        };

        // Erstelle Rust PolyTri
        let inner = PolyTri::new(rust_points, rust_boundaries, delaunay, holes, rust_border)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(PyPolyTri { inner })
    }

    // Property: points
    #[getter]
    fn points(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(points_to_python(py, self.inner.points()))
    }

    // Property: triangles (als Methode, um Konflikt mit get_triangles zu vermeiden)
    #[getter]
    #[pyo3(name = "triangles")]
    fn triangles_getter(&self, py: Python) -> PyResult<PyObject> {
        triangles_to_python(py, &self.inner.get_triangles())
    }

    // Method: get_triangles() - alias für triangles property
    fn get_triangles(&self, py: Python) -> PyResult<PyObject> {
        triangles_to_python(py, &self.inner.get_triangles())
    }

    // Property: boundary_edges
    #[getter]
    fn boundary_edges(&self, py: Python) -> PyResult<PyObject> {
        let edges = self.inner.boundary_edges();
        boundary_edges_to_python(py, &edges)
    }

    // Property: delaunay
    #[getter]
    fn delaunay(&self) -> bool {
        self.inner.delaunay()
    }

    // Property: boundaries
    #[getter]
    fn boundaries(&self, py: Python) -> PyResult<PyObject> {
        match self.inner.boundaries() {
            None => Ok(py.None()),
            Some(boundaries) => {
                let list = PyList::empty_bound(py);
                for boundary in boundaries {
                    let boundary_list = PyList::empty_bound(py);
                    for idx in boundary {
                        boundary_list.append(idx)?;
                    }
                    list.append(boundary_list)?;
                }
                Ok(list.into())
            }
        }
    }

    // Property: border
    #[getter]
    fn border(&self, py: Python) -> PyResult<PyObject> {
        let border = self.inner.border();
        let list = PyList::empty_bound(py);
        for idx in border {
            list.append(idx)?;
        }
        Ok(list.into())
    }

    // Method: constrain_boundaries()
    fn constrain_boundaries(&mut self) -> PyResult<()> {
        self.inner
            .constrain_boundaries()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
    }

    // Method: remove_empty_triangles()
    fn remove_empty_triangles(&mut self) {
        self.inner.remove_empty_triangles();
    }

    // Method: remove_holes()
    fn remove_holes(&mut self) -> PyResult<()> {
        self.inner
            .remove_holes()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
    }

    // Method: flip_edges()
    fn flip_edges(&mut self) {
        self.inner.flip_edges();
    }
}

// ============================================================================
// PYTHON-MODUL
// ============================================================================

#[pymodule]
fn _rust(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPolyTri>()?;
    Ok(())
}
