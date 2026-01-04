# Rust Implementierungsplan für PolyTri

Dieses Dokument beschreibt den Plan zur Implementierung der PolyTri-Bibliothek in Rust, basierend auf `API.md` und `ALGORITHM.md`.

## Übersicht

Die Rust-Implementierung soll die gleiche Funktionalität wie die Python-Version bieten:
- Delaunay-Triangulation mit constrained boundaries
- Hole removal
- Unterstützung für nicht-konvexe Geometrien
- Gleiche API wie die Python-Version (soweit möglich)

## Projektstruktur

```
rust/
├── Cargo.toml              # Projekt-Konfiguration
├── plan_rust.md            # Dieser Plan
├── src/
│   ├── lib.rs              # Hauptmodul, re-exportiert polytri
│   └── polytri.rs          # Komplette PolyTri-Implementierung (alle Funktionen)
└── tests/
    └── integration_tests.rs # Integrationstests
```

**Hinweis**: Alle Funktionalität wird in `polytri.rs` implementiert. Die Datei wird in logische Abschnitte unterteilt, aber alles bleibt in einer Datei.

## 1. Datentypen und Datenstrukturen (`polytri.rs`)

### 1.1 Grundlegende Typen

```rust
// 2D-Punkt
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// Edge als normalisierter Key (immer i1 < i2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeKey(pub usize, pub usize);

impl EdgeKey {
    pub fn new(i1: usize, i2: usize) -> Self {
        if i1 < i2 {
            EdgeKey(i1, i2)
        } else {
            EdgeKey(i2, i1)
        }
    }
}

// Edge mit Orientierung (für geometrische Operationen)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge(pub usize, pub usize);

// Dreieck als Array von 3 Punkt-Indizes
pub type Triangle = [usize; 3];

// Boundary als Liste von Punkt-Indizes
pub type Boundary = Vec<usize>;
```

### 1.2 Hauptdatenstrukturen

```rust
use std::collections::{HashMap, HashSet};

pub struct PolyTri {
    // Eingabeparameter
    points: Vec<Point>,                    // Sortierte Punkte
    boundaries: Option<Vec<Boundary>>,    // Optionale Boundaries
    delaunay: bool,                        // Delaunay-Kriterium aktiviert?
    holes: bool,                           // Hole removal aktiviert?
    border: Vec<usize>,                    // Border-Indizes für hole removal
    
    // Interne Datenstrukturen
    triangles: Vec<Triangle>,              // Liste von Dreiecken
    edge_to_triangles: HashMap<EdgeKey, Vec<usize>>,  // Edge -> Triangle-Indizes
    point_to_triangles: HashMap<usize, HashSet<usize>>, // Punkt -> Triangle-Indizes
    boundary_edges: HashSet<Edge>,         // Boundary-Edges (mit Orientierung)
    
    // Index-Mapping
    point_order: Vec<usize>,               // Sortierter Index -> Originaler Index
    point_unorder: Vec<usize>,             // Originaler Index -> Sortierter Index
    
    // Konstanten
    eps: f64,                              // Numerisches Epsilon
}
```

### 1.3 Struktur der `polytri.rs` Datei

Die Datei `polytri.rs` wird in folgende Abschnitte unterteilt:

1. **Imports und Typen** (Zeilen 1-50)
   - Standard-Library Imports
   - Externe Crates
   - Datentypen (Point, EdgeKey, Edge, Triangle, Boundary)
   - Fehlertypen

2. **Geometrische Hilfsfunktionen** (Zeilen 51-200)
   - `compute_triangle_area()`
   - `is_visible_from_edge()`
   - `ensure_counter_clockwise()`
   - `edges_intersect()`
   - `distance_squared_from_center()`
   - `make_edge_key()` (Hilfsfunktion)

3. **Triangulations-Algorithmus** (Zeilen 201-500)
   - `sort_points_by_distance_from_center()`
   - `remove_collinear_points()`
   - `create_first_triangle()`
   - `add_point()`
   - `flip_edge()`
   - `flip_edges()`
   - `triangle_to_edges()`

4. **Constraint Enforcement** (Zeilen 501-700)
   - `create_boundary_list()`
   - `constrain_edge()`
   - `constrain_boundaries()`

5. **Hole Removal** (Zeilen 701-850)
   - `update_mappings()`
   - `remove_empty_triangles()`
   - `remove_holes()`

6. **PolyTri Implementation** (Zeilen 851-1200)
   - `new()` - Konstruktor
   - Öffentliche Methoden (Getter, constrain_boundaries, etc.)
   - Private Hilfsmethoden

### 1.4 Rust-spezifische Überlegungen

- **Ownership**: `PolyTri` besitzt alle Datenstrukturen
- **Borrowing**: Methoden verwenden `&self` für read-only, `&mut self` für Mutationen
- **Collections**: 
  - `Vec<Triangle>` für Dreiecke (Reihenfolge wichtig)
  - `HashMap<EdgeKey, Vec<usize>>` für Edge-zu-Dreieck-Mapping
  - `HashMap<usize, HashSet<usize>>` für Punkt-zu-Dreieck-Mapping
  - `HashSet<Edge>` für Boundary-Edges

## 2. Implementierung in `polytri.rs`

Alle Funktionen werden in einer Datei `polytri.rs` implementiert. Die Datei ist in logische Abschnitte unterteilt:

### 2.1 Abschnitt 1: Imports, Typen und Fehlerbehandlung

```rust
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// ... Datentypen (Point, EdgeKey, Edge, Triangle, Boundary) ...

#[derive(Debug, Error)]
pub enum PolyTriError {
    #[error("Not enough points: need at least 3, got {0}")]
    NotEnoughPoints(usize),
    
    #[error("All points are collinear")]
    AllPointsCollinear,
    
    #[error("Invalid boundary {0}: {1}")]
    InvalidBoundary(usize, String),
    
    #[error("Constraint error: {0}")]
    ConstraintError(String),
    
    #[error("Invalid point index: {0}")]
    InvalidPointIndex(usize),
}
```

### 2.2 Abschnitt 2: Geometrische Hilfsfunktionen

Alle geometrischen Funktionen als private Hilfsfunktionen:

```rust
// Berechnung der signierten Dreiecksfläche (2D-Kreuzprodukt)
fn compute_triangle_area(p0: Point, p1: Point, p2: Point) -> f64;

// Prüft, ob Punkt von Edge aus "sichtbar" ist
fn is_visible_from_edge(point: Point, edge_start: Point, edge_end: Point, eps: f64) -> bool;

// Stellt sicher, dass Dreieck gegen den Uhrzeigersinn orientiert ist
fn ensure_counter_clockwise(triangle: &mut Triangle, points: &[Point], eps: f64);

// Prüft, ob zwei Edges sich schneiden (Endpunkte ausgeschlossen)
fn edges_intersect(
    edge1_start: Point, edge1_end: Point,
    edge2_start: Point, edge2_end: Point
) -> bool;

// Berechnet quadrierte Entfernung vom Schwerpunkt
fn distance_squared_from_center(point: Point, center: Point) -> f64;

// Erstellt normalisierten Edge-Key
fn make_edge_key(i1: usize, i2: usize) -> EdgeKey;
```

### 2.3 Abschnitt 3: Triangulations-Algorithmus (private Methoden)

Alle Triangulations-Funktionen als private Methoden von `PolyTri`:

```rust
impl PolyTri {
    // Punkt-Sortierung
    fn sort_points_by_distance_from_center(&mut self);
    
    // Kollineare Punkte entfernen
    fn remove_collinear_points(&mut self) -> Result<(), PolyTriError>;
    
    // Erstes Dreieck erstellen
    fn create_first_triangle(&mut self) -> Result<(), PolyTriError>;
    
    // Punkt hinzufügen
    fn add_point(&mut self, point_idx: usize);
    
    // Edge-Flipping (Delaunay)
    fn flip_edge(
        &mut self,
        edge: EdgeKey,
        enforce_delaunay: bool,
        check_intersection: bool
    ) -> HashSet<EdgeKey>;
    
    // Alle Edges flippen
    fn flip_edges(&mut self);
    
    // Dreieck-Edges extrahieren
    fn triangle_to_edges(&self, triangle: Triangle, create_key: bool) -> Vec<EdgeOrKey>;
    
    // Index-Mapping berechnen
    fn compute_point_unorder(&mut self);
}
```

### 2.4 Abschnitt 4: Constraint Enforcement (private Methoden)

```rust
impl PolyTri {
    // Boundary-Liste erstellen
    fn create_boundary_list(&self, border_indices: Option<&[usize]>, create_key: bool) -> Vec<BoundaryEdge>;
    
    // Edge constrainten
    fn constrain_edge(&mut self, edge: EdgeKey) -> Result<(), PolyTriError>;
    
    // Alle Boundaries constrainten (öffentlich aufrufbar)
    pub fn constrain_boundaries(&mut self) -> Result<(), PolyTriError>;
}
```

### 2.5 Abschnitt 5: Hole Removal (private Methoden)

```rust
impl PolyTri {
    // Mappings aktualisieren
    fn update_mappings(&mut self);
    
    // Leere Dreiecke entfernen (öffentlich aufrufbar)
    pub fn remove_empty_triangles(&mut self);
    
    // Holes entfernen (öffentlich aufrufbar)
    pub fn remove_holes(&mut self) -> Result<(), PolyTriError>;
}
```

### 2.6 Abschnitt 6: Öffentliche API

```rust
impl PolyTri {
    // Konstruktor
    pub fn new(
        points: Vec<Point>,
        boundaries: Option<Vec<Boundary>>,
        delaunay: bool,
        holes: bool,
        border: Option<Vec<usize>>
    ) -> Result<Self, PolyTriError>;
    
    // Getter für Punkte
    pub fn points(&self) -> &[Point];
    
    // Getter für Dreiecke (als originale Indizes)
    pub fn get_triangles(&self) -> Vec<[usize; 3]>;
    
    // Getter für Boundary-Edges
    pub fn boundary_edges(&self) -> Vec<(usize, usize)>;
    
    // Flip edges (öffentliche Methode)
    pub fn flip_edges(&mut self);
}
```

### 2.7 Hilfs-Enum für Boundary-Edges

```rust
// Enum für Boundary-Edges (kann EdgeKey oder Edge sein)
enum BoundaryEdge {
    Key(EdgeKey),
    Oriented(Edge),
}
```

## 3. Implementierungsreihenfolge

### Phase 1: Grundlagen
1. ✅ Projektstruktur erstellen (`Cargo.toml`, `src/lib.rs`, `src/polytri.rs`)
2. ✅ Imports und Datentypen definieren (Point, EdgeKey, Edge, Triangle, Boundary)
3. ✅ Fehlertypen definieren (`PolyTriError`)
4. ✅ Geometrische Hilfsfunktionen implementieren
5. ✅ Basis-Triangulation ohne Delaunay

### Phase 2: Delaunay-Triangulation
6. ✅ Edge-Flipping implementieren (`flip_edge()`)
7. ✅ Delaunay-Kriterium prüfen
8. ✅ `flip_edges()` implementieren

### Phase 3: Constraints
9. ✅ Boundary-Liste erstellen (`create_boundary_list()`)
10. ✅ Edge constrainten (`constrain_edge()`)
11. ✅ `constrain_boundaries()` implementieren

### Phase 4: Hole Removal
12. ✅ Mappings aktualisieren (`update_mappings()`)
13. ✅ Leere Dreiecke entfernen (`remove_empty_triangles()`)
14. ✅ Holes entfernen (`remove_holes()`)

### Phase 5: API und Integration
15. ✅ `PolyTri`-Struktur vollständig implementieren
16. ✅ Konstruktor (`new()`) vervollständigen
17. ✅ Öffentliche API vervollständigen
18. ✅ Tests schreiben

## 4. Rust-spezifische Überlegungen

### 4.1 Ownership und Borrowing

- `PolyTri` besitzt alle Datenstrukturen
- Methoden verwenden `&self` für read-only Zugriff
- Methoden verwenden `&mut self` für Mutationen
- Keine `Rc` oder `Arc` nötig (single-threaded)

### 4.2 Collections

- `Vec<Triangle>` für Dreiecke (Reihenfolge wichtig, O(1) Zugriff)
- `HashMap<EdgeKey, Vec<usize>>` für Edge-zu-Dreieck-Mapping
- `HashMap<usize, HashSet<usize>>` für Punkt-zu-Dreieck-Mapping
- `HashSet<Edge>` für Boundary-Edges (schnelle Lookups)

### 4.3 Numerische Präzision

- `f64` für alle Fließkomma-Operationen
- Epsilon-Vergleiche: `abs(value) < EPS`
- Keine `f32`, um mit Python `float64` kompatibel zu bleiben

### 4.4 Fehlerbehandlung

- `Result<T, PolyTriError>` für fallible Operationen
- `thiserror` für Fehlertypen
- Validierung am Anfang von Funktionen

### 4.5 Performance

- `Vec` für sequentielle Zugriffe
- `HashMap`/`HashSet` für schnelle Lookups
- Vermeide unnötige Allokationen
- Nutze `Vec::with_capacity()` wenn Größe bekannt ist

## 5. API-Kompatibilität mit Python

### 5.1 Unterschiede

- Python verwendet `numpy.ndarray`, Rust verwendet `Vec<Point>`
- Python verwendet `list` von `numpy.ndarray` für Dreiecke, Rust verwendet `Vec<[usize; 3]>`
- Python verwendet `set` von `tuple`, Rust verwendet `Vec<(usize, usize)>` oder `HashSet<Edge>`

### 5.2 Konvertierungshilfen

Optional: Hilfsfunktionen für Konvertierung zwischen Rust- und Python-Formaten:

```rust
// Konvertierung von Vec<[f64; 2]> zu Vec<Point>
pub fn points_from_slice(points: &[[f64; 2]]) -> Vec<Point>;

// Konvertierung von Vec<Point> zu Vec<[f64; 2]>
pub fn points_to_slice(points: &[Point]) -> Vec<[f64; 2]>;
```

## 6. Testing

### 6.1 Unit Tests

- Geometrische Funktionen
- Edge-Flipping
- Constraint-Enforcement
- Hole Removal

Alle Tests können in `polytri.rs` mit `#[cfg(test)]` Modulen oder in separaten Test-Dateien sein.

### 6.2 Integration Tests

- Vergleich mit Python-Implementierung
- Test mit bekannten Punkt-Sets
- Test mit Boundaries und Holes
- Edge-Cases (kollineare Punkte, etc.)

### 6.3 Test-Daten

- Einfache Punkt-Sets (3-10 Punkte)
- Komplexe Geometrien mit Boundaries
- Geometrien mit Holes
- Edge-Cases aus Python-Tests

## 7. Dokumentation

### 7.1 Code-Dokumentation

- `///` Dokumentation für alle öffentlichen Funktionen
- `//!` Dokumentation für Module
- Beispiele in Dokumentation

### 7.2 README

- Installation
- Quick Start
- API-Referenz
- Beispiele
- Vergleich mit Python-Version

## 8. Python-Bindings (PyO3)

### 8.1 Übersicht

Die Python-Anbindung muss **exakt** der Python API entsprechen (siehe `API.md`). Sie verwendet PyO3 für die FFI-Schnittstelle zwischen Rust und Python.

### 8.2 Projektstruktur für Python-Bindings

```
rust/
├── Cargo.toml              # Projekt-Konfiguration (mit PyO3)
├── src/
│   ├── lib.rs              # PyO3-Modul-Definition
│   ├── polytri.rs          # Rust-Implementierung
│   └── python_bindings.rs  # Python-Bindings (PyO3-Wrapper)
└── pyproject.toml          # Python-Paket-Konfiguration (optional)
```

### 8.3 Abhängigkeiten (`Cargo.toml`)

```toml
[package]
name = "polytri"
version = "0.1.0"
edition = "2021"

[lib]
name = "polytri"
crate-type = ["cdylib", "rlib"]

[dependencies]
thiserror = "1.0"
pyo3 = { version = "0.21", features = ["extension-module", "numpy"] }
numpy = "0.21"  # PyO3 numpy support

[dev-dependencies]
# Für Tests, falls nötig
```

### 8.4 Python-Bindings Implementierung (`src/python_bindings.rs`)

#### 8.4.1 Konvertierungsfunktionen

```rust
use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray2, PyArray1, IntoPyArray};

// Konvertiert Python-Array (numpy oder list) zu Vec<Point>
fn points_from_python(py: Python, points: &PyAny) -> PyResult<Vec<Point>> {
    // Unterstützt numpy.ndarray und list von tuples/lists
    if let Ok(array) = points.downcast::<PyArray2<f64>>() {
        let array = array.readonly();
        let shape = array.shape();
        if shape[1] != 2 {
            return Err(PyErr::new::<PyValueError, _>("points must have shape (N, 2)"));
        }
        let mut result = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            result.push(Point {
                x: *array.get([i, 0]).unwrap(),
                y: *array.get([i, 1]).unwrap(),
            });
        }
        Ok(result)
    } else if let Ok(list) = points.downcast::<PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            let coords: Vec<f64> = item.extract()?;
            if coords.len() != 2 {
                return Err(PyErr::new::<PyValueError, _>("Each point must have 2 coordinates"));
            }
            result.push(Point { x: coords[0], y: coords[1] });
        }
        Ok(result)
    } else {
        Err(PyErr::new::<PyTypeError, _>("points must be numpy array or list"))
    }
}

// Konvertiert Vec<Point> zu numpy.ndarray
fn points_to_python(py: Python, points: &[Point]) -> Py<PyArray2<f64>> {
    let array = PyArray2::from_vec2(py, 
        points.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>()
    ).unwrap();
    array.into()
}

// Konvertiert Python-Boundaries (list of lists) zu Vec<Boundary>
fn boundaries_from_python(boundaries: &PyAny) -> PyResult<Option<Vec<Boundary>>> {
    if boundaries.is_none() {
        return Ok(None);
    }
    
    let list = boundaries.downcast::<PyList>()?;
    let mut result = Vec::with_capacity(list.len());
    
    for (idx, item) in list.iter().enumerate() {
        let boundary_list = item.downcast::<PyList>()
            .map_err(|_| PyErr::new::<PyTypeError, _>(
                format!("boundary {} must be a list", idx)
            ))?;
        
        let mut boundary = Vec::with_capacity(boundary_list.len());
        for item in boundary_list.iter() {
            let index: usize = item.extract()
                .map_err(|_| PyErr::new::<PyValueError, _>(
                    format!("boundary {} contains invalid index", idx)
                ))?;
            boundary.push(index);
        }
        
        if boundary.len() < 2 {
            return Err(PyErr::new::<PyValueError, _>(
                format!("boundary {} must have at least 2 points", idx)
            ));
        }
        
        result.push(boundary);
    }
    
    Ok(Some(result))
}

// Konvertiert Vec<Triangle> zu Python list von numpy arrays
fn triangles_to_python(py: Python, triangles: &[[usize; 3]]) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for triangle in triangles {
        let array = PyArray1::from_vec(py, triangle.to_vec());
        list.append(array)?;
    }
    Ok(list.into())
}

// Konvertiert Python border (list oder None) zu Option<Vec<usize>>
fn border_from_python(border: &PyAny) -> PyResult<Option<Vec<usize>>> {
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

// Konvertiert HashSet<Edge> zu Python set von tuples
fn boundary_edges_to_python(py: Python, edges: &HashSet<Edge>) -> PyResult<PyObject> {
    let set = pyo3::types::PySet::empty(py)?;
    for edge in edges {
        let tuple = PyTuple::new(py, &[edge.0, edge.1]);
        set.add(tuple)?;
    }
    Ok(set.into())
}
```

#### 8.4.2 Python-Klasse Definition

```rust
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyTypeError};

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
        points: &PyAny,
        boundaries: Option<&PyAny>,
        delaunay: bool,
        holes: bool,
        border: Option<&PyAny>,
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
        let inner = PolyTri::new(
            rust_points,
            rust_boundaries,
            delaunay,
            holes,
            rust_border,
        ).map_err(|e| PyErr::new::<PyValueError, _>(format!("{}", e)))?;
        
        Ok(PyPolyTri { inner })
    }
    
    // Property: points
    #[getter]
    fn points(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        Ok(points_to_python(py, self.inner.points()))
    }
    
    // Property: triangles
    #[getter]
    fn triangles(&self, py: Python) -> PyResult<PyObject> {
        triangles_to_python(py, &self.inner.get_triangles())
    }
    
    // Property: boundary_edges
    #[getter]
    fn boundary_edges(&self, py: Python) -> PyResult<PyObject> {
        let edges = self.inner.boundary_edges();
        let set = pyo3::types::PySet::empty(py)?;
        for (i, j) in edges {
            let tuple = PyTuple::new(py, &[i, j]);
            set.add(tuple)?;
        }
        Ok(set.into())
    }
    
    // Property: delaunay
    #[getter]
    fn delaunay(&self) -> bool {
        self.inner.delaunay
    }
    
    // Property: boundaries
    #[getter]
    fn boundaries(&self, py: Python) -> PyResult<PyObject> {
        match self.inner.boundaries() {
            None => Ok(py.None()),
            Some(boundaries) => {
                let list = PyList::empty(py);
                for boundary in boundaries {
                    let boundary_list = PyList::empty(py);
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
        let list = PyList::empty(py);
        for idx in border {
            list.append(idx)?;
        }
        Ok(list.into())
    }
    
    // Method: get_triangles()
    fn get_triangles(&self, py: Python) -> PyResult<PyObject> {
        self.triangles(py)
    }
    
    // Method: constrain_boundaries()
    fn constrain_boundaries(&mut self) -> PyResult<()> {
        self.inner.constrain_boundaries()
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("{}", e)))
    }
    
    // Method: remove_empty_triangles()
    fn remove_empty_triangles(&mut self) {
        self.inner.remove_empty_triangles();
    }
    
    // Method: remove_holes()
    fn remove_holes(&mut self) -> PyResult<()> {
        self.inner.remove_holes()
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("{}", e)))
    }
    
    // Method: flip_edges()
    fn flip_edges(&mut self) {
        self.inner.flip_edges();
    }
}
```

#### 8.4.3 Python-Modul Definition (`src/lib.rs`)

```rust
use pyo3::prelude::*;

mod polytri;
mod python_bindings;

use python_bindings::PyPolyTri;

#[pymodule]
fn polytri(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPolyTri>()?;
    Ok(())
}
```

### 8.5 Erweiterte Rust-API für Python-Bindings

Die Rust `PolyTri` Struktur muss zusätzliche Getter-Methoden haben, die von den Python-Bindings verwendet werden:

```rust
impl PolyTri {
    // ... bestehende Methoden ...
    
    // Zusätzliche Getter für Python-Bindings
    pub fn delaunay(&self) -> bool {
        self.delaunay
    }
    
    pub fn boundaries(&self) -> Option<&Vec<Boundary>> {
        self.boundaries.as_ref()
    }
    
    pub fn border(&self) -> &[usize] {
        &self.border
    }
}
```

### 8.6 Fehlerbehandlung in Python-Bindings

```rust
// Konvertiert Rust PolyTriError zu Python Exception
impl From<PolyTriError> for PyErr {
    fn from(err: PolyTriError) -> Self {
        match err {
            PolyTriError::NotEnoughPoints(n) => {
                PyErr::new::<PyValueError, _>(
                    format!("Not enough points: need at least 3, got {}", n)
                )
            }
            PolyTriError::AllPointsCollinear => {
                PyErr::new::<PyValueError, _>("All points are collinear")
            }
            PolyTriError::InvalidBoundary(idx, msg) => {
                PyErr::new::<PyValueError, _>(
                    format!("Invalid boundary {}: {}", idx, msg)
                )
            }
            PolyTriError::ConstraintError(msg) => {
                PyErr::new::<PyValueError, _>(
                    format!("Constraint error: {}", msg)
                )
            }
            PolyTriError::InvalidPointIndex(idx) => {
                PyErr::new::<PyValueError, _>(
                    format!("Invalid point index: {}", idx)
                )
            }
        }
    }
}
```

### 8.7 Python-Paket-Struktur (optional)

Für ein vollständiges Python-Paket:

```
polytri/
├── __init__.py          # Importiert Rust-Modul
├── _rust.pyd (Windows)  # Kompilierte Rust-Bibliothek
│   _rust.so (Linux)     # oder
│   _rust.dylib (macOS)
└── __version__.py
```

`__init__.py`:
```python
from ._rust import PolyTri

__all__ = ["PolyTri"]
```

### 8.8 Build-Konfiguration

#### 8.8.1 `pyproject.toml` (optional)

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "polytri"
version = "0.1.0"
description = "Delaunay triangulation with constrained boundaries"
requires-python = ">=3.8"
dependencies = ["numpy"]
```

#### 8.8.2 Build mit Maturin

```bash
# Entwicklung
maturin develop

# Release-Build
maturin build --release

# Installieren
pip install .
```

### 8.9 API-Kompatibilität Checkliste

Die Python-Bindings müssen folgende API **exakt** implementieren (entsprechend `API.md`):

#### Konstruktor
- [x] **Signatur**: `PolyTri(points, boundaries=None, delaunay=True, holes=True, border=None)`
- [x] **`points`**: Akzeptiert `numpy.ndarray` (Shape `(N, 2)`) oder `list` von `[x, y]` Tupeln
- [x] **`boundaries`**: Akzeptiert `None` oder `list` von `list` von Indizes
- [x] **`delaunay`**: `bool`, Standard `True`
- [x] **`holes`**: `bool`, Standard `True`
- [x] **`border`**: Akzeptiert `None` oder `list` von Indizes, Standard `None`
- [x] **Fehler**: Wirft `ValueError` bei ungültigen Eingaben

#### Properties (read-only)
- [x] **`points`**: Gibt `numpy.ndarray` mit Shape `(N, 2)` zurück
- [x] **`triangles`**: Gibt `list` von `numpy.ndarray` zurück, jedes Array enthält 3 Indizes
- [x] **`boundary_edges`**: Gibt `set` von `tuple` `(i, j)` zurück
- [x] **`delaunay`**: Gibt `bool` zurück
- [x] **`boundaries`**: Gibt `list` oder `None` zurück
- [x] **`border`**: Gibt `list` zurück (nie `None`, auch wenn leer)

#### Methods
- [x] **`get_triangles()`**: 
  - Keine Parameter
  - Gibt `list` von `numpy.ndarray` zurück (gleiche Form wie `triangles` Property)
  
- [x] **`constrain_boundaries()`**:
  - Keine Parameter
  - Kann `ValueError` werfen bei Fehlern
  - Wird automatisch während Initialisierung aufgerufen, wenn `boundaries` gegeben
  
- [x] **`remove_empty_triangles()`**:
  - Keine Parameter
  - Wirft keine Exceptions
  
- [x] **`remove_holes()`**:
  - Keine Parameter
  - Kann `ValueError` werfen bei Fehlern
  - Wird automatisch während Initialisierung aufgerufen, wenn `holes=True` und `boundaries` gegeben
  
- [x] **`flip_edges()`**:
  - Keine Parameter
  - Wirft keine Exceptions
  - Wird automatisch während Initialisierung aufgerufen, wenn `delaunay=True`

### 8.10 Beispiel: Python-Nutzung (muss identisch sein)

```python
import numpy as np
from polytri import PolyTri

# Beispiel 1: Einfache Triangulation
points = np.array([[0., 0.], [1., 0.], [0.5, 0.5]])
tri = PolyTri(points)
print(tri.points.shape)  # (3, 2)
print(len(tri.triangles))  # Anzahl der Dreiecke
print(tri.delaunay)  # True

# Beispiel 2: Mit Boundaries
outer_boundary = [0, 1, 2, 0]
inner_boundary = [3, 4, 5, 3]
points = np.array([...])  # 6 Punkte
tri = PolyTri(points, boundaries=[outer_boundary, inner_boundary], holes=True)
print(tri.boundaries)  # [[0, 1, 2, 0], [3, 4, 5, 3]]
print(len(tri.boundary_edges))  # Anzahl Boundary-Edges

# Beispiel 3: Methoden aufrufen
tri.constrain_boundaries()  # Manuell aufrufen
tri.remove_empty_triangles()
triangles = tri.get_triangles()  # Äquivalent zu tri.triangles
```

### 8.11 Test-Strategie für Python-Bindings

```python
# tests/test_python_bindings.py
import numpy as np
from polytri import PolyTri

def test_basic_triangulation():
    points = np.array([[0., 0.], [1., 0.], [0.5, 0.5]])
    tri = PolyTri(points)
    assert len(tri.triangles) > 0
    assert tri.points.shape == (3, 2)
    assert tri.delaunay == True

def test_with_boundaries():
    points = np.array([...])
    boundaries = [[0, 1, 2, 0], [3, 4, 5, 3]]
    tri = PolyTri(points, boundaries=boundaries, holes=True)
    assert tri.boundaries is not None
    assert len(tri.boundary_edges) > 0
```

## 9. Abhängigkeiten (`Cargo.toml`)

```toml
[package]
name = "polytri"
version = "0.1.0"
edition = "2021"

[lib]
name = "polytri"
crate-type = ["cdylib", "rlib"]

[dependencies]
thiserror = "1.0"  # Für Fehlerbehandlung
pyo3 = { version = "0.21", features = ["extension-module", "numpy"] }
numpy = "0.21"  # PyO3 numpy support

[dev-dependencies]
# Für Tests, falls nötig
```

## 10. Dateistruktur Details

### 10.1 `src/lib.rs` (mit Python-Bindings)

```rust
use pyo3::prelude::*;

mod polytri;
mod python_bindings;

use python_bindings::PyPolyTri;

// Rust-API (für direkte Rust-Nutzung)
pub use polytri::{PolyTri, Point, PolyTriError};

// Python-Modul (für Python-Bindings)
#[pymodule]
fn polytri(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPolyTri>()?;
    Ok(())
}
```

### 10.2 `src/polytri.rs` - Struktur

Die Datei ist in logische Abschnitte unterteilt (durch Kommentare):

```rust
// ============================================================================
// IMPORTS
// ============================================================================
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// ============================================================================
// DATENTYPEN
// ============================================================================
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point { ... }

// ... weitere Typen ...

// ============================================================================
// FEHLERBEHANDLUNG
// ============================================================================
#[derive(Debug, Error)]
pub enum PolyTriError { ... }

// ============================================================================
// GEOMETRISCHE HILFSFUNKTIONEN
// ============================================================================
fn compute_triangle_area(...) { ... }
fn is_visible_from_edge(...) { ... }
// ... weitere geometrische Funktionen ...

// ============================================================================
// POLYTRI STRUKTUR
// ============================================================================
pub struct PolyTri { ... }

// ============================================================================
// PRIVATE METHODEN: TRIANGULATION
// ============================================================================
impl PolyTri {
    fn sort_points_by_distance_from_center(&mut self) { ... }
    fn add_point(&mut self, point_idx: usize) { ... }
    fn flip_edge(...) -> HashSet<EdgeKey> { ... }
    // ... weitere private Methoden ...
}

// ============================================================================
// PRIVATE METHODEN: CONSTRAINTS
// ============================================================================
impl PolyTri {
    fn create_boundary_list(...) -> Vec<BoundaryEdge> { ... }
    fn constrain_edge(...) -> Result<(), PolyTriError> { ... }
}

// ============================================================================
// PRIVATE METHODEN: HOLE REMOVAL
// ============================================================================
impl PolyTri {
    fn update_mappings(&mut self) { ... }
    fn remove_holes(&mut self) -> Result<(), PolyTriError> { ... }
}

// ============================================================================
// ÖFFENTLICHE API
// ============================================================================
impl PolyTri {
    pub fn new(...) -> Result<Self, PolyTriError> { ... }
    pub fn points(&self) -> &[Point] { ... }
    pub fn get_triangles(&self) -> Vec<[usize; 3]> { ... }
    // ... weitere öffentliche Methoden ...
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    // ... Tests ...
}
```

### 10.3 `src/python_bindings.rs` - Struktur

```rust
// ============================================================================
// IMPORTS
// ============================================================================
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numpy::{PyArray2, PyArray1};
use super::polytri::{PolyTri, Point, Boundary};

// ============================================================================
// KONVERTIERUNGSFUNKTIONEN
// ============================================================================
fn points_from_python(...) -> PyResult<Vec<Point>> { ... }
fn points_to_python(...) -> Py<PyArray2<f64>> { ... }
fn boundaries_from_python(...) -> PyResult<Option<Vec<Boundary>>> { ... }
fn triangles_to_python(...) -> PyResult<PyObject> { ... }
fn border_from_python(...) -> PyResult<Option<Vec<usize>>> { ... }
fn boundary_edges_to_python(...) -> PyResult<PyObject> { ... }

// ============================================================================
// PYTHON-KLASSE
// ============================================================================
#[pyclass(name = "PolyTri")]
pub struct PyPolyTri { ... }

#[pymethods]
impl PyPolyTri {
    #[new]
    fn new(...) -> PyResult<Self> { ... }
    
    // Properties
    #[getter] fn points(...) -> PyResult<Py<PyArray2<f64>>> { ... }
    #[getter] fn triangles(...) -> PyResult<PyObject> { ... }
    #[getter] fn boundary_edges(...) -> PyResult<PyObject> { ... }
    #[getter] fn delaunay(...) -> bool { ... }
    #[getter] fn boundaries(...) -> PyResult<PyObject> { ... }
    #[getter] fn border(...) -> PyResult<PyObject> { ... }
    
    // Methods
    fn get_triangles(...) -> PyResult<PyObject> { ... }
    fn constrain_boundaries(&mut self) -> PyResult<()> { ... }
    fn remove_empty_triangles(&mut self) { ... }
    fn remove_holes(&mut self) -> PyResult<()> { ... }
    fn flip_edges(&mut self) { ... }
}
```

## 11. Nächste Schritte

1. ✅ Projektstruktur erstellen (`Cargo.toml`, `src/lib.rs`, `src/polytri.rs`, `src/python_bindings.rs`)
2. ✅ Imports und Datentypen definieren
3. ✅ Geometrische Funktionen implementieren
4. ✅ Schrittweise Triangulation implementieren
5. ✅ Constraints und Hole Removal hinzufügen
6. ✅ Python-Bindings implementieren (PyO3)
7. ✅ Konvertierungsfunktionen zwischen Python- und Rust-Typen
8. ✅ Tests schreiben und validieren (sowohl Rust als auch Python)
9. ✅ Dokumentation vervollständigen

## 12. Offene Fragen / Entscheidungen

- **API-Design**: ✅ Rust-API ist Rust-idiomatisch, Python-Bindings entsprechen exakt der Python-API
- **Fehlerbehandlung**: ✅ `Result` für fallible Operationen, `Option` nur für optionale Werte
- **Performance**: Sollen zusätzliche Optimierungen vorgenommen werden (z.B. Spatial Index)?
- **FFI**: ✅ PyO3 wird für Python-Bindings verwendet
- **Build-System**: Maturin für Python-Paket-Build (optional)

---

## 13. Zusammenfassung: Python-Bindings Anforderungen

### Wichtigste Punkte:

1. **Exakte API-Kompatibilität**: Die Python-Bindings müssen **exakt** der Python API entsprechen (siehe `API.md`). Keine Abweichungen in Signatur, Rückgabetypen oder Verhalten.

2. **PyO3 als Basis**: Verwendung von PyO3 für die FFI-Schnittstelle zwischen Rust und Python.

3. **Typ-Konvertierungen**: 
   - Python `numpy.ndarray` ↔ Rust `Vec<Point>`
   - Python `list` von `list` ↔ Rust `Vec<Boundary>`
   - Python `set` von `tuple` ↔ Rust `HashSet<Edge>`
   - Python `list` von `numpy.ndarray` ↔ Rust `Vec<Triangle>`

4. **Properties**: Alle Properties müssen als `#[getter]` implementiert werden und die exakten Python-Typen zurückgeben.

5. **Fehlerbehandlung**: Rust `PolyTriError` wird zu Python `ValueError` konvertiert.

6. **Optional-Parameter**: `None` in Python entspricht `Option::None` in Rust.

7. **Standardwerte**: PyO3 `#[pyo3(signature = ...)]` wird verwendet für Standardwerte in der `#[new]` Methode.

8. **Build**: Maturin kann für einfaches Build und Distribution verwendet werden.

---

**Status**: Plan erstellt, inklusive Python-Bindings-Spezifikation, bereit für Implementierung

