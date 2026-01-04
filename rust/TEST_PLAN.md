# Rust Test Plan

Dieses Dokument beschreibt die Test-Strategie für die Rust-Implementierung von PolyTri, basierend auf den Python-Tests.

## Test-Struktur

Die Tests sind in mehrere Kategorien unterteilt:

### 1. Unit Tests (in `src/polytri.rs`)

Diese Tests testen einzelne Funktionen und Methoden:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_make_edge_key() { ... }
    
    #[test]
    fn test_compute_triangle_area() { ... }
    
    #[test]
    fn test_is_visible_from_edge() { ... }
    
    #[test]
    fn test_ensure_counter_clockwise() { ... }
    
    #[test]
    fn test_edges_intersect() { ... }
    
    // ... weitere Unit Tests
}
```

### 2. Integration Tests (in `tests/integration_tests.rs`)

Diese Tests testen die vollständige Funktionalität:

- Basic Triangulation Tests
- Constraint Tests
- Delaunay Tests
- Hole Removal Tests
- Edge Cases
- Property Tests
- Method Tests

## Test-Kategorien

### Basic Triangulation Tests

Testen einfache Triangulationen ohne Constraints:

- `test_easy_1`: 5 Punkte in einer Linie
- `test_easy_2`: 4 Punkte mit Constraint-Edge
- `test_easy_3`: 5 Punkte ohne Constraints
- `test_easy3`: 6 Punkte in einer Kurve

### Constraint Tests

Testen constrained boundaries:

- `test_constraint_edge`: Constraint zwischen erstem und letztem Punkt
- `test_constraint_edge_2`: Geschlossene Boundary

### Delaunay Tests

Testen Delaunay-Triangulation:

- `test_delaunay_triangulation`: Mit Delaunay aktiviert
- `test_non_delaunay_triangulation`: Ohne Delaunay

### Hole Removal Tests

Testen hole removal Funktionalität:

- `test_hole_removal`: Circle in Circle
- `test_border_parameter`: Mit border Parameter

### Edge Cases

Testen Grenzfälle und Fehlerbehandlung:

- `test_minimum_points`: Genau 3 Punkte
- `test_too_few_points`: Zu wenige Punkte
- `test_invalid_boundary_index`: Ungültiger Boundary-Index
- `test_negative_boundary_index`: Negativer Boundary-Index
- `test_empty_boundary`: Leere Boundary
- `test_single_point_boundary`: Boundary mit nur einem Punkt

### Geometry Tests

Testen spezielle Geometrien:

- `test_square_triangulation`: Quadrat
- `test_rectangle_triangulation`: Rechteck

### Property Tests

Testen Properties der PolyTri-Struktur:

- `test_points_property`: points Property
- `test_boundary_edges_property`: boundary_edges Property
- `test_delaunay_property`: delaunay Property
- `test_boundaries_property`: boundaries Property
- `test_border_property`: border Property

### Method Tests

Testen öffentliche Methoden:

- `test_get_triangles_method`: get_triangles()
- `test_constrain_boundaries_method`: constrain_boundaries()
- `test_remove_empty_triangles_method`: remove_empty_triangles()
- `test_flip_edges_method`: flip_edges()

## Test-Helper-Funktionen

Die Tests verwenden Helper-Funktionen:

```rust
// Konvertiert Array zu Vec<Point>
fn points_from_array(arr: &[[f64; 2]]) -> Vec<Point>

// Vergleicht zwei Dreiecke (normalisiert)
fn triangles_equal(tri1: &[usize; 3], tri2: &[usize; 3]) -> bool

// Normalisiert ein Dreieck (sortiert Indizes)
fn normalize_triangle(tri: &[usize; 3]) -> [usize; 3]

// Prüft ob zwei Dreieck-Listen gleich sind
fn triangle_lists_equal(tris1: &[[usize; 3]], tris2: &[[usize; 3]]) -> bool
```

## Test-Ausführung

### Alle Tests ausführen:
```bash
cargo test
```

### Nur Unit Tests:
```bash
cargo test --lib
```

### Nur Integration Tests:
```bash
cargo test --test integration_tests
```

### Spezifischen Test ausführen:
```bash
cargo test test_easy_1
```

### Tests mit Ausgabe:
```bash
cargo test -- --nocapture
```

## Vergleich mit Python-Tests

Die Rust-Tests sind von den Python-Tests abgeleitet:

| Python Test | Rust Test | Status |
|------------|-----------|--------|
| `test_easy_1` | `test_easy_1` | ✅ |
| `test_easy_2` | `test_easy_2` | ✅ |
| `test_easy_3` | `test_easy_3` | ✅ |
| `test_easy3` | `test_easy3` | ✅ |
| `test_constraint_edge` | `test_constraint_edge` | ✅ |
| `test_constraint_edge_2` | `test_constraint_edge_2` | ✅ |
| `test_ellipse` | `test_hole_removal` | ✅ |
| `test_profile` | - | ⏳ (später) |

## Test-Coverage

Die Tests sollten folgende Bereiche abdecken:

- ✅ Grundlegende Triangulation
- ✅ Constraint Enforcement
- ✅ Delaunay-Triangulation
- ✅ Hole Removal
- ✅ Edge Cases
- ✅ Fehlerbehandlung
- ✅ Properties
- ✅ Methoden

## Nächste Schritte

1. ✅ Test-Struktur erstellt
2. ⏳ Unit Tests für geometrische Funktionen implementieren
3. ⏳ Integration Tests vollständig implementieren
4. ⏳ Tests mit Python-Ergebnissen vergleichen
5. ⏳ Performance-Tests hinzufügen

