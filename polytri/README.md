# PolyTri Package

Dieses Package bietet sowohl eine Rust-Implementierung (schneller) als auch eine Python-Implementierung (Fallback) für Delaunay-Triangulation.

## Verwendung

```python
from polytri import PolyTri

# Verwendet automatisch die Rust-Version, falls verfügbar
# Andernfalls wird die Python-Version verwendet
tri = PolyTri(points, boundaries=boundaries, delaunay=True)
```

## Version-Auswahl

Das Package wählt automatisch die beste verfügbare Implementierung:

1. **Rust-Version** (Standard, falls verfügbar): Schneller, kompiliert
2. **Python-Version** (Fallback): Reine Python-Implementierung

### Manuelle Auswahl

Sie können die Implementierung mit der Umgebungsvariable `POLYTRI_USE_RUST` steuern:

```bash
# Rust-Version erzwingen (falls verfügbar)
POLYTRI_USE_RUST=1 python script.py

# Python-Version erzwingen
POLYTRI_USE_RUST=0 python script.py

# Automatische Auswahl (Standard)
POLYTRI_USE_RUST=auto python script.py
```

### Programmgesteuerte Auswahl

```python
from polytri import PolyTri, get_implementation, is_rust_available

# Prüfen, welche Version verwendet wird
print("Aktuelle Implementierung:", get_implementation())
print("Rust verfügbar:", is_rust_available())

# Beide Versionen können auch direkt importiert werden
from polytri._rust import PolyTri as RustPolyTri  # Nur wenn verfügbar
from polytri._python import PolyTri as PythonPolyTri  # Immer verfügbar
```

## API

Die API ist für beide Versionen identisch:

- `PolyTri(points, boundaries=None, delaunay=True, holes=True, border=None)`
- `tri.triangles` - Property für Dreiecke
- `tri.get_triangles()` - Methode für Dreiecke
- `tri.points` - Property für Punkte
- `tri.boundary_edges` - Property für Boundary-Kanten
- `tri.delaunay` - Property für Delaunay-Flag
- `tri.boundaries` - Property für Boundaries
- `tri.border` - Property für Border-Indizes

## Installation

### Rust-Version installieren

```bash
pixi run maturin develop --release
```

### Nur Python-Version

Die Python-Version ist immer verfügbar und benötigt keine zusätzliche Installation.

## Entwicklung

- `polytri/_rust.py` - Rust-Implementierung (wird von maturin generiert)
- `polytri/_python.py` - Python-Implementierung
- `polytri/__init__.py` - Package-Initialisierung mit automatischer Version-Auswahl

