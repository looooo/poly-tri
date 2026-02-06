# poly-tri

Delaunay triangulation with constrained boundaries and hole removal.

Uses [1] for triangle creation and adds triangle deletion for holes and non-convex geometry.

## Features

- Delaunay triangulation of 2D point sets
- Constrained boundaries
- Hole removal
- Non-convex geometries
- Well-documented API

## Installation

For development (Python + Rust extension, recommended):

```bash
pixi install
pixi run build
```

For using only the Python implementation:

```bash
pip install numpy
```

## Quick Start

```python
from polytri import PolyTri
import numpy as np

# Simple triangulation
pts = np.array([[-1, 0], [1, 0], [0., 0.5], [0., -0.5]])
tri = PolyTri(pts, delaunay=True)
triangles = tri.get_triangles()
```

## Documentation

- [API.md](API.md) – API reference
- [ALGORITHM.md](ALGORITHM.md) – algorithm description
- [polytri/README.md](polytri/README.md) – usage details, `POLYTRI_USE_RUST`, implementation notes

## Minimal Example

```python
from polytri import PolyTri
import numpy as np

pts = np.array([[-1, 0], [1, 0], [0., 0.5], [0., -0.5]])
boundaries = [[0, 1]]  # Constrained edge
tri = PolyTri(pts, boundaries=boundaries, holes=False, delaunay=False)
triangles = tri.get_triangles()
```


## example usage:
![example](example.png)


[1] https://code.activestate.com/recipes/579021-delaunay-triangulation/

## Project layout

```
poly-tri/
├── polytri/           # Python package (PolyTri, PolyTriPy)
├── rust/              # Rust extension (maturin build)
├── tests/             # pytest tests, helpers; optional: speedtest (pixi run speedtest)
├── pixi.toml          # environment and tasks
├── API.md
└── ALGORITHM.md
```

## TODO

- Rust: Delaunay not yet working in some cases (find the bug).
- Python, Rust: Some edge cases not yet working; add more tests. 

