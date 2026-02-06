# poly-tri

**Delaunay triangulation with constrained boundaries and hole removal** for 2D point sets. Supports non-convex domains and holes; available as a Python package with an optional Rust extension for better performance.

## Features

- **Delaunay triangulation** of 2D point sets
- **Constrained boundaries** — enforce edges in the mesh
- **Hole removal** — triangulate only the region outside specified holes
- **Non-convex geometries** — arbitrary polygonal boundaries
- **Dual implementation** — pure Python (always available) and optional Rust extension (faster)

## Installation

**With Rust extension (recommended for performance):**

```bash
pixi install
pixi run build
```

**Python only** (no build step):

```bash
pip install numpy
```

The package automatically uses the Rust implementation when available; otherwise it falls back to Python.

## Usage

Basic triangulation:

```python
import numpy as np
from polytri import PolyTri

pts = np.array([[-1, 0], [1, 0], [0.0, 0.5], [0.0, -0.5]])
tri = PolyTri(pts, delaunay=True)
triangles = tri.get_triangles()
```

With constrained boundaries and without hole removal:

```python
pts = np.array([[-1, 0], [1, 0], [0.0, 0.5], [0.0, -0.5]])
boundaries = [[0, 1]]  # constrained edge
tri = PolyTri(pts, boundaries=boundaries, holes=False, delaunay=False)
triangles = tri.get_triangles()
```

**Example output:**

![Example triangulation](example.png)

For more options (e.g. `POLYTRI_USE_RUST`, explicit Python/Rust choice), see [polytri/README.md](polytri/README.md).

## Documentation

| Document | Description |
|----------|-------------|
| [API.md](API.md) | Full API reference |
| [ALGORITHM.md](ALGORITHM.md) | Algorithm and data structures |
| [polytri/README.md](polytri/README.md) | Implementation details, env vars, development |

## Development

**Project layout:**

```
poly-tri/
├── polytri/        # Python package (PolyTri, PolyTriPy)
├── rust/           # Rust extension (maturin)
├── tests/          # pytest suite; optional: pixi run speedtest
├── pixi.toml       # environment and tasks
├── API.md
└── ALGORITHM.md
```

**Commands:** `pixi run test` (tests), `pixi run lint` / `pixi run lint-rust`, `pixi run speedtest` (optional benchmark).

## Known limitations

- Delaunay mode may fail in some edge cases (Rust); under investigation.
- Certain degenerate or extreme inputs are not yet handled; more tests are planned.

## License

See [LICENSE](LICENSE).

## Acknowledgments

Triangle creation is based on the approach described in [Delaunay triangulation (ActiveState Recipe)](https://code.activestate.com/recipes/579021-delaunay-triangulation/). This project adds constrained boundaries, hole removal, and non-convex support.
