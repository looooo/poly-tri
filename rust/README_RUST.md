# PolyTri Rust Implementation

This directory contains a Rust implementation of PolyTri with Python bindings using PyO3.

## Building

### Prerequisites

1. Install Rust: https://www.rust-lang.org/tools/install
2. Install maturin: `pip install maturin`

### Build Python Extension

```bash
# Development build (faster, includes debug info)
maturin develop

# Release build (optimized)
maturin build --release
```

### Using from Python

After building, you can use it exactly like the Python version:

```python
import numpy as np
from polytri._rust import PolyTri

points = np.array([[0., 0.], [1., 0.], [0.5, 0.5]])
tri = PolyTri(points, delaunay=True)
triangles = tri.get_triangles()
```

## Performance

The Rust implementation should be significantly faster than the Python version, especially for:
- Large point sets (>1000 points)
- Complex geometries with many boundaries
- Repeated triangulation operations

## API Compatibility

The Rust implementation provides the same API as the Python version:
- Same constructor parameters
- Same properties (`points`, `triangles`, `boundary_edges`, etc.)
- Same methods (`get_triangles()`, `constrain_boundaries()`, etc.)

## Development

```bash
# Run tests
cargo test

# Check code
cargo clippy

# Format code
cargo fmt
```


