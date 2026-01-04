# poly_tri

Delaunay triangulation with constrained boundaries and hole removal.

Uses [1] for triangle creation and adds triangle deletion for holes and non-convex geometry.

## Features

- Delaunay triangulation of 2D point sets
- Constrained boundaries
- Hole removal
- Non-convex geometries
- Well-documented API

## Installation

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

For complete API documentation, see [API.md](API.md).

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


## TODO:
cpp: delaunay not yet working  (find the bug)
python, cpp: some cases not yet working: make more tests. 

