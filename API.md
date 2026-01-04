# PolyTri API Documentation

PolyTri is a Python library for Delaunay triangulation with support for constrained boundaries and hole removal.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Class Reference](#class-reference)
  - [PolyTri](#polytri-class)
  - [Properties](#properties)
  - [Methods](#methods)
- [Examples](#examples)
- [Algorithm Details](#algorithm-details)

## Installation

```bash
pip install numpy
```

## Quick Start

```python
import numpy as np
from polytri import PolyTri

# Create a simple triangulation
points = np.array([[0., 0.], [1., 0.], [0.5, 0.5], [0., 1.]])
tri = PolyTri(points, delaunay=True)

# Get triangles
triangles = tri.get_triangles()
print(f"Created {len(triangles)} triangles")
```

## Class Reference

### PolyTri Class

The main class for performing Delaunay triangulation.

#### Constructor

```python
PolyTri(points, boundaries=None, delaunay=True, holes=True, border=None)
```

**Parameters:**

- `points` (array-like, shape: N×2): Array of 2D points to triangulate. Can be a list of tuples, numpy array, or any array-like structure.
- `boundaries` (list, optional): List of boundary edge sequences. Each boundary is a list of point indices forming a closed loop. Default: `None`
- `delaunay` (bool, optional): If `True`, enforce Delaunay criterion (maximize minimum angles). Default: `True`
- `holes` (bool, optional): If `True`, remove triangles inside holes defined by boundaries. Default: `True`
- `border` (list, optional): List of boundary indices to use as borders for hole removal. Only relevant when `holes=True`. Default: `None` (uses all boundaries)

**Returns:**

A `PolyTri` instance with the computed triangulation.

**Example:**

```python
import numpy as np
from polytri import PolyTri

# Simple triangulation
points = np.array([[0., 0.], [1., 0.], [0.5, 0.5]])
tri = PolyTri(points)

# With constrained boundaries
outer_boundary = [0, 1, 2, 0]  # Closed loop
inner_boundary = [3, 4, 5, 3]  # Hole
points = np.array([...])  # 6 points
tri = PolyTri(points, boundaries=[outer_boundary, inner_boundary], holes=True)
```

---

## Properties

### `points`

**Type:** `numpy.ndarray` (read-only)

**Description:** Returns the input points array (may be reordered internally for optimization).

**Example:**

```python
tri = PolyTri(points)
print(tri.points.shape)  # (N, 2)
print(tri.points[0])     # [x0, y0]
```

---

### `triangles`

**Type:** `list` of `numpy.ndarray` (read-only)

**Description:** Returns a list of triangles, where each triangle is a numpy array of 3 point indices (referring to the original input points).

**Example:**

```python
tri = PolyTri(points)
for triangle in tri.triangles:
    print(triangle)  # e.g., [0, 1, 2]
    # Access triangle vertices:
    v0, v1, v2 = triangle
    p0 = points[v0]
    p1 = points[v1]
    p2 = points[v2]
```

---

### `boundary_edges`

**Type:** `set` of `tuple` (read-only)

**Description:** Set of boundary edges as tuples `(i, j)` where `i` and `j` are point indices.

**Example:**

```python
tri = PolyTri(points)
for edge in tri.boundary_edges:
    i, j = edge
    print(f"Boundary edge: {i} -> {j}")
```

---

### `delaunay`

**Type:** `bool` (read-only)

**Description:** Whether Delaunay criterion is enforced.

---

### `boundaries`

**Type:** `list` or `None` (read-only)

**Description:** The boundaries passed to the constructor.

---

### `border`

**Type:** `list` (read-only)

**Description:** The border indices used for hole removal.

---

## Methods

### `get_triangles()`

**Description:** Get triangles as arrays of original point indices. This is equivalent to accessing the `triangles` property.

**Returns:**

- `list` of `numpy.ndarray`: List of triangles, each containing 3 point indices.

**Example:**

```python
tri = PolyTri(points)
triangles = tri.get_triangles()
print(f"Number of triangles: {len(triangles)}")
for tri_idx, triangle in enumerate(triangles):
    print(f"Triangle {tri_idx}: {triangle}")
```

---

### `constrain_boundaries()`

**Description:** Constrain all specified boundaries to be present in the triangulation. This method is called automatically during initialization if boundaries are provided.

**Note:** This is typically called automatically. You only need to call it manually if you modify the triangulation after initialization.

**Example:**

```python
# Boundaries are automatically constrained
tri = PolyTri(points, boundaries=[boundary1, boundary2])

# Or constrain manually after modification
tri.constrain_boundaries()
```

---

### `remove_empty_triangles()`

**Description:** Remove triangles with zero or near-zero area (degenerate triangles).

**Example:**

```python
tri = PolyTri(points)
tri.remove_empty_triangles()
```

---

### `remove_holes()`

**Description:** Remove triangles inside holes defined by boundaries. This method is called automatically during initialization if `holes=True` and boundaries are provided.

**Note:** Requires boundaries to be set. The `border` parameter determines which boundaries are used as hole borders.

**Example:**

```python
# Holes are automatically removed
tri = PolyTri(points, boundaries=[outer, inner], holes=True)

# Or remove manually
tri.remove_holes()
```

---

### `flip_edges()`

**Description:** Flip all edges to satisfy the Delaunay criterion. This method is called automatically during point insertion if `delaunay=True`.

**Example:**

```python
tri = PolyTri(points, delaunay=True)
# Edges are automatically flipped during construction

# Or flip manually
tri.flip_edges()
```

---

## Examples

### Basic Triangulation

```python
import numpy as np
from polytri import PolyTri
import matplotlib.pyplot as plt

# Create points
points = np.array([
    [0., 0.],
    [1., 0.],
    [0.5, 0.5],
    [0., 1.],
    [1., 1.]
])

# Triangulate
tri = PolyTri(points, delaunay=True)

# Visualize
plt.triplot(*points.T, tri.get_triangles())
plt.plot(*points.T, 'ro')
plt.show()
```

### Constrained Boundaries

```python
import numpy as np
from polytri import PolyTri
import matplotlib.pyplot as plt

# Create points
n = 20
phi = np.linspace(0, 2*np.pi, n)[:-1]
outer = np.array([np.cos(phi), np.sin(phi)]).T
inner = outer * 0.5
points = np.vstack([outer, inner])

# Define boundaries
outer_boundary = list(range(len(outer))) + [0]
inner_boundary = list(range(len(outer), len(points))) + [len(outer)]

# Triangulate with constraints
tri = PolyTri(points, boundaries=[outer_boundary, inner_boundary], 
              delaunay=True, holes=True)

# Visualize
plt.triplot(*points.T, tri.get_triangles())
plt.plot(*outer.T, 'b-', linewidth=2)
plt.plot(*inner.T, 'r-', linewidth=2)
plt.axis('equal')
plt.show()
```

### Non-Delaunay Triangulation

```python
import numpy as np
from polytri import PolyTri

# Create points
points = np.array([[-1, 0], [1, 0], [0., 0.5], [0., -0.5]])

# Triangulate without Delaunay criterion
tri = PolyTri(points, delaunay=False)

# Get triangles
triangles = tri.get_triangles()
print(f"Triangles: {triangles}")
```

### Working with Boundary Edges

```python
import numpy as np
from polytri import PolyTri

points = np.array([[0., 0.], [1., 0.], [0.5, 0.5], [0., 1.]])
tri = PolyTri(points)

# Access boundary edges
print("Boundary edges:")
for edge in tri.boundary_edges:
    i, j = edge
    print(f"  Edge: {i} -> {j}")
    print(f"    Points: {tri.points[i]} -> {tri.points[j]}")
```

### Complex Geometry with Holes

```python
import numpy as np
from polytri import PolyTri
import matplotlib.pyplot as plt

# Create a complex shape with multiple holes
# Outer boundary (rectangle)
outer_points = np.array([
    [0., 0.], [1., 0.], [1., 1.], [0., 1.]
])
outer_boundary = [0, 1, 2, 3, 0]

# Inner hole (circle)
n_hole = 16
phi_hole = np.linspace(0, 2*np.pi, n_hole)[:-1]
hole_center = [0.5, 0.5]
hole_radius = 0.3
hole_points = np.array([
    [hole_center[0] + hole_radius * np.cos(phi), 
     hole_center[1] + hole_radius * np.sin(phi)] 
    for phi in phi_hole
])

# Combine points
points = np.vstack([outer_points, hole_points])
hole_boundary = list(range(len(outer_points), len(points))) + [len(outer_points)]

# Triangulate
tri = PolyTri(points, boundaries=[outer_boundary, hole_boundary], 
              delaunay=True, holes=True)

# Visualize
plt.triplot(*points.T, tri.get_triangles())
plt.plot(*outer_points.T, 'b-', linewidth=2)
plt.plot(*hole_points.T, 'r-', linewidth=2)
plt.axis('equal')
plt.show()
```

---

## Algorithm Details

### Delaunay Triangulation

The Delaunay triangulation maximizes the minimum angle of all triangles, resulting in well-shaped triangles. The algorithm uses edge flipping to maintain the Delaunay criterion.

### Point Insertion

Points are inserted incrementally:
1. Points are sorted by distance from the center of gravity
2. Each point is added to the triangulation
3. Edges are flipped to maintain Delaunay criterion (if enabled)

### Constrained Boundaries

Boundaries are enforced by:
1. Finding edges that intersect the constraint
2. Flipping edges until the constraint is satisfied
3. The constraint edge becomes part of the triangulation

### Hole Removal

Holes are removed by:
1. Identifying triangles inside hole boundaries
2. Recursively removing triangles connected to hole triangles
3. Preserving boundary edges

### Numerical Precision

The algorithm uses an epsilon value (`EPS = 1.23456789e-14`) for floating-point comparisons. Collinear points are automatically removed during initialization.

---

## Performance Notes

- Time complexity: O(N log N) for N points (typical)
- Space complexity: O(N) for N points
- The algorithm is optimized for 2D point sets
- Large point sets (>10,000 points) may take several seconds

---

## References

- Based on: [Delaunay Triangulation Recipe](https://code.activestate.com/recipes/579021-delaunay-triangulation/)
- Delaunay triangulation: [Wikipedia](https://en.wikipedia.org/wiki/Delaunay_triangulation)

---

## License

See LICENSE file for details.

