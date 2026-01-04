"""
PolyTri - Delaunay triangulation with constrained boundaries and hole removal.

This module provides a Delaunay triangulation implementation that supports:
- Constrained boundaries
- Hole removal
- Non-convex geometries

Based on: https://code.activestate.com/recipes/579021-delaunay-triangulation/
"""
import numpy as np
import copy
import math


def _make_edge_key(i1, i2):
    """
    Create a normalized edge key tuple such that i1 < i2.
    
    Args:
        i1: First vertex index
        i2: Second vertex index
        
    Returns:
        Tuple (min(i1, i2), max(i1, i2))
    """
    if i1 < i2:
        return (i1, i2)
    return (i2, i1)


# Backward compatibility alias
make_key = _make_edge_key


class PolyTri(object):
    """
    Delaunay triangulation with constrained boundaries and hole removal.
    
    This class performs Delaunay triangulation on a set of 2D points, with
    optional support for constrained boundaries and hole removal.
    
    Args:
        points: Array-like of 2D points (shape: Nx2)
        boundaries: Optional list of boundary edge sequences. Each boundary
            is a list of point indices forming a closed loop.
        delaunay: If True, enforce Delaunay criterion (default: True)
        holes: If True, remove triangles inside holes (default: True)
        border: Optional list of boundary indices to use as borders for hole removal
        
    Attributes:
        points: The input points (reordered internally)
        triangles: List of triangles as arrays of point indices
        edge_to_triangles: Dictionary mapping edges to triangle indices
        point_to_triangles: Dictionary mapping point indices to triangle indices
        boundary_edges: Set of boundary edges
    """
    
    EPS = 1.23456789e-14  # Numerical epsilon for floating point comparisons
    small = EPS  # Backward compatibility alias

    def __init__(self, points, boundaries=None, delaunay=True, holes=True, border=None):
        if border is None:
            border = []
            
        # Store original parameters
        self._delaunay = delaunay
        self._boundaries = boundaries
        self._border = border
        
        # Internal data structures
        self._points = np.array(points, dtype=np.float64)
        self._triangles = []  # List of triangles (each is list of 3 point indices)
        self._edge_to_triangles = {}  # edge -> list of triangle indices
        self._point_to_triangles = {}  # point index -> set of triangle indices
        self._boundary_edges = set()  # Set of boundary edges (as tuples)
        
        # Mapping between original and sorted point indices
        self._point_order = None
        self._point_unorder = None
        
        # Public API properties (with backward compatibility)
        self.pts = self._points  # Backward compatibility
        self.tris = self._triangles  # Backward compatibility
        self.edge2tris = self._edge_to_triangles  # Backward compatibility
        self.pnt2tris = self._point_to_triangles  # Backward compatibility
        self.boundary_edges = self._boundary_edges  # Backward compatibility
        self.delaunay = self._delaunay
        self.boundaries = self._boundaries
        self.border = self._border
        
        # Initialize triangulation
        self._initialize_triangulation()
        
        # Apply constraints if specified
        if self._boundaries:
            self.constrain_boundaries()
            if holes:
                self.remove_empty_triangles()
                self._update_mappings()
                self.remove_holes()

    def _initialize_triangulation(self):
        """Initialize the triangulation by sorting points and creating first triangle."""
        # Compute center of gravity
        center = np.mean(self._points, axis=0)
        
        # Sort points by distance from center
        def distance_squared(point):
            d = point - center
            return np.dot(d, d)
        
        points_with_indices = [(pt, i) for i, pt in enumerate(self._points)]
        points_with_indices.sort(key=lambda x: distance_squared(x[0]))
        
        # Reorder points and create mapping
        self._points = np.array([pt for pt, _ in points_with_indices])
        self._point_order = np.array([idx for _, idx in points_with_indices])
        self._point_unorder = np.argsort(self._point_order)
        
        # Update backward compatibility references
        self.pts = self._points
        self.order = self._point_order
        self.unorder = self._point_unorder
        
        # Create first triangle, removing collinear points
        index = 0
        while index + 2 < len(self._points):
            area = self._compute_triangle_area(index, index + 1, index + 2)
            if abs(area) < self.EPS:
                # Remove collinear point
                self._points = np.delete(self._points, index, axis=0)
                self._point_order = np.delete(self._point_order, index)
                self._point_unorder = np.argsort(self._point_order)
                self.pts = self._points
                self.order = self._point_order
                self.unorder = self._point_unorder
            else:
                break
        
        if index > len(self._points) - 3:
            # All points are collinear
            return
        
        # Create first triangle
        triangle = [index, index + 1, index + 2]
        self._ensure_counter_clockwise(triangle)
        self._triangles.append(triangle)
        self.tris = self._triangles
        
        # Initialize boundary edges
        e01 = (triangle[0], triangle[1])
        e12 = (triangle[1], triangle[2])
        e20 = (triangle[2], triangle[0])
        
        self._boundary_edges.add(e01)
        self._boundary_edges.add(e12)
        self._boundary_edges.add(e20)
        self.boundary_edges = self._boundary_edges
        
        # Initialize edge and point mappings
        e01_key = _make_edge_key(e01[0], e01[1])
        e12_key = _make_edge_key(e12[0], e12[1])
        e20_key = _make_edge_key(e20[0], e20[1])
        
        self._edge_to_triangles[e01_key] = [0]
        self._edge_to_triangles[e12_key] = [0]
        self._edge_to_triangles[e20_key] = [0]
        self.edge2tris = self._edge_to_triangles
        
        for i in triangle:
            self._point_to_triangles[i] = {0}
        self.pnt2tris = self._point_to_triangles
        
        # Add remaining points
        for i in range(3, len(self._points)):
            self._add_point(i)
        
        # Update unorder mapping
        self._point_unorder = np.argsort(self._point_order)
        self.unorder = self._point_unorder

    @property
    def points(self):
        """Get the points array."""
        return self._points
    
    @property
    def triangles(self):
        """Get triangles as arrays of original point indices."""
        return [np.array([self._point_order[i] for i in tri]) for tri in self._triangles]
    
    def get_triangles(self):
        """
        Get triangles as arrays of original point indices.
        
        Returns:
            List of numpy arrays, each containing 3 point indices.
        """
        return self.triangles
    
    def get_tris(self):
        """Backward compatibility alias for get_triangles()."""
        return [self._point_order[tri] for tri in self._triangles]

    def _compute_triangle_area(self, i0, i1, i2):
        """
        Compute the signed area of a triangle (2D cross product).
        
        Args:
            i0: Index of first vertex
            i1: Index of second vertex
            i2: Index of third vertex
            
        Returns:
            Signed area (positive for counter-clockwise, negative for clockwise)
        """
        d1 = self._points[i1] - self._points[i0]
        d2 = self._points[i2] - self._points[i0]
        return d1[0] * d2[1] - d1[1] * d2[0]
    
    def get_area(self, i0, i1, i2):
        """Backward compatibility alias for _compute_triangle_area()."""
        return self._compute_triangle_area(i0, i1, i2)

    def _is_visible_from_edge(self, point_idx, edge):
        """
        Check if a point is visible from an edge (lies to the right when edge points down).
        
        Args:
            point_idx: Index of the point
            edge: Tuple of two point indices (edge endpoints)
            
        Returns:
            True if point is visible from the edge
        """
        area = self._compute_triangle_area(point_idx, edge[0], edge[1])
        return area < self.EPS
    
    def is_edge_visible(self, point_idx, edge):
        """Backward compatibility alias for _is_visible_from_edge()."""
        return self._is_visible_from_edge(point_idx, edge)

    def _ensure_counter_clockwise(self, triangle_indices):
        """
        Reorder triangle vertices to ensure counter-clockwise orientation.
        
        Args:
            triangle_indices: List of 3 point indices (modified in-place)
        """
        area = self._compute_triangle_area(triangle_indices[0], 
                                          triangle_indices[1], 
                                          triangle_indices[2])
        if area < -self.EPS:
            # Swap last two vertices
            triangle_indices[1], triangle_indices[2] = triangle_indices[2], triangle_indices[1]
    
    def make_counter_clockwise(self, triangle_indices):
        """Backward compatibility alias for _ensure_counter_clockwise()."""
        self._ensure_counter_clockwise(triangle_indices)

    def _constrain_edge(self, edge):
        """
        Constrain an edge to be present in the triangulation.
        
        Args:
            edge: Tuple of two point indices (edge endpoints)
        """
        edge_key = _make_edge_key(edge[0], edge[1])
        
        # If edge already exists, nothing to do
        if edge_key in self._edge_to_triangles:
            return
        
        pt0, pt1 = edge
        
        # Find first intersecting edge
        intersecting_edge = None
        for tri_idx in self._point_to_triangles.get(pt1, set()):
            tri_vertices = list(self._triangles[tri_idx])
            tri_vertices.remove(pt1)
            candidate_edge = _make_edge_key(*tri_vertices)
            if self._edges_intersect(candidate_edge, edge_key):
                intersecting_edge = candidate_edge
                break
        
        if intersecting_edge is None:
            return
        
        # Flip edges until constraint is satisfied
        edges_to_check = self._flip_edge(intersecting_edge, 
                                        enforce_delaunay=False, 
                                        check_intersection=True)
        
        while True:
            found_intersection = False
            for e in edges_to_check:
                if self._edges_intersect(e, edge_key):
                    intersecting_edge = e
                    found_intersection = True
                    break
            
            if not found_intersection:
                break
            
            edges_to_check = self._flip_edge(intersecting_edge,
                                            enforce_delaunay=False,
                                            check_intersection=True)
            
            if not edges_to_check:
                if edge_key in self._edge_to_triangles:
                    break
                else:
                    # Recursively constrain sub-edges
                    self._constrain_edge(_make_edge_key(intersecting_edge[0], pt0))
                    self._constrain_edge(_make_edge_key(pt0, pt1))
    
    def constraint_edge(self, edge):
        """Backward compatibility alias for _constrain_edge()."""
        self._constrain_edge(edge)

    def _flip_edge(self, edge, enforce_delaunay=True, check_intersection=False):
        """
        Flip an edge between two triangles and update data structures.
        
        Args:
            edge: Tuple of two point indices (edge endpoints, must be normalized)
            enforce_delaunay: If True, only flip if Delaunay criterion is violated
            check_intersection: If True, only flip if edges intersect
            
        Returns:
            Set of edges that may need to be checked/flipped next
        """
        result_edges = set()
        
        triangles = self._edge_to_triangles.get(edge, [])
        if len(triangles) < 2:
            return result_edges
        
        tri1_idx, tri2_idx = triangles
        tri1 = self._triangles[tri1_idx]
        tri2 = self._triangles[tri2_idx]
        
        # Find opposite vertices
        opposite1 = None
        opposite2 = None
        for i in range(3):
            if tri1[i] not in edge:
                opposite1 = tri1[i]
            if tri2[i] not in edge:
                opposite2 = tri2[i]
        
        if check_intersection:
            diagonal = _make_edge_key(opposite1, opposite2)
            if not self._edges_intersect(edge, diagonal):
                return set()
        
        if enforce_delaunay:
            # Compute angles at opposite vertices
            da1 = self._points[edge[0]] - self._points[opposite1]
            db1 = self._points[edge[1]] - self._points[opposite1]
            da2 = self._points[edge[0]] - self._points[opposite2]
            db2 = self._points[edge[1]] - self._points[opposite2]
            
            cross1 = self._compute_triangle_area(opposite1, edge[0], edge[1])
            cross2 = self._compute_triangle_area(opposite2, edge[1], edge[0])
            dot1 = np.dot(da1, db1)
            dot2 = np.dot(da2, db2)
            
            angle1 = abs(math.atan2(cross1, dot1))
            angle2 = abs(math.atan2(cross2, dot2))
            
            # Delaunay criterion: flip if sum of opposite angles > pi
            if not (angle1 + angle2 > math.pi * (1.0 + self.EPS)):
                return result_edges
        
        # Flip the triangles
        new_tri1 = [opposite1, edge[0], opposite2]
        new_tri2 = [opposite1, opposite2, edge[1]]
        
        self._triangles[tri1_idx] = new_tri1
        self._triangles[tri2_idx] = new_tri2
        
        # Update edge mappings
        del self._edge_to_triangles[edge]
        
        new_edge = _make_edge_key(opposite1, opposite2)
        self._edge_to_triangles[new_edge] = [tri1_idx, tri2_idx]
        self._point_to_triangles[new_edge[0]] |= {tri1_idx, tri2_idx}
        self._point_to_triangles[new_edge[1]] |= {tri1_idx, tri2_idx}
        
        # Update edges that now connect to different triangles
        e1 = _make_edge_key(opposite1, edge[1])
        if e1 in self._edge_to_triangles:
            tris = self._edge_to_triangles[e1]
            for i in range(len(tris)):
                if tris[i] == tri1_idx:
                    tris[i] = tri2_idx
            result_edges.add(e1)
        
        e2 = _make_edge_key(opposite2, edge[0])
        if e2 in self._edge_to_triangles:
            tris = self._edge_to_triangles[e2]
            for i in range(len(tris)):
                if tris[i] == tri2_idx:
                    tris[i] = tri1_idx
            result_edges.add(e2)
        
        # Update point-to-triangle mappings
        for i in new_tri1:
            if i in edge:
                tris_set = list(self._point_to_triangles[i])
                for j in range(len(tris_set)):
                    if tris_set[j] == tri2_idx:
                        tris_set[j] = tri1_idx
                self._point_to_triangles[i] = set(tris_set)
        
        for i in new_tri2:
            if i in edge:
                tris_set = list(self._point_to_triangles[i])
                for j in range(len(tris_set)):
                    if tris_set[j] == tri1_idx:
                        tris_set[j] = tri2_idx
                self._point_to_triangles[i] = set(tris_set)
        
        # Edges that might need flipping next
        result_edges.add(_make_edge_key(opposite1, edge[0]))
        result_edges.add(_make_edge_key(opposite2, edge[1]))
        
        return result_edges
    
    def flipOneEdge(self, edge, delaunay=True, check_self_intersection=False):
        """Backward compatibility alias for _flip_edge()."""
        return self._flip_edge(edge, 
                              enforce_delaunay=delaunay,
                              check_intersection=check_self_intersection)

    def flip_edges(self):
        """Flip all edges to satisfy Delaunay criterion."""
        edge_set = set(self._edge_to_triangles.keys())
        
        while edge_set:
            new_edge_set = set()
            for edge in edge_set:
                result_edges = self._flip_edge(edge, enforce_delaunay=True)
                new_edge_set.update(result_edges)
            edge_set = new_edge_set

    def _add_point(self, point_idx):
        """
        Add a point to the triangulation.
        
        Args:
            point_idx: Index of the point to add
        """
        edges_to_remove = set()
        edges_to_add = set()
        
        for edge in self._boundary_edges:
            if self._is_visible_from_edge(point_idx, edge):
                # Create new triangle
                new_triangle = [edge[0], edge[1], point_idx]
                new_triangle.sort()
                self._ensure_counter_clockwise(new_triangle)
                self._triangles.append(new_triangle)
                tri_idx = len(self._triangles) - 1
                
                # Update edge mappings
                e0 = _make_edge_key(*edge)
                e1 = _make_edge_key(point_idx, edge[0])
                e2 = _make_edge_key(edge[1], point_idx)
                
                for e in (e0, e1, e2):
                    if e not in self._edge_to_triangles:
                        self._edge_to_triangles[e] = []
                    self._edge_to_triangles[e].append(tri_idx)
                
                # Update point-to-triangle mappings
                for i in new_triangle:
                    if i not in self._point_to_triangles:
                        self._point_to_triangles[i] = set()
                    self._point_to_triangles[i].add(tri_idx)
                
                # Track boundary edge updates
                edges_to_remove.add(edge)
                edges_to_add.add((edge[0], point_idx))
                edges_to_add.add((point_idx, edge[1]))
        
        # Update boundary edges
        for edge in edges_to_remove:
            self._boundary_edges.discard(edge)
        
        for edge in edges_to_add:
            edge_key = _make_edge_key(*edge)
            if len(self._edge_to_triangles.get(edge_key, [])) == 1:
                self._boundary_edges.add(edge)
        
        # Update backward compatibility references
        self.tris = self._triangles
        self.edge2tris = self._edge_to_triangles
        self.pnt2tris = self._point_to_triangles
        self.boundary_edges = self._boundary_edges
        
        # Enforce Delaunay criterion if requested
        if self._delaunay:
            self.flip_edges()
    
    def add_point(self, point_idx):
        """Backward compatibility alias for _add_point()."""
        self._add_point(point_idx)

    def _create_boundary_list(self, border_indices=None, create_key=True):
        """
        Create a list of boundary edges from boundary definitions.
        
        Args:
            border_indices: Optional list of boundary indices to include
            create_key: If True, return normalized edge keys; otherwise return tuples
            
        Returns:
            List of boundary edges
        """
        boundary_edges = []
        for k, boundary in enumerate(self._boundaries):
            if border_indices and k not in border_indices:
                continue
            boundary_original = self._point_unorder[boundary]
            for i, j in zip(boundary_original[:-1], boundary_original[1:]):
                if create_key:
                    boundary_edges.append(_make_edge_key(i, j))
                else:
                    boundary_edges.append((i, j))
        return boundary_edges
    
    def create_boundary_list(self, border=None, create_key=True):
        """Backward compatibility alias for _create_boundary_list()."""
        return self._create_boundary_list(border, create_key)

    def constrain_boundaries(self):
        """Constrain all specified boundaries."""
        boundary_edges = self._create_boundary_list()
        for edge in boundary_edges:
            self._constrain_edge(edge)
    
    def constraint_boundaries(self):
        """Backward compatibility alias for constrain_boundaries()."""
        return self.constrain_boundaries()

    def _update_mappings(self):
        """Rebuild edge-to-triangle and point-to-triangle mappings."""
        self._edge_to_triangles = {}
        self._point_to_triangles = {}
        
        for tri_idx, triangle in enumerate(self._triangles):
            for edge in self._triangle_to_edges(triangle):
                if edge not in self._edge_to_triangles:
                    self._edge_to_triangles[edge] = []
                self._edge_to_triangles[edge].append(tri_idx)
            
            for point_idx in triangle:
                if point_idx not in self._point_to_triangles:
                    self._point_to_triangles[point_idx] = set()
                self._point_to_triangles[point_idx].add(tri_idx)
        
        # Update backward compatibility references
        self.edge2tris = self._edge_to_triangles
        self.pnt2tris = self._point_to_triangles
    
    def update_mapping(self):
        """Backward compatibility alias for _update_mappings()."""
        self._update_mappings()

    def remove_empty_triangles(self):
        """Remove triangles with zero or near-zero area."""
        triangles_to_remove = []
        for i, triangle in enumerate(self._triangles):
            self._ensure_counter_clockwise(triangle)
            area = self._compute_triangle_area(*triangle)
            if abs(area) < self.EPS:
                triangles_to_remove.append(i)
        
        triangles_to_remove.sort(reverse=True)
        for i in triangles_to_remove:
            self._triangles.pop(i)
        
        # Update backward compatibility reference
        self.tris = self._triangles
    
    def remove_empty(self):
        """Backward compatibility alias for remove_empty_triangles()."""
        self.remove_empty_triangles()

    def _triangle_to_edges(self, triangle, create_key=True):
        """
        Get edges of a triangle.
        
        Args:
            triangle: List of 3 point indices
            create_key: If True, return normalized edge keys; otherwise return tuples
            
        Returns:
            List of edges
        """
        triangle_cyclic = triangle + [triangle[0]]
        if create_key:
            return [_make_edge_key(*edge) for edge in zip(triangle_cyclic[:-1], triangle_cyclic[1:])]
        else:
            return [tuple(edge) for edge in zip(triangle_cyclic[:-1], triangle_cyclic[1:])]
    
    def tri2edges(self, triangle, create_key=True):
        """Backward compatibility alias for _triangle_to_edges()."""
        return self._triangle_to_edges(triangle, create_key)

    def _edges_intersect(self, edge1, edge2):
        """
        Check if two edges intersect (excluding endpoints).
        
        Args:
            edge1: Tuple of two point indices
            edge2: Tuple of two point indices
            
        Returns:
            True if edges intersect
        """
        # If edges share a vertex, they don't intersect
        if edge1[0] in edge2 or edge1[1] in edge2:
            return False
        
        p11 = self._points[edge1[0]]
        p12 = self._points[edge1[1]]
        p21 = self._points[edge2[0]]
        p22 = self._points[edge2[1]]
        
        t = p12 - p11
        s = p22 - p21
        r = p21 - p11
        
        try:
            coeffs = np.linalg.solve(np.array([t, -s]).T, r)
            c1, c2 = coeffs
            return (0 < c1 < 1) and (0 < c2 < 1)
        except np.linalg.LinAlgError:
            return False
    
    def is_intersecting(self, edge1, edge2):
        """Backward compatibility alias for _edges_intersect()."""
        return self._edges_intersect(edge1, edge2)

    def remove_holes(self):
        """Remove triangles inside holes defined by boundaries."""
        boundary_keys = self._create_boundary_list(self._border, create_key=True)
        boundary_tuples = self._create_boundary_list(self._border, create_key=False)
        
        edges_to_remove = set()
        for b_key, b_tuple in zip(boundary_keys, boundary_tuples):
            triangles = self._edge_to_triangles.get(b_key, [])
            for tri_idx in triangles:
                tri_edges = self._triangle_to_edges(self._triangles[tri_idx], create_key=False)
                if b_tuple in tri_edges:
                    for edge in self._triangle_to_edges(self._triangles[tri_idx], create_key=True):
                        edges_to_remove.add(edge)
        
        # Don't remove boundary edges themselves
        for b_key in boundary_keys:
            edges_to_remove.discard(b_key)
        
        # Find all triangles to remove
        triangles_to_remove = set()
        for edge in edges_to_remove:
            triangles_to_remove.update(self._edge_to_triangles.get(edge, []))
        
        # Expand removal set iteratively
        prev_count = len(triangles_to_remove)
        while True:
            for tri_idx in triangles_to_remove:
                for edge in self._triangle_to_edges(self._triangles[tri_idx], create_key=True):
                    edges_to_remove.add(edge)
            
            # Don't remove boundary edges
            for b_key in boundary_keys:
                edges_to_remove.discard(b_key)
            
            for edge in edges_to_remove:
                triangles_to_remove.update(self._edge_to_triangles.get(edge, []))
            
            if len(triangles_to_remove) == prev_count:
                break
            prev_count = len(triangles_to_remove)
        
        # Remove triangles in reverse order
        triangles_to_remove = sorted(triangles_to_remove, reverse=True)
        for i in triangles_to_remove:
            self._triangles.pop(i)
        
        # Update backward compatibility reference
        self.tris = self._triangles
