//! PolyTri - Delaunay triangulation with constrained boundaries and hole removal.
//!
//! This module provides a Rust implementation of Delaunay triangulation
//! that supports constrained boundaries and hole removal, exposed as a Python extension module.

#![allow(non_local_definitions)] // Suppress warning from PyO3 macros

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray2;
use std::collections::{HashMap, HashSet};

/// Numerical epsilon for floating point comparisons
const EPS: f64 = 1.23456789e-14;

/// Create a normalized edge key tuple such that i1 < i2.
/// This ensures edges are stored consistently regardless of vertex order.
fn make_edge_key(i1: usize, i2: usize) -> (usize, usize) {
    if i1 < i2 {
        (i1, i2)
    } else {
        (i2, i1)
    }
}

/// Delaunay triangulation with constrained boundaries and hole removal.
///
/// This struct maintains the triangulation data structures and provides
/// methods for point insertion, edge flipping, and constraint enforcement.
#[pyclass]
pub struct PolyTri {
    /// Points in the triangulation (may be reordered internally)
    points: Vec<[f64; 2]>,
    /// Triangles as arrays of 3 point indices (using internal ordering)
    triangles: Vec<[usize; 3]>,
    /// Mapping from normalized edge keys to triangle indices
    edge_to_triangles: HashMap<(usize, usize), Vec<usize>>,
    /// Mapping from point indices to sets of triangle indices
    point_to_triangles: HashMap<usize, HashSet<usize>>,
    /// Set of boundary edges (as normalized keys)
    boundary_edges: HashSet<(usize, usize)>,
    /// Mapping from internal index to original point index
    point_order: Vec<usize>,
    /// Mapping from original point index to internal index
    point_unorder: Vec<usize>,
    /// Whether to enforce Delaunay criterion
    delaunay: bool,
    /// Optional list of boundary definitions
    boundaries: Option<Vec<Vec<usize>>>,
    /// List of boundary indices to use as borders for hole removal
    border: Vec<usize>,
    /// Cache for triangles property (mapped to original indices)
    triangles_cache: Option<Vec<Vec<usize>>>,
}

#[pymethods]
impl PolyTri {
    #[new]
    #[pyo3(signature = (points, boundaries=None, delaunay=true, holes=true, border=None))]
    fn new(
        _py: Python,
        points: PyReadonlyArray2<f64>,
        boundaries: Option<Vec<Vec<usize>>>,
        delaunay: bool,
        holes: bool,
        border: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let points_array = points.as_array();
        
        // Validate inputs
        if points_array.nrows() < 3 {
            return Err(PyValueError::new_err("At least 3 points are required for triangulation"));
        }
        if points_array.ncols() != 2 {
            return Err(PyValueError::new_err("points must be a 2D array with shape (N, 2)"));
        }
        
        let border = border.unwrap_or_default();
        
        // Validate boundaries if provided
        if let Some(ref boundaries) = boundaries {
            for (i, boundary) in boundaries.iter().enumerate() {
                if boundary.len() < 2 {
                    return Err(PyValueError::new_err(format!("boundary {} must have at least 2 points", i)));
                }
                if let Some(&max_idx) = boundary.iter().max() {
                    if max_idx >= points_array.nrows() {
                        return Err(PyValueError::new_err(
                            format!("boundary {} contains invalid point indices (max index {} >= {} points)", 
                            i, max_idx, points_array.nrows())
                        ));
                    }
                }
                if let Some(&min_idx) = boundary.iter().min() {
                    if (min_idx as i64) < 0 {
                        return Err(PyValueError::new_err(format!("boundary {} contains negative point indices", i)));
                    }
                }
            }
        }
        
        // Convert points to Vec<[f64; 2]>
        let mut points_vec = Vec::new();
        for row in points_array.rows() {
            points_vec.push([row[0], row[1]]);
        }
        
        let mut poly_tri = PolyTri {
            points: points_vec,
            triangles: Vec::new(),
            edge_to_triangles: HashMap::new(),
            point_to_triangles: HashMap::new(),
            boundary_edges: HashSet::new(),
            point_order: Vec::new(),
            point_unorder: Vec::new(),
            delaunay,
            boundaries: boundaries.clone(),
            border,
            triangles_cache: None,
        };
        
        poly_tri.initialize_triangulation()?;
        
        // Enforce Delaunay criterion if requested (before constraints)
        if delaunay {
            poly_tri.flip_edges();
        }
        
        // Apply constraints if specified
        if let Some(ref _boundaries) = poly_tri.boundaries {
            poly_tri.constrain_boundaries()?;
            if holes {
                poly_tri.remove_empty_triangles();
                poly_tri.update_mappings();
                poly_tri.remove_holes()?;
                poly_tri.update_mappings();
            }
        }
        
        Ok(poly_tri)
    }
    
    #[getter]
    fn points(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.points.iter().map(|p| vec![p[0], p[1]]).collect())
    }
    
    /// Get triangles as arrays of original point indices.
    fn get_triangles(&self) -> PyResult<Vec<Vec<usize>>> {
        if let Some(ref cache) = self.triangles_cache {
            return Ok(cache.clone());
        }
        
        let triangles: Vec<Vec<usize>> = self.triangles.iter()
            .map(|tri| {
                let orig_0 = if tri[0] < self.point_order.len() { self.point_order[tri[0]] } else { tri[0] };
                let orig_1 = if tri[1] < self.point_order.len() { self.point_order[tri[1]] } else { tri[1] };
                let orig_2 = if tri[2] < self.point_order.len() { self.point_order[tri[2]] } else { tri[2] };
                vec![orig_0, orig_1, orig_2]
            })
            .collect();
        
        // Cache the result
        // Note: We can't mutate self here, so we'll cache it on next access
        Ok(triangles)
    }
    
    /// Property accessor for triangles (aliases get_triangles).
    #[getter]
    #[pyo3(name = "triangles")]
    fn triangles_property(&self) -> PyResult<Vec<Vec<usize>>> {
        self.get_triangles()
    }
    
    #[getter]
    fn boundary_edges(&self) -> PyResult<Vec<(usize, usize)>> {
        // Return boundary edges with original orientation from triangles
        // Python version returns internal indices (not mapped), so we need to map them to original indices
        // But preserve orientation for consistency
        // Recalculate boundary edges directly from edge_to_triangles to ensure consistency
        let mut mapped_edges = Vec::new();
        
        // Find all edges that belong to exactly one triangle (like update_mappings does)
        for (edge_key, triangles) in &self.edge_to_triangles {
            if triangles.len() == 1 {
                if let Some(&tri_idx) = triangles.first() {
                    if tri_idx < self.triangles.len() {
                        let tri = self.triangles[tri_idx];
                        let edges = [
                            (tri[0], tri[1]),
                            (tri[1], tri[2]),
                            (tri[2], tri[0]),
                        ];
                        for &(i, j) in &edges {
                            if make_edge_key(i, j) == *edge_key {
                                // Map to original indices (like Python does)
                                let orig_i = if i < self.point_order.len() { self.point_order[i] } else { i };
                                let orig_j = if j < self.point_order.len() { self.point_order[j] } else { j };
                                mapped_edges.push((orig_i, orig_j));
                                break;
                            }
                        }
                    }
                }
            }
        }
        Ok(mapped_edges)
    }
    
    #[getter]
    fn delaunay(&self) -> bool {
        self.delaunay
    }
    
    #[getter]
    fn boundaries(&self) -> PyResult<Option<Vec<Vec<usize>>>> {
        Ok(self.boundaries.clone())
    }
    
    #[getter]
    fn border(&self) -> Vec<usize> {
        self.border.clone()
    }
    
    fn constrain_boundaries(&mut self) -> PyResult<()> {
        let boundary_edges = self.create_boundary_list(None, true)?;
        for edge in boundary_edges {
            self.constrain_edge(edge)?;
        }
        Ok(())
    }
    
    fn remove_empty_triangles(&mut self) {
        let mut triangles_to_remove = Vec::new();
        for (i, triangle) in self.triangles.iter().enumerate() {
            let area = self.compute_triangle_area(triangle[0], triangle[1], triangle[2]);
            if area.abs() < EPS {
                triangles_to_remove.push(i);
            }
        }
        
        if !triangles_to_remove.is_empty() {
            triangles_to_remove.sort_by(|a, b| b.cmp(a));
            for &i in &triangles_to_remove {
                self.triangles.remove(i);
            }
            self.triangles_cache = None;
        }
    }
    
    fn remove_holes(&mut self) -> PyResult<()> {
        if self.boundaries.is_none() {
            return Ok(());
        }
        
        // If border is empty, use None to process all boundaries (like Python)
        let border_indices = if self.border.is_empty() {
            None
        } else {
            Some(&self.border[..])
        };
        let boundary_keys = self.create_boundary_list(border_indices, true)?;
        let boundary_tuples = self.create_boundary_list(border_indices, false)?;
        
        let mut edges_to_remove = HashSet::new();
        for (b_key, b_tuple) in boundary_keys.iter().zip(boundary_tuples.iter()) {
            if let Some(triangles) = self.edge_to_triangles.get(b_key) {
                for &tri_idx in triangles {
                    let tri_edges = self.triangle_to_edges(&self.triangles[tri_idx], false);
                    // Check if triangle contains this boundary edge (in either orientation)
                    let b_tuple_reversed = (b_tuple.1, b_tuple.0);
                    if tri_edges.contains(b_tuple) || tri_edges.contains(&b_tuple_reversed) {
                        let tri_edges_keys = self.triangle_to_edges(&self.triangles[tri_idx], true);
                        for edge in tri_edges_keys {
                            edges_to_remove.insert(edge);
                        }
                    }
                }
            }
        }
        
        // Don't remove boundary edges themselves
        for b_key in &boundary_keys {
            edges_to_remove.remove(b_key);
        }
        
        // Find all triangles to remove
        let mut triangles_to_remove = HashSet::new();
        for edge in &edges_to_remove {
            if let Some(triangles) = self.edge_to_triangles.get(edge) {
                for &tri_idx in triangles {
                    triangles_to_remove.insert(tri_idx);
                }
            }
        }
        
        // Expand removal set iteratively
        let mut prev_count = triangles_to_remove.len();
        loop {
            // Add edges from triangles to remove
            for &tri_idx in &triangles_to_remove {
                let tri_edges = self.triangle_to_edges(&self.triangles[tri_idx], true);
                for edge in tri_edges {
                    edges_to_remove.insert(edge);
                }
            }
            
            // Don't remove boundary edges
            for b_key in &boundary_keys {
                edges_to_remove.remove(b_key);
            }
            
            // Find all triangles connected to edges_to_remove
            for edge in &edges_to_remove {
                if let Some(triangles) = self.edge_to_triangles.get(edge) {
                    for &tri_idx in triangles {
                        triangles_to_remove.insert(tri_idx);
                    }
                }
            }
            
            if triangles_to_remove.len() == prev_count {
                break;
            }
            prev_count = triangles_to_remove.len();
        }
        
        // Remove triangles in reverse order
        if !triangles_to_remove.is_empty() {
            let mut sorted: Vec<usize> = triangles_to_remove.iter().cloned().collect();
            sorted.sort_by(|a, b| b.cmp(a));
            for &i in &sorted {
                self.triangles.remove(i);
            }
            self.triangles_cache = None;
            // Update mappings after removing triangles
            self.update_mappings();
        }
        
        Ok(())
    }
    
    fn flip_edges(&mut self) {
        let mut edge_set: HashSet<(usize, usize)> = self.edge_to_triangles.keys().cloned().collect();
        
        while !edge_set.is_empty() {
            let mut new_edge_set = HashSet::new();
            for edge in &edge_set {
                let result_edges = self.flip_edge(*edge, true, false);
                new_edge_set.extend(result_edges);
            }
            edge_set = new_edge_set;
        }
    }
}

// Private methods
impl PolyTri {
    fn initialize_triangulation(&mut self) -> PyResult<()> {
        // Compute center of gravity
        let mut center = [0.0, 0.0];
        for point in &self.points {
            center[0] += point[0];
            center[1] += point[1];
        }
        let n = self.points.len() as f64;
        center[0] /= n;
        center[1] /= n;
        
        // Sort points by distance from center
        let mut points_with_indices: Vec<(usize, [f64; 2])> = self.points.iter()
            .enumerate()
            .map(|(i, p)| (i, *p))
            .collect();
        
        points_with_indices.sort_by(|a, b| {
            let dist_a = {
                let d = [a.1[0] - center[0], a.1[1] - center[1]];
                d[0] * d[0] + d[1] * d[1]
            };
            let dist_b = {
                let d = [b.1[0] - center[0], b.1[1] - center[1]];
                d[0] * d[0] + d[1] * d[1]
            };
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Reorder points and create mapping
        self.points = points_with_indices.iter().map(|(_, p)| *p).collect();
        self.point_order = points_with_indices.iter().map(|(i, _)| *i).collect();
        self.point_unorder = vec![0; self.point_order.len()];
        for (i, &idx) in self.point_order.iter().enumerate() {
            if idx < self.point_unorder.len() {
                self.point_unorder[idx] = i;
            }
        }
        
        // Create first triangle, removing collinear points
        let index = 0;
        while index + 2 < self.points.len() {
            let area = self.compute_triangle_area(index, index + 1, index + 2);
            if area.abs() < EPS {
                // Remove collinear point
                self.points.remove(index);
                self.point_order.remove(index);
                // Rebuild unorder mapping
                self.point_unorder = vec![0; self.point_order.len()];
                for (i, &idx) in self.point_order.iter().enumerate() {
                    if idx < self.point_unorder.len() {
                        self.point_unorder[idx] = i;
                    }
                }
            } else {
                break;
            }
        }
        
        if index > self.points.len().saturating_sub(3) {
            return Ok(()); // All points are collinear
        }
        
        // Create first triangle
        let mut triangle = [index, index + 1, index + 2];
        self.ensure_counter_clockwise(&mut triangle);
        self.triangles.push(triangle);
        self.triangles_cache = None;
        
        // Initialize boundary edges (use normalized keys for consistency)
        let e01_key = make_edge_key(triangle[0], triangle[1]);
        let e12_key = make_edge_key(triangle[1], triangle[2]);
        let e20_key = make_edge_key(triangle[2], triangle[0]);
        
        self.boundary_edges.insert(e01_key);
        self.boundary_edges.insert(e12_key);
        self.boundary_edges.insert(e20_key);
        
        // Initialize edge and point mappings
        self.edge_to_triangles.insert(e01_key, vec![0]);
        self.edge_to_triangles.insert(e12_key, vec![0]);
        self.edge_to_triangles.insert(e20_key, vec![0]);
        
        for &i in &triangle {
            let mut set = HashSet::new();
            set.insert(0);
            self.point_to_triangles.insert(i, set);
        }
        
        // Add remaining points
        for i in 3..self.points.len() {
            self.add_point(i)?;
        }
        
        // Update unorder mapping
        self.point_unorder = vec![0; self.point_order.len()];
        for (i, &idx) in self.point_order.iter().enumerate() {
            if idx < self.point_unorder.len() {
                self.point_unorder[idx] = i;
            }
        }
        
        Ok(())
    }
    
    fn compute_triangle_area(&self, i0: usize, i1: usize, i2: usize) -> f64 {
        let d1 = [self.points[i1][0] - self.points[i0][0], self.points[i1][1] - self.points[i0][1]];
        let d2 = [self.points[i2][0] - self.points[i0][0], self.points[i2][1] - self.points[i0][1]];
        d1[0] * d2[1] - d1[1] * d2[0]
    }
    
    fn is_visible_from_edge(&self, point_idx: usize, edge: (usize, usize)) -> bool {
        let area = self.compute_triangle_area(point_idx, edge.0, edge.1);
        area < EPS
    }
    
    fn ensure_counter_clockwise(&self, triangle: &mut [usize; 3]) {
        let area = self.compute_triangle_area(triangle[0], triangle[1], triangle[2]);
        if area < -EPS {
            triangle.swap(1, 2);
        }
    }
    
    /// Constrain an edge to be present in the triangulation.
    /// 
    /// This method finds intersecting edges and flips them until the constraint
    /// edge is satisfied. If necessary, it recursively constrains sub-edges.
    fn constrain_edge(&mut self, edge: (usize, usize)) -> PyResult<()> {
        let edge_key = make_edge_key(edge.0, edge.1);
        
        // If edge already exists, nothing to do
        if self.edge_to_triangles.contains_key(&edge_key) {
            return Ok(());
        }
        
        let (pt0, pt1) = edge;
        
        // Validate edge endpoints (check if indices are within valid range)
        if pt0 >= self.points.len() {
            return Err(PyValueError::new_err(
                format!("Edge endpoint {} is not a valid point index (max: {})", pt0, self.points.len() - 1)
            ));
        }
        if pt1 >= self.points.len() {
            return Err(PyValueError::new_err(
                format!("Edge endpoint {} is not a valid point index (max: {})", pt1, self.points.len() - 1)
            ));
        }
        
        // Find first intersecting edge
        let mut intersecting_edge = None;
        if let Some(triangles) = self.point_to_triangles.get(&pt1) {
            for &tri_idx in triangles {
                let tri = self.triangles[tri_idx];
                let mut tri_vertices = vec![tri[0], tri[1], tri[2]];
                if let Some(pos) = tri_vertices.iter().position(|&x| x == pt1) {
                    tri_vertices.remove(pos);
                    if tri_vertices.len() == 2 {
                        let candidate_edge = make_edge_key(tri_vertices[0], tri_vertices[1]);
                        if self.edges_intersect(candidate_edge, edge_key) {
                            intersecting_edge = Some(candidate_edge);
                            break;
                        }
                    }
                }
            }
        }
        
        let Some(mut intersecting_edge) = intersecting_edge else {
            return Ok(());
        };
        
        // Flip edges until constraint is satisfied
        let mut edges_to_check = self.flip_edge(intersecting_edge, false, true);
        
        loop {
            let mut found_intersection = false;
            for &e in &edges_to_check {
                if self.edges_intersect(e, edge_key) {
                    intersecting_edge = e;
                    found_intersection = true;
                    break;
                }
            }
            
            if !found_intersection {
                break;
            }
            
            edges_to_check = self.flip_edge(intersecting_edge, false, true);
            
            if edges_to_check.is_empty() {
                if self.edge_to_triangles.contains_key(&edge_key) {
                    break;
                } else {
                    // Recursively constrain sub-edges
                    self.constrain_edge(make_edge_key(intersecting_edge.0, pt0))?;
                    self.constrain_edge(make_edge_key(pt0, pt1))?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Flip an edge between two triangles and update data structures.
    /// 
    /// Args:
    /// - edge: Normalized edge key to flip
    /// - enforce_delaunay: If true, only flip if Delaunay criterion is violated
    /// - check_intersection: If true, only flip if edges intersect
    /// 
    /// Returns: Set of edges that may need to be checked/flipped next
    fn flip_edge(&mut self, edge: (usize, usize), enforce_delaunay: bool, check_intersection: bool) -> HashSet<(usize, usize)> {
        let mut result_edges = HashSet::new();
        
        let triangles = match self.edge_to_triangles.get(&edge) {
            Some(tris) if tris.len() >= 2 => tris.clone(),
            _ => return result_edges,
        };
        
        let tri1_idx = triangles[0];
        let tri2_idx = triangles[1];
        let tri1 = self.triangles[tri1_idx];
        let tri2 = self.triangles[tri2_idx];
        
        // Find opposite vertices
        let mut opposite1 = None;
        let mut opposite2 = None;
        for i in 0..3 {
            if tri1[i] != edge.0 && tri1[i] != edge.1 {
                opposite1 = Some(tri1[i]);
            }
            if tri2[i] != edge.0 && tri2[i] != edge.1 {
                opposite2 = Some(tri2[i]);
            }
        }
        
        let (opposite1, opposite2) = match (opposite1, opposite2) {
            (Some(o1), Some(o2)) => (o1, o2),
            _ => return result_edges,
        };
        
        if check_intersection {
            let diagonal = make_edge_key(opposite1, opposite2);
            if !self.edges_intersect(edge, diagonal) {
                return HashSet::new();
            }
        }
        
        if enforce_delaunay {
            // Compute angles at opposite vertices
            let da1 = [self.points[edge.0][0] - self.points[opposite1][0], 
                       self.points[edge.0][1] - self.points[opposite1][1]];
            let db1 = [self.points[edge.1][0] - self.points[opposite1][0],
                       self.points[edge.1][1] - self.points[opposite1][1]];
            let da2 = [self.points[edge.0][0] - self.points[opposite2][0],
                       self.points[edge.0][1] - self.points[opposite2][1]];
            let db2 = [self.points[edge.1][0] - self.points[opposite2][0],
                       self.points[edge.1][1] - self.points[opposite2][1]];
            
            let cross1 = self.compute_triangle_area(opposite1, edge.0, edge.1);
            let cross2 = self.compute_triangle_area(opposite2, edge.1, edge.0);
            let dot1 = da1[0] * db1[0] + da1[1] * db1[1];
            let dot2 = da2[0] * db2[0] + da2[1] * db2[1];
            
            let angle1 = (cross1.atan2(dot1)).abs();
            let angle2 = (cross2.atan2(dot2)).abs();
            
            // Delaunay criterion: flip if sum of opposite angles > pi
            if !(angle1 + angle2 > std::f64::consts::PI * (1.0 + EPS)) {
                return result_edges;
            }
        }
        
        // Flip the triangles
        let new_tri1 = [opposite1, edge.0, opposite2];
        let new_tri2 = [opposite1, opposite2, edge.1];
        
        self.triangles[tri1_idx] = new_tri1;
        self.triangles[tri2_idx] = new_tri2;
        self.triangles_cache = None;
        
        // Update edge mappings
        self.edge_to_triangles.remove(&edge);
        
        let new_edge = make_edge_key(opposite1, opposite2);
        self.edge_to_triangles.insert(new_edge, vec![tri1_idx, tri2_idx]);
        
        // Update point-to-triangle mappings for new edge vertices
        self.point_to_triangles.entry(opposite1)
            .or_insert_with(HashSet::new)
            .insert(tri1_idx);
        self.point_to_triangles.entry(opposite1)
            .or_insert_with(HashSet::new)
            .insert(tri2_idx);
        self.point_to_triangles.entry(opposite2)
            .or_insert_with(HashSet::new)
            .insert(tri1_idx);
        self.point_to_triangles.entry(opposite2)
            .or_insert_with(HashSet::new)
            .insert(tri2_idx);
        
        // Update edges that now connect to different triangles
        let e1 = make_edge_key(opposite1, edge.1);
        if let Some(tris) = self.edge_to_triangles.get_mut(&e1) {
            for tri in tris.iter_mut() {
                if *tri == tri1_idx {
                    *tri = tri2_idx;
                }
            }
            result_edges.insert(e1);
        }
        
        let e2 = make_edge_key(opposite2, edge.0);
        if let Some(tris) = self.edge_to_triangles.get_mut(&e2) {
            for tri in tris.iter_mut() {
                if *tri == tri2_idx {
                    *tri = tri1_idx;
                }
            }
            result_edges.insert(e2);
        }
        
        // Update point-to-triangle mappings for vertices on the flipped edge
        // Remove old triangle references and add new ones
        for &vertex in &[edge.0, edge.1] {
            if let Some(tris_set) = self.point_to_triangles.get_mut(&vertex) {
                // Remove references to old triangles
                tris_set.remove(&tri1_idx);
                tris_set.remove(&tri2_idx);
                // Add references to new triangles
                tris_set.insert(tri1_idx);
                tris_set.insert(tri2_idx);
            }
        }
        
        // Edges that might need flipping next
        result_edges.insert(make_edge_key(opposite1, edge.0));
        result_edges.insert(make_edge_key(opposite2, edge.1));
        
        result_edges
    }
    
    /// Add a point to the triangulation.
    /// 
    /// This method finds visible boundary edges and creates new triangles.
    /// If Delaunay criterion is enabled, it flips affected edges to maintain
    /// the Delaunay property.
    fn add_point(&mut self, point_idx: usize) -> PyResult<()> {
        let mut edges_to_remove = HashSet::new();
        let mut edges_to_add = Vec::new();
        
        // Collect boundary edges with original orientation from triangles
        // We need to reconstruct the original edge orientation from triangles
        let mut boundary_edges_with_orientation = Vec::new();
        for &edge_key in &self.boundary_edges {
            // Find the triangle that contains this edge to get original orientation
            if let Some(triangles) = self.edge_to_triangles.get(&edge_key) {
                if let Some(&tri_idx) = triangles.first() {
                    let tri = self.triangles[tri_idx];
                    // Find the edge in the triangle with original orientation
                    let edges = [
                        (tri[0], tri[1]),
                        (tri[1], tri[2]),
                        (tri[2], tri[0]),
                    ];
                    for &(i, j) in &edges {
                        if make_edge_key(i, j) == edge_key {
                            boundary_edges_with_orientation.push((i, j));
                            break;
                        }
                    }
                }
            }
        }
        
        for edge in boundary_edges_with_orientation {
            if self.is_visible_from_edge(point_idx, edge) {
                // Create new triangle
                let mut new_triangle = [edge.0, edge.1, point_idx];
                self.ensure_counter_clockwise(&mut new_triangle);
                self.triangles.push(new_triangle);
                let tri_idx = self.triangles.len() - 1;
                self.triangles_cache = None;
                
                // Update edge mappings
                let e0 = make_edge_key(edge.0, edge.1);
                let e1 = make_edge_key(point_idx, edge.0);
                let e2 = make_edge_key(edge.1, point_idx);
                
                for e in &[e0, e1, e2] {
                    self.edge_to_triangles.entry(*e)
                        .or_insert_with(Vec::new)
                        .push(tri_idx);
                }
                
                // Update point-to-triangle mappings
                for &i in &new_triangle {
                    self.point_to_triangles.entry(i)
                        .or_insert_with(HashSet::new)
                        .insert(tri_idx);
                }
                
                // Track boundary edge updates (use normalized keys)
                let edge_key = make_edge_key(edge.0, edge.1);
                edges_to_remove.insert(edge_key);
                edges_to_add.push(make_edge_key(edge.0, point_idx));
                edges_to_add.push(make_edge_key(point_idx, edge.1));
            }
        }
        
        // Update boundary edges
        for edge_key in &edges_to_remove {
            self.boundary_edges.remove(edge_key);
        }
        
        // Add new boundary edges - check if they belong to exactly one triangle
        // Note: edges_to_add contains normalized keys, so we can use them directly
        for edge_key in &edges_to_add {
            // The edge should already be in edge_to_triangles from the triangle creation above
            // But we need to check if it belongs to exactly one triangle
            match self.edge_to_triangles.get(edge_key) {
                Some(triangles) if triangles.len() == 1 => {
                    self.boundary_edges.insert(*edge_key);
                }
                _ => {
                    // Edge belongs to 0 or 2+ triangles, so it's not a boundary edge
                }
            }
        }
        
        // Enforce Delaunay criterion if requested
        // Only flip edges that were affected by the new point
        if self.delaunay {
            let mut edges_to_check: HashSet<(usize, usize)> = edges_to_add.iter().cloned().collect();
            
            // Also check edges of newly created triangles
            let num_new_triangles = edges_to_add.len() / 2;
            let start_tri_idx = self.triangles.len().saturating_sub(num_new_triangles);
            for tri_idx in start_tri_idx..self.triangles.len() {
                let tri_edges = self.triangle_to_edges(&self.triangles[tri_idx], true);
                edges_to_check.extend(tri_edges);
            }
            
            // Flip edges iteratively until no more flips are needed
            while !edges_to_check.is_empty() {
                let mut new_edges_to_check = HashSet::new();
                for edge in &edges_to_check {
                    let result_edges = self.flip_edge(*edge, true, false);
                    new_edges_to_check.extend(result_edges);
                }
                edges_to_check = new_edges_to_check;
            }
        }
        
        Ok(())
    }
    
    fn create_boundary_list(&self, border_indices: Option<&[usize]>, create_key: bool) -> PyResult<Vec<(usize, usize)>> {
        let boundaries = match &self.boundaries {
            Some(b) => b,
            None => return Ok(Vec::new()),
        };
        
        let mut boundary_edges = Vec::new();
        for (k, boundary) in boundaries.iter().enumerate() {
            if let Some(border_indices) = border_indices {
                if !border_indices.contains(&k) {
                    continue;
                }
            }
            
            let boundary_original: Vec<usize> = boundary.iter()
                .filter_map(|&idx| {
                    if idx < self.point_unorder.len() {
                        Some(self.point_unorder[idx])
                    } else {
                        None
                    }
                })
                .collect();
            
            // Create edges between consecutive points (like Python's zip(boundary[:-1], boundary[1:]))
            // This handles closed boundaries correctly (boundary already includes last->first edge)
            for i in 0..boundary_original.len().saturating_sub(1) {
                let i_idx = boundary_original[i];
                let j_idx = boundary_original[i + 1];
                if create_key {
                    boundary_edges.push(make_edge_key(i_idx, j_idx));
                } else {
                    boundary_edges.push((i_idx, j_idx));
                }
            }
        }
        Ok(boundary_edges)
    }
    
    fn update_mappings(&mut self) {
        self.edge_to_triangles.clear();
        self.point_to_triangles.clear();
        
        for (tri_idx, triangle) in self.triangles.iter().enumerate() {
            for edge in self.triangle_to_edges(triangle, true) {
                self.edge_to_triangles.entry(edge)
                    .or_insert_with(Vec::new)
                    .push(tri_idx);
            }
            
            for &point_idx in triangle {
                self.point_to_triangles.entry(point_idx)
                    .or_insert_with(HashSet::new)
                    .insert(tri_idx);
            }
        }
        
        // Update boundary edges based on edge-to-triangle mapping
        let mut new_boundary_edges = HashSet::new();
        for (edge, triangles) in &self.edge_to_triangles {
            if triangles.len() == 1 {
                new_boundary_edges.insert(*edge);
            }
        }
        self.boundary_edges = new_boundary_edges;
    }
    
    fn triangle_to_edges(&self, triangle: &[usize; 3], create_key: bool) -> Vec<(usize, usize)> {
        let edges = vec![
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ];
        
        if create_key {
            edges.iter().map(|&(i, j)| make_edge_key(i, j)).collect()
        } else {
            edges
        }
    }
    
    fn edges_intersect(&self, edge1: (usize, usize), edge2: (usize, usize)) -> bool {
        // If edges share a vertex, they don't intersect
        if edge1.0 == edge2.0 || edge1.0 == edge2.1 || edge1.1 == edge2.0 || edge1.1 == edge2.1 {
            return false;
        }
        
        let p11 = self.points[edge1.0];
        let p12 = self.points[edge1.1];
        let p21 = self.points[edge2.0];
        let p22 = self.points[edge2.1];
        
        let t = [p12[0] - p11[0], p12[1] - p11[1]];
        let s = [p22[0] - p21[0], p22[1] - p21[1]];
        let r = [p21[0] - p11[0], p21[1] - p11[1]];
        
        let det = t[0] * (-s[1]) - t[1] * (-s[0]);
        if det.abs() < EPS {
            return false;
        }
        
        let c1 = (r[0] * (-s[1]) - r[1] * (-s[0])) / det;
        let c2 = (t[0] * r[1] - t[1] * r[0]) / det;
        
        (0.0 < c1 && c1 < 1.0) && (0.0 < c2 && c2 < 1.0)
    }
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PolyTri>()?;
    Ok(())
}
