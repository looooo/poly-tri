//! PolyTri - Delaunay triangulation implementation
//!
//! This module contains the complete PolyTri implementation.

use std::collections::{HashMap, HashSet};
use thiserror::Error;

// ============================================================================
// IMPORTS
// ============================================================================

// Standard library imports are done above

// ============================================================================
// DATENTYPEN
// ============================================================================

/// 2D-Punkt
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Edge als normalisierter Key (immer i1 < i2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeKey(pub usize, pub usize);

impl EdgeKey {
    /// Erstellt einen normalisierten Edge-Key
    pub fn new(i1: usize, i2: usize) -> Self {
        if i1 < i2 {
            EdgeKey(i1, i2)
        } else {
            EdgeKey(i2, i1)
        }
    }
}

/// Edge mit Orientierung (für geometrische Operationen)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge(pub usize, pub usize);

/// Dreieck als Array von 3 Punkt-Indizes
pub type Triangle = [usize; 3];

/// Boundary als Liste von Punkt-Indizes
pub type Boundary = Vec<usize>;

// ============================================================================
// FEHLERBEHANDLUNG
// ============================================================================

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

// ============================================================================
// POLYTRI STRUKTUR
// ============================================================================

pub struct PolyTri {
    // Eingabeparameter
    original_points: Vec<Point>,       // Ursprüngliche Punkte (vor Sortierung)
    points: Vec<Point>,                 // Sortierte Punkte (intern verwendet)
    boundaries: Option<Vec<Boundary>>, // Optionale Boundaries
    delaunay: bool,                    // Delaunay-Kriterium aktiviert?
    holes: bool,                       // Hole removal aktiviert?
    border: Vec<usize>,                // Border-Indizes für hole removal
    
    // Interne Datenstrukturen
    triangles: Vec<Triangle>,                        // Liste von Dreiecken
    edge_to_triangles: HashMap<EdgeKey, Vec<usize>>, // Edge -> Triangle-Indizes
    point_to_triangles: HashMap<usize, HashSet<usize>>, // Punkt -> Triangle-Indizes
    boundary_edges: HashSet<Edge>,                   // Boundary-Edges (mit Orientierung)
    
    // Index-Mapping
    point_order: Vec<usize>,   // Sortierter Index -> Originaler Index
    point_unorder: Vec<usize>, // Originaler Index -> Sortierter Index
    
    // Konstanten
    eps: f64, // Numerisches Epsilon
}

// ============================================================================
// GEOMETRISCHE HILFSFUNKTIONEN
// ============================================================================

/// Berechnet die signierte Dreiecksfläche (2D-Kreuzprodukt)
/// 
/// Positive Fläche bedeutet gegen den Uhrzeigersinn orientiert,
/// negative Fläche bedeutet im Uhrzeigersinn orientiert.
fn compute_triangle_area(p0: Point, p1: Point, p2: Point) -> f64 {
    let d1_x = p1.x - p0.x;
    let d1_y = p1.y - p0.y;
    let d2_x = p2.x - p0.x;
    let d2_y = p2.y - p0.y;
    
    d1_x * d2_y - d1_y * d2_x
}

/// Prüft, ob ein Punkt von einer Edge aus "sichtbar" ist
/// 
/// Ein Punkt ist sichtbar, wenn er rechts der Edge liegt (wenn man entlang der Edge schaut).
fn is_visible_from_edge(point: Point, edge_start: Point, edge_end: Point, eps: f64) -> bool {
    let area = compute_triangle_area(point, edge_start, edge_end);
    area < eps
}

/// Stellt sicher, dass ein Dreieck gegen den Uhrzeigersinn orientiert ist
/// 
/// Modifiziert das Dreieck in-place, falls nötig.
fn ensure_counter_clockwise(triangle: &mut Triangle, points: &[Point], eps: f64) {
    let area = compute_triangle_area(
        points[triangle[0]],
        points[triangle[1]],
        points[triangle[2]],
    );
    
    if area < -eps {
        // Dreieck ist im Uhrzeigersinn, vertausche letzte zwei Vertices
        triangle.swap(1, 2);
    }
}

/// Prüft, ob zwei Edges sich schneiden (Endpunkte ausgeschlossen)
/// 
/// Gibt `true` zurück, wenn sich die Edges schneiden, `false` sonst.
/// Parallele Edges schneiden sich nicht.
fn edges_intersect(
    edge1_start: Point,
    edge1_end: Point,
    edge2_start: Point,
    edge2_end: Point,
) -> bool {
    // Wenn Edges gemeinsame Vertices haben, schneiden sie sich nicht
    if edge1_start == edge2_start
        || edge1_start == edge2_end
        || edge1_end == edge2_start
        || edge1_end == edge2_end
    {
        return false;
    }
    
    // Vektoren berechnen
    let t_x = edge1_end.x - edge1_start.x;
    let t_y = edge1_end.y - edge1_start.y;
    let s_x = edge2_end.x - edge2_start.x;
    let s_y = edge2_end.y - edge2_start.y;
    let r_x = edge2_start.x - edge1_start.x;
    let r_y = edge2_start.y - edge1_start.y;
    
    // Löse: t * c1 - s * c2 = r
    // Das ist ein 2x2 Gleichungssystem: [t_x, -s_x; t_y, -s_y] * [c1; c2] = [r_x; r_y]
    
    let det = t_x * (-s_y) - t_y * (-s_x);
    
    // Wenn Determinante nahe Null, sind die Edges parallel
    if det.abs() < 1e-14 {
        return false;
    }
    
    // Löse das Gleichungssystem
    let c1 = (r_x * (-s_y) - r_y * (-s_x)) / det;
    let c2 = (t_x * r_y - t_y * r_x) / det;
    
    // Prüfe, ob Schnittpunkt innerhalb beider Edges liegt
    (0.0 < c1 && c1 < 1.0) && (0.0 < c2 && c2 < 1.0)
}

/// Berechnet die quadrierte Entfernung vom Schwerpunkt
fn distance_squared_from_center(point: Point, center: Point) -> f64 {
    let dx = point.x - center.x;
    let dy = point.y - center.y;
    dx * dx + dy * dy
}

/// Erstellt einen normalisierten Edge-Key (Hilfsfunktion)
fn make_edge_key(i1: usize, i2: usize) -> EdgeKey {
    EdgeKey::new(i1, i2)
}

// ============================================================================
// PRIVATE METHODEN: TRIANGULATION
// ============================================================================

impl PolyTri {
    /// Konstante für numerisches Epsilon
    const EPS: f64 = 1.23456789e-14;

    /// Sortiert Punkte nach Entfernung vom Schwerpunkt
    /// 
    /// Gibt zurück: (sortierte Punkte, point_order, point_unorder)
    fn sort_points_by_distance_from_center(
        points: &[Point],
    ) -> (Vec<Point>, Vec<usize>, Vec<usize>) {
        // Berechne Schwerpunkt
        let n = points.len() as f64;
        let center = Point {
            x: points.iter().map(|p| p.x).sum::<f64>() / n,
            y: points.iter().map(|p| p.y).sum::<f64>() / n,
        };
        
        // Erstelle Paare von (Punkt, Original-Index)
        let mut points_with_indices: Vec<(Point, usize)> = 
            points.iter().enumerate().map(|(i, &pt)| (pt, i)).collect();
        
        // Sortiere nach quadrierter Entfernung vom Schwerpunkt
        points_with_indices.sort_by(|a, b| {
            let dist_a = distance_squared_from_center(a.0, center);
            let dist_b = distance_squared_from_center(b.0, center);
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Reordne Punkte und erstelle Mapping
        let sorted_points: Vec<Point> = points_with_indices.iter().map(|(pt, _)| *pt).collect();
        let point_order: Vec<usize> = points_with_indices.iter().map(|(_, idx)| *idx).collect();
        
        // Berechne point_unorder (inverse Mapping)
        let mut point_unorder = vec![0; points.len()];
        for (sorted_idx, &original_idx) in point_order.iter().enumerate() {
            point_unorder[original_idx] = sorted_idx;
        }
        
        (sorted_points, point_order, point_unorder)
    }
    
    /// Entfernt kollineare Punkte am Anfang
    fn remove_collinear_points(
        points: &mut Vec<Point>,
        point_order: &mut Vec<usize>,
        eps: f64,
    ) -> Result<(), PolyTriError> {
        let index = 0;
        
        while index + 2 < points.len() {
            let area = compute_triangle_area(points[index], points[index + 1], points[index + 2]);
            
            if area.abs() < eps {
                // Entferne kollinearen Punkt
                points.remove(index);
                point_order.remove(index);
            } else {
                break;
            }
        }
        
        if points.len() < 3 {
            return Err(PolyTriError::AllPointsCollinear);
        }
        
        Ok(())
    }
    
    /// Erstellt das erste Dreieck aus den ersten drei nicht-kollinearen Punkten
    fn create_first_triangle(
        points: &[Point],
        eps: f64,
    ) -> Result<(Triangle, Vec<Edge>), PolyTriError> {
        if points.len() < 3 {
            return Err(PolyTriError::NotEnoughPoints(points.len()));
        }
        
        let mut triangle = [0, 1, 2];
        
        // Stelle sicher, dass Dreieck gegen den Uhrzeigersinn orientiert ist
        ensure_counter_clockwise(&mut triangle, points, eps);
        
        // Erstelle drei Boundary-Edges (mit Orientierung)
        let edges = vec![
            Edge(triangle[0], triangle[1]),
            Edge(triangle[1], triangle[2]),
            Edge(triangle[2], triangle[0]),
        ];
        
        Ok((triangle, edges))
    }
    
    /// Extrahiert alle drei Edges eines Dreiecks
    /// 
    /// Wenn create_key=true, werden EdgeKeys zurückgegeben (normalisiert).
    /// Wenn create_key=false, werden Edges mit Orientierung zurückgegeben.
    fn triangle_to_edges_keys(&self, triangle: Triangle) -> Vec<EdgeKey> {
        let mut edges = Vec::with_capacity(3);
        let triangle_cyclic = [triangle[0], triangle[1], triangle[2], triangle[0]];
        
        for i in 0..3 {
            edges.push(make_edge_key(triangle_cyclic[i], triangle_cyclic[i + 1]));
        }
        
        edges
    }
    
    /// Extrahiert alle drei Edges eines Dreiecks mit Orientierung
    fn triangle_to_edges_oriented(&self, triangle: Triangle) -> Vec<Edge> {
        vec![
            Edge(triangle[0], triangle[1]),
            Edge(triangle[1], triangle[2]),
            Edge(triangle[2], triangle[0]),
        ]
    }
    
    /// Berechnet point_unorder aus point_order
    fn compute_point_unorder(point_order: &[usize]) -> Vec<usize> {
        let mut point_unorder = vec![0; point_order.len()];
        for (sorted_idx, &original_idx) in point_order.iter().enumerate() {
            if original_idx < point_unorder.len() {
                point_unorder[original_idx] = sorted_idx;
            }
        }
        point_unorder
    }
    
    /// Fügt einen Punkt zur Triangulation hinzu
    fn add_point(&mut self, point_idx: usize) {
        let mut edges_to_remove = HashSet::new();
        let mut edges_to_add = Vec::new();
        
        // Finde alle sichtbaren Boundary-Edges
        let boundary_edges_copy: Vec<Edge> = self.boundary_edges.iter().copied().collect();
        
        for edge in boundary_edges_copy {
            let point = self.points[point_idx];
            let edge_start = self.points[edge.0];
            let edge_end = self.points[edge.1];
            
            if is_visible_from_edge(point, edge_start, edge_end, self.eps) {
                // Erstelle neues Dreieck
                let mut new_triangle = [edge.0, edge.1, point_idx];
                ensure_counter_clockwise(&mut new_triangle, &self.points, self.eps);
                
                self.triangles.push(new_triangle);
                let tri_idx = self.triangles.len() - 1;
                
                // Update Edge-Mappings
                let e0 = make_edge_key(edge.0, edge.1);
                let e1 = make_edge_key(point_idx, edge.0);
                let e2 = make_edge_key(edge.1, point_idx);
                
                for e in [e0, e1, e2] {
                    self.edge_to_triangles
                        .entry(e)
                        .or_insert_with(Vec::new)
                        .push(tri_idx);
                }
                
                // Update Punkt-zu-Dreieck-Mappings
                for &i in &new_triangle {
                    self.point_to_triangles
                        .entry(i)
                        .or_insert_with(HashSet::new)
                        .insert(tri_idx);
                }
                
                // Track Boundary-Edge-Updates
                edges_to_remove.insert(edge);
                edges_to_add.push(Edge(edge.0, point_idx));
                edges_to_add.push(Edge(point_idx, edge.1));
            }
        }
        
        // Update Boundary-Edges
        for edge in &edges_to_remove {
            self.boundary_edges.remove(edge);
        }
        
        for edge in edges_to_add {
            let edge_key = make_edge_key(edge.0, edge.1);
            if self
                .edge_to_triangles
                .get(&edge_key)
                .map(|v| v.len())
                .unwrap_or(0)
                == 1
            {
                self.boundary_edges.insert(edge);
            }
        }
        
        // Enforce Delaunay-Kriterium wenn aktiviert
        if self.delaunay {
            self.flip_edges();
        }
    }
    
    /// Initialisiert die Triangulation
    fn initialize_triangulation(&mut self) -> Result<(), PolyTriError> {
        // Sortiere Punkte nach Entfernung vom Schwerpunkt
        let (sorted_points, point_order, point_unorder) = 
            Self::sort_points_by_distance_from_center(&self.points);
        
        self.points = sorted_points;
        self.point_order = point_order;
        self.point_unorder = point_unorder;
        
        // Entferne kollineare Punkte
        Self::remove_collinear_points(&mut self.points, &mut self.point_order, self.eps)?;
        
        // Aktualisiere point_unorder nach Entfernung kollinearer Punkte
        self.point_unorder = Self::compute_point_unorder(&self.point_order);
        
        // Erstelle erstes Dreieck
        let (triangle, edges) = Self::create_first_triangle(&self.points, self.eps)?;
        
        self.triangles.push(triangle);
        
        // Initialisiere Boundary-Edges
        for edge in edges {
            self.boundary_edges.insert(edge);
        }
        
        // Initialisiere Edge- und Punkt-Mappings
        let tri_idx = 0;
        let e01_key = make_edge_key(triangle[0], triangle[1]);
        let e12_key = make_edge_key(triangle[1], triangle[2]);
        let e20_key = make_edge_key(triangle[2], triangle[0]);
        
        self.edge_to_triangles.insert(e01_key, vec![tri_idx]);
        self.edge_to_triangles.insert(e12_key, vec![tri_idx]);
        self.edge_to_triangles.insert(e20_key, vec![tri_idx]);
        
        for &i in &triangle {
            self.point_to_triangles
                .entry(i)
                .or_insert_with(HashSet::new)
                .insert(tri_idx);
        }
        
        // Füge alle weiteren Punkte hinzu
        for i in 3..self.points.len() {
            self.add_point(i);
        }
        
        // Aktualisiere point_unorder Mapping
        self.point_unorder = Self::compute_point_unorder(&self.point_order);
        
        Ok(())
    }
    
    /// Flippt eine Edge zwischen zwei Dreiecken
    /// 
    /// Gibt ein Set von Edges zurück, die möglicherweise geflippt werden müssen.
    fn flip_edge(
        &mut self,
        edge: EdgeKey,
        enforce_delaunay: bool,
        check_intersection: bool,
    ) -> HashSet<EdgeKey> {
        let mut result_edges = HashSet::new();
        
        // Prüfe, ob Edge geflippt werden kann (muss genau zwei Dreiecken angehören)
        let triangles = match self.edge_to_triangles.get(&edge) {
            Some(tris) if tris.len() >= 2 => tris.clone(),
            _ => return result_edges, // Edge ist Boundary-Edge oder existiert nicht
        };
        
        let tri1_idx = triangles[0];
        let tri2_idx = triangles[1];
        
        if tri1_idx >= self.triangles.len() || tri2_idx >= self.triangles.len() {
            return result_edges;
        }
        
        let tri1 = self.triangles[tri1_idx];
        let tri2 = self.triangles[tri2_idx];
        
        // Finde gegenüberliegende Vertices
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
        
        let opposite1 = match opposite1 {
            Some(v) => v,
            None => return result_edges,
        };
        
        let opposite2 = match opposite2 {
            Some(v) => v,
            None => return result_edges,
        };
        
        // Prüfe Intersektion (wenn check_intersection = true)
        if check_intersection {
            let diagonal = make_edge_key(opposite1, opposite2);
            let edge_start = self.points[edge.0];
            let edge_end = self.points[edge.1];
            let diag_start = self.points[diagonal.0];
            let diag_end = self.points[diagonal.1];
            
            if !edges_intersect(edge_start, edge_end, diag_start, diag_end) {
                return result_edges;
            }
        }
        
        // Prüfe Delaunay-Kriterium (wenn enforce_delaunay = true)
        if enforce_delaunay {
            // Berechne Vektoren
            let p_edge0 = self.points[edge.0];
            let p_edge1 = self.points[edge.1];
            let p_opp1 = self.points[opposite1];
            let p_opp2 = self.points[opposite2];
            
            let da1_x = p_edge0.x - p_opp1.x;
            let da1_y = p_edge0.y - p_opp1.y;
            let db1_x = p_edge1.x - p_opp1.x;
            let db1_y = p_edge1.y - p_opp1.y;
            
            let da2_x = p_edge0.x - p_opp2.x;
            let da2_y = p_edge0.y - p_opp2.y;
            let db2_x = p_edge1.x - p_opp2.x;
            let db2_y = p_edge1.y - p_opp2.y;
            
            // Berechne Winkel
            let cross1 = compute_triangle_area(p_opp1, p_edge0, p_edge1);
            let cross2 = compute_triangle_area(p_opp2, p_edge1, p_edge0);
            let dot1 = da1_x * db1_x + da1_y * db1_y;
            let dot2 = da2_x * db2_x + da2_y * db2_y;
            
            let angle1 = cross1.atan2(dot1).abs();
            let angle2 = cross2.atan2(dot2).abs();
            
            // Delaunay-Kriterium: Flip wenn Summe der gegenüberliegenden Winkel > π
            let pi = std::f64::consts::PI;
            if !(angle1 + angle2 > pi * (1.0 + self.eps)) {
                return result_edges; // Delaunay-Kriterium erfüllt, kein Flip nötig
            }
        }
        
        // Flippe die Dreiecke
        let mut new_tri1 = [opposite1, edge.0, opposite2];
        let mut new_tri2 = [opposite1, opposite2, edge.1];
        
        // Stelle sicher, dass beide Dreiecke gegen den Uhrzeigersinn orientiert sind
        ensure_counter_clockwise(&mut new_tri1, &self.points, self.eps);
        ensure_counter_clockwise(&mut new_tri2, &self.points, self.eps);
        
        self.triangles[tri1_idx] = new_tri1;
        self.triangles[tri2_idx] = new_tri2;
        
        // Update Edge-Mappings
        self.edge_to_triangles.remove(&edge);
        
        let new_edge = make_edge_key(opposite1, opposite2);
        self.edge_to_triangles
            .insert(new_edge, vec![tri1_idx, tri2_idx]);
        
        // Update Punkt-zu-Dreieck-Mappings für neue Edge
        if let Some(tris_set) = self.point_to_triangles.get_mut(&new_edge.0) {
            tris_set.insert(tri1_idx);
            tris_set.insert(tri2_idx);
        } else {
            let mut new_set = HashSet::new();
            new_set.insert(tri1_idx);
            new_set.insert(tri2_idx);
            self.point_to_triangles.insert(new_edge.0, new_set);
        }
        
        if let Some(tris_set) = self.point_to_triangles.get_mut(&new_edge.1) {
            tris_set.insert(tri1_idx);
            tris_set.insert(tri2_idx);
        } else {
            let mut new_set = HashSet::new();
            new_set.insert(tri1_idx);
            new_set.insert(tri2_idx);
            self.point_to_triangles.insert(new_edge.1, new_set);
        }
        
        // Update Edges, die jetzt zu verschiedenen Dreiecken gehören
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
        
        // Update Punkt-zu-Dreieck-Mappings für gemeinsame Vertices
        for &i in &new_tri1 {
            if i == edge.0 || i == edge.1 {
                if let Some(tris_set) = self.point_to_triangles.get_mut(&i) {
                    let mut new_set = HashSet::new();
                    for &tri_idx in tris_set.iter() {
                        if tri_idx == tri2_idx {
                            new_set.insert(tri1_idx);
                        } else {
                            new_set.insert(tri_idx);
                        }
                    }
                    *tris_set = new_set;
                }
            }
        }
        
        for &i in &new_tri2 {
            if i == edge.0 || i == edge.1 {
                if let Some(tris_set) = self.point_to_triangles.get_mut(&i) {
                    let mut new_set = HashSet::new();
                    for &tri_idx in tris_set.iter() {
                        if tri_idx == tri1_idx {
                            new_set.insert(tri2_idx);
                        } else {
                            new_set.insert(tri_idx);
                        }
                    }
                    *tris_set = new_set;
                }
            }
        }
        
        // Edges, die möglicherweise geflippt werden müssen
        result_edges.insert(make_edge_key(opposite1, edge.0));
        result_edges.insert(make_edge_key(opposite2, edge.1));
        
        result_edges
    }
    
    /// Flippt alle Edges, bis das Delaunay-Kriterium erfüllt ist
    pub fn flip_edges(&mut self) {
        let mut edge_set: HashSet<EdgeKey> = self.edge_to_triangles.keys().copied().collect();
        
        while !edge_set.is_empty() {
            let mut new_edge_set = HashSet::new();
            for edge in edge_set {
                let result_edges = self.flip_edge(edge, true, false);
                new_edge_set.extend(result_edges);
            }
            edge_set = new_edge_set;
        }
    }
}

// ============================================================================
// PRIVATE METHODEN: CONSTRAINTS
// ============================================================================

impl PolyTri {
    /// Erstellt eine Liste von Boundary-Edges aus Boundary-Definitionen
    /// 
    /// Wenn create_key=true, werden normalisierte EdgeKeys zurückgegeben.
    /// Wenn create_key=false, werden Edges mit Orientierung zurückgegeben.
    fn create_boundary_list(
        &self,
        border_indices: Option<&[usize]>,
        create_key: bool,
    ) -> Vec<EdgeKey> {
        let boundaries = match &self.boundaries {
            Some(b) => b,
            None => return Vec::new(),
        };
        
        let mut boundary_edges = Vec::new();
        
        for (k, boundary) in boundaries.iter().enumerate() {
            // Prüfe, ob Boundary verwendet werden soll
            if let Some(border) = border_indices {
                if !border.contains(&k) {
                    continue;
                }
            }
            
            // Mappe Boundary-Indizes von original zu sortiert
            let boundary_mapped: Vec<usize> = boundary
                .iter()
                .filter_map(|&idx| {
                    if idx < self.point_unorder.len() {
                        Some(self.point_unorder[idx])
                    } else {
                        None
                    }
                })
                .collect();
            
            // Erstelle Edges aus aufeinanderfolgenden Punkten
            for i in 0..boundary_mapped.len().saturating_sub(1) {
                let j = i + 1;
                if create_key {
                    boundary_edges.push(make_edge_key(boundary_mapped[i], boundary_mapped[j]));
                } else {
                    // Für nicht-Key-Version würde man Edge zurückgeben
                    // Hier verwenden wir EdgeKey für Konsistenz
                    boundary_edges.push(make_edge_key(boundary_mapped[i], boundary_mapped[j]));
                }
            }
        }
        
        boundary_edges
    }
    
    /// Constraint eine Edge, sodass sie in der Triangulation vorhanden ist
    fn constrain_edge(&mut self, edge: EdgeKey) -> Result<(), PolyTriError> {
        // Wenn Edge bereits existiert, nichts zu tun
        if self.edge_to_triangles.contains_key(&edge) {
            return Ok(());
        }
        
        let pt0 = edge.0;
        let pt1 = edge.1;
        
        // Validiere Edge-Endpunkte
        if !self.point_to_triangles.contains_key(&pt0)
            || !self.point_to_triangles.contains_key(&pt1)
        {
            return Err(PolyTriError::InvalidPointIndex(
                if !self.point_to_triangles.contains_key(&pt0) {
                    pt0
                } else {
                    pt1
                },
            ));
        }
        
        // Finde erste schneidende Edge
        let mut intersecting_edge = None;
        
        if let Some(tri_indices) = self.point_to_triangles.get(&pt1) {
            for &tri_idx in tri_indices {
                if tri_idx >= self.triangles.len() {
                    continue;
                }
                let tri_vertices = self.triangles[tri_idx];
                let mut other_vertices = Vec::new();
                for &v in &tri_vertices {
                    if v != pt1 {
                        other_vertices.push(v);
                    }
                }
                
                if other_vertices.len() == 2 {
                    let candidate_edge = make_edge_key(other_vertices[0], other_vertices[1]);
                    let edge_start = self.points[edge.0];
                    let edge_end = self.points[edge.1];
                    let cand_start = self.points[candidate_edge.0];
                    let cand_end = self.points[candidate_edge.1];
                    
                    if edges_intersect(edge_start, edge_end, cand_start, cand_end) {
                        intersecting_edge = Some(candidate_edge);
                        break;
                    }
                }
            }
        }
        
        let intersecting_edge = match intersecting_edge {
            Some(e) => e,
            None => return Ok(()), // Edge bereits constraint oder keine Intersektion gefunden
        };
        
        // Flippe Edges, bis Constraint erfüllt ist
        let mut edges_to_check = self.flip_edge(intersecting_edge, false, true);
        
        loop {
            let mut found_intersection = false;
            let mut next_intersecting_edge = None;
            
            for e in &edges_to_check {
                let edge_start = self.points[edge.0];
                let edge_end = self.points[edge.1];
                let e_start = self.points[e.0];
                let e_end = self.points[e.1];
                
                if edges_intersect(edge_start, edge_end, e_start, e_end) {
                    next_intersecting_edge = Some(*e);
                    found_intersection = true;
                    break;
                }
            }
            
            if !found_intersection {
                break;
            }
            
            let intersecting_edge = next_intersecting_edge.unwrap();
            edges_to_check = self.flip_edge(intersecting_edge, false, true);
            
            if edges_to_check.is_empty() {
                if self.edge_to_triangles.contains_key(&edge) {
                    break;
                } else {
                    // Rekursiv constraint Sub-Edges
                    self.constrain_edge(make_edge_key(intersecting_edge.0, pt0))?;
                    self.constrain_edge(make_edge_key(pt0, pt1))?;
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    /// Constraint alle spezifizierten Boundaries
    pub fn constrain_boundaries(&mut self) -> Result<(), PolyTriError> {
        let boundary_edges = self.create_boundary_list(None, true);
        for edge in boundary_edges {
            self.constrain_edge(edge)?;
        }
        Ok(())
    }
}

// ============================================================================
// PRIVATE METHODEN: HOLE REMOVAL
// ============================================================================

impl PolyTri {
    /// Baut Edge-zu-Dreieck- und Punkt-zu-Dreieck-Mappings neu auf
    fn update_mappings(&mut self) {
        // Lösche alte Mappings
        self.edge_to_triangles.clear();
        self.point_to_triangles.clear();
        
        // Baue Mappings neu auf
        for (tri_idx, triangle) in self.triangles.iter().enumerate() {
            // Edge-zu-Dreieck-Mapping
            let edges = self.triangle_to_edges_keys(*triangle);
            for edge in edges {
                self.edge_to_triangles
                    .entry(edge)
                    .or_insert_with(Vec::new)
                    .push(tri_idx);
            }
            
            // Punkt-zu-Dreieck-Mapping
            for &point_idx in triangle {
                self.point_to_triangles
                    .entry(point_idx)
                    .or_insert_with(HashSet::new)
                    .insert(tri_idx);
            }
        }
        
        // Update Boundary-Edges
        // Boundary-Edges sind Edges, die genau einem Dreieck angehören
        let mut new_boundary_edges = HashSet::new();
        for (edge_key, triangles) in &self.edge_to_triangles {
            if triangles.len() == 1 {
                // Konvertiere EdgeKey zu Edge mit Orientierung
                // Für Boundary-Edges verwenden wir die Orientierung aus dem Dreieck
                let tri_idx = triangles[0];
                if tri_idx < self.triangles.len() {
                    let tri_edges = self.triangle_to_edges_oriented(self.triangles[tri_idx]);
                    for edge in tri_edges {
                        let edge_key_from_edge = make_edge_key(edge.0, edge.1);
                        if edge_key_from_edge == *edge_key {
                            new_boundary_edges.insert(edge);
                            break;
                        }
                    }
                }
            }
        }
        
        self.boundary_edges = new_boundary_edges;
    }
    
    /// Entfernt Dreiecke mit null oder nahezu null Fläche
    fn remove_empty_triangles_impl(&mut self) {
        let mut triangles_to_remove = Vec::new();
        
        for (i, triangle) in self.triangles.iter().enumerate() {
            let mut tri = *triangle;
            ensure_counter_clockwise(&mut tri, &self.points, self.eps);
            let area = compute_triangle_area(
                self.points[tri[0]],
                self.points[tri[1]],
                self.points[tri[2]],
            );
            
            if area.abs() < self.eps {
                triangles_to_remove.push(i);
            }
        }
        
        // Entferne in umgekehrter Reihenfolge (damit Indizes stabil bleiben)
        if !triangles_to_remove.is_empty() {
            triangles_to_remove.sort_by(|a, b| b.cmp(a)); // Reverse sort
            for &i in &triangles_to_remove {
                self.triangles.remove(i);
            }
        }
    }
    
    /// Entfernt Dreiecke innerhalb von Löchern, die durch Boundaries definiert sind.
    /// 
    /// Hinweis: Der `border`-Parameter (falls gesetzt) bestimmt, welche Boundaries verwendet werden.
    /// Für typische Anwendungen wird `border` nicht benötigt und kann None bleiben.
    fn remove_holes_impl(&mut self) -> Result<(), PolyTriError> {
        if self.boundaries.is_none() {
            return Ok(());
        }
        
        // WICHTIG: Wenn border leer ist ([]), verwende None für alle Boundaries
        let border_indices = if self.border.is_empty() {
            None
        } else {
            Some(self.border.as_slice())
        };
        
        let boundary_keys = self.create_boundary_list(border_indices, true);
        
        // Erstelle auch Boundary-Tuples für Vergleich
        let mut boundary_tuples = Vec::new();
        if let Some(boundaries) = &self.boundaries {
            for (k, boundary) in boundaries.iter().enumerate() {
                if let Some(border) = border_indices {
                    if !border.contains(&k) {
                        continue;
                    }
                }
                
                let boundary_mapped: Vec<usize> = boundary
                    .iter()
                    .filter_map(|&idx| {
                        if idx < self.point_unorder.len() {
                            Some(self.point_unorder[idx])
                        } else {
                            None
                        }
                    })
                    .collect();
                
                for i in 0..boundary_mapped.len().saturating_sub(1) {
                    let j = i + 1;
                    boundary_tuples.push(Edge(boundary_mapped[i], boundary_mapped[j]));
                }
            }
        }
        
        let mut edges_to_remove = HashSet::new();
        
        // Finde Start-Dreiecke (die Boundary-Edges enthalten)
        for b_key in &boundary_keys {
            if let Some(triangles) = self.edge_to_triangles.get(b_key) {
                for &tri_idx in triangles {
                    if tri_idx >= self.triangles.len() {
                        continue;
                    }
                    let tri_edges = self.triangle_to_edges_oriented(self.triangles[tri_idx]);
                    
                    // Prüfe, ob Dreieck diese Boundary-Edge enthält (mit Orientierung)
                    // boundary_tuples enthält alle Edges aller Boundaries
                    for b_tuple in &boundary_tuples {
                        let b_key_from_tuple = make_edge_key(b_tuple.0, b_tuple.1);
                        if b_key_from_tuple == *b_key && tri_edges.contains(b_tuple) {
                            // Markiere ALLE Edges dieses Dreiecks zur Entfernung
                            let tri_edge_keys =
                                self.triangle_to_edges_keys(self.triangles[tri_idx]);
                            for edge in tri_edge_keys {
                                edges_to_remove.insert(edge);
                            }
                            break; // Nur einmal pro Boundary-Edge
                        }
                    }
                }
            }
        }
        
        // Entferne Boundary-Edges selbst aus der Entfernung-Liste
        for b_key in &boundary_keys {
            edges_to_remove.remove(b_key);
        }
        
        // Finde alle initialen Dreiecke zum Entfernen
        let mut triangles_to_remove = HashSet::new();
        for edge in &edges_to_remove {
            if let Some(tris) = self.edge_to_triangles.get(edge) {
                triangles_to_remove.extend(tris.iter().copied());
            }
        }
        
        // Expandiere iterativ (Flood-Fill)
        let mut prev_count = triangles_to_remove.len();
        loop {
            // Erweitere edges_to_remove um alle Edges der markierten Dreiecke
            for &tri_idx in &triangles_to_remove {
                if tri_idx < self.triangles.len() {
                    let tri_edge_keys = self.triangle_to_edges_keys(self.triangles[tri_idx]);
                    for edge in tri_edge_keys {
                        edges_to_remove.insert(edge);
                    }
                }
            }
            
            // Schütze Boundary-Edges erneut
            for b_key in &boundary_keys {
                edges_to_remove.remove(b_key);
            }
            
            // Finde alle neuen Dreiecke, die diese Edges enthalten
            for edge in &edges_to_remove {
                if let Some(tris) = self.edge_to_triangles.get(edge) {
                    triangles_to_remove.extend(tris.iter().copied());
                }
            }
            
            // Prüfe, ob noch neue Dreiecke gefunden wurden
            if triangles_to_remove.len() == prev_count {
                break; // Keine neuen Dreiecke mehr
            }
            
            prev_count = triangles_to_remove.len();
        }
        
        // Entferne Dreiecke in umgekehrter Reihenfolge
        if !triangles_to_remove.is_empty() {
            let mut sorted_indices: Vec<usize> = triangles_to_remove.into_iter().collect();
            sorted_indices.sort_by(|a, b| b.cmp(a)); // Reverse sort
            
            for i in sorted_indices {
                if i < self.triangles.len() {
                    self.triangles.remove(i);
                }
            }
        }
        
        Ok(())
    }
}

// ============================================================================
// ÖFFENTLICHE API
// ============================================================================

impl PolyTri {
    /// Erstellt eine neue PolyTri-Instanz
    /// 
    /// # Arguments
    /// 
    /// * `points` - Vektor von 2D-Punkten
    /// * `boundaries` - Optionale Liste von Boundaries (jede Boundary ist eine Liste von Punkt-Indizes)
    /// * `delaunay` - Wenn `true`, wird das Delaunay-Kriterium erzwungen
    /// * `holes` - Wenn `true`, werden Löcher entfernt
    /// * `border` - Optionale Liste von Boundary-Indizes. Für typische Anwendungen
    ///   wird dieser Parameter nicht benötigt und kann None bleiben. Wird für spezielle
    ///   Fälle verwendet, wo bestimmte Boundaries bei der Hole-Removal anders behandelt
    ///   werden sollen. Wenn nicht angegeben (None oder leer), werden alle Boundaries verwendet.
    pub fn new(
        points: Vec<Point>,
        boundaries: Option<Vec<Boundary>>,
        delaunay: bool,
        holes: bool,
        border: Option<Vec<usize>>,
    ) -> Result<Self, PolyTriError> {
        // Validierung
        if points.len() < 3 {
            return Err(PolyTriError::NotEnoughPoints(points.len()));
        }
        
        let border = border.unwrap_or_default();
        
        // Validiere Boundaries
        if let Some(ref boundaries) = boundaries {
            for (i, boundary) in boundaries.iter().enumerate() {
                if boundary.len() < 2 {
                    return Err(PolyTriError::InvalidBoundary(
                        i,
                        "boundary must have at least 2 points".to_string(),
                    ));
                }
                
                // Prüfe, ob Boundary-Indizes gültig sind
                for &idx in boundary {
                    if idx >= points.len() {
                        return Err(PolyTriError::InvalidBoundary(
                            i,
                            format!("boundary contains invalid point index {} (max index {} >= {} points)", 
                                   idx, idx, points.len()),
                        ));
                    }
                }
            }
        }

        // Speichere ursprüngliche Punkte (vor Sortierung)
        let original_points = points.clone();
        
        // Erstelle PolyTri-Instanz
        let mut polytri = PolyTri {
            original_points: original_points.clone(),
            points,
            boundaries,
            delaunay,
            holes,
            border,
            triangles: Vec::new(),
            edge_to_triangles: HashMap::new(),
            point_to_triangles: HashMap::new(),
            boundary_edges: HashSet::new(),
            point_order: Vec::new(),
            point_unorder: Vec::new(),
            eps: Self::EPS,
        };
        
        // Initialisiere Triangulation
        polytri.initialize_triangulation()?;
        
        // Wende Constraints an, wenn spezifiziert
        if polytri.boundaries.is_some() {
            polytri.constrain_boundaries()?;
            
            if polytri.holes {
                polytri.remove_empty_triangles_impl();
                polytri.update_mappings();
                polytri.remove_holes_impl()?;
            }
        }
        
        Ok(polytri)
    }
    
    /// Gibt die Punkte zurück in ursprünglicher Reihenfolge (nicht sortiert)
    pub fn points(&self) -> &[Point] {
        &self.original_points
    }
    
    /// Gibt Dreiecke als Arrays von originalen Punkt-Indizes zurück
    pub fn get_triangles(&self) -> Vec<[usize; 3]> {
        self.triangles
            .iter()
            .map(|tri| {
                [
                    self.point_order[tri[0]],
                    self.point_order[tri[1]],
                    self.point_order[tri[2]],
                ]
            })
            .collect()
    }
    
    /// Gibt Boundary-Edges zurück (als originale Indizes)
    pub fn boundary_edges(&self) -> Vec<(usize, usize)> {
        self.boundary_edges
            .iter()
            .map(|edge| (self.point_order[edge.0], self.point_order[edge.1]))
            .collect()
    }
    
    /// Gibt zurück, ob Delaunay-Kriterium aktiviert ist
    pub fn delaunay(&self) -> bool {
        self.delaunay
    }
    
    /// Gibt die Boundaries zurück
    pub fn boundaries(&self) -> Option<&Vec<Boundary>> {
        self.boundaries.as_ref()
    }
    
    /// Gibt die Border-Indizes zurück
    pub fn border(&self) -> &[usize] {
        &self.border
    }
    
    /// Entfernt leere Dreiecke (öffentliche Methode)
    pub fn remove_empty_triangles(&mut self) {
        self.remove_empty_triangles_impl();
    }
    
    /// Entfernt Holes (öffentliche Methode)
    pub fn remove_holes(&mut self) -> Result<(), PolyTriError> {
        self.remove_holes_impl()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compute_triangle_area() {
        let p0 = Point { x: 0.0, y: 0.0 };
        let p1 = Point { x: 1.0, y: 0.0 };
        let p2 = Point { x: 0.0, y: 1.0 };
        
        // compute_triangle_area gibt die Determinante zurück (2 * Fläche)
        let area = compute_triangle_area(p0, p1, p2);
        assert!(
            area > 0.0,
            "Triangle should have positive area (counter-clockwise)"
        );
        assert!(
            (area - 1.0).abs() < 1e-10,
            "Determinant should be 1.0 (area = 0.5)"
        );
        
        // Test im Uhrzeigersinn
        let area_ccw = compute_triangle_area(p0, p2, p1);
        assert!(
            area_ccw < 0.0,
            "Clockwise triangle should have negative area"
        );
        assert!(
            (area_ccw + 1.0).abs() < 1e-10,
            "Determinant should be -1.0 (area = -0.5)"
        );
    }
    
    #[test]
    fn test_is_visible_from_edge() {
        let point = Point { x: 0.5, y: 0.5 };
        let edge_start = Point { x: 0.0, y: 0.0 };
        let edge_end = Point { x: 1.0, y: 0.0 };
        
        // Punkt liegt oberhalb der Edge (von start nach end gesehen links)
        // Punkt oberhalb = links = positive Fläche, also nicht sichtbar (rechts)
        assert!(
            !is_visible_from_edge(point, edge_start, edge_end, 1e-14),
            "Point above edge should be on left, not visible from right"
        );
        
        let point_below = Point { x: 0.5, y: -0.5 };
        // Punkt liegt unterhalb der Edge (von start nach end gesehen rechts)
        // Punkt unterhalb = rechts = negative Fläche, also sichtbar
        assert!(
            is_visible_from_edge(point_below, edge_start, edge_end, 1e-14),
            "Point below edge should be on right, visible"
        );
    }
    
    #[test]
    fn test_ensure_counter_clockwise() {
        let mut triangle = [0, 1, 2];
        let points = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
        ];
        
        // Test mit bereits gegen den Uhrzeigersinn orientiertem Dreieck
        ensure_counter_clockwise(&mut triangle, &points, 1e-14);
        assert_eq!(triangle, [0, 1, 2]);
        
        // Test mit im Uhrzeigersinn orientiertem Dreieck
        let mut triangle_cw = [0, 2, 1];
        ensure_counter_clockwise(&mut triangle_cw, &points, 1e-14);
        assert_eq!(
            triangle_cw,
            [0, 1, 2],
            "Should swap vertices to counter-clockwise"
        );
    }
    
    #[test]
    fn test_edges_intersect() {
        // Zwei sich schneidende Edges
        let edge1_start = Point { x: 0.0, y: 0.0 };
        let edge1_end = Point { x: 1.0, y: 1.0 };
        let edge2_start = Point { x: 0.0, y: 1.0 };
        let edge2_end = Point { x: 1.0, y: 0.0 };
        
        assert!(edges_intersect(
            edge1_start,
            edge1_end,
            edge2_start,
            edge2_end
        ));
        
        // Zwei nicht-schneidende Edges
        let edge3_start = Point { x: 0.0, y: 0.0 };
        let edge3_end = Point { x: 1.0, y: 0.0 };
        let edge4_start = Point { x: 0.0, y: 1.0 };
        let edge4_end = Point { x: 1.0, y: 1.0 };
        
        assert!(!edges_intersect(
            edge3_start,
            edge3_end,
            edge4_start,
            edge4_end
        ));
        
        // Edges mit gemeinsamem Vertex schneiden sich nicht
        assert!(!edges_intersect(
            edge1_start,
            edge1_end,
            edge1_start,
            edge2_end
        ));
    }
    
    #[test]
    fn test_make_edge_key() {
        let key1 = make_edge_key(5, 3);
        let key2 = make_edge_key(3, 5);
        
        assert_eq!(key1, key2, "Edge keys should be normalized");
        assert_eq!(key1.0, 3);
        assert_eq!(key1.1, 5);
    }
    
    #[test]
    fn test_edge_key_new() {
        let key1 = EdgeKey::new(5, 3);
        let key2 = EdgeKey::new(3, 5);
        
        assert_eq!(key1, key2);
        assert_eq!(key1.0, 3);
        assert_eq!(key1.1, 5);
    }
    
    #[test]
    fn test_basic_triangulation() {
        let points = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.5, y: 0.5 },
        ];
        
        let tri = PolyTri::new(points.clone(), None, true, false, None).unwrap();
        let triangles = tri.get_triangles();
        
        assert_eq!(triangles.len(), 1, "Should have exactly one triangle");
        // Die Indizes können in beliebiger Reihenfolge sein, müssen aber alle drei Punkte enthalten
        let tri = triangles[0];
        let mut sorted_tri = [tri[0], tri[1], tri[2]];
        sorted_tri.sort();
        assert_eq!(
            sorted_tri,
            [0, 1, 2],
            "Triangle should contain all three points"
        );
    }
    
    #[test]
    fn test_too_few_points() {
        let points = vec![Point { x: 0.0, y: 0.0 }, Point { x: 1.0, y: 0.0 }];
        
        let result = PolyTri::new(points, None, true, false, None);
        assert!(result.is_err());
        
        match result {
            Err(PolyTriError::NotEnoughPoints(n)) => assert_eq!(n, 2),
            _ => panic!("Wrong error type"),
        }
    }
    
    #[test]
    fn test_delaunay_property() {
        let points = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.5, y: 0.5 },
        ];
        
        let tri_delaunay = PolyTri::new(points.clone(), None, true, false, None).unwrap();
        assert_eq!(tri_delaunay.delaunay(), true);
        
        let tri_no_delaunay = PolyTri::new(points, None, false, false, None).unwrap();
        assert_eq!(tri_no_delaunay.delaunay(), false);
    }
    
    #[test]
    fn test_points_property() {
        let points = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.5, y: 0.5 },
        ];
        
        let tri = PolyTri::new(points.clone(), None, true, false, None).unwrap();
        let returned_points = tri.points();
        
        assert_eq!(returned_points.len(), points.len());
    }
}
