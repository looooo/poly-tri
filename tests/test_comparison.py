# -*- coding: utf-8 -*-
"""
Vergleichstests für Rust- und Python-Version von PolyTri.

Diese Tests führen beide Implementierungen aus und vergleichen die Ergebnisse,
um Fehler in der Rust-Version zu identifizieren.
"""

import unittest
import sys
import os
import numpy as np
import copy

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polytri._python import PolyTri as PythonPolyTri
from polytri import is_rust_available

# Try to import Rust version
try:
    from polytri._rust import PolyTri as RustPolyTri
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustPolyTri = None


def normalize_triangles(triangles):
    """
    Normalisiere Dreiecke für Vergleich.
    Konvertiert zu sortierten Listen von Tupeln.
    """
    normalized = []
    for tri in triangles:
        # Konvertiere zu Liste falls numpy array
        if hasattr(tri, 'tolist'):
            tri = tri.tolist()
        elif not isinstance(tri, list):
            tri = list(tri)
        
        # Sortiere Indizes innerhalb jedes Dreiecks für konsistenten Vergleich
        tri_sorted = tuple(sorted(tri))
        normalized.append(tri_sorted)
    
    # Sortiere die Liste der Dreiecke
    normalized.sort()
    return normalized


def compare_triangulations(python_tri, rust_tri, test_name="", tolerance=1e-10):
    """
    Vergleiche zwei Triangulationen und gebe detaillierte Informationen aus.
    
    Returns:
        (bool, dict): (sind_gleich, fehler_details)
    """
    errors = []
    
    # Vergleiche Anzahl der Dreiecke
    python_triangles = python_tri.get_triangles()
    rust_triangles = rust_tri.get_triangles()
    
    python_count = len(python_triangles)
    rust_count = len(rust_triangles)
    
    if python_count != rust_count:
        errors.append(f"Anzahl der Dreiecke unterschiedlich: Python={python_count}, Rust={rust_count}")
    
    # Normalisiere Dreiecke für Vergleich
    python_norm = normalize_triangles(python_triangles)
    rust_norm = normalize_triangles(rust_triangles)
    
    # Finde fehlende Dreiecke
    python_set = set(python_norm)
    rust_set = set(rust_norm)
    
    missing_in_rust = python_set - rust_set
    extra_in_rust = rust_set - python_set
    
    if missing_in_rust:
        errors.append(f"Fehlende Dreiecke in Rust: {missing_in_rust}")
    
    if extra_in_rust:
        errors.append(f"Zusätzliche Dreiecke in Rust: {extra_in_rust}")
    
    # Vergleiche Punkte
    python_points = python_tri.points
    rust_points = rust_tri.points
    
    if hasattr(python_points, 'shape'):
        python_points = python_points.tolist()
    if hasattr(rust_points, 'shape'):
        rust_points = rust_points.tolist()
    
    if len(python_points) != len(rust_points):
        errors.append(f"Anzahl der Punkte unterschiedlich: Python={len(python_points)}, Rust={len(rust_points)}")
    else:
        for i, (p_py, p_rust) in enumerate(zip(python_points, rust_points)):
            if isinstance(p_py, np.ndarray):
                p_py = p_py.tolist()
            if isinstance(p_rust, np.ndarray):
                p_rust = p_rust.tolist()
            
            diff = np.array(p_py) - np.array(p_rust)
            max_diff = np.max(np.abs(diff))
            if max_diff > tolerance:
                errors.append(f"Punkt {i} unterschiedlich: Python={p_py}, Rust={p_rust}, Diff={max_diff}")
    
    # Vergleiche Boundary Edges
    # Python gibt interne Indizes zurück, Rust gibt gemappte Indizes zurück
    # Mappe Python Boundary Edges zu ursprünglichen Indizes für Vergleich
    def normalize_edge(edge):
        """Normalisiere Edge für Vergleich."""
        if edge[0] < edge[1]:
            return edge
        return (edge[1], edge[0])
    
    # Python boundary edges sind interne Indizes - mappe sie
    if hasattr(python_tri, '_point_order'):
        python_boundary_mapped = set()
        for e in python_tri.boundary_edges:
            mapped = (python_tri._point_order[e[0]], python_tri._point_order[e[1]])
            python_boundary_mapped.add(normalize_edge(mapped))
    else:
        python_boundary_mapped = set(normalize_edge(e) for e in python_tri.boundary_edges)
    
    rust_boundary = set(normalize_edge(e) for e in rust_tri.boundary_edges)
    
    if python_boundary_mapped != rust_boundary:
        missing_boundary = python_boundary_mapped - rust_boundary
        extra_boundary = rust_boundary - python_boundary_mapped
        if missing_boundary:
            errors.append(f"Fehlende Boundary Edges in Rust: {missing_boundary}")
        if extra_boundary:
            errors.append(f"Zusätzliche Boundary Edges in Rust: {extra_boundary}")
    
    return len(errors) == 0, {
        'python_triangles': python_triangles,
        'rust_triangles': rust_triangles,
        'python_normalized': python_norm,
        'rust_normalized': rust_norm,
        'errors': errors,
        'python_count': python_count,
        'rust_count': rust_count,
    }


class ComparisonTests(unittest.TestCase):
    """Vergleichstests für Rust- und Python-Version."""
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_easy_1(self):
        """Einfacher Test mit 5 Punkten."""
        pts = np.array([
            [0.,  0. ],
            [0.2, 0.1],
            [0.5, 0.1],
            [0.8, 0.1],
            [1.,  0. ]
        ])
        
        python_tri = PythonPolyTri(pts, delaunay=False)
        rust_tri = RustPolyTri(pts, delaunay=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "test_easy_1")
        
        if not are_equal:
            print(f"\n=== Fehler in test_easy_1 ===")
            print(f"Python Dreiecke ({details['python_count']}): {details['python_normalized']}")
            print(f"Rust Dreiecke ({details['rust_count']}): {details['rust_normalized']}")
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_easy_2(self):
        """Test mit constrained edge."""
        pts = np.array([[-1, 0], [1, 0], [0., 0.5], [0., -0.5]])
        boundaries = [[2, 3]]
        
        python_tri = PythonPolyTri(pts, boundaries, holes=False, delaunay=False)
        rust_tri = RustPolyTri(pts, boundaries, holes=False, delaunay=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "test_easy_2")
        
        if not are_equal:
            print(f"\n=== Fehler in test_easy_2 ===")
            print(f"Python Dreiecke ({details['python_count']}): {details['python_normalized']}")
            print(f"Rust Dreiecke ({details['rust_count']}): {details['rust_normalized']}")
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_easy_3(self):
        """Test mit 5 Punkten ohne Constraints."""
        pts = np.array([[-1., 0.], [1., 0.], [0., 0.5], [0., -0.5], [0., 1.]])
        
        python_tri = PythonPolyTri(pts, holes=False, delaunay=False)
        rust_tri = RustPolyTri(pts, holes=False, delaunay=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "test_easy_3")
        
        if not are_equal:
            print(f"\n=== Fehler in test_easy_3 ===")
            print(f"Python Dreiecke ({details['python_count']}): {details['python_normalized']}")
            print(f"Rust Dreiecke ({details['rust_count']}): {details['rust_normalized']}")
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_easy_4(self):
        """Test mit Delaunay."""
        pts = np.array([
            [0., 0.],
            [0.2, 0.5],
            [0.4, 0.7],
            [0.6, 0.7],
            [0.8, 0.5],
            [1.0, 0.]
        ])
        
        python_tri = PythonPolyTri(pts, holes=False, delaunay=True)
        rust_tri = RustPolyTri(pts, holes=False, delaunay=True)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "test_easy_4")
        
        if not are_equal:
            print(f"\n=== Fehler in test_easy_4 ===")
            print(f"Python Dreiecke ({details['python_count']}): {details['python_normalized']}")
            print(f"Rust Dreiecke ({details['rust_count']}): {details['rust_normalized']}")
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_constraint_edge(self):
        """Test mit constrained boundary."""
        n = 20
        x = np.linspace(0, 1, n)
        y = np.array([0] * n)
        pts = np.array([x, y]).T
        pts_upper = pts[1:-1].copy()
        pts_upper[:, 1] += 0.1
        pts[1:-1, 1] -= 0.1
        boundaries = [[0, n-1]]
        additional_pts = np.array([[-1., 0], [2, 0.]])
        pts = np.array(list(pts) + list(pts_upper) + list(additional_pts))
        
        python_tri = PythonPolyTri(pts, boundaries, holes=False, delaunay=False)
        rust_tri = RustPolyTri(pts, boundaries, holes=False, delaunay=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "test_constraint_edge")
        
        if not are_equal:
            print(f"\n=== Fehler in test_constraint_edge ===")
            print(f"Python Dreiecke ({details['python_count']}):")
            for i, tri in enumerate(details['python_normalized'][:10]):  # Erste 10
                print(f"  {i}: {tri}")
            if len(details['python_normalized']) > 10:
                print(f"  ... ({len(details['python_normalized']) - 10} weitere)")
            print(f"Rust Dreiecke ({details['rust_count']}):")
            for i, tri in enumerate(details['rust_normalized'][:10]):  # Erste 10
                print(f"  {i}: {tri}")
            if len(details['rust_normalized']) > 10:
                print(f"  ... ({len(details['rust_normalized']) - 10} weitere)")
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_constraint_edge_2(self):
        """Test mit geschlossener Boundary."""
        n = 15
        x = np.linspace(0, np.pi, n)
        y = abs(np.sin(x)) - 1.1
        pts = list(np.array([x, y]).T)
        pts.reverse()
        pts += list(np.array([[0., 0.], [np.pi, 0.]]))
        pts = np.array(pts)
        boundaries = [list(range(len(pts))) + [0]]
        
        python_tri = PythonPolyTri(pts, boundaries=boundaries, holes=True, delaunay=False)
        rust_tri = RustPolyTri(pts, boundaries=boundaries, holes=True, delaunay=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "test_constraint_edge_2")
        
        if not are_equal:
            print(f"\n=== Fehler in test_constraint_edge_2 ===")
            print(f"Python Dreiecke ({details['python_count']}): {details['python_normalized'][:5]}...")
            print(f"Rust Dreiecke ({details['rust_count']}): {details['rust_normalized'][:5]}...")
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_ellipse(self):
        """Test mit Ellipse und Hole."""
        phi = np.linspace(0, 2 * np.pi, 20)[:-1]
        outer_pts = np.array([np.cos(phi), np.sin(phi)]).T
        inner_pts = copy.copy(outer_pts)
        outer_pts *= np.array([1., 1.])
        inner_pts *= 0.5
        pts = np.array(list(inner_pts) + list(outer_pts))
        boundaries = [list(range(len(inner_pts))) + [0]]
        
        python_tri = PythonPolyTri(pts, boundaries, delaunay=True, holes=True)
        rust_tri = RustPolyTri(pts, boundaries, delaunay=True, holes=True)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "test_ellipse")
        
        if not are_equal:
            print(f"\n=== Fehler in test_ellipse ===")
            print(f"Python Dreiecke ({details['python_count']}): {details['python_normalized'][:5]}...")
            print(f"Rust Dreiecke ({details['rust_count']}): {details['rust_normalized'][:5]}...")
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_profile2(self):
        """Test mit komplexem Profil und mehreren Holes."""
        pts = np.array([
            [1., 0.], [0.93079585, 0.01364559], [0.8615917, 0.02693948],
            [0.79238754, 0.03983196], [0.72318339, 0.05224763], [0.65397924, 0.06411011],
            [0.58477509, 0.07533044], [0.51557093, 0.08571854], [0.44636678, 0.09508809],
            [0.37716263, 0.10309254], [0.30795848, 0.10914777], [0.23875433, 0.11209604],
            [0.16961802, 0.10886508], [0.10855553, 0.09829529], [0.06106249, 0.08212217],
            [0.02713888, 0.06031021], [0.00678472, 0.03289393], [0., 0.],
            [0.00678472, -0.01869534], [0.02713888, -0.03406389], [0.06106249, -0.04609344],
            [0.10855553, -0.05478371], [0.16961802, -0.06033743], [0.23875433, -0.06399067],
            [0.30795848, -0.06603619], [0.37716263, -0.06691995], [0.44636678, -0.06690913],
            [0.51557093, -0.06617305], [0.58477509, -0.06412989], [0.65397924, -0.0596694],
            [0.72318339, -0.05275214], [0.79238754, -0.04337563], [0.8615917, -0.03145862],
            [0.93079585, -0.0170007],
            [0.22424641, 0.02420158], [0.21770592, 0.04433111], [0.20058272, 0.05677185],
            [0.17941728, 0.05677185], [0.16229408, 0.04433111], [0.15575359, 0.02420158],
            [0.16229408, 0.00407205], [0.17941728, -0.00836869], [0.20058272, -0.00836869],
            [0.21770592, 0.00407205],
            [0.43347349, 0.01676735], [0.42708062, 0.03644257], [0.41034388, 0.04860253],
            [0.38965612, 0.04860253], [0.37291938, 0.03644257], [0.36652651, 0.01676735],
            [0.37291938, -0.00290788], [0.38965612, -0.01506784], [0.41034388, -0.01506784],
            [0.42708062, -0.00290788],
            [0.67016282, 0.0026589], [0.66535715, 0.01744923], [0.65277574, 0.02659016],
            [0.63722426, 0.02659016], [0.62464285, 0.01744923], [0.61983718, 0.0026589],
            [0.62464285, -0.01213144], [0.63722426, -0.02127236], [0.65277574, -0.02127236],
            [0.66535715, -0.01213144]
        ])
        boundaries = [
            [0, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 
             15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 34],
            [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 44],
            [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 54]
        ]
        
        try:
            python_tri = PythonPolyTri(pts, boundaries, delaunay=False, holes=True)
            rust_tri = RustPolyTri(pts, boundaries, delaunay=False, holes=True)
            
            are_equal, details = compare_triangulations(python_tri, rust_tri, "test_profile2")
            
            if not are_equal:
                print(f"\n=== Fehler in test_profile2 ===")
                print(f"Python Dreiecke ({details['python_count']}): {details['python_normalized'][:5]}...")
                print(f"Rust Dreiecke ({details['rust_count']}): {details['rust_normalized'][:5]}...")
                for error in details['errors']:
                    print(f"  - {error}")
            
            self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
        except Exception as e:
            self.fail(f"Fehler beim Ausführen von test_profile2: {e}")
    
    @unittest.skipIf(not RUST_AVAILABLE, "Rust-Version nicht verfügbar")
    def test_profile3(self):
        """Test mit komplexem Profil und Delaunay."""
        pts = np.array([
            [1., 0.], [0.88235294, 0.02298626], [0.76470588, 0.0448548],
            [0.64705882, 0.06526509], [0.52941176, 0.08372709], [0.41176471, 0.09930623],
            [0.29411765, 0.11005194], [0.17647059, 0.10961876], [0.07843137, 0.08925774],
            [0.01960784, 0.05267707], [0., 0.], [0.01960784, -0.02975812],
            [0.07843137, -0.04997027], [0.17647059, -0.06078761], [0.29411765, -0.06573109],
            [0.41176471, -0.06701299], [0.52941176, -0.06593259], [0.64705882, -0.06023152],
            [0.76470588, -0.04741638], [0.88235294, -0.02739101],
            [0.22420494, 0.02415623], [0.21767237, 0.04426139], [0.20056991, 0.05668706],
            [0.17943009, 0.05668706], [0.16232763, 0.04426139], [0.15579506, 0.02415623],
            [0.16232763, 0.00405107], [0.17943009, -0.0083746], [0.20056991, -0.0083746],
            [0.21767237, 0.00405107],
            [0.43345312, 0.016748], [0.42706414, 0.03641125], [0.41033758, 0.04856381],
            [0.38966242, 0.04856381], [0.37293586, 0.03641125], [0.36654688, 0.016748],
            [0.37293586, -0.00291525], [0.38966242, -0.01506781], [0.41033758, -0.01506781],
            [0.42706414, -0.00291525],
            [0.67018389, 0.00262844], [0.6653742, 0.01743116], [0.65278225, 0.02657975],
            [0.63721775, 0.02657975], [0.6246258, 0.01743116], [0.61981611, 0.00262844],
            [0.6246258, -0.01217428], [0.63721775, -0.02132286], [0.65278225, -0.02132286],
            [0.6653742, -0.01217428]
        ])
        boundaries = [
            [0, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 30],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 40]
        ]
        
        try:
            python_tri = PythonPolyTri(pts, boundaries, delaunay=True, holes=True)
            rust_tri = RustPolyTri(pts, boundaries, delaunay=True, holes=True)
            
            are_equal, details = compare_triangulations(python_tri, rust_tri, "test_profile3")
            
            if not are_equal:
                print(f"\n=== Fehler in test_profile3 ===")
                print(f"Python Dreiecke ({details['python_count']}): {details['python_normalized'][:5]}...")
                print(f"Rust Dreiecke ({details['rust_count']}): {details['rust_normalized'][:5]}...")
                for error in details['errors']:
                    print(f"  - {error}")
            
            self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
        except Exception as e:
            self.fail(f"Fehler beim Ausführen von test_profile3: {e}")


if __name__ == '__main__':
    if not RUST_AVAILABLE:
        print("WARNUNG: Rust-Version nicht verfügbar. Vergleichstests werden übersprungen.")
        print("Führen Sie 'pixi run maturin develop --release' aus, um die Rust-Version zu installieren.")
    
    unittest.main(verbosity=2)

