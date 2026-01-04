#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vergleichstest für einfaches Beispiel: Kreis im Kreis
Testet alle Features: Delaunay, Hole Removal, Constraints
"""

import unittest
import sys
import os
import numpy as np
import copy

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polytri import PolyTri, get_implementation, is_rust_available, is_python_available
from polytri._python import PolyTri as PythonPolyTri
from polytri._rust import PolyTri as RustPolyTri


def normalize_triangles(triangles):
    """Normalisiere Dreiecke für Vergleich (sortiere Indizes innerhalb jedes Dreiecks)."""
    return [tuple(sorted(tri)) for tri in triangles]


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
    
    # Vergleiche Boundary Edges (normalisiert)
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
        'errors': errors,
        'python_count': python_count,
        'rust_count': rust_count,
        'python_normalized': python_norm,
        'rust_normalized': rust_norm,
    }


class CircleInCircleTests(unittest.TestCase):
    """Teste einfaches Beispiel: Kreis im Kreis mit allen Features."""
    
    def setUp(self):
        """Erstelle Testdaten: äußerer und innerer Kreis."""
        # Äußerer Kreis
        self.n_outer = 5
        self.phi_outer = np.linspace(0, 2 * np.pi, self.n_outer)[:-1]
        self.outer_pts = np.array([np.cos(self.phi_outer), np.sin(self.phi_outer)]).T
        
        # Innerer Kreis (Hole)
        self.n_inner = 5
        self.phi_inner = np.linspace(0, 2 * np.pi, self.n_inner)[:-1]
        self.inner_pts = copy.copy(self.outer_pts[:self.n_inner]) * 0.5
        
        # Kombiniere Punkte
        self.pts = np.array(list(self.inner_pts) + list(self.outer_pts))
        
        # Boundary: innerer Kreis als geschlossene Schleife
        self.boundaries = [list(range(self.n_inner)) + [0]]
    
    def test_basic_triangulation(self):
        """Teste grundlegende Triangulation ohne Constraints."""
        print("\n=== Test: Grundlegende Triangulation ===")
        
        python_tri = PythonPolyTri(self.pts, delaunay=False, holes=False)
        rust_tri = RustPolyTri(self.pts, delaunay=False, holes=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "basic")
        
        print(f"Python Dreiecke: {details['python_count']}")
        print(f"Rust Dreiecke: {details['rust_count']}")
        
        if not are_equal:
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    def test_delaunay_triangulation(self):
        """Teste Delaunay-Triangulation."""
        print("\n=== Test: Delaunay-Triangulation ===")
        
        python_tri = PythonPolyTri(self.pts, delaunay=True, holes=False)
        rust_tri = RustPolyTri(self.pts, delaunay=True, holes=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "delaunay")
        
        print(f"Python Dreiecke: {details['python_count']}")
        print(f"Rust Dreiecke: {details['rust_count']}")
        
        if not are_equal:
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    def test_hole_removal(self):
        """Teste Hole-Entfernung."""
        print("\n=== Test: Hole-Entfernung ===")
        
        python_tri = PythonPolyTri(self.pts, self.boundaries, delaunay=False, holes=True)
        rust_tri = RustPolyTri(self.pts, self.boundaries, delaunay=False, holes=True)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "holes")
        
        print(f"Python Dreiecke: {details['python_count']}")
        print(f"Rust Dreiecke: {details['rust_count']}")
        
        if not are_equal:
            for error in details['errors']:
                print(f"  - {error}")
        
        # Hole-Entfernung sollte weniger Dreiecke ergeben
        self.assertLess(details['python_count'], len(self.pts) * 2, 
                       "Hole-Entfernung sollte Dreiecke entfernen")
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    def test_delaunay_with_holes(self):
        """Teste Delaunay-Triangulation mit Hole-Entfernung."""
        print("\n=== Test: Delaunay mit Hole-Entfernung ===")
        
        python_tri = PythonPolyTri(self.pts, self.boundaries, delaunay=True, holes=True)
        rust_tri = RustPolyTri(self.pts, self.boundaries, delaunay=True, holes=True)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "delaunay_holes")
        
        print(f"Python Dreiecke: {details['python_count']}")
        print(f"Rust Dreiecke: {details['rust_count']}")
        
        if not are_equal:
            for error in details['errors']:
                print(f"  - {error}")
        
        # Hole-Entfernung sollte weniger Dreiecke ergeben
        self.assertLess(details['python_count'], len(self.pts) * 2, 
                       "Hole-Entfernung sollte Dreiecke entfernen")
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    def test_constrained_boundaries(self):
        """Teste constrained boundaries."""
        print("\n=== Test: Constrained Boundaries ===")
        
        python_tri = PythonPolyTri(self.pts, self.boundaries, delaunay=False, holes=False)
        rust_tri = RustPolyTri(self.pts, self.boundaries, delaunay=False, holes=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "constrained")
        
        print(f"Python Dreiecke: {details['python_count']}")
        print(f"Rust Dreiecke: {details['rust_count']}")
        
        if not are_equal:
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    def test_constrained_boundaries_delaunay(self):
        """Teste constrained boundaries mit Delaunay."""
        print("\n=== Test: Constrained Boundaries mit Delaunay ===")
        
        python_tri = PythonPolyTri(self.pts, self.boundaries, delaunay=True, holes=False)
        rust_tri = RustPolyTri(self.pts, self.boundaries, delaunay=True, holes=False)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "constrained_delaunay")
        
        print(f"Python Dreiecke: {details['python_count']}")
        print(f"Rust Dreiecke: {details['rust_count']}")
        
        if not are_equal:
            for error in details['errors']:
                print(f"  - {error}")
        
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")
    
    def test_all_features_combined(self):
        """Teste alle Features kombiniert: Delaunay + Constraints + Holes."""
        print("\n=== Test: Alle Features kombiniert ===")
        
        python_tri = PythonPolyTri(self.pts, self.boundaries, delaunay=True, holes=True)
        rust_tri = RustPolyTri(self.pts, self.boundaries, delaunay=True, holes=True)
        
        are_equal, details = compare_triangulations(python_tri, rust_tri, "all_features")
        
        print(f"Python Dreiecke: {details['python_count']}")
        print(f"Rust Dreiecke: {details['rust_count']}")
        
        if not are_equal:
            for error in details['errors']:
                print(f"  - {error}")
        
        # Mit allen Features sollte eine sinnvolle Anzahl von Dreiecken entstehen
        self.assertGreater(details['python_count'], 0, "Sollte mindestens ein Dreieck haben")
        self.assertLess(details['python_count'], len(self.pts) * 2, 
                       "Sollte weniger Dreiecke haben als maximale Triangulation")
        self.assertTrue(are_equal, f"Verschiedene Ergebnisse: {details['errors']}")


if __name__ == '__main__':
    print("=" * 70)
    print("Vergleichstest: Kreis im Kreis")
    print("=" * 70)
    print(f"Verfügbare Implementierungen:")
    print(f"  - Rust: {'Ja' if is_rust_available() else 'Nein'}")
    print(f"  - Python: {'Ja' if is_python_available() else 'Nein'}")
    print(f"  - Aktuell verwendet: {get_implementation()}")
    print("=" * 70)
    
    unittest.main(verbosity=2)

