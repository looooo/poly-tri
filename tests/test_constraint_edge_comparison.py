# -*- coding: utf-8 -*-
"""
Vergleichstest für test_constraint_edge: Python vs Rust ohne visuelle Ausgaben.

Dieser Test extrahiert den test_constraint_edge Test und vergleicht die Ergebnisse
beider Implementierungen ohne visuelle Darstellung.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polytri._python import PolyTri as PythonPolyTri

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


def normalize_edge(edge):
    """Normalisiere Edge für Vergleich."""
    if edge[0] < edge[1]:
        return edge
    return (edge[1], edge[0])


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
        'python_boundary_edges': python_boundary_mapped,
        'rust_boundary_edges': rust_boundary,
    }


def test_constraint_edge_comparison(n=5):
    """
    Führe den test_constraint_edge Test aus und vergleiche beide Implementierungen.
    
    Args:
        n: Anzahl der Punkte für den Test (Standard: 5)
    """
    print("=" * 80)
    print(f"Test: test_constraint_edge (n={n})")
    print("=" * 80)
    
    # Erstelle Testdaten (wie im originalen test_constraint_edge)
    x = np.linspace(0, 1, n)
    y = np.array([0] * n)
    pts = np.array([x, y]).T
    pts_upper = pts[1:-1].copy()
    pts_upper[:, 1] += 0.1
    pts[1:-1, 1] -= 0.1
    cb = [[0, n-1]]
    additional_pts = np.array([[-1., 0], [2, 0.]])
    pts = np.array(list(pts) + list(pts_upper) + list(additional_pts))
    
    print(f"\nAnzahl Punkte: {len(pts)}")
    print(f"Constraint boundaries: {cb}")
    print(f"Parameter: holes=False, delaunay=False")
    
    # Führe Python-Version aus
    print("\n--- Python-Version ---")
    try:
        python_tri = PythonPolyTri(pts, cb, holes=False, delaunay=False)
        python_triangles = python_tri.get_triangles()
        print(f"✓ Python-Version erfolgreich")
        print(f"  Anzahl Dreiecke: {len(python_triangles)}")
        print(f"  Anzahl Boundary Edges: {len(python_tri.boundary_edges)}")
    except Exception as e:
        print(f"✗ Fehler in Python-Version: {e}")
        return False
    
    # Führe Rust-Version aus
    print("\n--- Rust-Version ---")
    if not RUST_AVAILABLE:
        print("✗ Rust-Version nicht verfügbar")
        return False
    
    try:
        rust_tri = RustPolyTri(pts, cb, holes=False, delaunay=False)
        rust_triangles = rust_tri.get_triangles()
        print(f"✓ Rust-Version erfolgreich")
        print(f"  Anzahl Dreiecke: {len(rust_triangles)}")
        print(f"  Anzahl Boundary Edges: {len(rust_tri.boundary_edges)}")
    except Exception as e:
        print(f"✗ Fehler in Rust-Version: {e}")
        return False
    
    # Vergleiche Ergebnisse
    print("\n--- Vergleich ---")
    are_equal, details = compare_triangulations(python_tri, rust_tri, "test_constraint_edge")
    
    if are_equal:
        print("✓ Beide Versionen liefern identische Ergebnisse!")
        print(f"\n  Gemeinsame Dreiecke: {details['python_count']}")
        print(f"  Gemeinsame Boundary Edges: {len(details['python_boundary_edges'])}")
    else:
        print("✗ Unterschiede gefunden:")
        print(f"\n  Python Dreiecke ({details['python_count']}):")
        for i, tri in enumerate(details['python_normalized'][:10]):  # Zeige erste 10
            print(f"    {i+1}. {tri}")
        if len(details['python_normalized']) > 10:
            print(f"    ... und {len(details['python_normalized']) - 10} weitere")
        
        print(f"\n  Rust Dreiecke ({details['rust_count']}):")
        for i, tri in enumerate(details['rust_normalized'][:10]):  # Zeige erste 10
            print(f"    {i+1}. {tri}")
        if len(details['rust_normalized']) > 10:
            print(f"    ... und {len(details['rust_normalized']) - 10} weitere")
        
        print(f"\n  Python Boundary Edges ({len(details['python_boundary_edges'])}):")
        for edge in sorted(details['python_boundary_edges'])[:10]:
            print(f"    {edge}")
        if len(details['python_boundary_edges']) > 10:
            print(f"    ... und {len(details['python_boundary_edges']) - 10} weitere")
        
        print(f"\n  Rust Boundary Edges ({len(details['rust_boundary_edges'])}):")
        for edge in sorted(details['rust_boundary_edges'])[:10]:
            print(f"    {edge}")
        if len(details['rust_boundary_edges']) > 10:
            print(f"    ... und {len(details['rust_boundary_edges']) - 10} weitere")
        
        print("\n  Fehlerdetails:")
        for error in details['errors']:
            print(f"    - {error}")
    
    print("\n" + "=" * 80)
    return are_equal


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Vergleiche Python und Rust Implementierung von test_constraint_edge')
    parser.add_argument('-n', '--num-points', type=int, default=20,
                       help='Anzahl der Punkte für den Test (Standard: 20)')
    
    args = parser.parse_args()
    
    success = test_constraint_edge_comparison(n=args.num_points)
    exit(0 if success else 1)

