# -*- coding: utf-8 -*-
"""
Debug-Script für test_constraint_edge um Unterschiede zu identifizieren.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polytri._python import PolyTri as PythonPolyTri

try:
    from polytri._rust import PolyTri as RustPolyTri
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustPolyTri = None

# Erstelle Testdaten
n = 10
x = np.linspace(0, 1, n)
y = np.array([0] * n)
pts = np.array([x, y]).T
pts_upper = pts[1:-1].copy()
pts_upper[:, 1] += 0.1
pts[1:-1, 1] -= 0.1
cb = [[0, n-1]]
additional_pts = np.array([[-1., 0], [2, 0.]])
pts = np.array(list(pts) + list(pts_upper) + list(additional_pts))

print("=" * 80)
print("DEBUG: test_constraint_edge")
print("=" * 80)
print(f"\nPunkte ({len(pts)}):")
for i, p in enumerate(pts):
    print(f"  {i}: {p}")

print(f"\nConstraint boundaries: {cb}")
print(f"Parameter: holes=False, delaunay=False")

# Python-Version
print("\n" + "=" * 80)
print("PYTHON VERSION")
print("=" * 80)
python_tri = PythonPolyTri(pts, cb, holes=False, delaunay=False)
print(f"\nAnzahl Punkte nach Initialisierung: {len(python_tri.points)}")
print(f"Anzahl Dreiecke: {len(python_tri.get_triangles())}")
print(f"Anzahl Boundary Edges: {len(python_tri.boundary_edges)}")

print("\nPython Dreiecke:")
for i, tri in enumerate(python_tri.get_triangles()):
    print(f"  {i+1}. {tri}")

print("\nPython Boundary Edges:")
for edge in sorted(python_tri.boundary_edges):
    if hasattr(python_tri, '_point_order'):
        mapped = (python_tri._point_order[edge[0]], python_tri._point_order[edge[1]])
        print(f"  {edge} -> {mapped}")
    else:
        print(f"  {edge}")

# Rust-Version
if RUST_AVAILABLE:
    print("\n" + "=" * 80)
    print("RUST VERSION")
    print("=" * 80)
    rust_tri = RustPolyTri(pts, cb, holes=False, delaunay=False)
    print(f"\nAnzahl Punkte nach Initialisierung: {len(rust_tri.points)}")
    print(f"Anzahl Dreiecke: {len(rust_tri.get_triangles())}")
    print(f"Anzahl Boundary Edges: {len(rust_tri.boundary_edges)}")
    
    print("\nRust Dreiecke:")
    for i, tri in enumerate(rust_tri.get_triangles()):
        print(f"  {i+1}. {tri}")
    
    print("\nRust Boundary Edges:")
    for edge in sorted(rust_tri.boundary_edges):
        print(f"  {edge}")
    
    # Vergleiche Punkte
    print("\n" + "=" * 80)
    print("VERGLEICH PUNKTE")
    print("=" * 80)
    python_points = python_tri.points
    rust_points = rust_tri.points
    
    if hasattr(python_points, 'tolist'):
        python_points = python_points.tolist()
    if hasattr(rust_points, 'tolist'):
        rust_points = rust_points.tolist()
    
    print(f"Python: {len(python_points)} Punkte")
    print(f"Rust: {len(rust_points)} Punkte")
    
    if len(python_points) != len(rust_points):
        print("⚠️  Unterschiedliche Anzahl von Punkten!")
    else:
        for i, (p_py, p_rust) in enumerate(zip(python_points, rust_points)):
            if isinstance(p_py, np.ndarray):
                p_py = p_py.tolist()
            if isinstance(p_rust, np.ndarray):
                p_rust = p_rust.tolist()
            diff = np.array(p_py) - np.array(p_rust)
            max_diff = np.max(np.abs(diff))
            if max_diff > 1e-10:
                print(f"  Punkt {i}: Python={p_py}, Rust={p_rust}, Diff={max_diff}")
else:
    print("\n⚠️  Rust-Version nicht verfügbar")

