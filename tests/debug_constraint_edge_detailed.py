# -*- coding: utf-8 -*-
"""
Detailliertes Debug-Script für test_constraint_edge.
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
print("DETAILLIERTES DEBUG: test_constraint_edge")
print("=" * 80)
print(f"\nPunkte ({len(pts)}):")
for i, p in enumerate(pts):
    print(f"  {i}: {p}")

print(f"\nConstraint boundaries: {cb}")
print(f"Parameter: holes=False, delaunay=False")

# Python-Version - Schritt für Schritt
print("\n" + "=" * 80)
print("PYTHON VERSION - SCHRITT FÜR SCHRITT")
print("=" * 80)

# Erstelle Python-Version ohne Constraints zuerst
print("\n1. Erstelle Triangulation OHNE Constraints:")
python_tri_no_constraints = PythonPolyTri(pts, boundaries=None, holes=False, delaunay=False)
print(f"   Anzahl Dreiecke: {len(python_tri_no_constraints.get_triangles())}")
print(f"   Anzahl Boundary Edges: {len(python_tri_no_constraints.boundary_edges)}")

print("\n2. Erstelle Triangulation MIT Constraints:")
python_tri = PythonPolyTri(pts, cb, holes=False, delaunay=False)
print(f"   Anzahl Dreiecke: {len(python_tri.get_triangles())}")
print(f"   Anzahl Boundary Edges: {len(python_tri.boundary_edges)}")

print("\nPython Dreiecke:")
for i, tri in enumerate(python_tri.get_triangles()):
    print(f"  {i+1:2d}. {tri}")

# Rust-Version - Schritt für Schritt
if RUST_AVAILABLE:
    print("\n" + "=" * 80)
    print("RUST VERSION - SCHRITT FÜR SCHRITT")
    print("=" * 80)
    
    print("\n1. Erstelle Triangulation OHNE Constraints:")
    rust_tri_no_constraints = RustPolyTri(pts, boundaries=None, holes=False, delaunay=False)
    print(f"   Anzahl Dreiecke: {len(rust_tri_no_constraints.get_triangles())}")
    print(f"   Anzahl Boundary Edges: {len(rust_tri_no_constraints.boundary_edges)}")
    
    print("\n2. Erstelle Triangulation MIT Constraints:")
    rust_tri = RustPolyTri(pts, cb, holes=False, delaunay=False)
    print(f"   Anzahl Dreiecke: {len(rust_tri.get_triangles())}")
    print(f"   Anzahl Boundary Edges: {len(rust_tri.boundary_edges)}")
    
    print("\nRust Dreiecke:")
    for i, tri in enumerate(rust_tri.get_triangles()):
        print(f"  {i+1:2d}. {tri}")
    
    # Vergleiche ohne Constraints
    print("\n" + "=" * 80)
    print("VERGLEICH OHNE CONSTRAINTS")
    print("=" * 80)
    py_tris_no = set(tuple(sorted(tri)) for tri in python_tri_no_constraints.get_triangles())
    rust_tris_no = set(tuple(sorted(tri)) for tri in rust_tri_no_constraints.get_triangles())
    
    print(f"Python ohne Constraints: {len(py_tris_no)} Dreiecke")
    print(f"Rust ohne Constraints: {len(rust_tris_no)} Dreiecke")
    
    missing_no_constraints = py_tris_no - rust_tris_no
    extra_no_constraints = rust_tris_no - py_tris_no
    
    if missing_no_constraints:
        print(f"\nFehlende in Rust (ohne Constraints): {len(missing_no_constraints)}")
        for tri in list(missing_no_constraints)[:5]:
            print(f"  {tri}")
    
    if extra_no_constraints:
        print(f"\nZusätzliche in Rust (ohne Constraints): {len(extra_no_constraints)}")
        for tri in list(extra_no_constraints)[:5]:
            print(f"  {tri}")
    
    # Vergleiche mit Constraints
    print("\n" + "=" * 80)
    print("VERGLEICH MIT CONSTRAINTS")
    print("=" * 80)
    py_tris = set(tuple(sorted(tri)) for tri in python_tri.get_triangles())
    rust_tris = set(tuple(sorted(tri)) for tri in rust_tri.get_triangles())
    
    print(f"Python mit Constraints: {len(py_tris)} Dreiecke")
    print(f"Rust mit Constraints: {len(rust_tris)} Dreiecke")
    
    missing = py_tris - rust_tris
    extra = rust_tris - py_tris
    
    if missing:
        print(f"\nFehlende in Rust (mit Constraints): {len(missing)}")
        for tri in list(missing)[:10]:
            print(f"  {tri}")
    
    if extra:
        print(f"\nZusätzliche in Rust (mit Constraints): {len(extra)}")
        for tri in list(extra)[:10]:
            print(f"  {tri}")
else:
    print("\n⚠️  Rust-Version nicht verfügbar")

