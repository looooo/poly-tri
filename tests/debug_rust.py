#!/usr/bin/env python
"""Debug-Skript für Rust-Version Fehler."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polytri._rust import PolyTri as RustPolyTri
from polytri._python import PolyTri as PythonPolyTri
import numpy as np

def debug_triangulation():
    pts = np.array([
        [0.,  0. ],
        [0.2, 0.1],
        [0.5, 0.1],
        [0.8, 0.1],
        [1.,  0. ]
    ])
    
    print("=" * 60)
    print("DEBUG: Einfacher Testfall")
    print("=" * 60)
    
    print("\n--- Python Version ---")
    py_tri = PythonPolyTri(pts, delaunay=False)
    print(f"Points: {py_tri.points}")
    print(f"Point order: {py_tri._point_order}")
    print(f"Internal triangles: {py_tri._triangles}")
    print(f"Mapped triangles: {py_tri.get_triangles()}")
    print(f"Boundary edges: {sorted(py_tri.boundary_edges)}")
    
    print("\n--- Rust Version ---")
    rust_tri = RustPolyTri(pts, delaunay=False)
    print(f"Points: {rust_tri.points}")
    print(f"Mapped triangles: {rust_tri.get_triangles()}")
    print(f"Boundary edges: {sorted(rust_tri.boundary_edges)}")
    
    print("\n--- Vergleich ---")
    py_tris = set(tuple(sorted(t)) for t in py_tri.get_triangles())
    rust_tris = set(tuple(sorted(t)) for t in rust_tri.get_triangles())
    
    print(f"Python triangles (set): {py_tris}")
    print(f"Rust triangles (set): {rust_tris}")
    print(f"Missing in Rust: {py_tris - rust_tris}")
    print(f"Extra in Rust: {rust_tris - py_tris}")

if __name__ == '__main__':
    debug_triangulation()

