#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test to verify points mapping issue."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polytri import PolyTri, PolyTriPy

# Simple test with 5 points
original_points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.5]
])

# Test both implementations
for impl_name, impl_class in [("Python", PolyTriPy), ("Rust (default)", PolyTri)]:
    print(f"\n{'='*60}")
    print(f"Testing {impl_name} implementation:")
    print(f"{'='*60}")
    print("Original points:")
    print(original_points)
    print()

    # Create triangulation
    try:
        tri = impl_class(original_points, delaunay=True, holes=False)

        print(f"tri.points (from {impl_name}):")
        print(tri.points)
        print()

        print("Are original_points and tri.points equal?")
        are_equal = np.allclose(original_points, tri.points)
        print(are_equal)
        print()

        if not are_equal:
            print("ERROR: Points don't match!")
            print("Original points shape:", original_points.shape)
            print("tri.points shape:", tri.points.shape)
            print()
        else:
            print("✓ Points match correctly!")
            print()

        print("Triangles (indices refer to original points):")
        triangles = tri.get_triangles()
        for i, tri_idx in enumerate(triangles[:3]):
            print(f"Triangle {i}: {tri_idx}")
            print(f"  Using original_points: {original_points[tri_idx]}")
            print(f"  Using tri.points: {tri.points[tri_idx]}")
            # Verify that the points match
            if np.allclose(original_points[tri_idx], tri.points[tri_idx]):
                print(f"  ✓ Points match!")
            else:
                print(f"  ✗ ERROR: Points don't match!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
