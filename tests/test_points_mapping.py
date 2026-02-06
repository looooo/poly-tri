#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test that tri.points matches original points and triangle indices are correct."""

import numpy as np
import pytest


ORIGINAL_POINTS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.5],
])


def _get_impl(impl_name):
    from polytri import PolyTri, PolyTriPy, is_python_available, is_rust_available

    if impl_name == "Python":
        if not is_python_available() or PolyTriPy is None:
            pytest.skip("Python implementation not available")
        return PolyTriPy
    if impl_name == "Rust":
        if not is_rust_available():
            pytest.skip("Rust implementation not available")
        return PolyTri
    raise ValueError(impl_name)


@pytest.mark.parametrize("impl_name", ["Python", "Rust"])
def test_points_match_original(impl_name):
    """Original points and tri.points must be equal for both implementations."""
    impl = _get_impl(impl_name)
    tri = impl(ORIGINAL_POINTS, delaunay=True, holes=False)

    assert np.allclose(ORIGINAL_POINTS, tri.points), (
        f"{impl_name}: tri.points should match original points"
    )


@pytest.mark.parametrize("impl_name", ["Python", "Rust"])
def test_triangle_indices_refer_to_original_points(impl_name):
    """Triangle vertex indices must refer to original point order."""
    impl = _get_impl(impl_name)
    tri = impl(ORIGINAL_POINTS, delaunay=True, holes=False)
    triangles = tri.get_triangles()

    for i, tri_idx in enumerate(triangles[:3]):
        assert np.allclose(
            ORIGINAL_POINTS[tri_idx],
            tri.points[tri_idx],
        ), f"{impl_name} triangle {i}: points at indices should match tri.points"
