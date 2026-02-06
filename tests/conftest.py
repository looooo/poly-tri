# -*- coding: utf-8 -*-
"""Pytest configuration and shared fixtures for poly-tri tests."""

import os
import sys

import numpy as np
import pytest

# Add project root to path so "polytri" can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import implementations once at collection time
try:
    from polytri._python import PolyTri as PythonPolyTri
    _python_available = True
except ImportError:
    _python_available = False
    PythonPolyTri = None

try:
    from polytri._rust import PolyTri as RustPolyTri
    _rust_available = True
except ImportError:
    _rust_available = False
    RustPolyTri = None


@pytest.fixture(scope="session")
def figures():
    """Session-scoped list to collect comparison figures; written to PDF at end."""
    from tests.helpers import _import_matplotlib

    collected = []
    yield collected
    if collected:
        plt, PdfPages = _import_matplotlib()
        pdf_path = os.path.join(os.path.dirname(__file__), "test_results.pdf")
        with PdfPages(pdf_path) as pdf:
            for fig in collected:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"\n✓ Test comparison figures saved to '{pdf_path}' ({len(collected)} pages)")


@pytest.fixture
def an_int():
    """Reproducible random number of points for parameterized tests."""
    np.random.seed(42)
    return int(np.random.randint(30, 200))


@pytest.fixture
def phi(an_int):
    """Angle array for circle/ellipse-style point sets."""
    return np.linspace(0, 2 * np.pi, an_int)[:-1]


@pytest.fixture
def python_available():
    """Whether the Python implementation is available."""
    return _python_available


@pytest.fixture
def rust_available():
    """Whether the Rust implementation is available."""
    return _rust_available


@pytest.fixture
def python_poly_tri_class():
    """PolyTri Python implementation class, or None if not available."""
    return PythonPolyTri


@pytest.fixture
def rust_poly_tri_class():
    """PolyTri Rust implementation class, or None if not available."""
    return RustPolyTri
