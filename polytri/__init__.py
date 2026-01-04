"""
PolyTri - Delaunay triangulation with constrained boundaries and hole removal.

This package provides both a Rust implementation (faster) and a Python implementation
(fallback). The Rust version is used by default if available, otherwise falls back
to the Python implementation.

Usage:
    from polytri import PolyTri
    
    # Uses Rust version if available, otherwise Python version
    tri = PolyTri(points, boundaries=boundaries, delaunay=True)
"""

import os
import sys

# Try to import Rust version first (faster)
_USE_RUST = os.getenv("POLYTRI_USE_RUST", "auto").lower()
_rust_available = False
_python_available = True

if _USE_RUST in ("auto", "1", "true", "yes"):
    try:
        from polytri._rust import PolyTri as RustPolyTri
        _rust_available = True
    except ImportError:
        _rust_available = False
elif _USE_RUST in ("0", "false", "no"):
    _rust_available = False

# Import Python version (always available as fallback)
try:
    from polytri._python import PolyTri as PythonPolyTri
    _python_available = True
except ImportError:
    _python_available = False

# Determine which version to use
if _rust_available:
    PolyTri = RustPolyTri
    _implementation = "rust"
elif _python_available:
    PolyTri = PythonPolyTri
    _implementation = "python"
else:
    raise ImportError(
        "Neither Rust nor Python implementation of PolyTri is available. "
        "Please ensure polytri._python is available."
    )

# Export the selected implementation
__all__ = ["PolyTri", "_implementation", "_rust_available", "_python_available"]

# Provide information about which implementation is being used
def get_implementation():
    """Return the name of the implementation being used."""
    return _implementation

def is_rust_available():
    """Check if Rust implementation is available."""
    return _rust_available

def is_python_available():
    """Check if Python implementation is available."""
    return _python_available

# Add these to __all__ as well
__all__.extend(["get_implementation", "is_rust_available", "is_python_available"])

