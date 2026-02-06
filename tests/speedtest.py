#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optional benchmark: compare Python vs Rust PolyTri performance.
Run with: pixi run speedtest
"""

import sys
import os
import time
import numpy as np
from statistics import mean, median, stdev

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both implementations
try:
    from polytri._python import PolyTri as PythonPolyTri
    _python_available = True
except ImportError:
    _python_available = False
    PythonPolyTri = None
    print("Warning: Python implementation not available")

try:
    from polytri._rust import PolyTri as RustPolyTri
    _rust_available = True
except ImportError:
    _rust_available = False
    RustPolyTri = None
    print("Warning: Rust implementation not available")

if not _python_available and not _rust_available:
    print("Error: Neither implementation is available!")
    sys.exit(1)


def generate_random_points(n, seed=42):
    """Generate n random points in a unit square."""
    np.random.seed(seed)
    return np.random.rand(n, 2)


def generate_circle_points(n, radius=1.0, seed=42):
    """Generate n points on a circle."""
    np.random.seed(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    noise = np.random.normal(0, 0.05, n)
    angles += noise
    return np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])


def generate_grid_points(n):
    """Generate approximately n points in a grid."""
    side = int(np.sqrt(n))
    x = np.linspace(0, 1, side)
    y = np.linspace(0, 1, side)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points[:n]


def benchmark_triangulation(impl_class, points, boundaries=None, border=None, 
                            delaunay=True, num_runs=3):
    """
    Benchmark a triangulation implementation.
    
    Args:
        impl_class: PolyTri class to test
        points: Points array
        boundaries: Optional boundaries
        border: Optional border indices
        delaunay: Whether to use Delaunay criterion
        num_runs: Number of runs for averaging (reduced for faster execution)
    
    Returns:
        List of execution times in seconds
    """
    if impl_class is None:
        return None
    
    times = []
    for _ in range(num_runs):
        try:
            start = time.perf_counter()
            # Pass boundaries and border to constructor, not to methods
            # remove_holes() is called automatically if holes=True and boundaries are provided
            tri = impl_class(points, boundaries=boundaries, delaunay=delaunay, 
                           holes=(border is not None), border=border)
            _ = tri.get_triangles()  # Force computation
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    return times


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    else:
        return f"{seconds:.3f}s"


def print_results(test_name, python_times, rust_times, num_points):
    """Print benchmark results."""
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"Points: {num_points}")
    print(f"{'='*70}")
    
    if python_times:
        py_mean = mean(python_times)
        py_median = median(python_times)
        py_stdev = stdev(python_times) if len(python_times) > 1 else 0
        print(f"Python:  mean={format_time(py_mean):>10}, "
              f"median={format_time(py_median):>10}, "
              f"std={format_time(py_stdev):>10}")
    else:
        print("Python:  Not available")
        py_mean = None
    
    if rust_times:
        rust_mean = mean(rust_times)
        rust_median = median(rust_times)
        rust_stdev = stdev(rust_times) if len(rust_times) > 1 else 0
        print(f"Rust:    mean={format_time(rust_mean):>10}, "
              f"median={format_time(rust_median):>10}, "
              f"std={format_time(rust_stdev):>10}")
    else:
        print("Rust:    Not available")
        rust_mean = None
    
    if python_times and rust_times and py_mean and rust_mean:
        speedup = py_mean / rust_mean
        if speedup > 1:
            print(f"Speedup: {speedup:.2f}x faster (Rust)")
        else:
            print(f"Speedup: {1/speedup:.2f}x faster (Python)")


def run_benchmark_suite():
    """Run a comprehensive benchmark suite (optimized for <10s execution)."""
    print("PolyTri Performance Benchmark")
    print("=" * 70)
    
    results = []
    
    # Test 1: Small random points
    print("\n[1/4] Small random points (100 points)...")
    points = generate_random_points(100)
    python_times = benchmark_triangulation(PythonPolyTri, points, num_runs=3) if _python_available else None
    rust_times = benchmark_triangulation(RustPolyTri, points, num_runs=3) if _rust_available else None
    print_results("Small random", python_times, rust_times, 100)
    if python_times and rust_times:
        results.append(("Small random (100)", mean(python_times), mean(rust_times)))
    
    # Test 2: Medium random points
    print("\n[2/4] Medium random points (500 points)...")
    points = generate_random_points(500)
    python_times = benchmark_triangulation(PythonPolyTri, points, num_runs=3) if _python_available else None
    rust_times = benchmark_triangulation(RustPolyTri, points, num_runs=3) if _rust_available else None
    print_results("Medium random", python_times, rust_times, 500)
    if python_times and rust_times:
        results.append(("Medium random (500)", mean(python_times), mean(rust_times)))
    
    # Test 3: Large random points (reduced size)
    print("\n[3/4] Large random points (1000 points)...")
    points = generate_random_points(1000)
    python_times = benchmark_triangulation(PythonPolyTri, points, num_runs=2) if _python_available else None
    rust_times = benchmark_triangulation(RustPolyTri, points, num_runs=2) if _rust_available else None
    print_results("Large random", python_times, rust_times, 1000)
    if python_times and rust_times:
        results.append(("Large random (1000)", mean(python_times), mean(rust_times)))
    
    # Test 4: Circle points with boundary (reduced size)
    print("\n[4/4] Circle with boundary (500 points)...")
    points = generate_circle_points(500)
    # Create boundary for circle
    boundary = list(range(len(points))) + [0]  # Close the circle
    boundaries = [boundary]
    python_times = benchmark_triangulation(PythonPolyTri, points, boundaries=boundaries, num_runs=2) if _python_available else None
    rust_times = benchmark_triangulation(RustPolyTri, points, boundaries=boundaries, num_runs=2) if _rust_available else None
    print_results("Circle with boundary", python_times, rust_times, 500)
    if python_times and rust_times:
        results.append(("Circle with boundary (500)", mean(python_times), mean(rust_times)))
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Test':<35} {'Python':>15} {'Rust':>15} {'Speedup':>10}")
        print("-" * 70)
        for name, py_time, rust_time in results:
            speedup = py_time / rust_time
            print(f"{name:<35} {format_time(py_time):>15} {format_time(rust_time):>15} {speedup:>9.2f}x")
        
        avg_speedup = mean([py / rust for _, py, rust in results])
        print("-" * 70)
        print(f"{'Average speedup':<35} {'':>15} {'':>15} {avg_speedup:>9.2f}x")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    run_benchmark_suite()

