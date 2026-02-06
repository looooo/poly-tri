# -*- coding: utf-8 -*-
"""Shared test helpers (timing, plotting) for triangulation tests."""

import time

import numpy as np


def measure_triangulation_time(func, *args, **kwargs):
    """Measure the time taken for a triangulation function."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    return plt, PdfPages


def plot_boundaries(ax, pts, boundaries, border=None, linewidth=3, color="red", alpha=0.8):
    """Plot boundary edges as thick lines."""
    if boundaries is None or len(boundaries) == 0:
        return
    for k, boundary in enumerate(boundaries):
        if border is not None and k not in border:
            continue
        boundary = np.asarray(boundary)
        if len(boundary) < 2:
            continue
        for i in range(len(boundary)):
            idx1 = boundary[i]
            idx2 = boundary[(i + 1) % len(boundary)]
            if idx1 < len(pts) and idx2 < len(pts):
                ax.plot(
                    [pts[idx1][0], pts[idx2][0]],
                    [pts[idx1][1], pts[idx2][1]],
                    color=color, linewidth=linewidth, alpha=alpha, zorder=10,
                )


def plot_comparison(
    test_name,
    pts,
    python_tri,
    rust_tri,
    annotate_points=False,
    python_time=None,
    rust_time=None,
    delaunay=None,
    holes=None,
    boundaries=None,
    border=None,
):
    """Plot Python and Rust triangulations side by side; returns the figure."""
    plt, _ = _import_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    params = []
    if delaunay is not None:
        params.append(f"delaunay={delaunay}")
    if holes is not None:
        params.append(f"holes={holes}")
    if boundaries is not None:
        if isinstance(boundaries, list) and len(boundaries) > 0:
            params.append(f"boundaries={len(boundaries)}")
        elif boundaries:
            params.append("boundaries=True")
        else:
            params.append("boundaries=False")
    if border is not None and len(border) > 0:
        params.append(f"border={border}")
    param_str = ", ".join(params) if params else ""

    time_str = ""
    if python_time is not None and rust_time is not None:
        speedup = python_time / rust_time if rust_time > 0 else 0
        time_str = f" | Python: {python_time*1000:.2f}ms, Rust: {rust_time*1000:.2f}ms"
        if speedup > 1:
            time_str += f" ({speedup:.2f}x faster)"
        elif speedup < 1 and speedup > 0:
            time_str += f" ({1/speedup:.2f}x slower)"
    elif python_time is not None:
        time_str = f" | Python: {python_time*1000:.2f}ms"
    elif rust_time is not None:
        time_str = f" | Rust: {rust_time*1000:.2f}ms"

    full_title = f"{test_name}\n({param_str}){time_str}" if param_str else f"{test_name}{time_str}"
    fig.suptitle(full_title, fontsize=16, fontweight="bold")

    if python_tri is not None:
        python_triangles = python_tri.get_triangles()
        if len(python_triangles) > 0:
            triangles_arr = np.array([t for t in python_triangles])
            ax1.triplot(*pts.T, triangles_arr, "b-", linewidth=0.5)
        ax1.plot(*pts.T, "ro", markersize=4)
        if boundaries is not None:
            plot_boundaries(ax1, pts, boundaries, border=border, linewidth=3, color="red", alpha=0.8)
        if annotate_points:
            for i, p in enumerate(pts):
                ax1.annotate(str(i), p, fontsize=8)
        title = f"Python Implementation\n({len(python_triangles)} triangles)"
        if python_time is not None:
            title += f"\nTime: {python_time*1000:.2f}ms"
        ax1.set_title(title, fontsize=12, fontweight="bold")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Python version\nnot available", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Python Implementation (not available)", fontsize=12)

    if rust_tri is not None:
        rust_triangles = rust_tri.get_triangles()
        if len(rust_triangles) > 0:
            triangles_arr = np.array([t for t in rust_triangles])
            ax2.triplot(*pts.T, triangles_arr, "g-", linewidth=0.5)
        ax2.plot(*pts.T, "ro", markersize=4)
        if boundaries is not None:
            plot_boundaries(ax2, pts, boundaries, border=border, linewidth=3, color="red", alpha=0.8)
        if annotate_points:
            for i, p in enumerate(pts):
                ax2.annotate(str(i), p, fontsize=8)
        title = f"Rust Implementation\n({len(rust_triangles)} triangles)"
        if rust_time is not None:
            title += f"\nTime: {rust_time*1000:.2f}ms"
        ax2.set_title(title, fontsize=12, fontweight="bold")
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Rust version\nnot available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Rust Implementation (not available)", fontsize=12)

    plt.tight_layout()
    return fig
