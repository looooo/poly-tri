# -*- coding: utf-8 -*-

import unittest
import copy
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both implementations for comparison
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

def measure_triangulation_time(func, *args, **kwargs):
    """Measure the time taken for a triangulation function."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def plot_boundaries(ax, pts, boundaries, border=None, linewidth=3, color='red', alpha=0.8):
    """
    Plot boundary edges as thick lines.
    
    Args:
        ax: Matplotlib axis
        pts: Points array
        boundaries: List of boundary sequences (each is a list of point indices)
        border: Optional list of boundary indices to plot (if None, plot all)
        linewidth: Line width for boundaries
        color: Color for boundaries
        alpha: Alpha transparency
    """
    if boundaries is None or len(boundaries) == 0:
        return
    
    for k, boundary in enumerate(boundaries):
        # Skip if border is specified and this boundary is not in it
        if border is not None and k not in border:
            continue
        
        boundary = np.asarray(boundary)
        if len(boundary) < 2:
            continue
        
        # Plot edges between consecutive points (closed loop)
        for i in range(len(boundary)):
            idx1 = boundary[i]
            idx2 = boundary[(i + 1) % len(boundary)]  # Wrap around for closed loop
            
            if idx1 < len(pts) and idx2 < len(pts):
                ax.plot([pts[idx1][0], pts[idx2][0]], 
                       [pts[idx1][1], pts[idx2][1]], 
                       color=color, linewidth=linewidth, alpha=alpha, zorder=10)

def plot_comparison(test_name, pts, python_tri, rust_tri, annotate_points=False,
                   python_time=None, rust_time=None, delaunay=None, holes=None, 
                   boundaries=None, border=None):
    """
    Plot Python and Rust triangulations side by side for comparison.
    
    Args:
        test_name: Name of the test (used as title)
        pts: Points array
        python_tri: Python PolyTri instance
        rust_tri: Rust PolyTri instance
        annotate_points: If True, annotate points with their indices
        python_time: Time taken for Python triangulation (in seconds)
        rust_time: Time taken for Rust triangulation (in seconds)
        delaunay: Whether Delaunay criterion was used (for display)
        holes: Whether hole removal was used (for display)
        boundaries: Boundaries used (for display)
        border: Border indices used (for display)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Build parameter string
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
    
    # Build time comparison string
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
    
    if param_str:
        full_title = f"{test_name}\n({param_str}){time_str}"
    else:
        full_title = f"{test_name}{time_str}"
    
    fig.suptitle(full_title, fontsize=16, fontweight='bold')
    
    # Plot Python version (left)
    if python_tri is not None:
        python_triangles = python_tri.get_triangles()
        # Nur plotten wenn Dreiecke vorhanden sind
        if len(python_triangles) > 0:
            ax1.triplot(*pts.T, python_triangles, 'b-', linewidth=0.5)
        ax1.plot(*pts.T, 'ro', markersize=4)
        # Plot boundaries as thick lines
        if boundaries is not None:
            plot_boundaries(ax1, pts, boundaries, border=border, linewidth=3, color='red', alpha=0.8)
        if annotate_points:
            for i, p in enumerate(pts):
                ax1.annotate(str(i), p, fontsize=8)
        title = f'Python Implementation\n({len(python_triangles)} triangles)'
        if python_time is not None:
            title += f'\nTime: {python_time*1000:.2f}ms'
        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Python version\nnot available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Python Implementation (not available)', fontsize=12)
    
    # Plot Rust version (right)
    if rust_tri is not None:
        rust_triangles = rust_tri.get_triangles()
        # Nur plotten wenn Dreiecke vorhanden sind
        if len(rust_triangles) > 0:
            ax2.triplot(*pts.T, rust_triangles, 'g-', linewidth=0.5)
        ax2.plot(*pts.T, 'ro', markersize=4)
        # Plot boundaries as thick lines
        if boundaries is not None:
            plot_boundaries(ax2, pts, boundaries, border=border, linewidth=3, color='red', alpha=0.8)
        if annotate_points:
            for i, p in enumerate(pts):
                ax2.annotate(str(i), p, fontsize=8)
        title = f'Rust Implementation\n({len(rust_triangles)} triangles)'
        if rust_time is not None:
            title += f'\nTime: {rust_time*1000:.2f}ms'
        ax2.set_title(title, fontsize=12, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Rust version\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Rust Implementation (not available)', fontsize=12)
    
    plt.tight_layout()
    return fig  # Return figure instead of showing

class TriangleTests(unittest.TestCase):
    figures = []  # Liste zum Sammeln aller Figures
    
    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests aufgerufen"""
        cls.figures = []
    
    @classmethod
    def tearDownClass(cls):
        """Wird einmal nach allen Tests aufgerufen - speichert PDF"""
        if cls.figures:
            pdf_path = os.path.join(os.path.dirname(__file__), 'test_results.pdf')
            with PdfPages(pdf_path) as pdf:
                for fig in cls.figures:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            print(f"\n✓ Alle Test-Ergebnisse wurden in '{pdf_path}' gespeichert")
            print(f"  ({len(cls.figures)} Seiten)")
        cls.figures = []
    
    @property
    def an_int(self):
        return np.random.randint(9, 50)

    def setUp(self):
        self.phi = np.linspace(0, 2 * np.pi, self.an_int)[:-1]

    def test_constraint_edge(self):
        n = self.an_int
        x = np.linspace(0, 1, n)
        y = np.array([0] * n)
        pts = np.array([x, y]).T
        pts_upper = pts[1:-1].copy()
        pts_upper[:,1] += 0.1
        pts[1:-1, 1] -= 0.1
        cb = [[0, n-1]]
        additional_pts = np.array([[-1., 0], [2, 0.]])
        pts = np.array(list(pts) + list(pts_upper) + list(additional_pts))
        
        python_time = None
        rust_time = None
        
        if _python_available:
            start = time.perf_counter()
            python_tri = PythonPolyTri(pts, cb, holes=False, delaunay=False)
            python_time = time.perf_counter() - start
        else:
            python_tri = None
        
        if _rust_available:
            start = time.perf_counter()
            rust_tri = RustPolyTri(pts, cb, holes=False, delaunay=False)
            rust_time = time.perf_counter() - start
        else:
            rust_tri = None
        
        fig = plot_comparison('test_constraint_edge', pts, python_tri, rust_tri, 
                       annotate_points=True, python_time=python_time, rust_time=rust_time,
                       delaunay=False, holes=False, boundaries=cb)
        self.__class__.figures.append(fig)

    def test_constraint_edge_1(self):
        n = self.an_int
        x = np.linspace(0, np.pi, n)
        y = abs(np.sin(x)) - 1.1
        pts_inner = list(np.array([x, y]).T)
        pts_inner.reverse()
        
        pts_outer = list(np.array(pts_inner) * np.array([1., -1.]))
        pts_outer.reverse()
        pts = pts_inner + pts_outer
        cb = [list(range(len(pts_inner)))]
        cb += [list(np.array(range(len(pts_outer))) + max(cb[0]) + 1)]
        pts += list(np.array([[0., 0.], [np.pi, 0.]]))
        cb += [[len(pts) - 2, len(pts) - 1]]
        pts = np.array(pts)
        
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, cb, border=[0, 1], holes=True, delaunay=False)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, cb, border=[0, 1], holes=True, delaunay=False)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_constraint_edge_1', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=False, holes=True, boundaries=cb, border=[0, 1])
        self.__class__.figures.append(fig)


    def test_constraint_edge_2(self):
        n = self.an_int
        x = np.linspace(0, np.pi, n)
        y = abs(np.sin(x)) - 1.1
        pts = list(np.array([x, y]).T)
        pts.reverse()
        pts += list(np.array([[0., 0.], [np.pi, 0.]]))
        pts = np.array(pts)
        
        boundaries = [list(range(len(pts))) + [0]]
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, boundaries=boundaries, holes=True, delaunay=False)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, boundaries=boundaries, holes=True, delaunay=False)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_constraint_edge_2', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=False, holes=True, boundaries=boundaries)
        self.__class__.figures.append(fig)

    def test_easy3(self):
        pts =np.array([
            [ 0., 0. ],
            [ 0.2, 0.5],
            [ 0.4, 0.7],
            [ 0.6, 0.7],
            [ 0.8, 0.5],
            [ 1.0, 0.]])
        
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, holes=False, delaunay=False)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, holes=False, delaunay=False)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_easy3', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=False, holes=False, boundaries=None)
        self.__class__.figures.append(fig)
       

    def test_easy_1(self):
        ul = [1, 2, 3, 4, 0, 1]
        pts = np.array([[0.,  0. ],
                        [0.2, 0.1],
                        [0.5, 0.1],
                        [0.8, 0.1],
                        [1.,  0. ]])
        
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, holes=False, delaunay=True)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, holes=False, delaunay=True)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_easy_1', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=True, holes=False, boundaries=None)
        self.__class__.figures.append(fig)


    def test_easy_2(self):
        pts = np.array([[-1, 0], [1, 0], [0., 0.5], [0., -0.5]])
        edge = [np.array([2, 3])]
        
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, edge, holes=False, delaunay=False)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, edge, holes=False, delaunay=False)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_easy_2', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=False, holes=False, boundaries=edge)
        self.__class__.figures.append(fig)


    def test_easy_3(self):
        pts = np.array([[-1., 0.], [1., 0.], [0., 0.5], [0., -0.5], [0., 1.]])
        edge = [np.array([0, 1])]
        
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, holes=False, delaunay=False)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, holes=False, delaunay=False)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_easy_3', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=False, holes=False, boundaries=None)
        self.__class__.figures.append(fig)

    def test_ellipse(self):
        outer_pts = np.array([np.cos(self.phi), np.sin(self.phi)]).T
        inner_pts = copy.copy(outer_pts)
        outer_pts *= np.array([1., 1.])
        inner_pts *= 0.5
        pts = np.array(list(inner_pts) + list(outer_pts))
        
        boundaries = [list(range(len(inner_pts))) + [0]]
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, boundaries, delaunay=False, holes=True)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, boundaries, delaunay=False, holes=True)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_ellipse', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=True, holes=True, boundaries=boundaries)
        self.__class__.figures.append(fig)


    def test_profile(self):
        profile = [0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
                     9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        hole = [24, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24]
        hole.reverse()
        pts = np.array([[1., 0.], 
                    [0.90196078, 0.01922532], 
                    [0.80392157, 0.0377041 ],
                    [0.70588235, 0.05526607], 
                    [0.60784314, 0.07167488], 
                    [0.50980392, 0.08654832], 
                    [0.41176471, 0.09930623],
                    [0.31372549, 0.10875978], 
                    [0.21568627, 0.11197072], 
                    [0.12254902, 0.10145591], 
                    [0.05446623, 0.07889333], 
                    [0.01361656, 0.04499772], 
                    [0., 0.], 
                    [ 0.01361656, -0.02548616], 
                    [ 0.05446623, -0.04430561], 
                    [ 0.12254902, -0.05642389], 
                    [ 0.21568627, -0.06297434], 
                    [ 0.31372549, -0.0661538 ], 
                    [ 0.41176471, -0.06701299], 
                    [ 0.50980392, -0.06625902], 
                    [ 0.60784314, -0.0629272 ], 
                    [ 0.70588235, -0.05471569], 
                    [ 0.80392157, -0.04154998], 
                    [ 0.90196078, -0.02334161], 
                    [0.22404773, 0.02395152], 
                    [0.21754519, 0.04396428], 
                    [0.20052133, 0.05633284], 
                    [0.17947867, 0.05633284], 
                    [0.16245481, 0.04396428], 
                    [0.15595227, 0.02395152], 
                    [0.16245481, 0.00393877], 
                    [ 0.17947867, -0.00842979], 
                    [ 0.20052133, -0.00842979], 
                    [0.21754519, 0.00393877]])
        
        boundaries = [hole]
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, boundaries, delaunay=False, holes=False)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, boundaries, delaunay=False, holes=False)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_profile', pts, python_tri, rust_tri, annotate_points=True,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=False, holes=False, boundaries=boundaries)
        self.__class__.figures.append(fig)
    

    def test_profile2(self):
        pts = np.array([
                 [ 1.        ,  0.        ],
                 [ 0.93079585,  0.01364559],
                 [ 0.8615917 ,  0.02693948],
                 [ 0.79238754,  0.03983196],
                 [ 0.72318339,  0.05224763],
                 [ 0.65397924,  0.06411011],
                 [ 0.58477509,  0.07533044],
                 [ 0.51557093,  0.08571854],
                 [ 0.44636678,  0.09508809],
                 [ 0.37716263,  0.10309254],
                 [ 0.30795848,  0.10914777],
                 [ 0.23875433,  0.11209604],
                 [ 0.16961802,  0.10886508],
                 [ 0.10855553,  0.09829529],
                 [ 0.06106249,  0.08212217],
                 [ 0.02713888,  0.06031021],
                 [ 0.00678472,  0.03289393],
                 [ 0.        , 0.        ],
                 [ 0.00678472, -0.01869534],
                 [ 0.02713888, -0.03406389],
                 [ 0.06106249, -0.04609344],
                 [ 0.10855553, -0.05478371],
                 [ 0.16961802, -0.06033743],
                 [ 0.23875433, -0.06399067],
                 [ 0.30795848, -0.06603619],
                 [ 0.37716263, -0.06691995],
                 [ 0.44636678, -0.06690913],
                 [ 0.51557093, -0.06617305],
                 [ 0.58477509, -0.06412989],
                 [ 0.65397924, -0.0596694 ],
                 [ 0.72318339, -0.05275214],
                 [ 0.79238754, -0.04337563],
                 [ 0.8615917 , -0.03145862],
                 [ 0.93079585, -0.0170007 ],
                 [ 0.22424641,  0.02420158],
                 [ 0.21770592,  0.04433111],
                 [ 0.20058272,  0.05677185],
                 [ 0.17941728,  0.05677185],
                 [ 0.16229408,  0.04433111],
                 [ 0.15575359,  0.02420158],
                 [ 0.16229408,  0.00407205],
                 [ 0.17941728, -0.00836869],
                 [ 0.20058272, -0.00836869],
                 [ 0.21770592,  0.00407205],
                 [ 0.43347349,  0.01676735],
                 [ 0.42708062,  0.03644257],
                 [ 0.41034388,  0.04860253],
                 [ 0.38965612,  0.04860253],
                 [ 0.37291938,  0.03644257],
                 [ 0.36652651,  0.01676735],
                 [ 0.37291938, -0.00290788],
                 [ 0.38965612, -0.01506784],
                 [ 0.41034388, -0.01506784],
                 [ 0.42708062, -0.00290788],
                 [ 0.67016282,  0.0026589 ],
                 [ 0.66535715,  0.01744923],
                 [ 0.65277574,  0.02659016],
                 [ 0.63722426,  0.02659016],
                 [ 0.62464285,  0.01744923],
                 [ 0.61983718,  0.0026589 ],
                 [ 0.62464285, -0.01213144],
                 [ 0.63722426, -0.02127236],
                 [ 0.65277574, -0.02127236],
                 [ 0.66535715, -0.01213144]])
        boundaries = [[0, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 
                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [34, 35, 36, 37, 38,
                       39, 40, 41, 42, 43, 34], [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 44], [54, 
                       55, 56, 57, 58, 59, 60, 61, 62, 63, 54]]

        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, boundaries, delaunay=False, holes=True)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, boundaries, delaunay=False, holes=True)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_profile2', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=False, holes=True, boundaries=boundaries)
        self.__class__.figures.append(fig)


    def test_profile3(self):
        pts = np.array([
[ 1.        ,  0.        ],
[ 0.88235294,  0.02298626],
[ 0.76470588,  0.0448548 ],
[ 0.64705882,  0.06526509],
[ 0.52941176,  0.08372709],
[ 0.41176471,  0.09930623],
[ 0.29411765,  0.11005194],
[ 0.17647059,  0.10961876],
[ 0.07843137,  0.08925774],
[ 0.01960784,  0.05267707],
[ 0.        ,  0.        ],
[ 0.01960784, -0.02975812],
[ 0.07843137, -0.04997027],
[ 0.17647059, -0.06078761],
[ 0.29411765, -0.06573109],
[ 0.41176471, -0.06701299],
[ 0.52941176, -0.06593259],
[ 0.64705882, -0.06023152],
[ 0.76470588, -0.04741638],
[ 0.88235294, -0.02739101],
[ 0.22420494,  0.02415623],
[ 0.21767237,  0.04426139],
[ 0.20056991,  0.05668706],
[ 0.17943009,  0.05668706],
[ 0.16232763,  0.04426139],
[ 0.15579506,  0.02415623],
[ 0.16232763,  0.00405107],
[ 0.17943009, -0.0083746 ],
[ 0.20056991, -0.0083746 ],
[ 0.21767237,  0.00405107],
[ 0.43345312,  0.016748  ],
[ 0.42706414,  0.03641125],
[ 0.41033758,  0.04856381],
[ 0.38966242,  0.04856381],
[ 0.37293586,  0.03641125],
[ 0.36654688,  0.016748  ],
[ 0.37293586, -0.00291525],
[ 0.38966242, -0.01506781],
[ 0.41033758, -0.01506781],
[ 0.42706414, -0.00291525],
[ 0.67018389,  0.00262844],
[ 0.6653742 ,  0.01743116],
[ 0.65278225,  0.02657975],
[ 0.63721775,  0.02657975],
[ 0.6246258 ,  0.01743116],
[ 0.61981611,  0.00262844],
[ 0.6246258 , -0.01217428],
[ 0.63721775, -0.02132286],
[ 0.65278225, -0.02132286],
[ 0.6653742 , -0.01217428]
            ])
        boundaries = [[0, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20],
                      [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 30], [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 40]]
        
        python_time = None
        rust_time = None
        
        if _python_available:
            python_tri, python_time = measure_triangulation_time(
                PythonPolyTri, pts, boundaries, delaunay=True, holes=True)
        else:
            python_tri = None
        
        if _rust_available:
            rust_tri, rust_time = measure_triangulation_time(
                RustPolyTri, pts, boundaries, delaunay=True, holes=True)
        else:
            rust_tri = None
        
        fig = plot_comparison('test_profile3', pts, python_tri, rust_tri,
                       python_time=python_time, rust_time=rust_time,
                       delaunay=True, holes=True, boundaries=boundaries)
        self.__class__.figures.append(fig)

if __name__ == '__main__':
    unittest.main()


