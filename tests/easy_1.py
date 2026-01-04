# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from polytri import PolyTri

ul = [[1, 2, 3, 4, 0, 1]]
pts = np.array([[0.,  0. ],
        [0.2, 0.1],
        [0.5, 0.1],
        [0.8, 0.1],
        [1.,  0. ]])
tri = PolyTri(pts, delaunay=False)
print("Triangles:", tri.get_triangles())