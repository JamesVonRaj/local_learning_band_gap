# topology.py

import numpy as np
from typing import Sequence, Tuple
from .geometry import Positions

def build_edges(pos: Positions, R_c: float) -> np.ndarray:
    """
    Given a Positions object and cutoff R_c, return an array of
    shape (M,2) of all index‐pairs (i,j) with i<j and ||r_j–r_i||_PBC <= R_c.
    """
    N = len(pos.coords)
    edges = []
    for i in range(N - 1):
        # compute all displacements from i to j>i
        delta = pos.coords[i+1:] - pos.coords[i]  
        # wrap via minimum‐image
        delta -= np.round(delta / pos.box) * pos.box
        dist = np.linalg.norm(delta, axis=1)
        close = np.where(dist <= R_c)[0]
        for k in close:
            j = i + 1 + int(k)
            edges.append((i, j))
    return np.array(edges, dtype=np.int32)

# (Optional) helper to get adjacency list
def adjacency_list(edges: np.ndarray, N: int) -> list[Sequence[int]]:
    adj = [[] for _ in range(N)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj
