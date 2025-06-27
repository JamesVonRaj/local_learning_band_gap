# assembly.py

import numpy as np
from .geometry import Positions
from typing import Sequence, Tuple, Dict

def spring_block(k_ij: float, n_hat: np.ndarray) -> np.ndarray:
    """
    2×2 stiffness sub-matrix for a spring of strength k_ij
    along unit vector n_hat.
    """
    return k_ij * np.outer(n_hat, n_hat)

def assemble_D(
    pos: Positions,
    edges: Sequence[Tuple[int,int]],
    k_dict: Dict[Tuple[int,int], float]
) -> np.ndarray:
    """
    Build the 2N×2N dynamical matrix D for a unit-mass network.

    Parameters
    ----------
    pos    : Positions
       Node positions with periodic box info.
    edges  : list of (i,j)
       All connected node pairs, i<j.
    k_dict : { (i,j): k_ij }
       Spring constants for each edge.

    Returns
    -------
    D : ndarray, shape (2N,2N)
       Global stiffness/dynamical matrix.
    """
    N = pos.coords.shape[0]
    D = np.zeros((2*N, 2*N), dtype=float)

    for (i, j) in edges:
        # 1) get minimum-image displacement
        disp = pos.pbc_delta(i, j)             # 2-vector
        dist = np.linalg.norm(disp)
        if dist == 0:
            continue  # skip degenerate
        n_hat = disp / dist                   # unit vector

        # 2) local 2×2 stiffness block
        K = spring_block(k_dict[(i, j)], n_hat)

        # 3) insert into global D
        bi, bj = slice(2*i, 2*i+2), slice(2*j, 2*j+2)
        D[bi, bi] += K
        D[bj, bj] += K
        D[bi, bj] -= K
        D[bj, bi] -= K

    return D
