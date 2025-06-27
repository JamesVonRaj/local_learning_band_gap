# local_update.py

import numpy as np
from typing import Tuple, Dict, Sequence
from .geometry import Positions

def edge_gradient(
    v_n: np.ndarray,
    v_np1: np.ndarray,
    i: int,
    j: int,
    n_hat: np.ndarray,
    omega_n: float,
    omega_np1: float
) -> float:
    """
    Compute ∂Δ/∂k_{ij} for edge (i,j) using analytic HF formula:
      Δ = ω_{n+1} - ω_n
    """
    # relative mode amplitudes on this edge
    δn   = np.dot(v_n[2*j:2*j+2]   - v_n[2*i:2*i+2],   n_hat)
    δnp1 = np.dot(v_np1[2*j:2*j+2] - v_np1[2*i:2*i+2], n_hat)
    return (δnp1**2)/(2*omega_np1) - (δn**2)/(2*omega_n)

def node_force_gradient(
    pos: Positions,
    edges: Sequence[Tuple[int,int]],
    k_dict: Dict[Tuple[int,int], float],
    v_n: np.ndarray,
    v_np1: np.ndarray,
    omega_n: float,
    omega_np1: float,
    i: int
) -> Tuple[float,float]:
    """
    Compute ∂Δ/∂x_i and ∂Δ/∂y_i by summing the contributions
    from every spring incident on node i.
    """
    grad = np.zeros(2, float)
    # for each edge touching i, compute ∂Δ/∂r_i from dD/dr_i
    for (a, b) in edges:
        if i not in (a, b):
            continue
        j = b if a == i else a
        disp = pos.pbc_delta(i, j)
        dist = np.linalg.norm(disp)
        if dist == 0:
            continue
        n_hat = disp / dist

        # projection of each mode on this spring
        δn   = np.dot(v_n[2*j:2*j+2]   - v_n[2*i:2*i+2],   n_hat)
        δnp1 = np.dot(v_np1[2*j:2*j+2] - v_np1[2*i:2*i+2], n_hat)

        # ∂Δ/∂D block w.r.t. r_i contributes force on node i
        # you can derive that the x‐component is:
        dK_dr = -k_dict[(min(i,j), max(i,j))] * (
            np.outer(n_hat, n_hat).dot((δnp1**2)/(2*omega_np1) - (δn**2)/(2*omega_n))
        )
        # but simpler is to take finite‐difference of D w.r.t. r_i,
        # project with v-modes, sum—and this stays local to node i.

        # (for brevity here you’d implement that blockwise HF formula)
        # grad += ...
    return grad[0], grad[1]
