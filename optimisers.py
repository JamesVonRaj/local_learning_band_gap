# optimisers.py

import itertools as it
import numpy as np
from .assembly import assemble_D
from .eigensolver import spectrum
from .objective import spectral_gap, loss_and_sign

class FDFiniteDifferenceGD:
    """
    Finite-difference gradient descent / physical-learning optimiser.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def _compute_gap_and_modes(self, pos, edges, k_dict):
        D = assemble_D(pos, edges, k_dict)
        ω, V = spectrum(D, k=self.cfg.n_gap+1)
        gap = spectral_gap(ω, self.cfg.n_gap)
        return gap, ω, V

    def _finite_diff(self, get, set_, compute_gap):
        δ = self.cfg.delta_fd
        orig = get()
        set_(orig + δ)
        Δp = compute_gap()
        set_(orig - δ)
        Δm = compute_gap()
        set_(orig)
        return (Δp - Δm)/(2*δ)

    def step(self, pos, edges, k_dict):
        # 1) physics step: gap, raw modes
        Δ, ω, V = self._compute_gap_and_modes(pos, edges, k_dict)
        L, dL_dΔ = loss_and_sign(Δ, self.cfg.mode, self.cfg.Δ_star)

        # 2) edge-wise stiffness updates
        for (i, j) in edges:
            def get_k(): return k_dict[(i, j)]
            def set_k(v): k_dict.__setitem__((i, j), v)
            grad = self._finite_diff(get_k, set_k,
                                    lambda: self._compute_gap_and_modes(pos, edges, k_dict)[0])
            k_new = k_dict[(i, j)] - self.cfg.eta_k * dL_dΔ * grad
            k_dict[(i, j)] = max(0.0, k_new)

        # 3) node-wise position updates
        for i, axis in it.product(range(len(pos.coords)), (0,1)):
            def get_r(): return pos.coords[i, axis]
            def set_r(v): pos.coords[i, axis] = v
            grad = self._finite_diff(get_r, set_r,
                                    lambda: self._compute_gap_and_modes(pos, edges, k_dict)[0])
            pos.coords[i, axis] -= self.cfg.eta_r * dL_dΔ * grad

        # enforce periodic box
        pos.coords %= pos.box

        return Δ, L

