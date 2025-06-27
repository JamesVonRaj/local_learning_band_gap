# objective.py

import numpy as np
from typing import Tuple

def spectral_gap(omega: np.ndarray, n: int) -> float:
    """
    Compute the gap Δₙ = ωₙ₊₁ – ωₙ.

    Parameters
    ----------
    omega : array_like, shape (m,)
        Sorted eigenfrequencies (ω₁ ≤ ω₂ ≤ …).
    n     : int
        Index of the lower mode in the gap (1-based).

    Returns
    -------
    float
        Δₙ = ω[n] – ω[n-1]   (zero-indexed in Python).
    """
    return float(omega[n] - omega[n-1])


def loss_maximise(Δ: float) -> float:
    """
    Loss for gap maximisation: L = –Δ.
    """
    return -Δ

def loss_target(Δ: float, Δ_star: float) -> float:
    """
    Loss for targeting a specific gap: L = (Δ – Δ*)².
    """
    return (Δ - Δ_star)**2

def loss_and_sign(Δ: float, mode: str, Δ_star: float = None) -> Tuple[float, float]:
    """
    Combined helper: given the current Δ, returns (L, dL_dΔ).

    Parameters
    ----------
    Δ      : float
        Current spectral gap.
    mode   : {'maximise','target'}
    Δ_star : float, optional
        Desired gap if mode=='target'.

    Returns
    -------
    L       : float
        Value of the loss function.
    sign    : float
        dL/dΔ, which you multiply by your local dΔ/dp to get dL/dp.
    """
    if mode == "maximise":
        return -Δ, -1.0
    elif mode == "target":
        if Δ_star is None:
            raise ValueError("Δ_star must be provided when mode='target'")
        return (Δ - Δ_star)**2, 2*(Δ - Δ_star)
    else:
        raise ValueError(f"Unknown mode: {mode}")
