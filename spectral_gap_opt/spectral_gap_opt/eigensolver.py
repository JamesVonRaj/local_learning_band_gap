# eigensolver.py

import numpy as np
from scipy.linalg import eigh
# For large N you can later swap in:
# from scipy.sparse.linalg import eigsh

def spectrum(
    D: np.ndarray,
    k: int = None,
    which: str = "SM",
    tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenfrequencies ω and mode shapes v of the dynamical matrix D.

    Parameters
    ----------
    D     : (2N×2N) array
        The assembled dynamical (stiffness) matrix.
    k     : int or None
        If None, compute the full spectrum (2N modes).
        If an integer m, compute the m smallest-magnitude modes.
    which : str
        Passed to sparse eigensolver ('SM'=smallest magnitude), ignored by dense eigh.
    tol   : float
        Threshold below which ω are treated as zero modes.

    Returns
    -------
    ω : ndarray, shape (m,)
        Sorted eigenfrequencies (non-negative).
    V : ndarray, shape (2N, m)
        Corresponding eigenvectors (column-stacked).
    """
    if k is None or k >= D.shape[0]:
        # dense full solve
        ω2, V = eigh(D)
        ω2 = np.clip(ω2, 0.0, None)
        ω = np.sqrt(ω2)
    else:
        # sparse partial solve (future; placeholder)
        ω2, V = eigsh(D, k=k, which=which)
        ω2 = np.clip(ω2, 0.0, None)
        ω  = np.sqrt(ω2)
        # sort ascending
        idx = np.argsort(ω)
        ω, V = ω[idx], V[:, idx]

    # zero-mode mask (optional use downstream)
    zero_modes = ω < tol

    return ω, V
