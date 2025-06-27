# geometry.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Positions:
    coords: np.ndarray      # shape (N, 2)
    box:    np.ndarray = np.array([1.0, 1.0])

    @classmethod
    def poisson(cls, N: int, rng: np.random.Generator, box=(1.0, 1.0)):
        """Sample N points uniformly in a periodic box."""
        pts = rng.random((N, 2)) * np.asarray(box)
        return cls(coords=pts, box=np.asarray(box))

    def pbc_delta(self, i: int, j: int) -> np.ndarray:
        """
        Minimum-image displacement from node i to node j.
        Returns a 2-vector ∈ [-box/2, box/2].
        """
        Δ = self.coords[j] - self.coords[i]
        Δ -= np.round(Δ / self.box) * self.box
        return Δ
