# io.py

import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np

def save_checkpoint(path: str,
                    pos_coords: np.ndarray,
                    k_dict: Dict[tuple, float],
                    cfg: Any) -> None:
    """
    Atomically write out the current state of the optimiser:
      - pos_coords: (N×2) array of node positions
      - k_dict:     dict mapping (i,j)→k_ij
      - cfg:        your OptimConfig dataclass

    The file will be a pickle of a dict.
    """
    tmp = Path(path).with_suffix('.tmp')
    data = {
        'pos_coords': pos_coords,
        'k_dict':     k_dict,
        'config':     cfg,
    }
    with tmp.open('wb') as f:
        pickle.dump(data, f)
    tmp.replace(path)   # atomic move

def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a checkpoint previously written by save_checkpoint.
    Returns a dict with keys 'pos_coords', 'k_dict', and 'config'.
    Raises FileNotFoundError if not present.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No checkpoint found at {path!r}")
    with p.open('rb') as f:
        data = pickle.load(f)
    return data
