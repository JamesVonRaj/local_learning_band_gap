from dataclasses import dataclass, field
import yaml
from pathlib import Path

@dataclass
class OptimConfig:
    # network geometry
    N:       int     = 100         # number of nodes
    R_c:     float   = 0.2         # connection radius
    n_gap:   int     = 3           # gap index (ω_n ↔ ω_{n+1})
    # learning rates
    eta_k:   float   = 1e-2        # stiffness step size
    eta_r:   float   = 1e-3        # position step size
    # finite‐difference offset
    delta_fd: float  = 1e-4
    # objective
    mode:    str     = "maximise"  # or "target"
    Δ_star:  float   = None        # required if mode=="target"
    # stopping criteria
    tol:     float   = 1e-6
    max_it:  int     = 500
    # random seed & I/O
    seed:    int     = 0
    log_every:           int     = 10
    checkpoint_every:    int     = 50

def load_config(path: str) -> OptimConfig:
    """Load an OptimConfig from a YAML file."""
    data = yaml.safe_load(Path(path).read_text())
    return OptimConfig(**data)
