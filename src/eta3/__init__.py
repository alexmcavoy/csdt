
from .core import (
    solve_eta_system,
    solve_eta_ij,
    solve_eta_ijk,
    compute_transition_matrix,
    compute_sum_pij_eta_ij_einsum,
    compute_sum_pij_pjk_eta_ijk_einsum,
)

__version__ = "3.0.0"
__all__ = [
    "solve_eta_system",
    "solve_eta_ij",
    "solve_eta_ijk",
    "compute_transition_matrix",
    "compute_sum_pij_eta_ij_einsum",
    "compute_sum_pij_pjk_eta_ijk_einsum",
]
