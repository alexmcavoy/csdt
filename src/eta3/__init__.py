from .core import (
    compute_c_values,
    compute_sum_pij_eta_ij_einsum,
    compute_sum_pij_pjk_eta_ijk_einsum,
    compute_transition_matrix,
    solve_eta_ij,
    solve_eta_ijk,
    solve_eta_system,
)

__version__ = '1.0.0'
__all__ = [
    'compute_c_values',
    'compute_sum_pij_eta_ij_einsum',
    'compute_sum_pij_pjk_eta_ijk_einsum',
    'compute_transition_matrix',
    'solve_eta_ij',
    'solve_eta_ijk',
    'solve_eta_system',
]
