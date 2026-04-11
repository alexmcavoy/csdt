import time

import eta3
from eta3.utils import create_large_graph
import numpy as np

for N in [10, 20, 30, 40]:
    adj_matrix = create_large_graph('erdos_renyi', N, p=0.4)
    solver_method = 'direct' if N <= 25 else 'gmres'

    start = time.time()
    result = eta3.solve_eta_system(adj_matrix, solver_method=solver_method)
    elapsed = time.time() - start

    P = result['transition_matrix']
    eta_ij = result['eta_ij']
    eta_ijk = result['eta_ijk']

    f1 = eta3.compute_sum_pij_eta_ij_einsum(P, eta_ij, m=1)
    g11 = eta3.compute_sum_pij_pjk_eta_ijk_einsum(P, eta_ijk, m=1, n=1)

    print(f'\nN={N}, time={elapsed:.4f}s')
    print(f'sum_pij_eta_ij = {f1:.6f}')
    print(f'um_pij_pjk_eta_ijk = {g11:.6f}')
