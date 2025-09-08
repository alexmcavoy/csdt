import eta3
from eta3.utils import create_large_graph
import numpy as np
import time

def main():
    
    for N in [10, 20, 30, 40]:
        print(f'\nTesting with N={N} nodes')
        
        adj_matrix = create_large_graph('erdos_renyi', N, p=0.4)
        
        start_time = time.time()
        solver_method = 'direct' if N <= 25 else 'gmres'
        result = eta3.solve_eta_system(adj_matrix, solver_method=solver_method)
        elapsed = time.time() - start_time
        
        eta_ij = result['eta_ij']
        eta_ijk = result['eta_ijk'] 
        P = result['transition_matrix']
        
        sum1 = eta3.compute_sum_pij_eta_ij_einsum(P, eta_ij, m=1)
        sum2 = eta3.compute_sum_pij_pjk_eta_ijk_einsum(P, eta_ijk, m=1, n=1)
        
        print(f'{sum1:.6f}')
        print(f'{sum2:.6f}')

if __name__ == "__main__":
    main()