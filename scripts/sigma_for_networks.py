import os
import time

import eta3
import numpy as np

NETWORKS_DIR = os.path.join(os.path.dirname(__file__), 'networks')
NETWORK_FILES = [
	'barabasi_albert.txt', 'ceiling_fan.txt', 'circle.txt', 'erdos_renyi.txt',
	'office.txt', 'star_graph.txt', 'small_world.txt', 'two_stars.txt',
]

def compute_sigma(filepath):
	adj_matrix = np.loadtxt(filepath)
	n = adj_matrix.shape[0]
	solver_method = 'direct' if n <= 25 else 'gmres'

	start = time.time()
	result = eta3.solve_eta_system(adj_matrix, solver_method=solver_method)
	elapsed = time.time() - start

	eta_ij = result['eta_ij']
	eta_ijk = result['eta_ijk']
	P = result['transition_matrix']

	sum1 = eta3.compute_sum_pij_eta_ij_einsum(P, eta_ij, m=1)
	sum2 = eta3.compute_sum_pij_eta_ij_einsum(P, eta_ij, m=2)
	sum3 = eta3.compute_sum_pij_eta_ij_einsum(P, eta_ij, m=3)
	sum21 = eta3.compute_sum_pij_pjk_eta_ijk_einsum(P, eta_ijk, m=2, n=1)

	sigma = (sum3 - sum21) / (2 * sum21 - sum1 - sum2 - sum3)
	return n, elapsed, sigma

if __name__ == '__main__':
	for filename in NETWORK_FILES:
		filepath = os.path.join(NETWORKS_DIR, filename)
		print(f'\n{filename}')
		n, elapsed, sigma = compute_sigma(filepath)
		print(f'N={n}, time={elapsed:.4f}s')
		print(f'\\sigma={sigma:.6f}')
