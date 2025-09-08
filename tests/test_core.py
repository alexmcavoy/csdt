import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import eta3
from eta3.utils import create_test_graphs


class TestEtaCore(unittest.TestCase):
    
    def setUp(self):
        self.graphs = create_test_graphs()
        self.triangle = self.graphs['triangle']
        self.path = self.graphs['path']
        
    def test_solve_eta_system_triangle(self):
        result = eta3.solve_eta_system(self.triangle)
        
        self.assertIn('eta_ij', result)
        self.assertIn('eta_ijk', result)
        self.assertIn('transition_matrix', result)
        
        eta_ij = result['eta_ij']
        eta_ijk = result['eta_ijk']
        
        self.assertEqual(eta_ij.shape, (3, 3))
        self.assertEqual(eta_ijk.shape, (3, 3, 3))
        
        self.assertTrue(np.allclose(eta_ij, eta_ij.T))
        
        self.assertTrue(np.allclose(np.diag(eta_ij), 0))
        
        off_diag = eta_ij[np.triu_indices(3, k=1)]
        self.assertTrue(np.all(off_diag > 0))
        
    def test_solve_eta_system_path(self):
        result = eta3.solve_eta_system(self.path)
        
        eta_ij = result['eta_ij']
        eta_ijk = result['eta_ijk']
        
        self.assertEqual(eta_ij.shape, (4, 4))
        self.assertEqual(eta_ijk.shape, (4, 4, 4))
        
        self.assertTrue(np.allclose(eta_ij, eta_ij.T))
        
        self.assertTrue(np.allclose(np.diag(eta_ij), 0))
        
    def test_different_solver_methods(self):
        methods = ['direct', 'gmres', 'bicgstab']
        results = []
        
        for method in methods:
            try:
                result = eta3.solve_eta_system(self.triangle, solver_method=method)
                results.append(result)
            except Exception as e:
                self.fail(f"Solver method {method} failed: {e}")
        
        for i in range(1, len(results)):
            self.assertTrue(np.allclose(results[0]['eta_ij'], results[i]['eta_ij'], rtol=1e-6),
                          f"eta_ij mismatch between {methods[0]} and {methods[i]}")
            
    def test_compute_sum_pij_eta_ij(self):
        result = eta3.solve_eta_system(self.triangle)
        P = result['transition_matrix']
        eta_ij = result['eta_ij']
        
        for m in [1, 2]:
            sum_result = eta3.compute_sum_pij_eta_ij_einsum(P, eta_ij, m=m)
            self.assertIsInstance(sum_result, (float, np.floating))
            self.assertTrue(np.isfinite(sum_result))
            
    def test_compute_sum_pij_pjk_eta_ijk(self):
        result = eta3.solve_eta_system(self.triangle)
        P = result['transition_matrix']
        eta_ijk = result['eta_ijk']
        
        for m, n in [(1, 1), (1, 2), (2, 1)]:
            sum_result = eta3.compute_sum_pij_pjk_eta_ijk_einsum(P, eta_ijk, m=m, n=n)
            self.assertIsInstance(sum_result, (float, np.floating))
            self.assertTrue(np.isfinite(sum_result))
            
    def test_deterministic_graphs(self):
        for name, adj in self.graphs.items():
            with self.subTest(graph=name):
                self.assertEqual(adj.shape[0], adj.shape[1])
                self.assertTrue(np.allclose(adj, adj.T))
                self.assertTrue(np.all(np.diag(adj) == 0))
                self.assertTrue(np.all((adj == 0) | (adj == 1)))
                
    def test_transition_probabilities(self):
        P = eta3.compute_transition_matrix(self.triangle)
        
        self.assertEqual(P.shape, (3, 3))
        
        P_dense = P.toarray() if hasattr(P, 'toarray') else P
        row_sums = np.sum(P_dense, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))
        
        self.assertTrue(np.all(P_dense >= 0))
        
    def test_invalid_solver_method(self):
        with self.assertRaises(ValueError):
            eta3.solve_eta_system(self.triangle, solver_method='invalid_method')
            
    def test_disconnected_graph_error(self):
        disconnected = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
        with self.assertRaises(ValueError):
            eta3.solve_eta_system(disconnected)


if __name__ == '__main__':
    unittest.main()