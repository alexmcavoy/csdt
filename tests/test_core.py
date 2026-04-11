import unittest

import numpy as np

import eta3
from eta3.core import compute_c_values
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

    def test_triangle_eta_ij_analytical(self):
        result = eta3.solve_eta_system(self.triangle)
        eta_ij = result['eta_ij']
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertAlmostEqual(eta_ij[i, j], 1.0, places=10)
                else:
                    self.assertEqual(eta_ij[i, j], 0.0)

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
                self.fail(f'Solver method {method} failed: {e}')

        for i in range(1, len(results)):
            self.assertTrue(
                np.allclose(results[0]['eta_ij'], results[i]['eta_ij'], rtol=1e-6),
                f'eta_ij mismatch between {methods[0]} and {methods[i]}',
            )

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

    def test_compute_c_values(self):
        c = compute_c_values(self.triangle, solver_method='direct')

        self.assertEqual(set(c.keys()), {'c_xx', 'c_xy', 'c_yx', 'c_yy'})
        for v in c.values():
            self.assertTrue(np.isfinite(v))

        # Algebraic identity: c_xx + c_xy + c_yx + c_yy == 2 * f2
        result = eta3.solve_eta_system(self.triangle)
        f2 = eta3.compute_sum_pij_eta_ij_einsum(result['transition_matrix'], result['eta_ij'], m=2)
        self.assertAlmostEqual(
            c['c_xx'] + c['c_xy'] + c['c_yx'] + c['c_yy'],
            2 * f2,
            places=10,
        )

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
            [0, 0, 1, 0],
        ])

        with self.assertRaises(ValueError):
            eta3.solve_eta_system(disconnected)

if __name__ == '__main__':
    unittest.main()
