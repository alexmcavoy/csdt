import warnings
from itertools import combinations
from typing import Dict as PyDict, Literal, Tuple, TypedDict, Union

import networkx as nx
import numpy as np
from numba import jit, types
from numba.typed import Dict
from numpy.linalg import matrix_power
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab, gmres, spsolve

class EtaSystemResult(TypedDict):
    eta_ij: np.ndarray
    eta_ijk: np.ndarray
    transition_matrix: np.ndarray

SolverMethod = Literal['direct', 'gmres', 'bicgstab']
SparseMatrix = Union[csr_matrix, np.ndarray]

@jit(nopython=True)
def _sort_triple(a, b, c):
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return a, b, c

@jit(nopython=True)
def _count_unique(a, b, c):
    if a == b:
        return 1 if b == c else 2
    return 2 if b == c else 3

@jit(nopython=True)
def _get_eta_val(m, n, eta_ij_flat, N):
    return eta_ij_flat[m * N + n] if m != n else 0.0

def _build_lookup_tables(N: int) -> Tuple[np.ndarray, Dict, np.ndarray, Dict]:
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pair_lookup = Dict.empty(key_type=types.int64, value_type=types.int64)
    for idx, (i, j) in enumerate(pairs):
        pair_lookup[i * N + j] = idx

    triplets = list(combinations(range(N), 3))
    triplet_lookup = Dict.empty(key_type=types.int64, value_type=types.int64)
    for idx, (i, j, k) in enumerate(triplets):
        triplet_lookup[i * N * N + j * N + k] = idx

    return (np.array(pairs, dtype=np.int32), pair_lookup,
            np.array(triplets, dtype=np.int32), triplet_lookup)

@jit(nopython=True)
def _assemble_eta_ij_matrix(N, P, pairs_array, pair_lookup):
    n_vars = len(pairs_array)
    rows, cols, data = [], [], []
    rhs = np.full(n_vars, 0.5)

    for eq_idx in range(n_vars):
        i, j = pairs_array[eq_idx]

        rows.append(eq_idx)
        cols.append(eq_idx)
        data.append(1.0)

        for l in range(N):
            if l != j:
                key = l * N + j if l < j else j * N + l
                if key in pair_lookup:
                    rows.append(eq_idx)
                    cols.append(pair_lookup[key])
                    data.append(-0.5 * P[i, l])

            if l != i:
                key = i * N + l if i < l else l * N + i
                if key in pair_lookup:
                    rows.append(eq_idx)
                    cols.append(pair_lookup[key])
                    data.append(-0.5 * P[j, l])

    return np.array(rows), np.array(cols), np.array(data), rhs

@jit(nopython=True)
def _assemble_eta_ijk_matrix(N, P, triplets_array, triplet_lookup, eta_ij_flat):
    n_vars = len(triplets_array)
    rows, cols, data = [], [], []
    rhs = np.full(n_vars, 1.0 / 3.0)

    for eq_idx in range(n_vars):
        i, j, k = triplets_array[eq_idx]

        rows.append(eq_idx)
        cols.append(eq_idx)
        data.append(1.0)

        for l in range(N):
            for node_idx, (fixed1, fixed2, var_node, coeff_node) in enumerate([
                (j, k, l, i),
                (i, k, l, j),
                (i, j, l, k),
            ]):
                x, y, z = _sort_triple(fixed1, fixed2, var_node)
                n_unique = _count_unique(x, y, z)

                if n_unique == 3:
                    key = x * N * N + y * N + z
                    if key in triplet_lookup:
                        rows.append(eq_idx)
                        cols.append(triplet_lookup[key])
                        data.append(-1.0 / 3.0 * P[coeff_node, var_node])

                elif n_unique == 2:
                    m, n = (x, z) if x == y else (x, y)
                    eta_val = _get_eta_val(m, n, eta_ij_flat, N)
                    rhs[eq_idx] += 1.0 / 3.0 * P[coeff_node, var_node] * eta_val

    return np.array(rows), np.array(cols), np.array(data), rhs

def _solve_linear_system(A: csr_matrix, b: np.ndarray, method: SolverMethod = 'direct') -> np.ndarray:
    if method == 'direct':
        return spsolve(A, b)
    elif method == 'gmres':
        solver_func = gmres
    elif method == 'bicgstab':
        solver_func = bicgstab
    else:
        raise ValueError(f'Unknown solver method: {method}')

    solution, info = solver_func(A, b, rtol=1e-8)

    if info != 0:
        warnings.warn(f'{method.upper()} did not converge (info={info}), using direct solver')
        return spsolve(A, b)

    return solution

def _expand_eta_ij(solution_vec: np.ndarray, pairs_array: np.ndarray, N: int) -> np.ndarray:
    eta_ij = np.zeros((N, N))
    for idx, (i, j) in enumerate(pairs_array):
        eta_ij[i, j] = eta_ij[j, i] = solution_vec[idx]
    return eta_ij

def _expand_eta_ijk(solution_vec: np.ndarray, triplets_array: np.ndarray, eta_ij: np.ndarray, N: int) -> np.ndarray:
    eta_ijk = np.zeros((N, N, N))

    for idx, (i, j, k) in enumerate(triplets_array):
        value = solution_vec[idx]
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            eta_ijk[perm] = value

    for d in range(N):
        eta_ijk[d, d, :] = eta_ij[d, :]
        eta_ijk[d, :, d] = eta_ij[d, :]
        eta_ijk[:, d, d] = eta_ij[:, d]

    return eta_ijk

def _solve_eta_ij_core(
    P_dense: np.ndarray, N: int, method: SolverMethod,
    pairs_array: np.ndarray, pair_lookup: Dict,
) -> np.ndarray:
    rows, cols, data, rhs = _assemble_eta_ij_matrix(N, P_dense, pairs_array, pair_lookup)
    A = csr_matrix((data, (rows, cols)), shape=(len(pairs_array), len(pairs_array)))
    return _expand_eta_ij(_solve_linear_system(A, rhs, method), pairs_array, N)

def _solve_eta_ijk_core(
    P_dense: np.ndarray, N: int, eta_ij: np.ndarray, method: SolverMethod,
    triplets_array: np.ndarray, triplet_lookup: Dict,
) -> np.ndarray:
    rows, cols, data, rhs = _assemble_eta_ijk_matrix(
        N, P_dense, triplets_array, triplet_lookup, eta_ij.flatten()
    )
    A = csr_matrix((data, (rows, cols)), shape=(len(triplets_array), len(triplets_array)))
    return _expand_eta_ijk(_solve_linear_system(A, rhs, method), triplets_array, eta_ij, N)

def compute_transition_matrix(adj_matrix: np.ndarray, k_steps: int = 1) -> csr_matrix:
    A = csr_matrix(adj_matrix)
    degrees = np.array(A.sum(axis=1)).flatten()
    P = A.toarray() / degrees[:, np.newaxis]
    return csr_matrix(matrix_power(P, k_steps))

def solve_eta_ij(P: SparseMatrix, N: int, method: SolverMethod = 'direct') -> np.ndarray:
    pairs_array, pair_lookup, _, _ = _build_lookup_tables(N)
    P_dense = P.toarray() if hasattr(P, 'toarray') else P
    return _solve_eta_ij_core(P_dense, N, method, pairs_array, pair_lookup)

def solve_eta_ijk(P: SparseMatrix, N: int, eta_ij: np.ndarray, method: SolverMethod = 'direct') -> np.ndarray:
    _, _, triplets_array, triplet_lookup = _build_lookup_tables(N)
    P_dense = P.toarray() if hasattr(P, 'toarray') else P
    return _solve_eta_ijk_core(P_dense, N, eta_ij, method, triplets_array, triplet_lookup)

def solve_eta_system(
    adj_matrix: np.ndarray,
    k_steps: int = 1,
    solver_method: SolverMethod = 'direct',
    check_connectivity: bool = True,
    verbose: bool = False,
) -> EtaSystemResult:
    N = len(adj_matrix)

    if check_connectivity:
        G = nx.from_numpy_array(adj_matrix)
        if not nx.is_connected(G):
            raise ValueError(
                f'Graph must be connected. Found {nx.number_connected_components(G)} components.'
            )

    if verbose:
        print(f'Solving eta system for N={N} nodes...')

    P = compute_transition_matrix(adj_matrix, k_steps)
    P_dense = P.toarray() if hasattr(P, 'toarray') else P
    pairs_array, pair_lookup, triplets_array, triplet_lookup = _build_lookup_tables(N)

    eta_ij = _solve_eta_ij_core(P_dense, N, solver_method, pairs_array, pair_lookup)
    eta_ijk = _solve_eta_ijk_core(P_dense, N, eta_ij, solver_method, triplets_array, triplet_lookup)

    return {
        'eta_ij': eta_ij,
        'eta_ijk': eta_ijk,
        'transition_matrix': P_dense,
    }

def compute_sum_pij_eta_ij_einsum(P: SparseMatrix, eta_ij: np.ndarray, m: int = 1) -> float:
    N = P.shape[0]
    P_dense = P.toarray() if hasattr(P, 'toarray') else P
    P_m = matrix_power(P_dense, m)
    return np.einsum('ij,ij->', P_m, eta_ij) / N

def compute_sum_pij_pjk_eta_ijk_einsum(P: SparseMatrix, eta_ijk: np.ndarray, m: int = 1, n: int = 1) -> float:
    N = P.shape[0]
    P_dense = P.toarray() if hasattr(P, 'toarray') else P
    P_m = matrix_power(P_dense, m)
    P_n = matrix_power(P_dense, n)
    return np.einsum('ij,jk,ijk->', P_m, P_n, eta_ijk) / N

def compute_c_values(
    network: Union[np.ndarray, nx.Graph],
    k_steps: int = 1,
    solver_method: SolverMethod = 'gmres',
) -> PyDict[str, float]:
    if isinstance(network, nx.Graph):
        adj_matrix = nx.to_numpy_array(network)
    else:
        adj_matrix = network

    result = solve_eta_system(adj_matrix, k_steps=k_steps, solver_method=solver_method)
    P = result['transition_matrix']
    eta_ij = result['eta_ij']
    eta_ijk = result['eta_ijk']

    f1 = compute_sum_pij_eta_ij_einsum(P, eta_ij, m=1)
    f2 = compute_sum_pij_eta_ij_einsum(P, eta_ij, m=2)
    f3 = compute_sum_pij_eta_ij_einsum(P, eta_ij, m=3)
    g21 = compute_sum_pij_pjk_eta_ijk_einsum(P, eta_ijk, m=2, n=1)

    return {
        'K_xx': f2 + f3 - g21,
        'K_xy': g21 - f3,
        'K_yx': f1 + f2 - g21,
        'K_yy': g21 - f1,
    }
