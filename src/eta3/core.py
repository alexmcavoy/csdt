import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, gmres, bicgstab
from numpy.linalg import matrix_power
import networkx as nx
from itertools import combinations
import warnings
from numba import jit, types
from numba.typed import Dict
from typing import Literal, Union

SolverMethod = Literal["direct", "gmres", "bicgstab"]

@jit(nopython=True)
def _sort_triple(a, b, c):
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return a, b, c

@jit(nopython=True)
def _count_unique(a, b, c):
    if a == b:
        return 1 if b == c else 2
    return 2 if b == c else 3

@jit(nopython=True)
def _get_eta_val(m, n, eta_ij_flat, N):
    return eta_ij_flat[m * N + n] if m != n else 0.0

def _build_lookup_tables(N):
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pair_lookup = Dict.empty(key_type=types.int64, value_type=types.int64)
    for idx, (i, j) in enumerate(pairs):
        pair_lookup[i * N + j] = idx

    triplets = list(combinations(range(N), 3))
    triplet_lookup = Dict.empty(key_type=types.int64, value_type=types.int64)
    for idx, (i, j, k) in enumerate(triplets):
        triplet_lookup[i * N * N + j * N + k] = idx

    return (
        np.array(pairs, dtype=np.int32),
        pair_lookup,
        np.array(triplets, dtype=np.int32),
        triplet_lookup,
    )

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
        diag_value = 1.0

        for l in range(N):
            for fixed1, fixed2, var_node, coeff_node in [(j, k, l, i), (i, k, l, j), (i, j, l, k)]:
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
                    diag_value += 1.0 / 3.0 * P[coeff_node, var_node] * eta_val

        rows.append(eq_idx)
        cols.append(eq_idx)
        data.append(diag_value)

    return np.array(rows), np.array(cols), np.array(data), rhs

def _solve_linear_system(A, b, method="direct"):
    if method == "direct":
        return spsolve(A, b)
    elif method == "gmres":
        solver = gmres
    elif method == "bicgstab":
        solver = bicgstab
    else:
        raise ValueError(f"Invalid solver method: {method}. Use 'direct', 'gmres', or 'bicgstab'")

    solution, info = solver(A, b, rtol=1e-8)

    if info != 0:
        warnings.warn(f"{method.upper()} failed, using direct solver")
        return spsolve(A, b)

    return solution

def _expand_eta_ij(solution_vec, pairs_array, N):
    eta_ij = np.zeros((N, N))
    for idx, (i, j) in enumerate(pairs_array):
        eta_ij[i, j] = eta_ij[j, i] = solution_vec[idx]
    return eta_ij

def _expand_eta_ijk(solution_vec, triplets_array, eta_ij, N):
    eta_ijk = np.zeros((N, N, N))

    for idx, (i, j, k) in enumerate(triplets_array):
        value = solution_vec[idx]
        for perm in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]:
            eta_ijk[perm] = value

    for i in range(N):
        for j in range(N):
            for k in range(N):
                unique = set([i, j, k])
                if len(unique) == 2:
                    m, n = list(unique)
                    eta_ijk[i, j, k] = eta_ij[m, n]
                elif len(unique) == 1:
                    eta_ijk[i, j, k] = 0.0

    return eta_ijk

def compute_transition_matrix(adj_matrix, k_steps=1):
    A = csr_matrix(adj_matrix)
    degrees = np.array(A.sum(axis=1)).flatten()
    P = A.toarray() / degrees[:, np.newaxis]
    return csr_matrix(matrix_power(P, k_steps))

def solve_eta_ij(P, N, method="direct"):
    pairs_array, pair_lookup, _, _ = _build_lookup_tables(N)
    P_dense = P.toarray() if hasattr(P, "toarray") else P

    rows, cols, data, rhs = _assemble_eta_ij_matrix(N, P_dense, pairs_array, pair_lookup)
    A = csr_matrix((data, (rows, cols)), shape=(len(pairs_array), len(pairs_array)))

    solution = _solve_linear_system(A, rhs, method)
    return _expand_eta_ij(solution, pairs_array, N)

def solve_eta_ijk(P, N, eta_ij, method="direct"):
    _, _, triplets_array, triplet_lookup = _build_lookup_tables(N)
    P_dense = P.toarray() if hasattr(P, "toarray") else P
    eta_ij_flat = eta_ij.flatten()

    rows, cols, data, rhs = _assemble_eta_ijk_matrix(
        N, P_dense, triplets_array, triplet_lookup, eta_ij_flat
    )
    A = csr_matrix((data, (rows, cols)), shape=(len(triplets_array), len(triplets_array)))

    solution = _solve_linear_system(A, rhs, method)
    return _expand_eta_ijk(solution, triplets_array, eta_ij, N)

def solve_eta_system(
    adj_matrix, k_steps=1, solver_method="direct", check_connectivity=True, verbose=True
):
    N = len(adj_matrix)

    if check_connectivity:
        G = nx.from_numpy_array(adj_matrix)
        if not nx.is_connected(G):
            raise ValueError(
                f"Graph must be connected ({nx.number_connected_components(G)} components found)"
            )

    if verbose:
        print(f"Solving eta system for N={N} nodes...")

    P = compute_transition_matrix(adj_matrix, k_steps)
    eta_ij = solve_eta_ij(P, N, solver_method)
    eta_ijk = solve_eta_ijk(P, N, eta_ij, solver_method)

    return {
        "eta_ii": np.zeros(N),
        "eta_ij": eta_ij,
        "eta_ijk": eta_ijk,
        "transition_matrix": P.toarray() if hasattr(P, "toarray") else P,
    }

def compute_sum_pij_eta_ij_einsum(P, eta_ij, m=1):
    N = P.shape[0]
    P_dense = P.toarray() if hasattr(P, "toarray") else P
    P_m = matrix_power(P_dense, m)
    return np.einsum("ij,ij->", P_m, eta_ij) / N

def compute_sum_pij_pjk_eta_ijk_einsum(P, eta_ijk, m=1, n=1):
    N = P.shape[0]
    P_dense = P.toarray() if hasattr(P, "toarray") else P
    P_m = matrix_power(P_dense, m)
    P_n = matrix_power(P_dense, n)
    return np.einsum("ij,jk,ijk->", P_m, P_n, eta_ijk) / N
