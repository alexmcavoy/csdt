from sympy import symbols, Matrix, simplify, factor, expand, series, solve, Eq
from itertools import product

pCD, pDC, pDD = symbols('p_CD p_DC p_DD', real=True)
eps, s = symbols('epsilon s', positive=True)
R, S, T, P = symbols('R S T P', real=True)

def transition_matrix(p, q):
	pCC, pCD, pDC, pDD = p
	qCC, qCD, qDC, qDD = q
	return Matrix([
		[pCC*qCC, pCC*(1-qCC), (1-pCC)*qCC, (1-pCC)*(1-qCC)],
		[pCD*qDC, pCD*(1-qDC), (1-pCD)*qDC, (1-pCD)*(1-qDC)],
		[pDC*qCD, pDC*(1-qCD), (1-pDC)*qCD, (1-pDC)*(1-qCD)],
		[pDD*qDD, pDD*(1-qDD), (1-pDD)*qDD, (1-pDD)*(1-qDD)]
	])

def press_dyson_det(M, f):
	A = (M - Matrix.eye(4)).copy()
	for i in range(4):
		A[i, 3] = f[i]
	return A.det()

def tremble(p):
	return p * (1 - eps) + (1 - p) * eps

def get_selection(x, q, use_trembling=False):
	if use_trembling:
		x = tuple(tremble(xi) for xi in x)
		q = tuple(tremble(qi) for qi in q)
	M = transition_matrix(x, q)
	D = press_dyson_det(M, [1, 1, 1, 1])
	num = [press_dyson_det(M, [int(i==j) for j in range(4)]) for i in range(4)]
	s_xy_num = R*num[0] + S*num[1] + T*num[2] + P*num[3]
	s_yx_num = R*num[0] + T*num[1] + S*num[2] + P*num[3]
	N = (1 + s) * (R * D - s_yx_num) - s * (R * D - s_xy_num)
	if use_trembling:
		return simplify(series(expand(N) / expand(D), eps, 0, n=1).removeO())
	return simplify(N), simplify(D)

def analyze_case(x, case_name):
	N_1, _ = get_selection(x, (0,0,0,0))
	N_2, _ = get_selection(x, (0,1,1,1))

	print(f'\n{case_name}')
	print('-' * len(case_name))
	print(f'\nN_1 [q = (0,0,0,0)] = {factor(N_1)}')
	print(f'N_2 [q = (0,1,1,1)] = {factor(N_2)}')
	print('\nnumerator = a * N_1 + b * N_2')
	print('-' * 29)

	for q in product([0, 1], repeat=4):
		N, D = get_selection(x, q)

		if D == 0:
			sel_cond = get_selection(x, q, use_trembling=True)
			print(f'\nq = {q} [trembling]')
			print(f'  (1+s)*(R - s_yx) - s*(R - s_xy) = {factor(sel_cond)}')
			continue

		if simplify(N) == 0:
			print(f'\nq = {q}')
			print(f'  a = 0\n  b = 0\n  D = {factor(D)}')
			continue

		a, b = symbols('a b')
		eqs = []
		for var in [R, S, T, P]:
			c_t, c_e = expand(N).coeff(var), expand(a * N_1 + b * N_2).coeff(var)
			if c_t != 0 or c_e != 0:
				eqs.append(Eq(c_t, c_e))
				if len(eqs) >= 2:
					break

		if len(eqs) < 2:
			print(f'\nq = {q}\n  insufficient equations')
			continue

		sol = solve(eqs, [a, b])
		if not sol:
			print(f'\nq = {q}\n  no solution')
			continue

		a_val, b_val = simplify(sol[a]), simplify(sol[b])
		if simplify(expand(N - a_val * N_1 - b_val * N_2)) != 0:
			print(f'\nq = {q}\n  decomposition failed')
			continue

		print(f'\nq = {q}')
		print(f'  a = {a_val}\n  b = {b_val}\n  D = {factor(D)}')

if __name__ == '__main__':
	analyze_case((1, pCD, pDC, pDD), 'general: pCD, pDC, pDD in (0,1)')
	analyze_case((1, pCD, 0, pDD), 'case pDC = 0')
	analyze_case((1, 0, pDC, pDD), 'case pCD = 0')
	analyze_case((1, 0, 0, pDD), 'case pDC = 0, pCD = 0')
	analyze_case((1, pCD, 1, pDD), 'case pDC = 1')
	analyze_case((1, 0, 1, pDD), 'case pDC = 1, pCD = 0')
