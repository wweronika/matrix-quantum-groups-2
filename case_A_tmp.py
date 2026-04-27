import sympy as sp
import numpy as np
from sympy import I, sqrt, exp, pi
from scipy.linalg import eig
from util import simplify_q_uv_obj_any_G, get_ABCD, permute_basis, save_pickle
np.set_printoptions(linewidth=np.inf)
from scipy.linalg import null_space

def solve_intertwiner(A1, A2, tol=1e-10):
    n = A1.shape[0]
    
    K = np.kron(A1.T, np.eye(n)) - np.kron(np.eye(n), A2)
    
    ns = null_space(K)  # columns are basis vectors of solution space
    
    # reshape each solution vector into matrix form
    solutions = [v.reshape(n, n) for v in ns.T]
    
    return solutions

def simultaneous_intertwiners(A_list, D_list, rcond=1e-12):
    """
    Solve S A_i = D_i S for all i.

    A_list: list of (n,n) arrays
    D_list: list of (m,m) arrays

    Returns:
        basis: list of (m,n) matrices forming a basis of the solution space
    """
    if len(A_list) != len(D_list):
        raise ValueError("A_list and D_list must have the same length")
    if len(A_list) == 0:
        raise ValueError("Need at least one equation")

    n = A_list[0].shape[0]
    m = D_list[0].shape[0]

    for A in A_list:
        if A.shape != (n, n):
            raise ValueError("All A_i must have the same square shape")
    for D in D_list:
        if D.shape != (m, m):
            raise ValueError("All D_i must have the same square shape")

    blocks = []
    I_n = np.eye(n)
    I_m = np.eye(m)

    for A, D in zip(A_list, D_list):
        K = np.kron(A.T, I_m) - np.kron(I_n, D)
        blocks.append(K)

    M = np.vstack(blocks)
    ns = null_space(M, rcond=rcond)

    return [ns[:, j].reshape(m, n) for j in range(ns.shape[1])]

def reorder_basis_matrix(A, perm):
    perm = np.asarray(perm)
    return A[np.ix_(perm, perm)]


def solve_in_span_mod_groebner(v: sp.Matrix, basis_vecs, groeb, simplify_fn):
    """
    Solve A*c = v modulo the relations encoded by groeb/simplify_fn.

    Parameters
    ----------
    v : sp.Matrix
        Column vector.
    basis_vecs : list[sp.Matrix] | tuple[sp.Matrix] | sp.Matrix
        Spanning vectors. If a matrix is passed, its columns are used.
    groeb : sympy Groebner basis
        Passed only for external consistency/documentation.
    simplify_fn : callable
        Example: lambda expr: simplify_q_uv_obj_any_G(expr, G)

    Returns
    -------
    dict with keys:
        "in_span" : bool | None
        "coeffs" : sp.Matrix | None
        "residual" : sp.Matrix | None
        "basis_matrix" : sp.Matrix
        "basis_matrix_reduced" : sp.Matrix
        "target_reduced" : sp.Matrix
        "rank" : int
    """
    v = sp.Matrix(v)
    if v.cols != 1:
        raise ValueError("v must be a column vector")

    if isinstance(basis_vecs, sp.Matrix):
        A = sp.Matrix(basis_vecs)
    else:
        basis_vecs = [sp.Matrix(b) for b in basis_vecs]
        if len(basis_vecs) == 0:
            A = sp.zeros(v.rows, 0)
        else:
            if any(b.cols != 1 for b in basis_vecs):
                raise ValueError("All spanning vectors must be column vectors")
            if any(b.rows != v.rows for b in basis_vecs):
                raise ValueError("All vectors must have the same dimension as v")
            A = sp.Matrix.hstack(*basis_vecs)

    if A.rows != v.rows:
        raise ValueError("Basis vectors must have the same dimension as v")

    # Reduce basis and target BEFORE solving
    A_red = simplify_fn(A)
    v_red = simplify_fn(v)

    # Solve reduced system
    try:
        sol = A_red.gauss_jordan_solve(v_red)
        coeffs = sol[0] if isinstance(sol, tuple) else sol
    except ValueError:
        return {
            "in_span": False,
            "coeffs": None,
            "residual": None,
            "basis_matrix": A,
            "basis_matrix_reduced": A_red,
            "target_reduced": v_red,
            "rank": A_red.rank(),
        }

    coeffs = simplify_fn(coeffs)
    residual = simplify_fn(v_red - A_red * coeffs)

    zero_flags = [entry.equals(0) for entry in residual]
    if all(z is True for z in zero_flags):
        in_span = True
    elif any(z is False for z in zero_flags):
        in_span = False
    else:
        in_span = None

    return {
        "in_span": in_span,
        "coeffs": coeffs,
        "residual": residual,
        "basis_matrix": A,
        "basis_matrix_reduced": A_red,
        "target_reduced": v_red,
        "rank": A_red.rank(),
    }


# Global symbolic variables
x_1, x_2, y_1, y_2, z_1, z_2, lam = sp.symbols("x_1 x_2 y_1 y_2 z_1 z_2 lam")
X, q, k, d = sp.symbols("X q k d")

outdir = "output_commutant_calcs"

G = sp.groebner(
[
    q**2 + q + 1,
    # x_1*x_2 + y_1*z_2,
    # z_1*y_2 + t_1*t_2,
    # (z_1 * y_2) * (x_1 * x_2) + (1 - y_1 * z_1) *  (1 - y_2 * z_2)
],
q, x_1, x_2, y_1, y_2, z_1, z_2, lam,
order='grevlex',
domain=sp.QQ
)

v1_0 = sp.Matrix([1, 0, 0, 0, 0, 1, 0, 1, 0])
v1_1 = sp.Matrix([0, 1, 0, 1, 0, 0, 0, 0, 1])
v1_2 = sp.Matrix([0, 0, 1, 0, 1, 0, 1, 0, 0])

vq_0 = sp.Matrix([1, 0, 0, 0, 0, q**2, 0, q, 0])
vq_1 = sp.Matrix([0, 1, 0, q**2, 0, 0, 0, 0, q])
vq_2 = sp.Matrix([0, 0, 1, 0, q**2, 0, q, 0, 0])

vq2_0 = sp.Matrix([1, 0, 0, 0, 0, q, 0, q**2, 0])
vq2_1 = sp.Matrix([0, 1, 0, q, 0, 0, 0, 0, q**2])
vq2_2 = sp.Matrix([0, 0, 1, 0, q, 0, q**2, 0, 0])

A, B, C, D = get_ABCD()

S = sp.Matrix([
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,0],
])

# Checks OK i.e. S commutes with A, D
# sp.pprint(simplify_q_uv_obj_any_G(D @ S - S @ D, G))
# sp.pprint(simplify_q_uv_obj_any_G(A @ S - S @ A, G))

# Checks OK i.e. the vectors above are indeed the eigenvectors of S with appropriate eigenvalues
# sp.pprint(simplify_q_uv_obj_any_G(S @ v1_0 - 1 * v1_0, G))
# sp.pprint(simplify_q_uv_obj_any_G(S @ v1_1 - 1 * v1_1, G))
# sp.pprint(simplify_q_uv_obj_any_G(S @ v1_2 - 1 * v1_2, G))
# input()
# sp.pprint(simplify_q_uv_obj_any_G(S @ vq_0 - q * vq_0, G))
# sp.pprint(simplify_q_uv_obj_any_G(S @ vq_1 - q * vq_1, G))
# sp.pprint(simplify_q_uv_obj_any_G(S @ vq_2 - q * vq_2, G))
# input()
# sp.pprint(simplify_q_uv_obj_any_G(S @ vq2_0 - q**2 * vq2_0, G))
# sp.pprint(simplify_q_uv_obj_any_G(S @ vq2_1 - q**2 * vq2_1, G))
# sp.pprint(simplify_q_uv_obj_any_G(S @ vq2_2 - q**2 * vq2_2, G))
# input()


# sp.pprint(simplify_q_uv_obj_any_G(S @ D @ v1_0 - 1 *  D @ v1_0, G))
# sp.pprint(simplify_q_uv_obj_any_G(S @ v1_1 - 1 * v1_1, G))
# sp.pprint(simplify_q_uv_obj_any_G(S @ v1_2 - 1 * v1_2, G))

vecs_1 = [v1_0, v1_1, v1_2]
vecs_q = [vq_0, vq_1, vq_2]
vecs_q2 = [vq2_0, vq2_1, vq2_2]
basis = vecs_1 + vecs_q + vecs_q2

eigvals = [1, q, q**2]

D_reduced = simplify_q_uv_obj_any_G(D, G)
A_reduced = simplify_q_uv_obj_any_G(A, G)
B_reduced = simplify_q_uv_obj_any_G(B, G)
C_reduced = simplify_q_uv_obj_any_G(C, G)

# print(simplify_q_uv_obj_any_G(S, G))
# input()
# print(simplify_q_uv_obj_any_G(S @ A - A @ S, G))
# exit()

subs_dict = {
    # x_1: 1/sqrt(2) + I/2,
    # y_1: I/2,
    # z_1: I/2,

    # x_2: 1/sqrt(2) - I/2,
    # y_2: I/2,
    # z_2: I/2,

    x_1: 1,
    y_1: 2,
    z_1: 3,

    x_2: 4,
    y_2: 5,
    z_2: 6,

    q: exp(2 * pi * I / 3)
}

# for i in range(9):
#     u = basis[i]
#     lam = eigvals[i // 3]
#     v = simplify_q_uv_obj_any_G(S @ A_reduced @ u - lam * A_reduced @ u,  G)
#     sp.pprint(f"i={i}, A, {v == sp.zeros(9, 1)}")
#     v = simplify_q_uv_obj_any_G(S @ D_reduced @ u - lam * D_reduced @ u,  G)
#     sp.pprint(f"i={i}, D, {v == sp.zeros(9, 1)}")

# v = simplify_q_uv_obj_any_G(A_reduced @ vq_0, G)

P_AD = sp.Matrix.hstack(*basis)
P_AD_inv = P_AD.inv()
print("Calculated P_AD_inv")

D_new = simplify_q_uv_obj_any_G(P_AD_inv @ D_reduced @ P_AD, G)
A_new = simplify_q_uv_obj_any_G(P_AD_inv @ A_reduced @ P_AD, G)
B_new = simplify_q_uv_obj_any_G(P_AD_inv @ B_reduced @ P_AD, G)
C_new = simplify_q_uv_obj_any_G(P_AD_inv @ C_reduced @ P_AD, G)

# sp.pprint(simplify_q_uv_obj_any_G(B_new @ C_new, G))
# exit()

# MISSION CRITICAL, the original working version used _new
A_num = A_new.subs(subs_dict)
B_num = B_new.subs(subs_dict)
C_num = C_new.subs(subs_dict)
D_num = D_new.subs(subs_dict)


# A_num = A_reduced.subs(subs_dict)
# B_num = B_reduced.subs(subs_dict)
# C_num = C_reduced.subs(subs_dict)
# D_num = D_reduced.subs(subs_dict)

A_numpy = sp.lambdify((), A_num, modules="numpy")()
B_numpy = sp.lambdify((), B_num, modules="numpy")()
C_numpy = sp.lambdify((), C_num, modules="numpy")()
D_numpy = sp.lambdify((), D_num, modules="numpy")()

BC = B_numpy @ C_numpy
q_num = np.exp(2 * np.pi * 1.0j / 3)
# print(np.round(D_numpy @ B_numpy - q_num**(1) * B_numpy @ D_numpy, 8))

# exit()
eigvals, eigvecs = eig(BC)
Q = eigvecs 
print(eigvals)
# print(np.linalg.inv(Q) @ BC @ Q)
print()
D_simple = permute_basis(np.linalg.inv(Q) @ D_numpy @ Q, [0,1,2,3,4,5,6,7,8])
A_simple = permute_basis(np.linalg.inv(Q) @ A_numpy @ Q, [0,1,2,3,4,5,6,7,8])
B_simple = permute_basis(np.linalg.inv(Q) @ B_numpy @ Q, [0,1,2,3,4,5,6,7,8])
C_simple = permute_basis(np.linalg.inv(Q) @ C_numpy @ Q, [0,1,2,3,4,5,6,7,8])
A_simple = sp.lambdify((), A_simple, modules="numpy")()
D_simple = sp.lambdify((), D_simple, modules="numpy")()
B_simple = sp.lambdify((), B_simple, modules="numpy")()
C_simple = sp.lambdify((), C_simple, modules="numpy")()
print(A_simple)
input()
save_pickle(A_simple, "A_simple", outdir)
save_pickle(B_simple, "B_simple", outdir)
save_pickle(C_simple, "C_simple", outdir)
save_pickle(D_simple, "D_simple", outdir)
exit()
AD = A_simple @ D_simple
# print(np.round(AD, 8))
A1 = A_simple[0:3, 0:3]
A2 = A_simple[3:6, 3:6]
D1 = D_simple[0:3, 0:3]
D2 = D_simple[3:6, 3:6]
sols = simultaneous_intertwiners([A1, A2], [D1, D2])

print("dimension =", len(sols))
for S in sols:
    print(S)
# eigvals, eigvecs = eig(BC[0:3,0:3])
# print(eigvals)
# eigvals, eigvecs = eig(BC[3:6,3:6])
# print(eigvals)
# print(B_numpy)
exit()
eigvals, eigvecs = eig(D_numpy)
print(eigvals)
print(np.linalg.det(A_numpy))
exit()


D1  = D_new[:3, :3]
Dq  = D_new[3:6, 3:6]
Dq2 = D_new[6:9, 6:9]
# sp.pprint(simplify_q_uv_obj_any_G(Dq2 * y_1 * z_2, G))

A1  = A_new[:3, :3]
Aq  = A_new[3:6, 3:6]
Aq2 = A_new[6:9, 6:9]

# print(simplify_q_uv_obj_any_G(D1 @ D1 @ D1 - Dq2 @ Dq2 @ Dq2, G))
# print(simplify_q_uv_obj_any_G(Aq2 @ Aq2 @ Aq2, G))


D1_num = D1.subs(subs_dict)
Dq_num = Dq.subs(subs_dict)
Dq2_num = Dq2.subs(subs_dict)

D1_numpy = sp.lambdify((), D1_num, modules="numpy")()
Dq_numpy = sp.lambdify((), Dq_num, modules="numpy")()
Dq2_numpy = sp.lambdify((), Dq2_num, modules="numpy")()


eigvals_1, eigvecs_1 = eig(D1_numpy)
eigvals_q, eigvecs_q = eig(Dq_numpy)
eigvals_q2, eigvecs_q2 = eig(Dq2_numpy)

# print(eigvals_1)
# print(eigvals_q)
# print(eigvals_q2)
# exit()

V_full = np.block([
    [eigvecs_1,              np.zeros((3, 3), dtype=complex), np.zeros((3, 3), dtype=complex)],
    [np.zeros((3, 3), dtype=complex), eigvecs_q,              np.zeros((3, 3), dtype=complex)],
    [np.zeros((3, 3), dtype=complex), np.zeros((3, 3), dtype=complex), eigvecs_q2]
])

vs_lam = [V_full[:, 0], V_full[:, 3], V_full[:, 6]]
vs_qlam = [V_full[:, 1], V_full[:, 4], V_full[:, 7]] 
vs_q2lam = [V_full[:, 2], V_full[:, 5], V_full[:, 8]]

def is_proportional(v, w, tol=1e-10):
    """
    Check if v is proportional to w.
    Returns (bool, scalar alpha, residual).
    """
    v = np.asarray(v, dtype=complex)
    w = np.asarray(w, dtype=complex)

    # handle zero vectors explicitly
    if np.linalg.norm(v) < tol or np.linalg.norm(w) < tol:
        return False, None, np.inf

    # best scalar alpha (least squares)
    alpha = np.vdot(w, v) / np.vdot(w, w)

    residual = np.linalg.norm(v - alpha * w)

    return residual < tol, alpha, residual


def solve_sector_generalized_eig(B, C, basis_vecs):
    """
    basis_vecs: list of 3 full-space vectors spanning one sector
    Returns:
        W_vals      : generalized eigenvalues
        beta_vecs   : coefficients in the sector basis
        V_sector    : matrix with sector basis as columns
        VW_sector   : full-space generalized eigenvectors in this sector
        B_red, C_red: reduced 3x3 matrices
    """
    V_sector = np.column_stack(basis_vecs)   # shape (9,3)

    # reduced pencil on this sector
    B_red = V_sector.conj().T @ B @ V_sector
    C_red = V_sector.conj().T @ C @ V_sector

    # solve B_red beta = W C_red beta
    W_vals, beta_vecs = eig(B_red, C_red)

    # full-space vectors v^{X,W} = V_sector @ beta
    VW_sector = V_sector @ beta_vecs

    return W_vals, beta_vecs, V_sector, VW_sector, B_red, C_red

def find_proportional(target, candidates, tol=1e-10):
    """
    target: vector
    candidates: list or array of vectors
    Returns list of matches (index, alpha, residual)
    """
    matches = []
    for i, w in enumerate(candidates):
        ok, alpha, res = is_proportional(target, w, tol)
        if ok:
            matches.append((i, alpha, res))
    return matches

def in_span(v, basis_vecs, tol=1e-10):
    """
    Check whether v lies in span(basis_vecs).
    Returns:
        ok         : bool
        coeffs     : least-squares coefficients
        residual   : ||v - B @ coeffs||
        rel_resid  : residual / ||v||
    """
    V = np.column_stack(basis_vecs).astype(complex)
    v = np.asarray(v, dtype=complex)

    coeffs, *_ = np.linalg.lstsq(V, v, rcond=None)
    resid_vec = v - V @ coeffs
    residual = np.linalg.norm(resid_vec)
    rel_resid = residual / max(np.linalg.norm(v), tol)

    return rel_resid < tol, coeffs, residual, rel_resid

print(B_numpy)
print()
print(C_numpy)
for i in range(3):
    ok, coeffs, res, rel = in_span(C_numpy @ vs_lam[i], vs_qlam)
    print("lam sector", i, ok, res, rel)
# exit()

# v_sets = [vs_lam, vs_qlam, vs_q2lam]


# # old lambda-subspace basis
# V = np.column_stack(vs_lam + vs_qlam + vs_q2lam)

# # reduced matrices in old basis
# MB = V.conj().T @ B_numpy @ V
# MC = V.conj().T @ C_numpy @ V

# # generalized eigensystem
# W_vals, beta_vecs = eig(MB, MC)

# # print(W_vals)
# # input()

# # B and C in the new basis v^{X,W}
# B_new = np.linalg.solve(beta_vecs, MB @ beta_vecs)
# C_new = np.linalg.solve(beta_vecs, MC @ beta_vecs)


# res =  np.linalg.solve(C_new, B_new)

# VW_full = V @ beta_vecs

# B_full_new = np.linalg.solve(VW_full, B_numpy @ VW_full)
# C_full_new = np.linalg.solve(VW_full, C_numpy @ VW_full)
# res_full = np.linalg.solve(C_full_new, B_full_new)

# B3 = B_full_new @ B_full_new @ B_full_new
# C3 = C_full_new @ C_full_new @ C_full_new
# print(np.linalg.norm(C3[0,0] * np.eye(9) - C3))

W_lam, beta_lam, V_lam, VW_lam, B_lam, C_lam = solve_sector_generalized_eig(
    B_numpy, C_numpy, vs_lam
)

W_qlam, beta_qlam, V_qlam, VW_qlam, B_qlam, C_qlam = solve_sector_generalized_eig(
    B_numpy, C_numpy, vs_qlam
)

W_q2lam, beta_q2lam, V_q2lam, VW_q2lam, B_q2lam, C_q2lam = solve_sector_generalized_eig(
    B_numpy, C_numpy, vs_q2lam
)

for i in range(3):
    ok, coeffs, res, rel = in_span(VW_lam[:, i], vs_lam)
    print("lam sector", i, ok, res, rel)

for i in range(3):
    ok, coeffs, res, rel = in_span(VW_qlam[:, i], vs_qlam)
    print("qlam sector", i, ok, res, rel)

for i in range(3):
    ok, coeffs, res, rel = in_span(VW_q2lam[:, i], vs_q2lam)
    print("q2lam sector", i, ok, res, rel)

print(W_lam)
print(W_qlam)
print(W_qlam)
input()

B_lam_new = np.linalg.solve(beta_lam, B_lam @ beta_lam)
C_lam_new = np.linalg.solve(beta_lam, C_lam @ beta_lam)

B_qlam_new = np.linalg.solve(beta_qlam, B_qlam @ beta_qlam)
C_qlam_new = np.linalg.solve(beta_qlam, C_qlam @ beta_qlam)

B_q2lam_new = np.linalg.solve(beta_q2lam, B_q2lam @ beta_q2lam)
C_q2lam_new = np.linalg.solve(beta_q2lam, C_q2lam @ beta_q2lam)

# print(np.linalg.solve(C_lam_new, B_lam_new))
# print(np.linalg.solve(C_qlam_new, B_qlam_new))
# print(np.linalg.solve(C_q2lam_new, B_q2lam_new))

VW_full_sectorwise = np.column_stack([VW_lam, VW_qlam, VW_q2lam])
W_all = np.concatenate([W_lam, W_qlam, W_q2lam])

A_full_new = np.linalg.solve(VW_full_sectorwise, A_numpy @ VW_full_sectorwise)
D_full_new = np.linalg.solve(VW_full_sectorwise, D_numpy @ VW_full_sectorwise)
B_full_new = np.linalg.solve(VW_full_sectorwise, B_numpy @ VW_full_sectorwise)
C_full_new = np.linalg.solve(VW_full_sectorwise, C_numpy @ VW_full_sectorwise)
res_full = np.linalg.solve(C_full_new, B_full_new)

perm = [0, 5, 8, 1, 4, 6, 2, 3, 7]

VW_reordered = VW_full_sectorwise[:, perm]
B_reordered = reorder_basis_matrix(B_full_new, perm)
C_reordered = reorder_basis_matrix(C_full_new, perm)
A_reordered = reorder_basis_matrix(A_full_new, perm)
D_reordered = reorder_basis_matrix(D_full_new, perm)
# print("max offdiag abs:", np.max(np.abs(res_full - np.diag(np.diag(res_full)))))
# print("diag:", np.diag(res_full))
# print("W_all:", W_all)
# print(B_reordered)
# exit()

print("A reordered rounded")
res = np.round(A_reordered, 8)
for i in range(9):
    print(res[i, :])
print("B reordered rounded")
res = np.round(B_reordered, 8)
for i in range(9):
    print(res[i, :])
print()
print("C reordered rounded")
res = np.round(C_reordered, 8)
for i in range(9):
    print(res[i, :])
print()
print("D reordered rounded")
res = np.round(D_reordered, 8)
for i in range(9):
    print(res[i, :])

exit()
