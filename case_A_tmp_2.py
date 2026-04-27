import sympy as sp
import numpy as np
from sympy import I, sqrt, exp, pi
from scipy.linalg import eig
from util import simplify_q_uv_obj_any_G, get_ABCD, permute_basis
from commutant_analysis import *
np.set_printoptions(linewidth=np.inf)

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

a, b, c = sp.symbols("a b c")
# generic commutant entry
M_block = sp.Matrix([[0, 0, a], 
                [b, 0, 0],
                [0, c, 0]])
M = sp.Matrix(sp.permutedims(sp.tensorproduct(sp.eye(3), M_block), (0,2,1,3)).reshape(9,9))


subs_dict = {
    x_1: 1/sqrt(2) + I/2,
    y_1: I/2 - 11,
    z_1: I/2,

    x_2: 1/sqrt(2) - I/2 + 7,
    y_2: I/2 - 5,
    z_2: I/2,

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

# Commutant checks - is M in the commutant?
M_num = M.subs({a: 1, b: 2, c: 3})
M_numpy = sp.lambdify((), M_num, modules="numpy")()

# MISSION CRITICAL, the original working version used _new
A_num = A_new.subs(subs_dict)
B_num = B_new.subs(subs_dict)
C_num = C_new.subs(subs_dict)
D_num = D_new.subs(subs_dict)


A_numpy = sp.lambdify((), A_num, modules="numpy")()
B_numpy = sp.lambdify((), B_num, modules="numpy")()
C_numpy = sp.lambdify((), C_num, modules="numpy")()
D_numpy = sp.lambdify((), D_num, modules="numpy")()

BC_numpy = B_numpy @ C_numpy
eigvals_BC, eigvecs_BC = eig(BC_numpy)
# perm = [0, 5, 7, 1, 3, 8, 2, 4, 6]
# V_BC = eigvecs_BC[:, perm]
perm = [0,3,6,1,4,7,2,5,8]
V_BC = eigvecs_BC[:, perm]
# V_BC = eigvecs_BC 
# print(V_BC.shape)
# V_BC_inv = np.inv(V_BC) 
# BC_new = V_BC_inv @ BC_numpy @ V_BC

BC_new = np.linalg.solve(V_BC, BC_numpy @ V_BC)
print(np.round(BC_new, 5))
print("========== B: ==========")
B_new = np.linalg.solve(V_BC, B_numpy @ V_BC)
print(np.round(B_new, 5))
print("========== C: ==========")
C_new = np.linalg.solve(V_BC, C_numpy @ V_BC)
print(np.round(C_new, 5))
print("========== A: ==========")
A_new = np.linalg.solve(V_BC, A_numpy @ V_BC)
print(np.round(A_new, 5))
print("========== D: ==========")
D_new = np.linalg.solve(V_BC, D_numpy @ V_BC)
print(np.round(D_new, 5))

# print("Commutant checks")
# print(np.round(M_numpy @ A_new - A_new @ M_numpy, 5))
# print(np.round(M_numpy @ D_new - D_new @ M_numpy, 5))
# print(np.round(M_numpy @ B_new - B_new @ M_numpy, 5))
# print(np.round(M_numpy @ C_new - C_new @ M_numpy, 5))

print(B_new[0,1] * C_new[1,0])
print(B_new[1,2] * C_new[2,1])
print(B_new[2,0] * C_new[0,2])


# Example: choose 9 random coefficients
rng = np.random.default_rng(12345)
params = rng.standard_normal(9)

alpha1, beta1, gamma1, alpha2, beta2, gamma2, alpha3, beta3, gamma3 = params

# Extract the three 3x3 blocks
B1 = B_new[0:3, 0:3]
B2 = B_new[3:6, 3:6]
B3 = B_new[6:9, 6:9]

def commutant_block_from_abc(a, b, c, alpha, beta, gamma):
    return np.array([
        [alpha,        beta*a,      gamma*a*b],
        [gamma*b*c,    alpha,       beta*b   ],
        [beta*c,       gamma*c*a,   alpha    ]
    ], dtype=complex)

# Read off (a,b,c) from each block
a1, b1, c1 = B1[0, 1], B1[1, 2], B1[2, 0]
a2, b2, c2 = B2[0, 1], B2[1, 2], B2[2, 0]
a3, b3, c3 = B3[0, 1], B3[1, 2], B3[2, 0]

# Build the three commutant blocks
X1 = commutant_block_from_abc(a1, b1, c1, alpha1, beta1, gamma1)
X2 = commutant_block_from_abc(a2, b2, c2, alpha2, beta2, gamma2)
X3 = commutant_block_from_abc(a3, b3, c3, alpha3, beta3, gamma3)

# Assemble full 9x9 block diagonal commutant element
X = np.block([
    [X1, np.zeros((3, 3), dtype=complex), np.zeros((3, 3), dtype=complex)],
    [np.zeros((3, 3), dtype=complex), X2, np.zeros((3, 3), dtype=complex)],
    [np.zeros((3, 3), dtype=complex), np.zeros((3, 3), dtype=complex), X3]
])

# print("params =", params)
# print("[B, X] =")
# print(np.round(B_new @ X - X @ B_new, 8))
# print(np.round(C_new @ X - X @ C_new, 8))
# print(np.round(A_new @ X - X @ A_new, 8))


# big matrix M, shape (9,9)

M = A_new
# entries from the previous matrix M
a = np.array([
    -0.68732-8.36468j,
    -6.61274+3.19251j,
     6.23714+3.91948j
], dtype=complex)

b = np.array([
    -0.17867-7.55113j,
    -6.89820+4.21477j,
     6.46725+4.21486j
], dtype=complex)

# entries from D
u = np.array([
    -0.72260-2.85524j,
     2.50646+1.15048j,
    -2.21797+1.43544j
], dtype=complex)

# permutation P for (1,4,7,2,5,8,3,6,9)
perm = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
P = np.eye(9, dtype=complex)[:, perm]

# similarity matrices S_i relative to sector 1
S = []
for i in range(3):
    Si = np.diag([
        1.0 + 0j,
        a[i] / a[0],
        (a[i] * b[i]) / (a[0] * b[0])
    ])
    S.append(Si)

# choose arbitrary 9 parameters
rng = np.random.default_rng(1234)
x = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))

# build X' in the permuted basis
Xprime_blocks = []
for i in range(3):
    row = []
    Si_inv = np.linalg.inv(S[i])
    for j in range(3):
        row.append(x[i, j] * (Si_inv @ S[j]))
    Xprime_blocks.append(row)

Xprime = np.block(Xprime_blocks)

# back to original basis
X = P @ Xprime @ P.T

print(np.allclose(X @ M, M @ X))
print(X)