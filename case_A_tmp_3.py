from util import load_pickle, get_ABCD, simplify_q_uv_obj_any_G
import sympy as sp
from sympy import I, sqrt, exp, pi
import numpy as np
import os
np.set_printoptions(linewidth=np.inf)

import sympy as sp

def to_numpy(object, subs_dict):
    return sp.lambdify((), object.subs(subs_dict), modules="numpy")()

def commutant_basis_rank(B, C, include_identity=True, simplify=True):
    """
    Checks linear independence of B^2 C and B C^2,
    optionally together with the identity.

    Returns:
        rank, M
    where M has vectorised matrices as columns.
    """
    n, m = B.shape
    if n != m or C.shape != (n, n):
        raise ValueError("B and C must be square matrices of the same size.")

    S1 = B**2 * C
    S2 = B * C**2

    mats = [S1, S2]
    if include_identity:
        mats = [sp.eye(n)] + mats

    cols = []
    for X in mats:
        if simplify:
            X = X.applyfunc(sp.simplify)
        cols.append(sp.Matrix(X).reshape(n * n, 1))

    M = sp.Matrix.hstack(*cols)

    if simplify:
        M = M.applyfunc(sp.simplify)

    return M.rank(), M

x_1, x_2, y_1, y_2, z_1, z_2 = sp.symbols("x_1 x_2 y_1 y_2 z_1 z_2")
X, q, k, d = sp.symbols("X q k d")

G = sp.groebner(
    [
        q**2 + q + 1,
        # x_1*x_2 + y_1*z_2,
        # z_1*y_2 + t_1*t_2,
        # (z_1 * y_2) * (x_1 * x_2) + (1 - y_1 * z_1) *  (1 - y_2 * z_2)
    ],
    q, x_1, x_2, y_1, y_2, z_1, z_2,
    order='grevlex',
    domain=sp.QQ
)

subs_dict = {
    x_1: 1,
    y_1: 1,
    z_1: 2,

    x_2: 3,
    y_2: 4,
    z_2: -3,

    # x_1: 1,
    # y_1: 1,
    # z_1: 0,

    # x_2: 1,
    # y_2: -1,
    # z_2: 1,

    # x_1: 1,
    # y_1: 2,
    # z_1: 3,

    # x_2: 4,
    # y_2: 5,
    # z_2: 6,

    q: exp(2 * pi * I / 3)
}

# outdir = "output_case_A_nilpotent"
outdir = "output_commutant_calcs"

# A = load_pickle(os.path.join(outdir, f"A_new_basis_reduced_permuted.pkl"))
# B = load_pickle(os.path.join(outdir, f"B_new_basis_reduced_permuted.pkl"))
# C = load_pickle(os.path.join(outdir, f"C_new_basis_reduced_permuted.pkl"))
# D = load_pickle(os.path.join(outdir, f"D_new_basis_reduced_permuted.pkl"))

# A = load_pickle(os.path.join(outdir, f"A_simple.pkl"))
# B = load_pickle(os.path.join(outdir, f"B_simple.pkl"))
# C = load_pickle(os.path.join(outdir, f"C_simple.pkl"))
# D = load_pickle(os.path.join(outdir, f"D_simple.pkl"))

A, B, C, D = get_ABCD()
A = simplify_q_uv_obj_any_G(A, G)
B = simplify_q_uv_obj_any_G(B, G)
C = simplify_q_uv_obj_any_G(C, G)
D = simplify_q_uv_obj_any_G(D, G)

B2C = B @ B @ C
BC2 = B @ C @ C

A3 = simplify_q_uv_obj_any_G(A @ A @ A, G)
B3 = simplify_q_uv_obj_any_G(B @ B @ B, G)
C3 = simplify_q_uv_obj_any_G(C @ C @ C, G)


I = sp.eye(9)

f = (x_1**3 * x_2**3 * y_2**3 + y_1**3 * z_2**3 * y_2**3 + y_1**3) / x_2**3
g = (x_1**3 * x_2**3 * z_1**3 + y_1**3 * z_2**3 * z_1**3 + z_2**3) / x_1**3
h = x_1**3 * x_2**3 + y_1**3 * z_2**3
# h_num = h.subs(subs_dict)
# h_numpy = sp.lambdify((), h_num, modules="numpy")()
# print(h_numpy)
# exit()

alpha = 1/3
beta3 = 1/(27 * f**2 * g) 
gamma3 = 1/(27 * f * g**2)

# beta = beta3 ** (1/3)
gamma = gamma3 ** (1/3)
beta = 3 * g * gamma**2

P1 = alpha * I + beta * B2C + gamma * BC2

print(to_numpy(f, subs_dict))
print(to_numpy(g, subs_dict))
print(to_numpy(beta3, subs_dict))
print(to_numpy(gamma3, subs_dict))
print(to_numpy(beta, subs_dict))
print(to_numpy(gamma, subs_dict))

# exit()

alpha = 2/3
beta3 = -1/(27 * f**2 * g) 
gamma3 = -1/(27 * f * g**2)

# beta = beta3 ** (1/3)
gamma = gamma3 ** (1/3)
beta = -3 * g * gamma**2

# print(gamma)
# print(beta)
# exit()

P2 = alpha * I + beta * B2C + gamma * BC2

P1_num = P1.subs(subs_dict)
P1_numpy = sp.lambdify((), P1_num, modules="numpy")()
P2_num = P2.subs(subs_dict)
P2_numpy = sp.lambdify((), P2_num, modules="numpy")()



A_num = A.subs(subs_dict)
B_num = B.subs(subs_dict)
C_num = C.subs(subs_dict)
D_num = D.subs(subs_dict)
A_numpy = sp.lambdify((), A_num, modules="numpy")()
B_numpy = sp.lambdify((), B_num, modules="numpy")()
C_numpy = sp.lambdify((), C_num, modules="numpy")()
D_numpy = sp.lambdify((), D_num, modules="numpy")()

# print(np.round(P1_numpy @ P1_numpy - P1_numpy, 8))
# print(np.round(P1_numpy @ A_numpy - A_numpy @ P1_numpy, 8))
# print(np.round(P1_numpy @ B_numpy - B_numpy @ P1_numpy, 8))
# print(np.round(P1_numpy @ C_numpy - C_numpy @ P1_numpy, 8))
# print(np.round(P1_numpy @ D_numpy - D_numpy @ P1_numpy, 8))

rank, M = commutant_basis_rank(B_numpy, C_numpy, include_identity=True, simplify=False)
M_np = np.array(M.tolist(), dtype=np.complex128)
s = np.linalg.svd(M_np, compute_uv=False)
print(s)

s = np.linalg.svd(P1_numpy, compute_uv=False)
print(s)

P2complement = np.eye(9) - P2_numpy
# P2complement = P2_numpy
s = np.linalg.svd(P2complement, compute_uv=False)
print(s)

P3_numpy = np.eye(9) - P1_numpy - P2complement
s = np.linalg.svd(P3_numpy, compute_uv=False)
print(s)

# print(np.round(P2complement @ P3_numpy, 6))

import numpy as np

def basis_from_projector(P, rank=3, tol=1e-10):
    """
    Returns a numerical basis for im(P).

    For an exact projector, the nonzero singular directions span im(P).
    """
    P = np.asarray(P, dtype=np.complex128)

    U, s, Vh = np.linalg.svd(P)

    # Robust numerical rank check
    r = np.sum(s > tol * max(s[0], 1.0))

    if r != rank:
        raise ValueError(f"Expected rank {rank}, got numerical rank {r}. Singular values: {s}")

    return U[:, :rank]


def block_diagonalising_basis_from_projectors(P1, P2, P3, rank_each=3, tol=1e-10):
    """
    Given three pairwise orthogonal rank-3 projectors on a 9-dim space,
    returns T = [basis(im P1) | basis(im P2) | basis(im P3)].

    Then T^{-1} X T should be block diagonal for every X commuting with all P_i.
    """
    Ps = [P1, P2, P3]
    Ps = [np.asarray(P, dtype=np.complex128) for P in Ps]

    n = Ps[0].shape[0]

    for i, P in enumerate(Ps):
        if P.shape != (n, n):
            raise ValueError(f"P{i+1} has shape {P.shape}, expected {(n, n)}")

        idem_err = np.linalg.norm(P @ P - P)
        if idem_err > tol:
            raise ValueError(f"P{i+1} is not idempotent. Error: {idem_err}")

    for i in range(3):
        for j in range(i + 1, 3):
            orth_err = np.linalg.norm(Ps[i] @ Ps[j])
            if orth_err > tol:
                raise ValueError(f"P{i+1} P{j+1} is not zero. Error: {orth_err}")

    bases = [basis_from_projector(P, rank=rank_each, tol=tol) for P in Ps]

    T = np.hstack(bases)

    if T.shape != (n, 3 * rank_each):
        raise ValueError(f"T has shape {T.shape}, expected {(n, 3 * rank_each)}")

    sT = np.linalg.svd(T, compute_uv=False)
    if sT[-1] < tol * max(sT[0], 1.0):
        raise ValueError(f"T is numerically singular. Singular values: {sT}")

    return T

T = block_diagonalising_basis_from_projectors(P1_numpy, P2complement, P3_numpy)

s = np.linalg.svd(T, compute_uv=False)
print(s)
print(np.round(np.linalg.inv(T) @ T - np.eye(9), 12))
exit()

A_blk = np.linalg.inv(T) @ A_numpy @ T
B_blk = np.linalg.inv(T) @ B_numpy @ T
C_blk = np.linalg.inv(T) @ C_numpy @ T
D_blk = np.linalg.inv(T) @ D_numpy @ T



print(np.round(A_blk, 3))
# print(np.round(B_blk, 8))
# print(np.round(C_blk, 8))
# print(np.round(D_blk, 8))

# print(np.round(A_numpy @ A_numpy @ A_numpy, 8))
print(np.round(A_blk @ A_blk @ A_blk, 8))
# print(np.round(D_numpy @ D_numpy @ D_numpy, 8))

# CONCLUSION: still block diagonalisable if D is not nilpotent even if A







# sp.pprint(A)
# sp.pprint(B)
# sp.pprint(C)
# sp.pprint(D)



# print(np.round(A, 4))
# print(np.round(B, 4))
# print(np.round(C, 4))
# print(np.round(D, 4))

# print(sp.latex(A))
# print(sp.latex(B))
# print(sp.latex(C))
# print(sp.latex(D))