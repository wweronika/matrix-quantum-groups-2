import sympy as sp
import numpy as np
from sympy import I, sqrt, exp, pi
from scipy.linalg import eig
from util import simplify_q_uv_obj_any_G, get_ABCD, permute_basis
np.set_printoptions(linewidth=np.inf)

def solve_sector_generalized_eig(B, C, V):
    # V = np.column_stack(basis_vecs)      # full-space basis for sector
    Q, _ = np.linalg.qr(V)               # orthonormalize

    B_red = Q.conj().T @ B @ Q
    C_red = Q.conj().T @ C @ Q

    W, beta = eig(B_red, C_red)
    v_full = Q @ beta

    return W, beta, v_full, B_red, C_red

def group_equal_eigs(A, tol=1e-8):
    # A is (d^2 x d^2), assumed diagonalizable with d distinct eigenvalues,
    # each appearing with multiplicity d.
    w, V = np.linalg.eig(A)

    used = np.zeros(len(w), dtype=bool)
    groups = []

    for i, lam in enumerate(w):
        if used[i]:
            continue
        idx = np.where(np.abs(w - lam) < tol)[0]
        used[idx] = True
        groups.append(idx)

    # reorder basis: all eigenvectors for the same eigenvalue adjacent
    perm = np.concatenate(groups)
    V_reordered = V[:, perm]
    w_reordered = w[perm]

    return w_reordered, V_reordered, perm

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

R = sp.Matrix([
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
R_inv = R.inv()

v1_0 = sp.Matrix([1, 0, 0, 0, 0, 1, 0, 1, 0])
v1_1 = sp.Matrix([0, 1, 0, 1, 0, 0, 0, 0, 1])
v1_2 = sp.Matrix([0, 0, 1, 0, 1, 0, 1, 0, 0])

vq_0 = sp.Matrix([1, 0, 0, 0, 0, q**2, 0, q, 0])
vq_1 = sp.Matrix([0, 1, 0, q**2, 0, 0, 0, 0, q])
vq_2 = sp.Matrix([0, 0, 1, 0, q**2, 0, q, 0, 0])

vq2_0 = sp.Matrix([1, 0, 0, 0, 0, q, 0, q**2, 0])
vq2_1 = sp.Matrix([0, 1, 0, q, 0, 0, 0, 0, q**2])
vq2_2 = sp.Matrix([0, 0, 1, 0, q, 0, q**2, 0, 0])

vecs_1 = [v1_0, v1_1, v1_2]
vecs_q = [vq_0, vq_1, vq_2]
vecs_q2 = [vq2_0, vq2_1, vq2_2]
basis = vecs_1 + vecs_q + vecs_q2
P_AD = sp.Matrix.hstack(*basis)
P_AD_inv = P_AD.inv()
print("Calculated P_AD_inv")



A, B, C, D = get_ABCD()
PDP = simplify_q_uv_obj_any_G(P_AD_inv @ D @ P_AD, G)
D3 = simplify_q_uv_obj_any_G(D  @ D @ D, G)
# evs = PDP.eigenvals()
# for e in evs:
#     sp.pprint(simplify_q_uv_obj_any_G(e, G))

# sp.pprint(D3)
# exit()


D_reduced = simplify_q_uv_obj_any_G(D, G)
A_reduced = simplify_q_uv_obj_any_G(A, G)
B_reduced = simplify_q_uv_obj_any_G(B, G)
C_reduced = simplify_q_uv_obj_any_G(C, G)

subs_dict = {
    x_1: 1/sqrt(2) + I/2,
    y_1: I/2,
    z_1: I/2,

    x_2: 1/sqrt(2) - I/2,
    y_2: I/2,
    z_2: I/2,

    q: exp(2 * pi * I / 3)
}



# D_new = simplify_q_uv_obj_any_G(P_AD_inv @ D_reduced @ P_AD, G)
# A_new = simplify_q_uv_obj_any_G(P_AD_inv @ A_reduced @ P_AD, G)
# B_new = simplify_q_uv_obj_any_G(P_AD_inv @ B_reduced @ P_AD, G)
# C_new = simplify_q_uv_obj_any_G(P_AD_inv @ C_reduced @ P_AD, G)

# MISSION CRITICAL, the original working version used _new
# A_num = A_new.subs(subs_dict)
# B_num = B_new.subs(subs_dict)
# C_num = C_new.subs(subs_dict)
# D_num = D_new.subs(subs_dict)


A_num = A_reduced.subs(subs_dict)
B_num = B_reduced.subs(subs_dict)
C_num = C_reduced.subs(subs_dict)
D_num = D_reduced.subs(subs_dict)

A_numpy = sp.lambdify((), A_num, modules="numpy")()
B_numpy = sp.lambdify((), B_num, modules="numpy")()
C_numpy = sp.lambdify((), C_num, modules="numpy")()
D_numpy = sp.lambdify((), D_num, modules="numpy")()


eigvals_D, eigvecs_D, perm = group_equal_eigs(D_numpy)
# columns of V_reordered are the reordered basis vectors
print(eigvals_D, perm)
P = eigvecs_D
P_inv = np.linalg.inv(P)

PDP = P_inv @ D_numpy @ P
PCP = P_inv @ C_numpy @ P
PBP = P_inv @ B_numpy @ P
PAP = P_inv @ A_numpy @ P
# print(np.round(PAP, 3))

# B and C do shift D eigenspaces by *(q)
# q_num = np.exp(2*1.0j*np.pi/3)
# print(np.round(D_numpy @ B_numpy - B_numpy @ D_numpy * q_num, 8))
# print(np.round(D_numpy @ C_numpy - C_numpy @ D_numpy * q_num, 8))

# print(V_X.T.conj() @ V_X)
final_basis = []

for i in range(3):
    V_X = P[:, i*3:(i+1)*3]
    W, beta, v_full, B_red, C_red = solve_sector_generalized_eig(
        B_numpy, C_numpy, V_X
    )

    idx = np.argsort(W)
    print(W[idx])

    final_basis.append(v_full[:, idx])

S = np.hstack(final_basis)
perm = [0,3,6,1,4,7,2,5,8]
S = S[:, perm]
S = S[:, ]
S_inv = np.linalg.inv(S)
SAS = S_inv @ A_numpy @ S
SBS = S_inv @ B_numpy @ S
SCS = S_inv @ C_numpy @ S
SDS = S_inv @ D_numpy @ S
print(np.round(SAS, 5))

A_block = SAS[0:3,0:3]



