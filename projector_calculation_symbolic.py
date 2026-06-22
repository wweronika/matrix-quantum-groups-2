from util import get_ABCD, simplify_q_uv_obj_any_G
import numpy as np
import sympy as sp
from sympy import exp, I, pi 
np.set_printoptions(linewidth=np.inf)


def real_cuberoot(expr):
    return sp.sign(expr) * sp.Abs(expr)**sp.Rational(1, 3)

def to_numpy(object, subs_dict):
    return sp.lambdify((), object.subs(subs_dict), modules="numpy")()

def check_projectors(P1, P2, subs_dict, tol=1e-10):
    P1_np = to_numpy(P1, subs_dict)
    P2_np = to_numpy(P2, subs_dict)

    n = P1_np.shape[0]
    I = np.eye(n, dtype=np.complex128)

    checks = {
        "P1^2 - P1": P1_np @ P1_np - P1_np,
        "P2^2 - P2": P2_np @ P2_np - P2_np,
        "I - P1 - P2": I - P1_np - P2_np,
        "P1 P2": P1_np @ P2_np,
    }

    for name, M in checks.items():
        err = np.linalg.norm(M)
        print(f"\n{name}")
        print(f"norm = {err:.3e}")
        # print(np.round(M, 10))
        if err > 1e-10:
            return False

    print(f"All checks passed")
    return checks

def check_projector_ranks(P1, P2, subs_dict, tol=1e-10):
    P1_np = to_numpy(P1, subs_dict)
    P2_np = to_numpy(P2, subs_dict)

    def svd_rank(P, name):
        s = np.linalg.svd(P, compute_uv=False)
        thresh = tol * max(s[0], 1.0)
        rank = np.sum(s > thresh)

        print(f"\n{name}")
        print(f"singular values = {s}")
        print(f"threshold = {thresh:.3e}")
        print(f"rank = {rank}")

        return rank, s

    r1, s1 = svd_rank(P1_np, "P1")
    r2, s2 = svd_rank(P2_np, "P2")

    return r1 == 3 and r2 == 6

x_1, x_2, y_1, y_2, z_1, z_2 = sp.symbols("x_1 x_2 y_1 y_2 z_1 z_2")
X, q, k, d = sp.symbols("X q k d")
a, b, c = sp.symbols("a b c")

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

    q: exp(2 * pi * I / 3)
}


A, B, C, D = get_ABCD()
A = simplify_q_uv_obj_any_G(A, G)
B = simplify_q_uv_obj_any_G(B, G)
C = simplify_q_uv_obj_any_G(C, G)
D = simplify_q_uv_obj_any_G(D, G)

A_numpy = to_numpy(A, subs_dict)
B_numpy = to_numpy(B, subs_dict)
C_numpy = to_numpy(C, subs_dict)
D_numpy = to_numpy(D, subs_dict)

B2C = simplify_q_uv_obj_any_G(B @ B @ C, G)
BC2 = simplify_q_uv_obj_any_G(B @ C @ C, G)


f = (x_1**3 * x_2**3 * y_2**3 + y_1**3 * z_2**3 * y_2**3 + y_1**3) / x_2**3
g = (x_1**3 * x_2**3 * z_1**3 + y_1**3 * z_2**3 * z_1**3 + z_2**3) / x_1**3

f_root = real_cuberoot(f)
g_root = real_cuberoot(g)

f_root_numpy = to_numpy(f_root, subs_dict)
g_root_numpy = to_numpy(g_root, subs_dict)
print(f_root_numpy)
print(g_root_numpy)

# exit()

def get_P1_P2_from_omega(omega, B2C, BC2):
    alpha_1 = sp.Rational(1, 3)
    alpha_2 = sp.Rational(2, 3)

    beta_1 = omega / (3 * f_root**2 * g_root)
    gamma_1  = 3 * f * beta_1**2
    gamma_1_v2 = omega**2 / (3 * g_root**2 * f_root)
    print(f"gamma_1: {to_numpy(gamma_1, subs_dict)}, gamma_1_v2: {to_numpy(gamma_1_v2, subs_dict)}")
    # input()

    beta_2 = -omega / (3 * f_root**2 * g_root)
    gamma_2  = -3 * f * beta_2**2
    gamma_2_v2 = -omega**2 / (3 * g_root**2 * f_root)
    print(f"gamma_2: {to_numpy(gamma_2, subs_dict)}, gamma_2_v2: {to_numpy(gamma_2_v2, subs_dict)}")
    # input()

    gamma_1 = gamma_1_v2
    gamma_2 = gamma_2_v2

    P1 = alpha_1 * sp.eye(9) + beta_1 * B2C + gamma_1 * BC2
    P2 = alpha_2 * sp.eye(9) + beta_2 * B2C + gamma_2 * BC2

    return P1, P2

def get_similarity_transformation_from_projectors(P1s, tol=1e-10):
    bases = []
    for P in P1s:
        P = to_numpy(P, subs_dict)
        P = np.asarray(P, dtype=np.complex128)
        U, s, Vh = np.linalg.svd(P)
        bases.append(U[:, :3])
    T = np.hstack(bases)

    sT = np.linalg.svd(T, compute_uv=False)
    if sT[-1] < tol * max(sT[0], 1.0):
        raise ValueError(f"T is numerically singular. Singular values: {sT}")

    return T

def print_block_structure(M, block_size=3):
    M = np.asarray(M, dtype=np.complex128)

    n = M.shape[0]
    n_blocks = n // block_size

    diag_norm_sq = 0.0
    offdiag_norm_sq = 0.0

    print("Block Frobenius norms:")

    for i in range(n_blocks):
        row = []
        for j in range(n_blocks):
            block = M[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ]

            norm = np.linalg.norm(block, ord='fro')
            row.append(f"{norm:.3e}")

            if i == j:
                diag_norm_sq += norm**2
            else:
                offdiag_norm_sq += norm**2

        print("  ".join(row))

    diag_norm = np.sqrt(diag_norm_sq)
    offdiag_norm = np.sqrt(offdiag_norm_sq)

    print()
    print(f"Diagonal part norm     : {diag_norm:.6e}")
    print(f"Off-diagonal part norm : {offdiag_norm:.6e}")

    if diag_norm > 0:
        print(f"Relative off-diagonal  : {offdiag_norm/diag_norm:.6e}")

    return diag_norm, offdiag_norm



P1s = []
P2s = []
for omega in [
    1,
    sp.exp(2*sp.pi*sp.I/3),
    sp.exp(4*sp.pi*sp.I/3),
]:
    print(f"\n===== omega = {omega} =====")
    P1, P2 = get_P1_P2_from_omega(omega, B2C, BC2)
    check_passed = check_projectors(P1, P2, subs_dict)
    check_ranks_passed = check_projector_ranks(P1, P2, subs_dict)
    if check_passed and check_ranks_passed:
        P1s.append(P1)
        P2s.append(P2)

T = get_similarity_transformation_from_projectors(P1s)
T_inv = np.linalg.inv(T)


A_blk = T_inv @ A_numpy @ T
B_blk = T_inv @ B_numpy @ T
C_blk = T_inv @ C_numpy @ T
D_blk = T_inv @ D_numpy @ T

print("\n A:")
print(np.round(A_blk, 12))
print("\n B:")
print(np.round(B_blk, 12))
print("\n C:")
print(np.round(C_blk, 12))
print("\n D:")
print(np.round(D_blk, 12))

# print_block_structure(A_blk)
# print_block_structure(B_blk)
# print_block_structure(C_blk)
# print_block_structure(D_blk)











