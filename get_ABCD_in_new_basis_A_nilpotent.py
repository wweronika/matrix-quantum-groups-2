import sympy as sp
import os
import pickle
from util import get_ABCD, load_pickle, save_pickle, simplify_q_uv_obj_any_G, permute_basis, get_non_zero_entries_in_columns
from render_latex import sympy_to_png

# Global symbolic variables
x_1, x_2, y_1, y_2, z_1, z_2 = sp.symbols("x_1 x_2 y_1 y_2 z_1 z_2")
X, q, k, d = sp.symbols("X q k d")

# Define output directory
outdir = "output_case_A_nilpotent"
os.makedirs(outdir, exist_ok=True)

A, B, C, D = get_ABCD()
# sp.pprint(D)
# exit()

# save_pickle(A, "A", outdir)
# save_pickle(B, "B", outdir)
# save_pickle(C, "C", outdir)
# save_pickle(D, "D", outdir)

xx = x_1*x_2

S_A = sp.Matrix([
    [xx**2,       xx,      1,       0,            0,          0,         0,             0,     0],
    [0,           0,       0,       xx**2,        xx,         1,         0,             0,     0],
    [0,           0,       0,       0,            0,          0,         xx**2,         0,     0],
    [0,           0,       0,       0,            0,          0,         xx**2*q**2,   -xx,    0],
    [xx**2,       0,       0,       0,            0,          0,         0,             0,     0],
    [0,           0,       0,       xx**2*q,      xx*(q-1),   q-1,       0,             0,     0],
    [0,           0,       0,       xx**2*q,     -xx,         0,         0,             0,     0],
    [0,           0,       0,       0,            0,          0,         xx**2,         xx,    1],
    [xx**2*q**2, -xx,      0,       0,            0,          0,         0,             0,     0],
])
# save_pickle(S_A, "S_A", outdir)

G = sp.groebner(
    [
        q**2 + q + 1,
        x_1*x_2 + y_1*z_2,
        # z_1*y_2 + t_1*t_2,
        (z_1 * y_2) * (x_1 * x_2) + (1 - y_1 * z_1) *  (1 - y_2 * z_2)
    ],
    q, x_1, x_2, y_1, y_2, z_1, z_2,
    order='grevlex',
    domain=sp.QQ
)

# ------------------------------------------------------------------
# Jordan form J = J3(0) ⊕ J3(0) ⊕ J3(0)
# ------------------------------------------------------------------
J3 = sp.Matrix([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
])
J = sp.diag(J3, J3, J3)

S_A_inv = S_A.inv()

A_new = S_A_inv * A * S_A
B_new = S_A_inv * B * S_A
C_new = S_A_inv * C * S_A
D_new = S_A_inv * D * S_A

A_new_reduced = simplify_q_uv_obj_any_G(A_new, G)
B_new_reduced = simplify_q_uv_obj_any_G(B_new, G)
C_new_reduced = simplify_q_uv_obj_any_G(C_new, G)
D_new_reduced = simplify_q_uv_obj_any_G(D_new, G)

# result = get_non_zero_entries_in_columns(D * x_1 * x_2)
# for i, r in enumerate(result):
#     print(f"{i}: {r}")
# exit()

w_1 = sp.Matrix([1, 0, 0, 0, 1, 0, 0, 0, q**2])
w_2 = sp.Matrix([0, 1, 0, 0, 0, q, q, 0, 0])
w_3 = sp.Matrix([0, 0, 1, q**2, 0, 0, 0, 1, 0])

v0 = sp.Matrix([
    1, 0, 0,
    0, 0, q**2,
    0, q, 0
])

v1 = sp.Matrix([
    0, 1, 0,
    q**2, 0, 0,
    0, 0, q
])

v2 = sp.Matrix([
    0, 0, 1,
    0, q**2, 0,
    q, 0, 0
])

P_AD = sp.Matrix([
    [1,0,0, 1,0,0, 1,0,0],
    [0,1,0, 0,1,0, 0,1,0],
    [0,0,1, 0,0,1, 0,0,1],
    [1,0,0, q**2,0,0, q,0,0],
    [0,0,1, 0,q**2,0, 0,q,0],
    [0,1,0, 0,0,q**2, 0,0,q],
    [0,0,1, 0,0,q, 0,0,q**2],
    [1,0,0, q,0,0, q**2,0,0],
    [0,1,0, 0,0,q, 0,0,q**2],
])

P_AD_inv = P_AD.inv()
save_pickle(P_AD, "P_AD", outdir)
save_pickle(P_AD_inv, "P_AD_inv", outdir)

# P_AD = load_pickle(os.path.join(outdir, f"P_AD.pkl"))

# print(x_1 * x_2 * D)
# input()
print(simplify_q_uv_obj_any_G(P_AD_inv @ A @ P_AD,  G))
input()
print(simplify_q_uv_obj_any_G(P_AD_inv @ D @ P_AD,  G))
exit()
# print(B_new_reduced)

# sympy_to_png(B_new_reduced, "img/B_new_reduced.png")
print(simplify_q_uv_obj_any_G(A @ D - q * B @ C, G)[0, 1])
exit()
# AD_comm = A_new_reduced @ D_new_reduced - D_new_reduced @ A_new_reduced
AD_comm = A @ D - D @ A
print(simplify_q_uv_obj_any_G(B @ C, G))
exit()
print(simplify_q_uv_obj_any_G(D @ w_1, G))
print()
input()
print(simplify_q_uv_obj_any_G(D @ w_2, G))
print()
input()
print(simplify_q_uv_obj_any_G(D @ w_3, G))
# print(simplify_q_uv_obj_any_G(AD_comm, G))
exit()

# print(A_new_reduced)
# sp.pprint(B_new_reduced)
# input()
# sp.pprint(C_new_reduced)
# input()
# sp.pprint(D_new_reduced)

# diff = simplify_q_uv_obj_any_G(A_new - J, G)
# print(diff == sp.zeros(9))

order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
# order = [3, 6, 0, 4, 7, 1, 5, 8, 2]

A_new_reduced_permuted = permute_basis(A_new_reduced, order)
B_new_reduced_permuted = permute_basis(B_new_reduced, order)
C_new_reduced_permuted = permute_basis(C_new_reduced, order)
D_new_reduced_permuted = permute_basis(D_new_reduced, order)

print(D_new_reduced_permuted)

# sp.pprint(A_new_reduced_permuted)
# input()
# sp.pprint(B_new_reduced_permuted)
# input()
# sp.pprint(C_new_reduced_permuted)
# input()
# sp.pprint(D_new_reduced_permuted)
# input()

# save_pickle(A_new_reduced_permuted, "A_new_basis_reduced_permuted", outdir)
# save_pickle(B_new_reduced_permuted, "B_new_basis_reduced_permuted", outdir)
# save_pickle(C_new_reduced_permuted, "C_new_basis_reduced_permuted", outdir)
# save_pickle(D_new_reduced_permuted, "D_new_basis_reduced_permuted", outdir)
