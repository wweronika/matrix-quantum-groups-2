import sympy as sp
import os
import pickle
from util import get_ABCD, load_pickle, save_pickle, simplify_q_uv_obj_any_G, permute_basis, get_non_zero_entries_in_columns, simplify_q_uv_obj
from render_latex import sympy_to_png

# Global symbolic variables
x_1, x_2, y_1, y_2, z_1, z_2 = sp.symbols("x_1 x_2 y_1 y_2 z_1 z_2")
X, q, k, d = sp.symbols("X q k d")

# Define output directory
outdir = "output_case_A_nilpotent"
os.makedirs(outdir, exist_ok=True)

A, B, C, D = get_ABCD()


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

A_reduced = simplify_q_uv_obj_any_G(A, G)
B_reduced = simplify_q_uv_obj_any_G(B, G)
C_reduced = simplify_q_uv_obj_any_G(C, G)
D_reduced = simplify_q_uv_obj_any_G(D, G)

# A2 = simplify_q_uv_obj_any_G(A @ A, G)
# A3 = simplify_q_uv_obj_any_G(A @ A @ A, G)

A2 = A @ A

w0 = A2[:, 0]
w1 = A2[:, 1]
w2 = A2[:, 2]

ws_ker_A = [w0, w1, w2]
names = ["1", "q", "q2"]

for name, w_X in zip(names, ws_ker_A):

    save_pickle(w_X, f"w_{name}", outdir)

    u_X_set = sp.linsolve((A, w_X))
    u_X_tuple = next(iter(u_X_set))
    u_X = sp.Matrix(u_X_tuple)
    print(u_X)

    save_pickle(u_X, f"u_{name}", outdir)

    v_X_set = sp.linsolve((A, u_X))
    v_X_tuple = next(iter(v_X_set))
    v_X = sp.Matrix(v_X_tuple)

    save_pickle(v_X, f"v_{name}", outdir)
    print(v_X)

    w_X_reduced = simplify_q_uv_obj_any_G(w_X, G)
    save_pickle(w_X_reduced, f"w_{name}_reduced", outdir)
    u_X_reduced = simplify_q_uv_obj_any_G(u_X, G)
    save_pickle(u_X_reduced, f"u_{name}_reduced", outdir)
    v_X_reduced = simplify_q_uv_obj_any_G(v_X, G)
    save_pickle(v_X_reduced, f"v_{name}_reduced", outdir)

print("saved basis")

w_1 = load_pickle(os.path.join(outdir, f"w_1_reduced.pkl"))
w_q = load_pickle(os.path.join(outdir, f"w_q_reduced.pkl"))
w_q2 = load_pickle(os.path.join(outdir, f"w_q2_reduced.pkl"))
u_1 = load_pickle(os.path.join(outdir, f"u_1_reduced.pkl"))
u_q = load_pickle(os.path.join(outdir, f"u_q_reduced.pkl"))
u_q2 = load_pickle(os.path.join(outdir, f"u_q2_reduced.pkl"))
v_1 = load_pickle(os.path.join(outdir, f"v_1_reduced.pkl"))
v_q = load_pickle(os.path.join(outdir, f"v_q_reduced.pkl"))
v_q2 = load_pickle(os.path.join(outdir, f"v_q2_reduced.pkl"))

basis = [w_1, u_1, v_1, w_q, u_q, v_q, w_q2, u_q2, v_q2]


S = sp.Matrix.hstack(*basis)
save_pickle(S, "S", outdir)

S_inv = S.inv()
print("S inv computed")
save_pickle(S_inv, "S_inv", outdir)
S_inv_reduced = simplify_q_uv_obj_any_G(S_inv, G)
save_pickle(S_inv_reduced, "S_inv_reduced", outdir)

A_new_basis = S_inv @ A_reduced @ S
save_pickle(A_new_basis, "A_new_basis", outdir)
A_new_basis_reduced = simplify_q_uv_obj_any_G(A_new_basis, G)
save_pickle(A_new_basis_reduced, "A_new_basis_reduced", outdir)
sp.pprint(A_new_basis_reduced)
input()

B_new_basis = S_inv @ B_reduced @ S
save_pickle(B_new_basis, "B_new_basis", outdir)
B_new_basis_reduced = simplify_q_uv_obj_any_G(B_new_basis, G)
save_pickle(B_new_basis_reduced, "B_new_basis_reduced", outdir)
sp.pprint(B_new_basis_reduced)
input()

C_new_basis = S_inv @ C_reduced @ S
save_pickle(C_new_basis, "C_new_basis", outdir)
C_new_basis_reduced = simplify_q_uv_obj_any_G(C_new_basis, G)
save_pickle(C_new_basis_reduced, "C_new_basis_reduced", outdir)
sp.pprint(C_new_basis_reduced)
input()

D_new_basis = S_inv @ D_reduced @ S
save_pickle(D_new_basis, "D_new_basis", outdir)
D_new_basis_reduced = simplify_q_uv_obj_any_G(D_new_basis, G)
save_pickle(D_new_basis_reduced, "D_new_basis_reduced", outdir)
sp.pprint(D_new_basis_reduced)
input()
