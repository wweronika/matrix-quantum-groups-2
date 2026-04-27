import sympy as sp
import os
import pickle
from util import load_pickle, save_pickle, simplify_q_uv_obj

# Define output directory
outdir = "output"
os.makedirs(outdir, exist_ok=True)

w_1 = load_pickle(os.path.join(outdir, f"w_1_reduced.pkl"))
w_q = load_pickle(os.path.join(outdir, f"w_q_reduced.pkl"))
w_q2 = load_pickle(os.path.join(outdir, f"w_q2_reduced.pkl"))
u_1 = load_pickle(os.path.join(outdir, f"u_1_reduced.pkl"))
u_q = load_pickle(os.path.join(outdir, f"u_q_reduced.pkl"))
u_q2 = load_pickle(os.path.join(outdir, f"u_q2_reduced.pkl"))
v_1 = load_pickle(os.path.join(outdir, f"v_1_reduced.pkl"))
v_q = load_pickle(os.path.join(outdir, f"v_q_reduced.pkl"))
v_q2 = load_pickle(os.path.join(outdir, f"v_q2_reduced.pkl"))

A_u1 = load_pickle(os.path.join(outdir, f"A_u1.pkl"))
B_u1 = load_pickle(os.path.join(outdir, f"B_u1.pkl"))
C_u1 = load_pickle(os.path.join(outdir, f"C_u1.pkl"))
D_u1 = load_pickle(os.path.join(outdir, f"D_u1.pkl"))

basis = [w_1, u_1, v_1, w_q, u_q, v_q, w_q2, u_q2, v_q2]

S = sp.Matrix.hstack(*basis)
save_pickle(S, "S", outdir)

S_inv = S.inv()
print("S inv computed")
save_pickle(S_inv, "S_inv", outdir)
S_inv_reduced = simplify_q_uv_obj(S_inv, use_uv_relation=True)
save_pickle(S_inv_reduced, "S_inv_reduced", outdir)

A_new_basis = S_inv @ A_u1 @ S
save_pickle(A_new_basis, "A_new_basis", outdir)
A_new_basis_reduced = simplify_q_uv_obj(A_new_basis)
save_pickle(A_new_basis_reduced, "A_new_basis_reduced", outdir)

B_new_basis = S_inv @ B_u1 @ S
save_pickle(B_new_basis, "B_new_basis", outdir)
B_new_basis_reduced = simplify_q_uv_obj(B_new_basis)
save_pickle(B_new_basis_reduced, "B_new_basis_reduced", outdir)

C_new_basis = S_inv @ C_u1 @ S
save_pickle(C_new_basis, "C_new_basis", outdir)
C_new_basis_reduced = simplify_q_uv_obj(C_new_basis)
save_pickle(C_new_basis_reduced, "C_new_basis_reduced", outdir)

D_new_basis = S_inv @ D_u1 @ S
save_pickle(D_new_basis, "D_new_basis", outdir)
D_new_basis_reduced = simplify_q_uv_obj(D_new_basis)
save_pickle(D_new_basis_reduced, "D_new_basis_reduced", outdir)
