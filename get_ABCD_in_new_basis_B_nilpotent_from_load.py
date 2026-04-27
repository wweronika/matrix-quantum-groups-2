from util import load_pickle, save_pickle, simplify_q_uv_obj
import os
import sympy as sp

# Define output directory
outdir = "output"
os.makedirs(outdir, exist_ok=True)

A_u1 = load_pickle(os.path.join(outdir, f"A_u1.pkl"))
B_u1 = load_pickle(os.path.join(outdir, f"B_u1.pkl"))
C_u1 = load_pickle(os.path.join(outdir, f"C_u1.pkl"))
D_u1 = load_pickle(os.path.join(outdir, f"D_u1.pkl"))

outdir = "output3"
os.makedirs(outdir, exist_ok=True)

S = load_pickle(os.path.join(outdir, f"S.pkl"))
S_inv = load_pickle(os.path.join(outdir, f"S_inv.pkl"))

A_new_basis = S_inv @ A_u1 @ S
save_pickle(A_new_basis, "A_new_basis", outdir)
A_new_basis_reduced = simplify_q_uv_obj(A_new_basis, use_uv_relation=True)
save_pickle(A_new_basis_reduced, "A_new_basis_reduced", outdir)

B_new_basis = S_inv @ B_u1 @ S
save_pickle(B_new_basis, "B_new_basis", outdir)
B_new_basis_reduced = simplify_q_uv_obj(B_new_basis, use_uv_relation=True)
save_pickle(B_new_basis_reduced, "B_new_basis_reduced", outdir)

C_new_basis = S_inv @ C_u1 @ S
save_pickle(C_new_basis, "C_new_basis", outdir)
C_new_basis_reduced = simplify_q_uv_obj(C_new_basis, use_uv_relation=True)
save_pickle(C_new_basis_reduced, "C_new_basis_reduced", outdir)

D_new_basis = S_inv @ D_u1 @ S
save_pickle(D_new_basis, "D_new_basis", outdir)
D_new_basis_reduced = simplify_q_uv_obj(D_new_basis, use_uv_relation=True)
save_pickle(D_new_basis_reduced, "D_new_basis_reduced", outdir)
