from util import load_pickle, simplify_q_uv_scalar
import sympy as sp
import os

# # Define output directory
# outdir = "output"
# os.makedirs(outdir, exist_ok=True)

M1 = load_pickle(os.path.join("output3/" f"C_new_basis_reduced.pkl"))
M2 = load_pickle(os.path.join("output/" f"C_new_basis_reduced.pkl"))
print(M1)
# M_simp = simplify_q_uv_scalar(M[0])
# sp.pprint(M_simp)
# sp.pprint(S[0, 2])