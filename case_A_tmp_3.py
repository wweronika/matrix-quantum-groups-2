from util import load_pickle
import sympy as sp
import numpy as np
import os
np.set_printoptions(linewidth=np.inf)

# outdir = "output_case_A_nilpotent"
outdir = "output_commutant_calcs"

# A = load_pickle(os.path.join(outdir, f"A_new_basis_reduced_permuted.pkl"))
# B = load_pickle(os.path.join(outdir, f"B_new_basis_reduced_permuted.pkl"))
# C = load_pickle(os.path.join(outdir, f"C_new_basis_reduced_permuted.pkl"))
# D = load_pickle(os.path.join(outdir, f"D_new_basis_reduced_permuted.pkl"))

A = load_pickle(os.path.join(outdir, f"A_simple.pkl"))
B = load_pickle(os.path.join(outdir, f"B_simple.pkl"))
C = load_pickle(os.path.join(outdir, f"C_simple.pkl"))
D = load_pickle(os.path.join(outdir, f"D_simple.pkl"))

# sp.pprint(A)
# sp.pprint(B)
# sp.pprint(C)
# sp.pprint(D)

print(np.round(A, 4))
print(np.round(B, 4))
print(np.round(C, 4))
print(np.round(D, 4))

# print(sp.latex(A))
# print(sp.latex(B))
# print(sp.latex(C))
# print(sp.latex(D))