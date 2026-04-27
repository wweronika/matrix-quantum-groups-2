from util import load_pickle, save_pickle, permute_basis, get_non_zero_entries_in_columns
import os
import sympy as sp

# Define output directory
outdir = "output"
os.makedirs(outdir, exist_ok=True)

A = load_pickle(os.path.join(outdir, f"A_new_basis_reduced.pkl"))
B = load_pickle(os.path.join(outdir, f"B_new_basis_reduced.pkl"))
C = load_pickle(os.path.join(outdir, f"C_new_basis_reduced.pkl"))
D = load_pickle(os.path.join(outdir, f"D_new_basis_reduced.pkl"))

# result = get_non_zero_entries_in_columns(C)
# for i, l in enumerate(result):
#     print(f"{i} : {l}")

order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
# order = [3, 6, 0, 4, 7, 1, 5, 8, 2]

A_new = permute_basis(A, order)
B_new = permute_basis(B, order)
C_new = permute_basis(C, order)
D_new = permute_basis(D, order)

save_pickle(A_new, "A_new_basis_reduced_permuted", outdir)
save_pickle(B_new, "B_new_basis_reduced_permuted", outdir)
save_pickle(C_new, "C_new_basis_reduced_permuted", outdir)
save_pickle(D_new, "D_new_basis_reduced_permuted", outdir)