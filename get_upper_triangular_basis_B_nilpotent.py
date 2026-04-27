import sympy as sp
import os
import pickle
from util import get_ABCD, save_pickle, simplify_q_uv_obj

# Global symbolic variables
x_1, x_2, y_1, y_2, z_1, z_2 = sp.symbols("x_1 x_2 y_1 y_2 z_1 z_2")
X, q, k, d = sp.symbols("X q k d")
u, v = sp.symbols('u v')

# Define output directory
outdir = "output"
os.makedirs(outdir, exist_ok=True)

subs_dict = {
    x_1: u,
    y_1: -v,
    z_1: v,
    x_2: u,
    y_2: v,
    z_2: -v,
    X: q

}

A, B, C, D = get_ABCD()

A_u1 = A.subs(subs_dict)
B_u1 = B.subs(subs_dict)
C_u1 = C.subs(subs_dict)
D_u1 = D.subs(subs_dict)
save_pickle(A_u1, "A_u1", outdir)
save_pickle(B_u1, "B_u1", outdir)
save_pickle(C_u1, "C_u1", outdir)
save_pickle(D_u1, "D_u1", outdir)

# Vectors spanning ker B in U(1) subgroup
k_1 = sp.Matrix([0, u**2 * v**2, 0, v**2/u**2, q**2*v**4/u**2, q* v**6/u**2, -q*v**4, 0, v**2])
k_2 = sp.Matrix([0, 0, u**2*v**2, q**2*v**6/u**2, q*v**2/u**2, v**4/u**2, q**2*v**2, -v**4, 0])
k_3 = sp.Matrix([u**2*v**2, 0, 0, q*v**4/u**2, v**6/u**2, q**2*v**2/u**2, 0, q*v**2, -q**2*v**4])

# Projectors onto A-eigenspaces in U(1) subgroup
I = sp.eye(9)
P_1 = (I + A_u1 + A_u1**2) / 3
P_q = (I + q**2 * A_u1 + q* A_u1**2) / 3
P_q_2 = (I + q * A_u1 + q**2 * A_u1**2) / 3

# Choose vector in ker(B)]
k = k_1

# Project vector in ker(B) onto A-eigenspaces
w_1 = P_1 * k_1
w_q = P_q * k_1
w_q_2 = P_q_2 * k_1

# Calculate the remaining basis vectors from Bu = w, Bv = u (i.e. compute preimages of w under B twice)
w_Xs = [w_1, w_q, w_q_2]
names = ["1", "q", "q2"]

for name, w_X in zip(names, w_Xs):

    save_pickle(w_X, f"w_{name}", outdir)

    u_X_set = sp.linsolve((B_u1, w_X))
    u_X_tuple = next(iter(u_X_set))
    u_X = sp.Matrix(u_X_tuple)

    save_pickle(u_X, f"u_{name}", outdir)

    v_X_set = sp.linsolve((B_u1, u_X))
    v_X_tuple = next(iter(v_X_set))
    v_X = sp.Matrix(v_X_tuple)

    save_pickle(v_X, f"v_{name}", outdir)

    w_X_reduced = simplify_q_uv_obj(w_X)
    save_pickle(w_X_reduced, f"w_{name}_reduced", outdir)
    u_X_reduced = simplify_q_uv_obj(u_X)
    save_pickle(u_X_reduced, f"u_{name}_reduced", outdir)
    v_X_reduced = simplify_q_uv_obj(v_X)
    save_pickle(v_X_reduced, f"v_{name}_reduced", outdir)




