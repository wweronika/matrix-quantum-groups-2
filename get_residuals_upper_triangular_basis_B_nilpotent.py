import sympy as sp
import os
import pickle
from util import simplify_q_uv_obj, load_pickle, save_pickle

# Define output directory
outdir = "output"
os.makedirs(outdir, exist_ok=True)

A_u1 = load_pickle(os.path.join(outdir, f"A_u1.pkl"))
B_u1 = load_pickle(os.path.join(outdir, f"B_u1.pkl"))
C_u1 = load_pickle(os.path.join(outdir, f"C_u1.pkl"))
D_u1 = load_pickle(os.path.join(outdir, f"D_u1.pkl"))


def check_case(name):
    print(f"\n===== CASE {name} =====")

    w_X = load_pickle(os.path.join(outdir, f"w_{name}_reduced.pkl"))
    u_X = load_pickle(os.path.join(outdir, f"u_{name}_reduced.pkl"))
    v_X = load_pickle(os.path.join(outdir, f"v_{name}_reduced.pkl"))

    # symbolic residuals
    r1 = B_u1 @ w_X
    r2 = B_u1 @ u_X - w_X
    r3 = B_u1 @ v_X - u_X

    save_pickle(r1, f"r1_{name}")
    save_pickle(r2, f"r2_{name}")
    save_pickle(r3, f"r3_{name}")

    r1_reduced = simplify_q_uv_obj(r1)
    r2_reduced = simplify_q_uv_obj(r2)
    r3_reduced = simplify_q_uv_obj(r3)

    save_pickle(r1_reduced, f"r1_reduced_{name}")
    save_pickle(r2_reduced, f"r2_reduced_{name}")
    save_pickle(r3_reduced, f"r3_reduced_{name}")


# --- run all cases ---
# for name in ["1", "q", "q2"]:
#     check_case(name)

# r3_reduced = load_pickle(os.path.join(outdir, f"r2_reduced_1.pkl"))
# sp.pprint(r3_reduced)