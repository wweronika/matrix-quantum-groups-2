import sympy as sp
import os
import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_pickle(var, var_name, outdir):
    with open(os.path.join(outdir, f"{var_name}.pkl"), "wb") as f:
        pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {var_name}.pkl")

def simplify_q_uv_scalar(expr, use_uv_relation=False):

    u, v, q = sp.symbols('u v q')
    polys = None

    if use_uv_relation:
        polys = [q**2 + q + 1, u**6 + v**6 - 1]
    else:
        polys = [q**2 + q + 1]

    G_QUV = sp.groebner(
    polys,
    q, u, v,
    order='grevlex',
    domain=sp.QQ
    )

    expr = sp.sympify(expr)

    if expr in (sp.nan, sp.zoo, sp.oo, -sp.oo):
        return expr

    # Normalize into a single rational expression first.
    expr = sp.cancel(sp.together(expr))

    num, den = sp.fraction(expr)
    num = sp.expand(sp.expand_power_base(num, force=True))
    den = sp.expand(sp.expand_power_base(den, force=True))

    # Only do Groebner reduction when both pieces are polynomial.
    if not num.is_polynomial(q, u, v):
        return expr
    if not den.is_polynomial(q, u, v):
        return expr

    num_red = G_QUV.reduce(num)[1]
    den_red = G_QUV.reduce(den)[1]

    if den_red == 0:
        return expr

    expr = sp.cancel(num_red / den_red)

    # Crucial final pass: recombine and reduce again.
    expr = sp.cancel(sp.together(expr))
    num2, den2 = sp.fraction(expr)
    num2 = sp.expand(num2)
    den2 = sp.expand(den2)

    if num2.is_polynomial(q, u, v) and den2.is_polynomial(q, u, v):
        num2_red = G_QUV.reduce(num2)[1]
        den2_red = G_QUV.reduce(den2)[1]
        if den2_red != 0:
            expr = sp.cancel(num2_red / den2_red)

    return expr

def simplify_q_uv_obj(obj, use_uv_relation=False):
    if isinstance(obj, sp.MatrixBase):
        return obj.applyfunc(lambda x: simplify_q_uv_scalar(x, use_uv_relation))
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            simplify_q_uv_obj(x, use_uv_relation) for x in obj
        )
    return simplify_q_uv_scalar(obj, use_uv_relation)

#  Simplification using a custom sp.Groebner object 
# (can specify polynomials, variables, order)
def simplify_q_uv_scalar_any_G(expr, G):

    # print(f"simplifying {expr}")

    vars = G.gens

    expr = sp.sympify(expr)

    if expr in (sp.nan, sp.zoo, sp.oo, -sp.oo):
        return expr

    # Normalize into a single rational expression first.
    expr = sp.cancel(sp.together(expr))

    num, den = sp.fraction(expr)
    num = sp.expand(sp.expand_power_base(num, force=True))
    den = sp.expand(sp.expand_power_base(den, force=True))

    # Only do Groebner reduction when both pieces are polynomial.
    if not num.is_polynomial(*vars):
        return expr
    if not den.is_polynomial(*vars):
        return expr

    num_red = G.reduce(num)[1]
    den_red = G.reduce(den)[1]

    if den_red == 0:
        return expr

    expr = sp.cancel(num_red / den_red)

    # Crucial final pass: recombine and reduce again.
    expr = sp.cancel(sp.together(expr))
    num2, den2 = sp.fraction(expr)
    num2 = sp.expand(num2)
    den2 = sp.expand(den2)

    if num2.is_polynomial(*vars) and den2.is_polynomial(*vars):
        num2_red = G.reduce(num2)[1]
        den2_red = G.reduce(den2)[1]
        if den2_red != 0:
            expr = sp.cancel(num2_red / den2_red)

    return expr

#  Simplification using a custom sp.Groebner object 
# (can specify polynomials, variables, order)
def simplify_q_uv_obj_any_G(obj, G):
    if isinstance(obj, sp.MatrixBase):
        return obj.applyfunc(lambda x: simplify_q_uv_scalar_any_G(x, G))
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            simplify_q_uv_obj(x, G) for x in obj
        )
    return simplify_q_uv_scalar(obj, G)

def permutation_matrix(order):
    n = len(order)
    P = sp.zeros(n)
    for new_idx, old_idx in enumerate(order):
        P[old_idx, new_idx] = 1
    return P

def permute_basis(M, order):
    # order uses 0-based indices
    P = permutation_matrix(order)
    return P.T * M * P

def get_non_zero_entries_in_columns(M):
    result = []
    for j in range(M.shape[1]):
        ids = []
        for i in range(M.shape[0]):
            if M[i, j] != 0:
                ids.append(i)
        result.append(ids)
    return result

def get_ABCD():

    x_1, x_2, y_1, y_2, z_1, z_2 = sp.symbols("x_1 x_2 y_1 y_2 z_1 z_2")
    X, q, k, d = sp.symbols("X q k d")

    sigma = sp.Matrix([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]])
    I_q = sp.Matrix([[1, 0, 0],
                    [0, q, 0],
                    [0, 0, q**2]])
    I_q_inv = sp.Matrix([[1, 0, 0],
                    [0, q**(-1), 0],
                    [0, 0, q**(-2)]])
    a_1 = sp.Matrix([[x_1, 0, 0],
                    [0, q * x_1, 0],
                    [0, 0, q**2 * x_1]])
    a_2 = sp.Matrix([[x_2, 0, 0],
                    [0, q * x_2, 0],
                    [0, 0, q**2 * x_2]])
    b_1 = sp.Matrix([[0, y_1, 0],
                    [0, 0, y_1],
                    [y_1, 0, 0]])
    b_2 = sp.Matrix([[0, y_2, 0],
                    [0, 0, y_2],
                    [y_2, 0, 0]])
    c_1 = sp.Matrix([[0, z_1, 0],
                    [0, 0, z_1],
                    [z_1, 0, 0]])
    c_2 = sp.Matrix([[0, z_2, 0],
                    [0, 0, z_2],
                    [z_2, 0, 0]])
    d_1 = I_q_inv * 1/x_1 + y_1 * z_1 / x_1 * (sigma @ I_q_inv @ sigma)
    d_2 = I_q_inv * 1/x_2 + y_2 * z_2 / x_2 * (sigma @ I_q_inv @ sigma)

    A = sp.permutedims(sp.tensorproduct(a_1, a_2) + sp.tensorproduct(b_1, c_2), (0,2,1,3)).reshape(9, 9)
    A = sp.Matrix(A.tolist())

    B = sp.permutedims(sp.tensorproduct(a_1, b_2) + sp.tensorproduct(b_1, d_2), (0,2,1,3)).reshape(9, 9)
    B = sp.Matrix(B.tolist())

    C = sp.permutedims(sp.tensorproduct(c_1, a_2) + sp.tensorproduct(d_1, c_2), (0,2,1,3)).reshape(9, 9)
    C = sp.Matrix(C.tolist())

    D = sp.permutedims(sp.tensorproduct(c_1, b_2) + sp.tensorproduct(d_1, d_2), (0,2,1,3)).reshape(9, 9)
    D = sp.Matrix(D.tolist())
    
    return A, B, C, D

def subs_ABCD(A, B, C, D, subs_dict):
    A_num = A.subs(subs_dict)
    B_num = B.subs(subs_dict)
    C_num = C.subs(subs_dict)
    D_num = D.subs(subs_dict)

    A_numpy = sp.lambdify((), A_num, modules="numpy")()
    B_numpy = sp.lambdify((), B_num, modules="numpy")()
    C_numpy = sp.lambdify((), C_num, modules="numpy")()
    D_numpy = sp.lambdify((), D_num, modules="numpy")()

    return A_numpy, B_numpy, C_numpy, D_numpy
