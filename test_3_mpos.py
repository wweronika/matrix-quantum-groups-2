import sympy as sp
import os
import pickle
import random
from simultaneous_block_diagonalisation_27x27 import simultaneous_block_diag_9x3
from math import sin, cos

x_1, x_2, x_3, y_1, y_2, y_3, z_1, z_2, z_3 = sp.symbols("x_1 x_2 x_3 y_1 y_2 y_3 z_1 z_2 z_3")
X, q, k, d = sp.symbols("X q k d")
u, v = sp.symbols('u v')

def get_ABCD():

    sigma = sp.Matrix([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]])
    I_q = sp.Matrix([[1, 0, 0],
                    [0, q, 0],
                    [0, 0, q**2]])
    I_q_inv = sp.Matrix([[1, 0, 0],
                    [0, q**(-1), 0],
                    [0, 0, q**(-2)]])
    a_1 = x_1 * I_q
    a_2 = x_2 * I_q
    a_3 = x_3 * I_q

    b_1 = y_1 * sigma
    b_2 = y_2 * sigma
    b_3 = y_3 * sigma

    c_1 = z_1 * sigma
    c_2 = z_2 * sigma
    c_3 = z_3 * sigma

    d_1 = I_q_inv * 1/x_1 + y_1 * z_1 / x_1 * (sigma @ I_q_inv @ sigma)
    d_2 = I_q_inv * 1/x_2 + y_2 * z_2 / x_2 * (sigma @ I_q_inv @ sigma)
    d_3 = I_q_inv * 1/x_3 + y_3 * z_3 / x_3 * (sigma @ I_q_inv @ sigma)

    A = sp.tensorproduct(a_1, a_2, a_3) + sp.tensorproduct(b_1, c_2, a_3) + sp.tensorproduct(a_1, b_2, c_3) + sp.tensorproduct(b_1, d_2, c_3)
    A = sp.permutedims(A, (0,2,4,1,3,5)).reshape(27, 27)
    A = sp.Matrix(A.tolist())

    B = sp.tensorproduct(a_1, a_2, b_3) + sp.tensorproduct(b_1, c_2, b_3) + sp.tensorproduct(a_1, b_2, d_3) + sp.tensorproduct(b_1, d_2, d_3)
    B = sp.permutedims(B, (0,2,4,1,3,5)).reshape(27, 27)
    B = sp.Matrix(B.tolist())

    C = sp.tensorproduct(c_1, a_2, a_3) + sp.tensorproduct(d_1, c_2, a_3) + sp.tensorproduct(c_1, b_2, c_3) + sp.tensorproduct(d_1, d_2, c_3)
    C = sp.permutedims(C, (0,2,4,1,3,5)).reshape(27, 27)
    C = sp.Matrix(C.tolist())

    D = sp.tensorproduct(c_1, a_2, b_3) + sp.tensorproduct(d_1, c_2, b_3) + sp.tensorproduct(c_1, b_2, d_3) + sp.tensorproduct(d_1, d_2, d_3)
    D = sp.permutedims(D, (0,2,4,1,3,5)).reshape(27, 27)
    D = sp.Matrix(D.tolist())
    
    return A, B, C, D

random.seed(0)

subs_dict = {
    x_1: random.randint(-10, 10),
    x_2: random.randint(-10, 10),
    x_3: random.randint(-10, 10),
    y_1: random.randint(-10, 10),
    y_2: random.randint(-10, 10),
    y_3: random.randint(-10, 10),
    z_1: random.randint(-10, 10),
    z_2: random.randint(-10, 10),
    z_3: random.randint(-10, 10),
    q: sp.exp(2*sp.pi*sp.I / 3)
}

theta = 2
d_num = 3
subs_dict = {
    x_1: cos(theta)**(1/d_num),
    x_2: cos(theta)**(1/d_num),
    x_3: random.randint(-10, 10),
    y_1: sin(theta)**(1/d_num),
    y_2: -sin(theta)**(1/d_num),
    y_3: random.randint(-10, 10),
    z_1: -sin(theta)**(1/d_num),
    z_2: sin(theta)**(1/d_num),
    z_3: random.randint(-10, 10),
    q: sp.exp(2*sp.pi*sp.I / 3)
}
subs_dict = {
    x_1: cos(theta)**(1/d_num),
    x_2: cos(theta)**(1/d_num),
    x_3: 1,
    y_1: sin(theta)**(1/d_num),
    y_2: -sin(theta)**(1/d_num),
    y_3: 1,
    z_1: -sin(theta)**(1/d_num),
    z_2: sin(theta)**(1/d_num),
    z_3: 1,
    q: sp.exp(2*sp.pi*sp.I / 3)
}

A, B, C, D = get_ABCD()
A_num = A.subs(subs_dict)
B_num = B.subs(subs_dict)
C_num = C.subs(subs_dict)
D_num = D.subs(subs_dict)

A_numpy = sp.lambdify((), A_num, modules="numpy")()
B_numpy = sp.lambdify((), B_num, modules="numpy")()
C_numpy = sp.lambdify((), C_num, modules="numpy")()
D_numpy = sp.lambdify((), D_num, modules="numpy")()

res = simultaneous_block_diag_9x3(A_numpy, B_numpy, C_numpy, D_numpy, trials=500, verify_tol=1e-8)

if res.success:
    print(res.message)
    S = res.S
    Ablk = res.A_block
    Bblk = res.B_block
    Cblk = res.C_block
    Dblk = res.D_block
else:
    print(res.message)

