import sympy as sp
from util import get_ABCD, subs_ABCD

# Global symbolic variables
x_1, x_2, y_1, y_2, z_1, z_2 = sp.symbols("x_1 x_2 y_1 y_2 z_1 z_2")
X, q, k, d = sp.symbols("X q k d")

A, B, C, D = get_ABCD()

subs_dict = {
    x_1: 1,
    y_1: 0,
    z_1: 0,
    x_2: 0,
    y_2: 1, 
    z_2: 1
}

A_numpy, B_numpy, C_numpy, D_numpy = subs_ABCD(A, B, C, D, subs_dict) 

print(A_numpy)