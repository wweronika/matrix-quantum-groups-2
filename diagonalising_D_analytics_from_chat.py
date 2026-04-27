import sympy as sp

# ============================================================
# symbols
# ============================================================
x1, x2, y1, y2, z1, z2, X = sp.symbols('x1 x2 y1 y2 z1 z2 X', nonzero=True)

# primitive cube root of unity
q = sp.exp(2 * sp.pi * sp.I / 3)

# ============================================================
# local d=3 matrices in the paper's convention
# basis = |1>, |2>, |3>
# sigma|1>=|3>, sigma|2>=|1>, sigma|3>=|2>
# I_{q^{-1}} = diag(1,q^{-1},q^{-2})
# ============================================================
sigma = sp.Matrix([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
])

Iqinv = sp.diag(1, q**(-1), q**(-2))

def d_block(x, y, z):
    return (1/x) * Iqinv + (y*z/x) * sigma * Iqinv * sigma

b2 = y2 * sigma
c1 = z1 * sigma
d1 = d_block(x1, y1, z1)
d2 = d_block(x2, y2, z2)

# fused D
D = sp.kronecker_product(c1, b2) + sp.kronecker_product(d1, d2)

# ============================================================
# ambient basis ordering:
# |1,1>, |1,2>, |1,3>, |2,1>, |2,2>, |2,3>, |3,1>, |3,2>, |3,3>
# ============================================================
def basis_ij(i, j):
    """
    i,j in {0,1,2} internally, corresponding to |i+1, j+1>
    """
    v = sp.zeros(9, 1)
    v[3*i + j, 0] = 1
    return v

# ============================================================
# sector basis:
# |m>_k = sum_{r=0}^2 q^{-k r} |r+1, ((m-r) mod 3)+1>
#
# k = 0,1,2 labels the sector
# m = 0,1,2 labels basis vectors inside the sector
# ============================================================
def sector_basis_vec(k, m):
    v = sp.zeros(9, 1)
    for r in range(3):
        s = (m - r) % 3
        v += q**(-k * r) * basis_ij(r, s)
    return v

def sector_basis_matrix(k):
    # 9x3 matrix whose columns span V_k
    return sp.Matrix.hstack(
        sector_basis_vec(k, 0),
        sector_basis_vec(k, 1),
        sector_basis_vec(k, 2),
    )

# ============================================================
# 3x3 block D_k on V_k
# Since S_k has full column rank, the block is:
#   D_k = (S^T S)^(-1) S^T D S
# ============================================================
def sector_block(k):
    S = sector_basis_matrix(k)   # 9x3
    DS = sp.expand(D * S)        # 9x3

    cols = []
    for j in range(3):
        rhs = DS[:, j]
        sol = S.gauss_jordan_solve(rhs)[0]   # 3x1 coefficient vector
        cols.append(sp.expand(sol))

    return sp.Matrix.hstack(*cols)

# ============================================================
# Solve (D_k - X I) alpha = 0 with alpha_1 = 1
#
# If alpha = (1, a2, a3)^T, use the first two rows to solve
# for a2, a3 explicitly. The third row should vanish
# automatically when X is an eigenvalue.
# ============================================================
def sector_coeffs(k):
    B = sector_block(k)
    M = sp.expand(B - X * sp.eye(3))

    # Unknown vector = [1, a2, a3]^T
    # Row equations:
    # M00 + M01 a2 + M02 a3 = 0
    # M10 + M11 a2 + M12 a3 = 0

    den = sp.expand(M[0,1]*M[1,2] - M[1,1]*M[0,2])

    a2 = sp.cancel((M[0,2]*M[1,0] - M[1,2]*M[0,0]) / den)
    a3 = sp.cancel((M[1,1]*M[0,0] - M[0,1]*M[1,0]) / den)

    return sp.Matrix([1, a2, a3]), B, M

# ============================================================
# Ambient 9-component eigenvector in sector k
# ============================================================
def D_eigvec_sector(k):
    S = sector_basis_matrix(k)
    alpha, B, M = sector_coeffs(k)
    v = sp.expand(S * alpha)
    return v, alpha, B, M

# build the 3 symbolic sector vectors
v0, alpha0, B0, M0 = D_eigvec_sector(0)
v1, alpha1, B1, M1 = D_eigvec_sector(1)
v2, alpha2, B2, M2 = D_eigvec_sector(2)

# ============================================================
# checks
# ============================================================
# 1) block-level residuals
block_res0 = sp.expand(M0 * alpha0)
block_res1 = sp.expand(M1 * alpha1)
block_res2 = sp.expand(M2 * alpha2)

# 2) ambient residuals
res0 = sp.expand(D * v0 - X * v0)
res1 = sp.expand(D * v1 - X * v1)
res2 = sp.expand(D * v2 - X * v2)

print("D_0 block =")
sp.pprint(B0)
print("\nalpha0 =")
sp.pprint(alpha0)
print("\nv0 =")
sp.pprint(v0)

print("\nblock residual k=0:")
sp.pprint(block_res0)
print("\nambient residual k=0:")
sp.pprint(res0)

print("\n" + "="*70 + "\n")

print("D_1 block =")
sp.pprint(B1)
print("\nalpha1 =")
sp.pprint(alpha1)
print("\nv1 =")
sp.pprint(v1)

print("\nblock residual k=1:")
sp.pprint(block_res1)
print("\nambient residual k=1:")
sp.pprint(res1)

print("\n" + "="*70 + "\n")

print("D_2 block =")
sp.pprint(B2)
print("\nalpha2 =")
sp.pprint(alpha2)
print("\nv2 =")
sp.pprint(v2)

print("\nblock residual k=2:")
sp.pprint(block_res2)
print("\nambient residual k=2:")
sp.pprint(res2)

char0 = sp.factor((B0 - X*sp.eye(3)).det())
char1 = sp.factor((B1 - X*sp.eye(3)).det())
char2 = sp.factor((B2 - X*sp.eye(3)).det())

print(char0)
print(char1)
print(char2)