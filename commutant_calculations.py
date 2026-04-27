from util import load_pickle
import os
import numpy as np
from scipy.linalg import block_diag
np.set_printoptions(linewidth=np.inf)

def chop_into_blocks(M):
    M1 = M[0:3, 0:3]
    M2 = M[3:6, 3:6]
    M3 = M[6:9, 6:9]
    return M1, M2, M3


def chop_into_blocks_B(M):
    M1 = M[0:3, 3:6]
    M2 = M[3:6, 6:9]
    M3 = M[6:9, 0:3]
    return M1, M2, M3

def chop_into_blocks_C(M):
    M1 = M[3:6, 0:3]
    M2 = M[6:9, 3:6]
    M3 = M[0:3, 6:9]
    return M1, M2, M3

def solve_for_K(T, B_simple, C_simple, tol=1e-10):
    I3 = np.eye(3, dtype=complex)
    I9 = np.eye(9, dtype=complex)

    # Correct convention for your ansatz X = T^{-1}(K⊗I)T
    B_tilde = T @ B_simple @ np.linalg.inv(T)
    C_tilde = T @ C_simple @ np.linalg.inv(T)

    # Basis for vec(K ⊗ I)
    basis = []
    for j in range(3):
        for i in range(3):
            Eij = np.zeros((3, 3), dtype=complex)
            Eij[i, j] = 1.0
            basis.append(np.kron(Eij, I3).reshape(-1, order='F'))
    M = np.column_stack(basis)   # 81 x 9

    # Commutator operators
    KB = np.kron(B_tilde.T, I9) - np.kron(I9, B_tilde)
    KC = np.kron(C_tilde.T, I9) - np.kron(I9, C_tilde)

    # Restrict to Y = K ⊗ I
    A = np.vstack([KB @ M, KC @ M])   # 162 x 9

    U, S, Vh = np.linalg.svd(A)
    # Relative tolerance is safer than absolute
    rtol = tol * S[0] if len(S) > 0 else tol
    rank = np.sum(S > rtol)
    null_basis =  Vh[rank:].conj().T

    K_basis = [null_basis[:, k].reshape(3, 3, order='F')
               for k in range(null_basis.shape[1])]

    return K_basis, A, B_tilde, C_tilde, S

import numpy as np

def proportionality_constant(A, B, tol=1e-10):
    """
    Check if A = lambda * B for some scalar lambda.

    Returns:
        is_proportional (bool)
        lambda (complex or None)
        residual (float)
    """

    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)

    # Frobenius inner product
    inner_BB = np.vdot(B, B)   # = sum conj(B_ij)*B_ij

    if abs(inner_BB) < tol:
        # B is (numerically) zero
        if np.linalg.norm(A) < tol:
            return True, 0.0, 0.0
        else:
            return False, None, np.linalg.norm(A)

    lam = np.vdot(B, A) / inner_BB

    residual = np.linalg.norm(A - lam * B)

    # relative error is better
    scale = max(np.linalg.norm(A), abs(lam) * np.linalg.norm(B), 1.0)
    rel_err = residual / scale

    return rel_err < tol, lam, rel_err

def check_all_proportional(blocks, tol=1e-10):
    ref = blocks[0]
    results = []

    for i, B in enumerate(blocks):
        ok, lam, err = proportionality_constant(B, ref, tol=tol)
        results.append((i, ok, lam, err))

    return results


def check_compatibility_B_and_C(mu, nu, tol=1e-10):
    mu = np.asarray(mu, dtype=complex)
    nu = np.asarray(nu, dtype=complex)
    d = len(mu)

    results = {}
    allowed = []

    for r in range(d):
        errs = []
        ok = True
        for j in range(d):
            lhs = mu[(j - 1) % d] * nu[j]
            rhs = mu[(j + r - 1) % d] * nu[(j + r) % d]
            # lhs = mu[(r - 1) % d] * nu[(r - 1) % d]
            # rhs = mu[j] * nu[(j + 1) % d]
            err = abs(lhs - rhs)
            errs.append(err)
            if err > tol:
                ok = False
        results[r] = {"compatible": ok, "errors": errs}
        if ok:
            allowed.append(r)

    return results, allowed

def print_compatibility_report(mu, nu, tol=1e-10):
    results, allowed = check_compatibility_B_and_C(mu, nu, tol=tol)

    print("allowed shifts:", allowed)
    for r, data in results.items():
        print(f"\nr = {r}")
        print("compatible:", data["compatible"])
        print("errors:", [f"{e:.3e}" for e in data["errors"]])

    return results, allowed

outdir = "output_commutant_calcs"

A_simple = load_pickle(os.path.join(outdir, f"A_simple.pkl"))
B_simple = load_pickle(os.path.join(outdir, f"B_simple.pkl"))
C_simple = load_pickle(os.path.join(outdir, f"C_simple.pkl"))
D_simple = load_pickle(os.path.join(outdir, f"D_simple.pkl"))



A0, A1, A2 = chop_into_blocks(A_simple) 
B0, B1, B2 = chop_into_blocks_B(B_simple) 
D0, D1, D2 = chop_into_blocks(D_simple) 
A_blocks = [A0, A1, A2]
B_blocks = [B0, B1, B2]
D_blocks = [D0, D1, D2]
# print(D1)
# for Ai in A_blocks:
#     print(Ai[0, 1] * Ai[1, 2] * Ai[2, 0])

# for Di in D_blocks:
#     print(Di[0, 2] * Di[1, 0] * Di[2, 1])

# for i in range(3):
#     print(A0[i, (i+1)%3] * D0[(i+1)%3, i])

s0 = 1
s1 = A1[0, 1] / A0[0, 1]
s2 = s1 * A1[1, 2] / A0[1, 2]
S0 = np.diag([s0, s1, s2])

s0 = 1
s1 = A2[0, 1] / A1[0, 1]
s2 = s1 * A2[1, 2] / A1[1, 2]
S1 = np.diag([s0, s1, s2])

s0 = 1
s1 = A0[0, 1] / A2[0, 1]
s2 = s1 * A0[1, 2] / A2[1, 2]
S2 = np.diag([s0, s1, s2])

S_blocks = [S0, S1, S2]
R0 = np.eye(3, dtype=complex)
R1 = S0
R2 = S0 @ S1

T = block_diag(R0, R1, R2)
TBT = T @ B_simple @ np.linalg.inv(T)
TCT  =  T @ C_simple @ np.linalg.inv(T)
TBT_blocks = chop_into_blocks_B(TBT)
TCT_blocks = chop_into_blocks_C(TCT)
# print(np.round(TBT, 5))
# print(TBT[0,3] / TBT[3,6])
# print(TBT[1,4] / TBT[4,7])
# print(TBT[2,5] / TBT[5,8])
# print(TCT[3,0] / TCT[6,3])
# print(TCT[4,1] / TCT[7,4])
# print(TCT[5,2] / TCT[8,5])
mu = []
nu = []
for i in range(3):
    ok, lam, err = proportionality_constant(TBT_blocks[i], TBT_blocks[0])
    print(i, ok, lam, err)
    nu.append(lam)
for i in range(3):
    ok, lam, err = proportionality_constant(TCT_blocks[i], TCT_blocks[0])
    print(i, ok, lam, err)
    mu.append(lam)

print(mu, nu)

# results, allowed = print_compatibility_report(mu, nu, tol=1e-8)

# mu = np.array([
#     1.0 + 0j,
#     -0.4698508647186603 - 0.834601808117218j,
#     -0.49710332878394164 + 0.8320723730573122j
# ], dtype=complex)

# nu = np.array([
#     1.0 + 0j,
#     -0.5121995137607969 + 0.9098262287066182j,
#     -0.5291394674839264 - 0.8856958038576996j
# ], dtype=complex)

print(check_compatibility_B_and_C(mu, nu))                 # probably only [0]
print(check_compatibility_B_and_C(mu, np.roll(nu, -1)))    # should give [0,1,2]

exit()
C = np.random.rand(3,3)
K = np.diag(np.random.rand(3))
X = np.linalg.inv(T) @ (np.kron(C, np.eye(3))) @ T
X = np.linalg.inv(T) @ (np.kron(K, np.eye(3))) @ T

exit()

K_basis, A, B_tilde, C_tilde, S = solve_for_K(T, B_simple, C_simple)
print("singular values =", S)
print("dim solution space =", len(K_basis))
for idx, K in enumerate(K_basis):
    Y = np.kron(K, np.eye(3))
    errB = np.linalg.norm(Y @ B_tilde - B_tilde @ Y)
    errC = np.linalg.norm(Y @ C_tilde - C_tilde @ Y)
    X = np.linalg.inv(T) @ Y @ T
    errB_orig = np.linalg.norm(X @ B_simple - B_simple @ X)
    errC_orig = np.linalg.norm(X @ C_simple - C_simple @ X)
    print(f"\nBasis {idx}")
    print("errB_tilde =", errB)
    print("errC_tilde =", errC)
    print("errB_orig  =", errB_orig)
    print("errC_orig  =", errC_orig)
    print(np.round(K, 6))

Ks = K_basis

# identify scalar part
for i, K in enumerate(Ks):
    lam = np.trace(K)/3
    print(i, np.linalg.norm(K - lam*np.eye(3)))

# Express products in the basis
B = np.column_stack([K.reshape(-1, order='F') for K in Ks])

for i, Ki in enumerate(Ks):
    for j, Kj in enumerate(Ks):
        prod = (Ki @ Kj).reshape(-1, order='F')
        coeffs, *_ = np.linalg.lstsq(B, prod, rcond=None)
        recon = sum(coeffs[a] * Ks[a] for a in range(len(Ks)))
        err = np.linalg.norm(Ki @ Kj - recon)
        print(f"{i},{j}: err={err:.3e}, coeffs={coeffs}")

from itertools import product

Ks = K_basis
B = Ks

def mat_from_coeffs(c):
    return sum(c[i]*B[i] for i in range(len(B)))

# crude random search
for _ in range(20000):
    c = np.random.randn(3) + 1j*np.random.randn(3)
    P = mat_from_coeffs(c)
    # normalize scale
    nrm = np.linalg.norm(P)
    if nrm < 1e-12:
        continue
    P = P / nrm

    err_idem = np.linalg.norm(P @ P - P)
    err_zero = np.linalg.norm(P)
    err_one = np.linalg.norm(P - np.eye(3))

    if err_idem < 1e-6 and err_zero > 1e-3 and err_one > 1e-3:
        print("candidate idempotent found")
        print(P)
        print("err_idem =", err_idem)
        break

for i, K in enumerate(K_basis):
    w, V = np.linalg.eig(K)
    print(f"K{i} eigenvalues:", w)
    print("rank(V) =", np.linalg.matrix_rank(V))

for i, K in enumerate(K_basis):
    print(i, np.linalg.norm(K @ K.conj().T - K.conj().T @ K))





exit()









































print(np.linalg.norm(np.linalg.inv(R1) @ A0 @ R1 - A1))
print(np.linalg.norm(np.linalg.inv(R2) @ A0 @ R2 - A2))
print(np.linalg.norm(np.linalg.inv(R1) @ D0 @ R1 - D1))
print(np.linalg.norm(np.linalg.inv(R2) @ D0 @ R2 - D2))

# print(B_simple)

print(np.linalg.inv(S1) @ D1 @ S0)
exit()
for i in range(3):
    res_A = A_blocks[i] @ S_blocks[i] - S_blocks[i] @ A_blocks[(i+1)%3]
    res_D = D_blocks[i] @ S_blocks[i] - S_blocks[i] @ D_blocks[(i+1)%3]
    print(np.round(res_A, 8))
    print(np.round(res_D, 8))
    print(S_blocks[i])
    input()

# print(D0 @ S0 - S0 @ D1)

# print(A1 @ S0 - S0 @ A1)
# print(D1 @ S0 - S0 @ D1)

# print(A0 @ S0 - S0 @ A1)
# print(D0 @ S0 - S0 @ D1)