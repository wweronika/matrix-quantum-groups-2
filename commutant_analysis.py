import numpy as np
from numpy.linalg import svd, eigvals

def commutant_basis(mats, tol=1e-10, return_residuals=False):
    """
    Compute a basis for the commutant
        { X : X M = M X for all M in mats }.

    Parameters
    ----------
    mats : list[np.ndarray]
        List of n x n matrices.
    tol : float
        Singular value threshold for nullspace detection.
    return_residuals : bool
        If True, also return max commutator residual for each basis element.

    Returns
    -------
    basis : list[np.ndarray]
        Basis matrices X_i spanning the commutant.
    residuals : list[float], optional
        max_j ||X_i M_j - M_j X_i||_F
    """
    if not mats:
        raise ValueError("mats must be nonempty")

    n = mats[0].shape[0]
    for M in mats:
        if M.shape != (n, n):
            raise ValueError("all matrices must have the same shape")

    I = np.eye(n, dtype=complex)

    # vec(XM - MX) = (I ⊗ M^T - M ⊗ I) vec(X)
    constraints = []
    for M in mats:
        K = np.kron(I, M.T) - np.kron(M, I)
        constraints.append(K)

    K_all = np.vstack(constraints)
    U, S, Vh = svd(K_all)
    tol_eff = tol * S[0]
    rank = np.sum(S > tol_eff)
    nullspace = Vh[rank:].conj().T   # columns span nullspace

    basis = []
    residuals = []
    for k in range(nullspace.shape[1]):
        x = nullspace[:, k]
        X = x.reshape((n, n), order='F')  # vec convention
        basis.append(X)

        if return_residuals:
            r = max(np.linalg.norm(X @ M - M @ X, ord='fro') for M in mats)
            residuals.append(r)

    if return_residuals:
        return basis, residuals
    return basis


def commutant_dimension(mats, tol=1e-10):
    """Return dim Comm(mats)."""
    return len(commutant_basis(mats, tol=tol))


def projector_onto_commutant(mats, tol=1e-10):
    """
    Return an orthonormal basis {Q_i} for the commutant with respect to
    the Frobenius inner product, as a list of n x n matrices.
    """
    basis = commutant_basis(mats, tol=tol)
    if not basis:
        return []

    n = basis[0].shape[0]
    B = np.column_stack([X.reshape(n*n, order='F') for X in basis])

    # Orthonormalize basis vectors in matrix space
    Q, _ = np.linalg.qr(B)
    out = []
    for j in range(Q.shape[1]):
        out.append(Q[:, j].reshape((n, n), order='F'))
    return out


def random_commutant_element(mats, tol=1e-10, seed=None):
    """
    Produce a random linear combination of commutant basis elements.
    Useful for probing eigenspaces / block structure.
    """
    rng = np.random.default_rng(seed)
    basis = commutant_basis(mats, tol=tol)
    if not basis:
        raise RuntimeError("commutant appears trivial / empty basis returned")

    coeffs = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    X = sum(c * B for c, B in zip(coeffs, basis))
    return X


def simultaneous_commutator_residual(X, mats):
    """max Frobenius norm of [X, M] over M in mats."""
    return max(np.linalg.norm(X @ M - M @ X, ord='fro') for M in mats)


def analyze_commutant(mats, tol=1e-10, name="family"):
    """
    Convenience report:
      - commutant dimension
      - residuals for basis elements
      - whether identity is in the span (it should be)
      - sample spectrum of a random commutant element
    """
    basis, residuals = commutant_basis(mats, tol=tol, return_residuals=True)
    n = mats[0].shape[0]

    print(f"Analysis for {name}")
    print(f"matrix size: {n}")
    print(f"number of matrices: {len(mats)}")
    print(f"commutant dimension: {len(basis)}")
    print(f"max basis residual: {max(residuals) if residuals else 0.0:.3e}")

    # Check whether I lies in the span of the basis
    B = np.column_stack([X.reshape(n*n, order='F') for X in basis])
    ivec = np.eye(n, dtype=complex).reshape(n*n, order='F')
    coeffs, *_ = np.linalg.lstsq(B, ivec, rcond=None)
    recon = (B @ coeffs).reshape((n, n), order='F')
    err_I = np.linalg.norm(recon - np.eye(n), ord='fro')
    print(f"identity reconstruction error from basis: {err_I:.3e}")

    # Random probe
    X = random_commutant_element(mats, tol=tol, seed=0)
    lam = eigvals(X)
    print("sample eigenvalues of a random commutant element:")
    print(lam)
    print("random commutant element:")
    print(X)

    return basis, X


# ---------- Optional: block-structure probe from the commutant ----------

def joint_eigenspace_partition(X, tol=1e-8):
    """
    Heuristic helper: cluster nearly equal eigenvalues of X.
    If the commutant is nontrivial, a generic Hermitian element often reveals
    the common block decomposition by its eigenspaces.
    """
    w = eigvals(X)
    groups = []
    used = np.zeros(len(w), dtype=bool)

    for i in range(len(w)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(w)):
            if not used[j] and abs(w[j] - w[i]) < tol:
                group.append(j)
                used[j] = True
        groups.append(group)
    return w, groups


# ---------- Example usage ----------
if __name__ == "__main__":
    # Replace these with your actual matrices

    from util import load_pickle
    import os

    outdir = "output_commutant_calcs"

    A = load_pickle(os.path.join(outdir, f"A_simple.pkl"))
    B = load_pickle(os.path.join(outdir, f"B_simple.pkl"))
    C = load_pickle(os.path.join(outdir, f"C_simple.pkl"))
    D = load_pickle(os.path.join(outdir, f"D_simple.pkl"))


    # Example:
    basis, X = analyze_commutant([A, B, C, D], tol=1e-9, name="{A,D,BC}")
    print(np.round(X @ A - A @ X, 5))
    # basis2 = analyze_commutant([A, B, C, D], tol=1e-9, name="{A,B,C,D}")

    pass