import numpy as np
import scipy.linalg as la
from dataclasses import dataclass


@dataclass
class SBDResult:
    success: bool
    S: np.ndarray | None = None
    A_block: np.ndarray | None = None
    B_block: np.ndarray | None = None
    C_block: np.ndarray | None = None
    D_block: np.ndarray | None = None
    block_sizes: list[int] | None = None
    commutant_dim: int | None = None
    separator: np.ndarray | None = None
    message: str = ""


def _nullspace_svd(M: np.ndarray, tol: float | None = None):
    """
    Numerical nullspace basis from SVD.
    Returns N whose columns form a basis of ker(M).
    """
    U, s, Vh = la.svd(M, full_matrices=False)
    if tol is None:
        tol = max(M.shape) * (s[0] if len(s) else 1.0) * np.finfo(M.dtype if np.issubdtype(M.dtype, np.floating) else np.float64).eps * 1e3
    r = np.sum(s > tol)
    N = Vh[r:].conj().T
    return N, s, tol


def _joint_commutant_basis(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, tol: float | None = None):
    """
    Compute basis of {X : AX=XA, BX=XB}.

    vec convention: column-major / Fortran order.
    """
    n = A.shape[0]
    I = np.eye(n, dtype=A.dtype)

    # vec(AX - XA) = (I ⊗ A - A^T ⊗ I) vec(X)
    M1 = np.kron(I, A) - np.kron(A.T, I)
    M2 = np.kron(I, B) - np.kron(B.T, I)
    M3 = np.kron(I, C) - np.kron(C.T, I)
    M4 = np.kron(I, D) - np.kron(D.T, I)
    M = np.vstack([M1, M2, M3, M4])

    N, s, used_tol = _nullspace_svd(M, tol=tol)
    basis = [N[:, j].reshape((n, n), order="F") for j in range(N.shape[1])]
    return basis, M, s, used_tol


def _cluster_eigenvalues(vals: np.ndarray, tol: float):
    """
    Greedy clustering of eigenvalues by absolute distance.
    Returns list of lists of indices.
    """
    n = len(vals)
    unused = set(range(n))
    clusters = []

    while unused:
        i = unused.pop()
        cluster = [i]
        changed = True
        while changed:
            changed = False
            current = vals[cluster]
            add_now = []
            for j in list(unused):
                if np.min(np.abs(vals[j] - current)) <= tol:
                    add_now.append(j)
            if add_now:
                for j in add_now:
                    unused.remove(j)
                cluster.extend(add_now)
                changed = True
        clusters.append(sorted(cluster))

    # sort clusters by cluster centroid for determinism
    clusters.sort(key=lambda idxs: (np.mean(vals[idxs]).real, np.mean(vals[idxs]).imag))
    return clusters


def _orthonormal_basis(V: np.ndarray, rank_tol: float = 1e-10):
    """
    Orthonormal basis for span(V) via SVD.
    """
    U, s, Vh = la.svd(V, full_matrices=False)
    r = np.sum(s > rank_tol)
    return U[:, :r], s


def _invariant_subspace_from_cluster(X: np.ndarray, lam: complex, subspace_dim: int, tol: float):
    """
    Numerically recover eigenspace/generalized eigenspace for eigenvalue lam.
    Here we use singular vectors of (X - lam I).
    For diagonalizable separator, this is the eigenspace.
    """
    n = X.shape[0]
    M = X - lam * np.eye(n, dtype=X.dtype)
    U, s, Vh = la.svd(M, full_matrices=False)

    # Vectors corresponding to smallest singular values approximate nullspace
    idx = np.argsort(s)
    V = Vh.conj().T[:, idx[:subspace_dim]]

    Q, s2 = _orthonormal_basis(V, rank_tol=tol)
    return Q


def _offblock_fro_norm(M: np.ndarray, block_sizes: list[int]):
    n = M.shape[0]
    mask = np.ones((n, n), dtype=bool)
    start = 0
    for bs in block_sizes:
        stop = start + bs
        mask[start:stop, start:stop] = False
        start = stop
    return np.linalg.norm(M[mask])


def _blockdiag_fro_norm(M: np.ndarray, block_sizes: list[int]):
    """
    Frobenius norm of the block-diagonal part under given partition.
    """
    n = M.shape[0]
    BD = np.zeros_like(M)
    start = 0
    for bs in block_sizes:
        stop = start + bs
        BD[start:stop, start:stop] = M[start:stop, start:stop]
        start = stop
    return la.norm(BD, ord='fro')


def _relative_offblock_error(M: np.ndarray, block_sizes: list[int]):
    off = _offblock_fro_norm(M, block_sizes)
    total = la.norm(M, ord='fro')
    if total == 0:
        return off
    return off / total


def _check_commutation(A: np.ndarray, X: np.ndarray):
    return la.norm(A @ X - X @ A, ord='fro') / max(1.0, la.norm(A, ord='fro') * la.norm(X, ord='fro'))


def simultaneous_block_diag_9x3(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    *,
    commutant_tol: float | None = None,
    eig_cluster_tol: float | None = None,
    verify_tol: float = 1e-8,
    trials: int = 200,
    rng: np.random.Generator | None = None,
    use_complex_separator: bool = True,
) -> SBDResult:
    """
    Attempt simultaneous block diagonalization of A, B into exactly 9 blocks of size 3.

    Returns SBDResult(success=False, ...) on failure.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)

    if A.shape != (27, 27) or B.shape != (27, 27):
        return SBDResult(False, message="A and B must both be 27x27.")
    if A.dtype.kind not in "fc":
        A = A.astype(np.complex128)
    if B.dtype.kind not in "fc":
        B = B.astype(np.complex128)
    if C.dtype.kind not in "fc":
        C = C.astype(np.complex128)
    if D.dtype.kind not in "fc":
        D = D.astype(np.complex128)

    block_sizes = [3] * 9
    n = 27

    # 1. Compute joint commutant basis
    basis, M, singvals, used_comm_tol = _joint_commutant_basis(A, B, C, D, tol=commutant_tol)
    d = len(basis)

    if d == 0:
        return SBDResult(False, commutant_dim=0, message="Joint commutant is numerically trivial.")

    # Trivial commutant = scalars only => impossible to get 9 distinct 3D blocks
    if d == 1:
        X0 = basis[0]
        # if basis is basically identity, fail immediately
        return SBDResult(
            False,
            commutant_dim=1,
            message="Joint commutant is numerically 1-dimensional; no 9x3 simultaneous block structure detected."
        )

    # 2. Search for separator in commutant
    for t in range(trials):
        if use_complex_separator:
            alpha = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        else:
            alpha = rng.standard_normal(d)

        X = sum(alpha[j] * basis[j] for j in range(d))

        # symmetrize only if desired? no: general commutant element is fine
        # But Schur/eig on general X is OK.
        vals, vecs = la.eig(X)

        if eig_cluster_tol is None:
            tol_eig = 1e3 * la.norm(X, ord=2) * np.finfo(float).eps
            # allow a slightly looser floor
            tol_eig = max(tol_eig, 1e-10)
        else:
            tol_eig = eig_cluster_tol

        clusters = _cluster_eigenvalues(vals, tol=tol_eig)

        if len(clusters) != 9:
            continue
        if any(len(c) != 3 for c in clusters):
            continue

        # Recover the 9 invariant subspaces robustly
        blocks = []
        ok = True
        cluster_reps = []

        for c in clusters:
            lam = np.mean(vals[c])
            Q = _invariant_subspace_from_cluster(X, lam, subspace_dim=3, tol=max(verify_tol, 1e-10))
            if Q.shape[1] != 3:
                ok = False
                break
            blocks.append(Q)
            cluster_reps.append(lam)

        if not ok:
            continue

        S = np.column_stack(blocks)
        if np.linalg.matrix_rank(S) < n:
            continue

        # 3. Transform and verify
        try:
            Ahat = la.solve(S, A @ S)
            Bhat = la.solve(S, B @ S)
            Chat = la.solve(S, C @ S)
            Dhat = la.solve(S, D @ S)
        except la.LinAlgError:
            continue

        errA = _relative_offblock_error(Ahat, block_sizes)
        errB = _relative_offblock_error(Bhat, block_sizes)
        errC = _relative_offblock_error(Chat, block_sizes)
        errD = _relative_offblock_error(Dhat, block_sizes)

        # Extra sanity: separator should commute well
        commA = _check_commutation(A, X)
        commB = _check_commutation(B, X)
        commC = _check_commutation(C, X)
        commD = _check_commutation(D, X)

        if errA <= verify_tol and errB <= verify_tol:
            return SBDResult(
                success=True,
                S=S,
                A_block=Ahat,
                B_block=Bhat,
                C_block=Chat,
                D_block=Dhat,
                block_sizes=block_sizes,
                commutant_dim=d,
                separator=X,
                message=(
                    f"Success on trial {t+1}. "
                    f"commutant_dim={d}, rel_offblock(A)={errA:.3e}, "
                    f"rel_offblock(B)={errB:.3e}, rel_offblock(C)={errC:.3e}, "
                    f"rel_offblock(D)={errD:.3e}, comm_res=({commA:.3e},{commB:.3e})"
                ),
            )

    return SBDResult(
        success=False,
        block_sizes=block_sizes,
        commutant_dim=d,
        message=(
            f"Failed after {trials} trials. "
            f"commutant_dim={d}. No separator with nine 3D spectral subspaces "
            f"was verified at tolerance {verify_tol:.1e}."
        ),
    )