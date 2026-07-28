"""Wahba's problem via Davenport's q-method (optimal attitude).

Solves for the quaternion maximizing q^T K q (the largest-eigenvector solution).
This is the exact q-method; QUEST is the fast approximation of the same result.
"""
import numpy as np


def davenport_q(vecs_b, vecs_i, weights=None):
    """Optimal attitude quaternion from N weighted vector pairs.

    vecs_b, vecs_i: lists/arrays of body- and inertial-frame unit vectors.
    weights: per-vector weights (default 1). Use w_k = 1/sigma_k^2.
    Returns quaternion q (scalar-first) for R_BI.
    """
    vb = np.asarray(vecs_b, float)
    vi = np.asarray(vecs_i, float)
    n = len(vb)
    w = np.ones(n) if weights is None else np.asarray(weights, float)

    # Attitude profile matrix B = sum w_k v_b,k v_i,k^T
    B = sum(w[k] * np.outer(vb[k], vi[k]) for k in range(n))
    S = B + B.T
    sigma = np.trace(B)
    z = sum(w[k] * np.cross(vb[k], vi[k]) for k in range(n))

    # Davenport K matrix (4x4)
    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = z
    K[1:, 0] = z
    K[1:, 1:] = S - sigma * np.eye(3)

    # Optimal quaternion = eigenvector of the largest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(K)
    q_dav = eigvecs[:, np.argmax(eigvals)]   # [scalar, vec] but check ordering

    # Davenport K is ordered [scalar; vector] as above, so eigvec is [qw,qx,qy,qz]
    q = q_dav / np.linalg.norm(q_dav)
    if q[0] < 0:
        q = -q
    return q
