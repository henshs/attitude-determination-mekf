"""Quaternion algebra and conversions.

Convention: scalar-first, q = [qw, qx, qy, qz], unit norm.
q represents the rotation from INERTIAL to BODY (q_BI), so that
    v_body = dcm(q) @ v_inertial.
Hamilton product used throughout.
"""
import numpy as np


def quat_mult(p, q):
    """Hamilton product p (x) q, both scalar-first [w,x,y,z]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_conj(q):
    """Conjugate = inverse for a unit quaternion (negate vector part)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    return np.asarray(q, float) / np.linalg.norm(q)


def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_axis_angle(axis, angle):
    """Unit quaternion for rotation `angle` [rad] about unit `axis`."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle/2), *(axis * np.sin(angle/2))])


def quat_to_axis_angle(q):
    q = quat_normalize(q)
    angle = 2 * np.arccos(np.clip(q[0], -1, 1))
    s = np.sqrt(max(1 - q[0]**2, 0.0))
    axis = q[1:] / s if s > 1e-12 else np.array([1.0, 0, 0])
    return axis, angle


def dcm(q):
    """Rotation matrix R_BI (inertial->body) from quaternion."""
    qw, qx, qy, qz = quat_normalize(q)
    return np.array([
        [qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy+qw*qz),         2*(qx*qz-qw*qy)],
        [2*(qx*qy-qw*qz),         qw*qw-qx*qx+qy*qy-qz*qz, 2*(qy*qz+qw*qx)],
        [2*(qx*qz+qw*qy),         2*(qy*qz-qw*qx),         qw*qw-qx*qx-qy*qy+qz*qz],
    ])


def quat_from_dcm(R):
    """Quaternion from a rotation matrix (numerically robust, Shepperd).

    Our dcm(q) is R_BI = (standard active matrix)^T, so transpose here to
    match the Shepperd off-diagonal sign convention and recover q (not q*).
    """
    R = np.asarray(R, float).T
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz])
    return quat_normalize(q) * np.sign(qw if qw != 0 else 1)


def rotate_vector(q, v):
    """Rotate v from inertial to body using dcm(q)."""
    return dcm(q) @ np.asarray(v, float)


def small_angle_quat(dtheta):
    """Error quaternion from a small rotation vector dtheta."""
    q = np.array([1.0, *(0.5 * np.asarray(dtheta, float))])
    return quat_normalize(q)


def quat_error_angle(q_est, q_true):
    """Multiplicative error angle vector (Topic 20): 2*vec(q_est* (x) q_true)."""
    dq = quat_mult(quat_conj(q_est), q_true)
    if dq[0] < 0:
        dq = -dq                       # keep short rotation (double cover)
    return 2 * dq[1:]


def skew(v):
    """Skew-symmetric cross-product matrix [v x]."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
