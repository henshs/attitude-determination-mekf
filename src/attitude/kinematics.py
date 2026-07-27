"""Attitude kinematics: quaternion propagation."""
import numpy as np
from .quaternion import quat_mult, quat_normalize, quat_from_axis_angle


def qdot(q, omega):
    """Quaternion rate: q_dot = 0.5 * q (x) [0, omega]."""
    return 0.5 * quat_mult(q, np.array([0.0, *omega]))


def propagate_quat(q, omega, dt):
    """Exact closed-form step for (approximately) constant body rate omega.

    q_{k+1} = q_k (x) dq,  dq = [cos(|w|dt/2), e sin(|w|dt/2)].
    """
    q = np.asarray(q, float)
    w = np.linalg.norm(omega)
    if w < 1e-12:
        return quat_normalize(q)
    dq = quat_from_axis_angle(np.asarray(omega) / w, w * dt)
    return quat_normalize(quat_mult(q, dq))
