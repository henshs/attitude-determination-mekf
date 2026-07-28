"""Attitude control: quaternion PD and LQR."""
import numpy as np
from scipy.linalg import solve_continuous_are
from ..attitude.quaternion import quat_mult, quat_conj, skew


def attitude_error_quat(q, q_desired):
    """Error quaternion dq = q_desired* (x) q ; force short rotation (dq0>=0)."""
    dq = quat_mult(quat_conj(q_desired), q)
    if dq[0] < 0:
        dq = -dq
    return dq


def pd_control(q, omega, q_desired, kp, kd, J=None, feedforward_gyro=False):
    """Quaternion feedback PD torque:  N = -kp dq_v - kd omega.

    If feedforward_gyro and J are given, add +omega x (J omega) to cancel the
    gyroscopic term for fast slews.
    """
    dq = attitude_error_quat(q, q_desired)
    N = -kp * dq[1:] - kd * np.asarray(omega, float)
    if feedforward_gyro and J is not None:
        Jm = np.diag(J) if np.ndim(J) == 1 else np.asarray(J, float)
        N = N + np.cross(omega, Jm @ omega)
    return N


def pd_gains(J_axis, wn, zeta):
    """PD gains for a target natural frequency wn and damping zeta (single axis)."""
    kp = 2 * J_axis * wn**2
    kd = 2 * zeta  * wn * J_axis
    return kp, kd


def lqr_gains(J, Qw, Rw):
    """Optimal LQR gain K for the linearized attitude system.

    State x = [dtheta; omega], u = torque.  dx = A x + B u,
    A = [[0, I],[0,0]], B = [[0],[J^-1]].
    Returns 3x6 gain K so that  u = -K x = -K_theta dtheta - K_omega omega.
    """
    Jm = np.diag(J) if np.ndim(J) == 1 else np.asarray(J, float)
    A = np.zeros((6, 6)); A[0:3, 3:6] = np.eye(3)
    B = np.zeros((6, 3)); B[3:6, :] = np.linalg.inv(Jm)
    Q = np.asarray(Qw, float); R = np.asarray(Rw, float)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


def lqr_control(q, omega, q_desired, K):
    """Apply the LQR gain. dtheta from the error quaternion vector part (x2)."""
    dq = attitude_error_quat(q, q_desired)
    dtheta = 2 * dq[1:]
    x = np.concatenate([dtheta, np.asarray(omega, float)])
    return -K @ x
