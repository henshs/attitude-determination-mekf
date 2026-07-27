"""Rigid-body rotational dynamics: Euler's equation."""
import numpy as np
from scipy.integrate import solve_ivp
from ..attitude.quaternion import skew, quat_normalize
from ..attitude.kinematics import qdot


def euler_eq(omega, J, N, h_w=None):
    """Angular acceleration from Euler's equation.

        J omega_dot = N - omega x (J omega + h_w)

    J: 3x3 inertia (or diag vector). N: external torque. h_w: wheel momentum.
    """
    J = np.diag(J) if np.ndim(J) == 1 else np.asarray(J, float)
    h = np.zeros(3) if h_w is None else np.asarray(h_w, float)
    Jw = J @ omega + h
    omega_dot = np.linalg.solve(J, N - np.cross(omega, Jw))
    return omega_dot


def propagate_attitude(q0, omega0, J, t_span, torque_fn=None, t_eval=None,
                       rtol=1e-10, atol=1e-12):
    """Integrate coupled attitude + angular velocity (Euler + kinematics).

    torque_fn(t, q, omega) -> external torque (default zero).
    Returns (t, Q [N,4], W [N,3]).
    """
    J = np.diag(J) if np.ndim(J) == 1 else np.asarray(J, float)
    if torque_fn is None:
        torque_fn = lambda t, q, w: np.zeros(3)

    def deriv(t, x):
        q = x[:4]; w = x[4:7]
        N = torque_fn(t, q, w)
        dq = qdot(q, w)
        dw = euler_eq(w, J, N)
        return np.concatenate([dq, dw])

    x0 = np.concatenate([quat_normalize(q0), omega0])
    sol = solve_ivp(deriv, t_span, x0, t_eval=t_eval, rtol=rtol, atol=atol,
                    method="DOP853")
    Q = sol.y[:4].T
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    return sol.t, Q, sol.y[4:7].T


def kinetic_energy(omega, J):
    J = np.diag(J) if np.ndim(J) == 1 else np.asarray(J, float)
    return 0.5 * omega @ J @ omega


def angular_momentum(omega, J):
    J = np.diag(J) if np.ndim(J) == 1 else np.asarray(J, float)
    return J @ omega
