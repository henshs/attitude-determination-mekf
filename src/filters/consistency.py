"""Filter consistency: NEES/NIS for attitude."""
import numpy as np
from scipy.stats import chi2
from ..attitude.quaternion import quat_error_angle


def attitude_error_state(q_est, b_est, q_true, b_true):
    """6-D error [dtheta (multiplicative); dbias (additive)]."""
    dtheta = quat_error_angle(q_est, q_true)
    dbias = np.asarray(b_true) - np.asarray(b_est)
    return np.concatenate([dtheta, dbias])


def nees(q_est, b_est, q_true, b_true, P):
    e = attitude_error_state(q_est, b_est, q_true, b_true)
    return float(e @ np.linalg.inv(P) @ e)


def nis(y, S):
    return float(y @ np.linalg.inv(S) @ y)


def chi2_bounds(dof, N=1, alpha=0.05):
    lo = chi2.ppf(alpha / 2, dof * N) / N
    hi = chi2.ppf(1 - alpha / 2, dof * N) / N
    return lo, hi
