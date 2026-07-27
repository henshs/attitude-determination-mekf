"""Multiplicative Extended Kalman Filter for attitude + gyro bias.

Convention: body-frame (local) multiplicative error,  q_true = q_ref (x) dq,
with dq ~ [1, dtheta/2]. Error state x = [dtheta (3); dbias (3)] (6-D).

Error dynamics:
    dtheta_dot = -[w_hat x] dtheta - dbias - eta_v
    dbias_dot  =  eta_u
    F = [[-[w_hat x], -I3], [0, 0]]

Star-tracker update:
    innovation y = 2*vec(q_ref* (x) q_meas)  ~= dtheta_meas
    H = [I3, 0]
"""
import numpy as np
from ..attitude.quaternion import (quat_mult, quat_conj, quat_normalize,
                                    small_angle_quat, skew)
from ..attitude.kinematics import propagate_quat


class MEKF:
    def __init__(self, sigma_v, sigma_u, R_meas):
        """sigma_v = ARW (rad/s/sqrt Hz), sigma_u = RRW (rad/s^2/sqrt Hz).
        R_meas = 3x3 star-tracker attitude-error noise covariance."""
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.R = np.asarray(R_meas, float)
        # reference state
        self.q = np.array([1.0, 0, 0, 0])
        self.b = np.zeros(3)
        # 6x6 error covariance
        self.P = np.eye(6)

    def initialize(self, q0, b0, P0):
        self.q = quat_normalize(q0)
        self.b = np.asarray(b0, float)
        self.P = np.asarray(P0, float)

    # ------------------------------------------------------------ predict
    def predict(self, omega_meas, dt):
        w_hat = np.asarray(omega_meas, float) - self.b     # bias-corrected rate
        # propagate reference quaternion (bias held constant)
        self.q = propagate_quat(self.q, w_hat, dt)
        # error-state transition matrix Phi = I + F dt
        F = np.zeros((6, 6))
        F[0:3, 0:3] = -skew(w_hat)
        F[0:3, 3:6] = -np.eye(3)
        Phi = np.eye(6) + F * dt
        # discrete process noise (ARW on attitude, RRW on bias + coupling)
        Q = self._process_noise(dt)
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _process_noise(self, dt):
        sv2, su2 = self.sigma_v**2, self.sigma_u**2
        I3 = np.eye(3)
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (sv2 * dt + su2 * dt**3 / 3) * I3
        Q[0:3, 3:6] = -(su2 * dt**2 / 2) * I3
        Q[3:6, 0:3] = -(su2 * dt**2 / 2) * I3
        Q[3:6, 3:6] = (su2 * dt) * I3
        return Q

    # ------------------------------------------------------------ update
    def update(self, q_meas):
        """Star-tracker (full-attitude) update; returns (innovation, nis)."""
        dq = quat_mult(quat_conj(self.q), quat_normalize(q_meas))
        if dq[0] < 0:
            dq = -dq
        y = 2.0 * dq[1:]                       # innovation ~ measured dtheta
        H = np.zeros((3, 6)); H[:, 0:3] = np.eye(3)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y                             # [dtheta; dbias]
        # apply multiplicatively / additively, then reset
        self.q = quat_normalize(quat_mult(self.q, small_angle_quat(dx[0:3])))
        self.b = self.b + dx[3:6]
        I6 = np.eye(6)
        self.P = (I6 - K @ H) @ self.P @ (I6 - K @ H).T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        nis = float(y @ np.linalg.inv(S) @ y)
        return y, nis
