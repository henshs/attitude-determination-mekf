"""Star tracker: noisy full-attitude quaternion measurement."""
import numpy as np
from ..attitude.quaternion import quat_mult, small_angle_quat, quat_normalize


class StarTracker:
    """Outputs a noisy attitude quaternion.

    sigma: 1-sigma per-axis measurement noise [rad].
    """
    def __init__(self, sigma=1.5e-5, rng=None):     # ~3 arcsec
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()

    def measure(self, q_true):
        """Perturb the true attitude by a small random rotation."""
        dtheta = self.rng.normal(0, self.sigma, 3)
        dq = small_angle_quat(dtheta)
        return quat_normalize(quat_mult(q_true, dq))

    @property
    def R(self):
        """3x3 measurement noise covariance for the attitude-error innovation."""
        return (self.sigma ** 2) * np.eye(3)
