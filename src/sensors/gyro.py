"""Gyroscope model: rate + bias + noise."""
import numpy as np


class Gyro:
    """Rate gyro with a drifting bias.

        omega_meas = omega_true + bias + white_noise (ARW)
        bias_dot   = white_noise (RRW)

    arw: angle random walk [rad/s/sqrt(Hz)]  (sigma_v)
    rrw: rate  random walk [rad/s^2/sqrt(Hz)] (sigma_u)
    """
    def __init__(self, arw=1e-4, rrw=1e-6, bias0=None, rng=None):
        self.arw = arw
        self.rrw = rrw
        self.bias = np.zeros(3) if bias0 is None else np.array(bias0, float)
        self.rng = rng or np.random.default_rng()

    def measure(self, omega_true, dt):
        """Return a noisy rate reading and advance the true bias by one step."""
        # white rate noise scaled to the step
        eta_v = self.rng.normal(0, self.arw / np.sqrt(dt), 3)
        meas = omega_true + self.bias + eta_v
        # advance true bias (random walk)
        self.bias = self.bias + self.rng.normal(0, self.rrw * np.sqrt(dt), 3)
        return meas
