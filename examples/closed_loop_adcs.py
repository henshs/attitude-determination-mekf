"""Capstone 2: full closed-loop ADCS.

Sense (gyro + star tracker) -> estimate (MEKF) -> control (quaternion PD) ->
actuate (torque on the rigid body). A 40-degree slew commanded, then hold,
flown entirely on the FILTER's estimate (not truth).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attitude.quaternion import (quat_identity, quat_from_axis_angle,
    quat_error_angle, small_angle_quat, quat_mult)
from src.attitude.kinematics import propagate_quat
from src.dynamics.rigid_body import euler_eq
from src.sensors.gyro import Gyro
from src.sensors.star_tracker import StarTracker
from src.filters.mekf import MEKF
from src.control.attitude_control import pd_control, pd_gains

rng = np.random.default_rng(3)
J = np.array([150., 120., 100.])          # kg m^2
dt, T = 0.05, 120.0
times = np.arange(0, T, dt)

# commanded attitude: 40 deg about a tilted axis
q_cmd = quat_from_axis_angle([0.3, 0.6, 0.74], np.radians(40))

# PD gains (per dominant axis)
kp, kd = pd_gains(np.max(J), wn=0.15, zeta=0.7)

# sensors + filter
sigma_v, sigma_u = 3e-4, 3e-6
gyro = Gyro(arw=sigma_v, rrw=sigma_u, bias0=[4e-4, -2e-4, 3e-4], rng=rng)
st = StarTracker(sigma=1.5e-5, rng=rng)
mekf = MEKF(sigma_v, sigma_u, st.R)
mekf.initialize(small_angle_quat(rng.normal(0, 0.01, 3)), np.zeros(3),
                np.diag([0.01**2]*3 + [1e-6]*3))

# true state
q_true = quat_identity()
w_true = np.zeros(3)
st_every = int(0.5 / dt)

pt_err_true, pt_err_est, torque_mag, rate_mag = [], [], [], []

for k, t in enumerate(times):
    # --- SENSE ---
    wm = gyro.measure(w_true, dt)
    # --- ESTIMATE (MEKF) ---
    mekf.predict(wm, dt)
    if k % st_every == 0 and k > 0:
        mekf.update(st.measure(q_true))
    q_est, w_est = mekf.q, wm - mekf.b        # bias-corrected rate estimate
    # --- CONTROL (on the estimate, as in flight) ---
    N = pd_control(q_est, w_est, q_cmd, kp, kd, J=J, feedforward_gyro=True)
    N = np.clip(N, -5.0, 5.0)                  # wheel torque limit
    # --- ACTUATE: integrate true rigid-body dynamics one step ---
    wdot = euler_eq(w_true, J, N)
    w_true = w_true + wdot * dt
    q_true = propagate_quat(q_true, w_true, dt)

    pt_err_true.append(np.degrees(np.linalg.norm(quat_error_angle(q_true, q_cmd))))
    pt_err_est.append(np.degrees(np.linalg.norm(quat_error_angle(q_est, q_cmd))))
    torque_mag.append(np.linalg.norm(N))
    rate_mag.append(np.degrees(np.linalg.norm(w_true)))

pt_err_true = np.array(pt_err_true)
print(f"settled pointing error (true): {np.mean(pt_err_true[-200:]):.4f} deg")
print(f"peak slew rate: {max(rate_mag):.3f} deg/s, peak torque: {max(torque_mag):.2f} N.m")

os.makedirs("examples/figures", exist_ok=True)
fig, ax = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
ax[0].plot(times, pt_err_true, color="#13315c", lw=1.4, label="true pointing error")
ax[0].plot(times, pt_err_est, color="#b8860b", lw=0.9, ls="--", label="filter-estimated")
ax[0].set_yscale("log"); ax[0].set_ylabel("pointing error [deg]")
ax[0].set_title("Closed-loop 40° slew: pointing error (flown on the MEKF estimate)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(times, rate_mag, color="#2e7d32", lw=1.3)
ax[1].set_ylabel("body rate [deg/s]"); ax[1].set_title("Slew rate profile"); ax[1].grid(alpha=0.3)
ax[2].plot(times, torque_mag, color="#c0392b", lw=1.3)
ax[2].set_ylabel("torque [N·m]"); ax[2].set_xlabel("time [s]")
ax[2].set_title("Commanded wheel torque (clipped at limit)"); ax[2].grid(alpha=0.3)
plt.tight_layout()
out = "examples/figures/closed_loop_adcs.png"
plt.savefig(out, dpi=130); print("saved", out)
