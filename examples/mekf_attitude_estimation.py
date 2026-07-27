"""Capstone 1: MEKF attitude + gyro-bias estimation with consistency validation.

Truth: a slowly rotating spacecraft with a drifting gyro bias.
Sensors: noisy gyro (fast) + star tracker (slow, accurate).
Output: attitude-error and bias-error convergence, NEES and NIS consistency plots.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attitude.quaternion import (quat_identity, quat_error_angle,
                                      small_angle_quat, quat_mult)
from src.attitude.kinematics import propagate_quat
from src.sensors.gyro import Gyro
from src.sensors.star_tracker import StarTracker
from src.filters.mekf import MEKF
from src.filters.consistency import nees, chi2_bounds

rng = np.random.default_rng(7)
dt, T = 0.1, 200.0
times = np.arange(0, T, dt)
w_true = np.array([0.02, -0.01, 0.03])      # slow rotation -> bias observable
sigma_v, sigma_u, st_sigma = 3e-4, 3e-6, 1.5e-5

gyro = Gyro(arw=sigma_v, rrw=sigma_u, bias0=[5e-4, -3e-4, 4e-4], rng=rng)
st = StarTracker(sigma=st_sigma, rng=rng)

mekf = MEKF(sigma_v, sigma_u, st.R)
mekf.initialize(quat_mult(quat_identity(), small_angle_quat(rng.normal(0, 0.02, 3))),
                np.zeros(3), np.diag([0.02**2]*3 + [1e-6]*3))

q_true = quat_identity()
st_every = int(1.0 / dt)
att_err, bias_err, sig3, nees_h, nis_h, nis_t = [], [], [], [], [], []

for k, t in enumerate(times):
    wm = gyro.measure(w_true, dt)
    q_true = propagate_quat(q_true, w_true, dt)
    mekf.predict(wm, dt)
    if k % st_every == 0 and k > 0:
        y, nis = mekf.update(st.measure(q_true))
        nis_h.append(nis); nis_t.append(t)
    att_err.append(np.degrees(np.linalg.norm(quat_error_angle(mekf.q, q_true))))
    bias_err.append(np.degrees(np.linalg.norm(gyro.bias - mekf.b)) * 3600)  # deg/hr
    sig3.append(np.degrees(3 * np.sqrt(np.trace(mekf.P[:3, :3]))))
    nees_h.append(nees(mekf.q, mekf.b, q_true, gyro.bias, mekf.P))

att_err = np.array(att_err); bias_err = np.array(bias_err); sig3 = np.array(sig3)
print(f"final attitude error: {att_err[-1]*3600:.1f} arcsec")
print(f"final bias error: {bias_err[-1]:.3f} deg/hr")
print(f"mean NEES (last half): {np.mean(nees_h[len(nees_h)//2:]):.2f} (target 6)")
print(f"mean NIS: {np.mean(nis_h):.2f} (target 3)")

os.makedirs("examples/figures", exist_ok=True)
fig, ax = plt.subplots(3, 1, figsize=(9, 9))

ax[0].plot(times, att_err, color="#13315c", lw=1.3, label="attitude error")
ax[0].plot(times, sig3, color="#b8860b", ls="--", lw=1.0, label="3σ envelope")
ax[0].set_yscale("log"); ax[0].set_ylabel("attitude error [deg]")
ax[0].set_title("MEKF convergence: attitude error &  3σ envelope")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(times, bias_err, color="#2e7d32", lw=1.3)
ax[1].set_ylabel("bias error [deg/hr]"); ax[1].set_xlabel("time [s]")
ax[1].set_title("Gyro-bias estimation error (converges as filter calibrates gyro)")
ax[1].grid(alpha=0.3)

lo, hi = chi2_bounds(6, N=1)
ax[2].plot(times, nees_h, ".", color="#13315c", ms=2, label="NEES")
ax[2].axhline(6, color="#b8860b", lw=1.2, label="expected (n=6)")
ax[2].axhline(lo, color="#888", ls=":", lw=0.8)
ax[2].axhline(hi, color="#888", ls=":", lw=0.8, label="95% χ² band")
ax[2].set_ylim(0, 20); ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("NEES")
ax[2].set_title(f"Consistency: mean NEES = {np.mean(nees_h[len(nees_h)//2:]):.2f} (target 6), "
                f"mean NIS = {np.mean(nis_h):.2f} (target 3)")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

plt.tight_layout()
out = "examples/figures/mekf_attitude_estimation.png"
plt.savefig(out, dpi=130); print("saved", out)
