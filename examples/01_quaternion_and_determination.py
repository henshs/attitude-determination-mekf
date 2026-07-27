"""Demo: quaternion basics, and TRIAD vs QUEST attitude determination."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attitude.quaternion import (quat_from_axis_angle, dcm, rotate_vector,
                                      quat_to_axis_angle)
from src.attitude.frames import quat_to_euler321
from src.determination.triad import triad_quat
from src.determination.quest import davenport_q

rng = np.random.default_rng(1)
q_true = quat_from_axis_angle([0.3, 0.5, 0.8], np.radians(40))
ax, ang = quat_to_axis_angle(q_true)
print(f"true attitude: {np.degrees(ang):.1f} deg about {np.round(ax,3)}")
print("Euler 3-2-1 [deg]:", np.round(np.degrees(quat_to_euler321(q_true)), 2))

# two reference directions
si, mi = np.array([1., 0, 0]), np.array([0., 1, 0])
sb, mb = rotate_vector(q_true, si), rotate_vector(q_true, mi)

# TRIAD (exact, noise-free)
q_triad = triad_quat(sb, mb, si, mi)
print(f"\nTRIAD error: {np.degrees(2*abs(np.arccos(np.clip(abs(q_triad@q_true),-1,1)))):.2e} deg")

# QUEST with 3 noisy vectors, star tracker weighted 100x the magnetometer
vi = [np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1])]
vb = [rotate_vector(q_true, v) + rng.normal(0, 0.002, 3) for v in vi]
q_quest = davenport_q(vb, vi, weights=[100, 100, 1])
print(f"QUEST error (noisy): {np.degrees(2*abs(np.arccos(np.clip(abs(q_quest@q_true),-1,1)))):.3f} deg")
