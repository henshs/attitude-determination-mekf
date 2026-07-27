"""Demo: torque-free tumble — conservation and the intermediate-axis instability."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attitude.quaternion import quat_identity, dcm
from src.dynamics.rigid_body import propagate_attitude, kinetic_energy, angular_momentum

J = np.array([100., 120., 80.])   # min=80(axis3), mid=100(axis1), max=120(axis2)
print("Principal inertias:", J, " (axis1=intermediate, unstable)")

for label, w0 in [("spin about MAX axis (2)", [0.01, 0.5, 0.01]),
                  ("spin about MIN axis (3)", [0.01, 0.01, 0.5]),
                  ("spin about INTERMEDIATE axis (1)", [0.5, 0.01, 0.01])]:
    t, Q, W = propagate_attitude(quat_identity(), w0, J, (0, 120),
                                 t_eval=np.linspace(0, 120, 600))
    E = np.array([kinetic_energy(w, J) for w in W])
    Hi = np.array([dcm(q).T @ angular_momentum(w, J) for q, w in zip(Q, W)])
    # transverse wobble growth: max deviation of the two non-spin components
    spin_axis = int(np.argmax(np.abs(w0)))
    transverse = np.delete(W, spin_axis, axis=1)
    wobble = np.max(np.abs(transverse)) / np.abs(w0[spin_axis])
    print(f"\n{label}")
    print(f"  energy drift: {E.max()-E.min():.2e},  H drift: {np.max(np.abs(Hi-Hi[0])):.2e}")
    print(f"  peak transverse/spin ratio: {wobble:.3f}  "
          f"({'UNSTABLE (large wobble / flips)' if wobble > 0.3 else 'stable (bounded nutation)'})")
