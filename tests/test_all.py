"""Unit tests. Run:  python3 tests/test_all.py   (or pytest -q)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attitude.quaternion import (quat_mult, quat_conj, quat_from_axis_angle,
    quat_from_dcm, dcm, rotate_vector, quat_identity, quat_error_angle, small_angle_quat)
from src.attitude.kinematics import propagate_quat
from src.attitude.frames import euler321_to_quat, quat_to_euler321
from src.dynamics.rigid_body import propagate_attitude, kinetic_energy, angular_momentum
from src.determination.triad import triad_quat
from src.determination.quest import davenport_q
from src.control.attitude_control import lqr_gains, pd_gains

rng = np.random.default_rng(0)


def test_quat_conj_inverse():
    q = quat_from_axis_angle([1, 2, 3], 0.9)
    assert np.allclose(quat_mult(q, quat_conj(q)), [1, 0, 0, 0], atol=1e-12)


def test_dcm_quat_roundtrip():
    for _ in range(50):
        q = quat_from_axis_angle(rng.normal(size=3), rng.uniform(0, np.pi))
        q2 = quat_from_dcm(dcm(q))
        assert np.allclose(q, q2, atol=1e-9) or np.allclose(q, -q2, atol=1e-9)


def test_dcm_proper_rotation():
    q = quat_from_axis_angle([1, 2, 3], 1.1)
    R = dcm(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1) < 1e-12


def test_euler_roundtrip():
    a = np.radians([20, 35, 50])
    assert np.allclose(a, quat_to_euler321(euler321_to_quat(*a)), atol=1e-9)


def test_propagation_single_axis():
    q = propagate_quat(quat_identity(), [0, 0, 0.1], 10.0)
    assert np.allclose(q, [np.cos(0.5), 0, 0, np.sin(0.5)], atol=1e-9)


def test_energy_momentum_conserved():
    J = np.array([100., 120., 80.])
    t, Q, W = propagate_attitude(quat_identity(), [0.1, 0.05, 0.5], J,
                                 (0, 40), t_eval=np.linspace(0, 40, 60))
    E = [kinetic_energy(w, J) for w in W]
    Hi = np.array([dcm(q).T @ angular_momentum(w, J) for q, w in zip(Q, W)])
    assert (max(E) - min(E)) < 1e-9
    assert np.max(np.abs(Hi - Hi[0])) < 1e-6


def test_triad_exact():
    q_true = quat_from_axis_angle([0.3, 0.5, 0.8], 0.7)
    si, mi = np.array([1., 0, 0]), np.array([0., 1, 0])
    sb, mb = rotate_vector(q_true, si), rotate_vector(q_true, mi)
    q_est = triad_quat(sb, mb, si, mi)
    err = 2 * abs(np.arccos(np.clip(abs(q_est @ q_true), -1, 1)))
    assert np.degrees(err) < 1e-6


def test_quest_recovers_attitude():
    q_true = quat_from_axis_angle([0.3, 0.5, 0.8], 0.7)
    vi = [np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1])]
    vb = [rotate_vector(q_true, v) for v in vi]
    q_q = davenport_q(vb, vi)
    err = 2 * abs(np.arccos(np.clip(abs(q_q @ q_true), -1, 1)))
    assert np.degrees(err) < 1e-6


def test_error_angle_small():
    qt = quat_from_axis_angle([0, 0, 1], np.radians(0.5))
    e = quat_error_angle(quat_identity(), qt)
    assert np.allclose(e, [0, 0, np.radians(0.5)], atol=1e-5)


def test_lqr_gain_shape_and_stability():
    J = np.array([150., 150., 150.])
    Q = np.diag([1, 1, 1, 0, 0, 0.]); R = 0.1 * np.eye(3)
    K = lqr_gains(J, Q, R)
    assert K.shape == (3, 6)
    # closed-loop A - B K should be stable (negative real parts)
    A = np.zeros((6, 6)); A[0:3, 3:6] = np.eye(3)
    B = np.zeros((6, 3)); B[3:6, :] = np.linalg.inv(np.diag(J))
    eig = np.linalg.eigvals(A - B @ K)
    assert np.all(eig.real < 0)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for tfn in tests:
        try:
            tfn(); print(f"PASS  {tfn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL  {tfn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
