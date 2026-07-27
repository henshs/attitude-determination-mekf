"""Euler-angle <-> DCM conversions. 3-2-1 (yaw-pitch-roll)."""
import numpy as np
from .quaternion import quat_from_dcm, dcm


def euler321_to_dcm(phi, theta, psi):
    """R_BI = Rx(phi) Ry(theta) Rz(psi)  (roll, pitch, yaw)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    Rx = np.array([[1, 0, 0], [0, cphi, sphi], [0, -sphi, cphi]])
    Ry = np.array([[cth, 0, -sth], [0, 1, 0], [sth, 0, cth]])
    Rz = np.array([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def dcm_to_euler321(R):
    """Recover (phi, theta, psi) from a 3-2-1 DCM. Note theta = -asin(R13)."""
    theta = -np.arcsin(np.clip(R[0, 2], -1, 1))
    phi = np.arctan2(R[1, 2], R[2, 2])
    psi = np.arctan2(R[0, 1], R[0, 0])
    return phi, theta, psi


def euler321_to_quat(phi, theta, psi):
    return quat_from_dcm(euler321_to_dcm(phi, theta, psi))


def quat_to_euler321(q):
    return dcm_to_euler321(dcm(q))
