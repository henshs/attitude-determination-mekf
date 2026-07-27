"""TRIAD deterministic two-vector attitude."""
import numpy as np
from ..attitude.quaternion import quat_from_dcm


def triad(v1_b, v2_b, v1_i, v2_i):
    """Attitude R_BI from two vector pairs; v1 is the trusted anchor.

    Returns the DCM R_BI such that v_body ~= R_BI v_inertial.
    """
    def unit(v): return np.asarray(v, float) / np.linalg.norm(v)
    # body triad
    t1b = unit(v1_b)
    t2b = unit(np.cross(v1_b, v2_b))
    t3b = np.cross(t1b, t2b)
    Mb = np.column_stack([t1b, t2b, t3b])
    # inertial triad
    t1i = unit(v1_i)
    t2i = unit(np.cross(v1_i, v2_i))
    t3i = np.cross(t1i, t2i)
    Mi = np.column_stack([t1i, t2i, t3i])
    return Mb @ Mi.T


def triad_quat(v1_b, v2_b, v1_i, v2_i):
    return quat_from_dcm(triad(v1_b, v2_b, v1_i, v2_i))
