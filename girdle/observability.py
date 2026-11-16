"""How well the cuffs see the shoulder girdle.

The cuffs fix the arm, and the arm joins the girdle only at the glenohumeral (GH) joint, so the robot senses girdle
motion only through the GH centre. With the thorax fixed and the AC loop closed, the girdle has 3 free coordinates.
The singular values of the GH-centre Jacobian over them (mm of GH motion per degree of girdle motion) say how well
each girdle motion is seen; the smallest one, sigma_3, is nearly zero over much of the arm elevation range.

A gravity-only IMU on the scapula (an accelerometer) senses the scapula tilt except rotations about gravity. Its
rows are added with a lever arm of 180/pi mm, so that 1 deg of IMU rotation counts as 1 mm.
"""
import numpy as np
import pinocchio as pin

from .model import UP

MM_PER_DEG = 1000 * np.pi / 180      # m/rad -> mm/deg
LEVER_MM = 180 / np.pi                # IMU lever arm (mm)


def observability(shoulder, q):
    """Singular values (mm/deg) without and with the scapula IMU, and the least observable girdle direction
    (unit vector in the 6 girdle coordinates)."""
    s = shoulder
    s.update(q)
    _, J_loop = s.ac_loop()
    _, sv, vt = np.linalg.svd(J_loop[:, s.GIRDLE])
    N = vt[np.sum(sv > 1e-9):].T                       # girdle velocities that keep the AC loop closed

    gh_rows = s.jacobian(s.GH_CENTRE)[:3, s.GIRDLE] * MM_PER_DEG
    g = s.frame(s.IMU).rotation.T @ UP                 # gravity direction in the IMU frame
    J_imu = s.jacobian(s.IMU, pin.LOCAL)[3:, s.GIRDLE]
    imu_rows = (np.eye(3) - np.outer(g, g)) @ J_imu    # rotations about gravity are invisible

    _, sigma, vt = np.linalg.svd(gh_rows @ N)
    sigma_imu = np.linalg.svd(np.vstack([gh_rows, LEVER_MM * np.pi / 180 * imu_rows]) @ N, compute_uv=False)
    return sigma, sigma_imu, N @ vt[-1]


def move_along(shoulder, q, direction, angle_deg):
    """Move the girdle by angle_deg along a girdle direction, with the AC loop closed and the humerus orientation
    kept (the cuffs hold the arm)."""
    s = shoulder
    s.update(q)
    R_humerus = s.frame("/bodyset/humerus").rotation.copy()
    step = np.zeros(s.model.nv)
    step[s.GIRDLE] = direction * np.radians(angle_deg)
    return s.close_loop(pin.integrate(s.model, q, step), R_humerus)
