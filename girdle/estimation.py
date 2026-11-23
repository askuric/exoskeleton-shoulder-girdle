"""Reconstruct the shoulder from the robot cuffs, with and without IMUs (model IK).

The solver is the one of the paper: differential IK with pink (QP solved by quadprog). In every frame the shoulder
model is fitted so that its arm sits in the measured cuffs (FrameTask), the AC loop is an equality constraint
(PointCoincidenceTask), the joint limits are inequality constraints, and a weak PostureTask keeps the pose close to
the previous frame. The user is assumed seated, so the thorax position is held in every case (pinned through its
limits). The three cases differ in the extra sensors:
    "robot"    cuffs only: the thorax orientation is estimated
    "scapula"  + gravity-only scapula IMU: the direction of gravity in the scapula is known (GravityTask); without a
               thorax IMU the thorax is held at its starting pose
    "both"     + thorax IMU and gravity-only scapula IMU: the thorax orientation is measured
"""
import numpy as np
import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask, Task

from .model import UP

CASES = ("robot", "scapula", "both")

W_ORIENTATION = 0.05   # cuff orientation, relative to cuff position (per rad vs per m)
W_GRAVITY = 0.2        # scapula IMU gravity direction
W_POSTURE = 1e-2       # change from the previous frame (per rad)
W_THORAX = 5.0         # thorax pose (per m, per rad)
DAMPING = 1e-2
DT = 0.05
ITERATIONS = 40


class PointCoincidenceTask(Task):
    """Equality constraint: the origins of two frames coincide (closes the AC loop)."""

    def __init__(self, frame_a, frame_b):
        super().__init__(cost=1.0)
        self.frame_a, self.frame_b = frame_a, frame_b

    def compute_error(self, configuration):
        return (configuration.get_transform_frame_to_world(self.frame_a).translation
                - configuration.get_transform_frame_to_world(self.frame_b).translation)

    def compute_jacobian(self, configuration):
        J = lambda f: pin.getFrameJacobian(configuration.model, configuration.data,
                                           configuration.model.getFrameId(f), pin.LOCAL_WORLD_ALIGNED)[:3]
        return J(self.frame_a) - J(self.frame_b)

    def __repr__(self):
        return f"PointCoincidenceTask({self.frame_a}, {self.frame_b})"


class GravityTask(Task):
    """The up direction seen by an IMU, R^T * up, matches the measured one (2 DOF: tilt only)."""

    def __init__(self, frame_name, cost):
        super().__init__(cost=cost)
        self.frame_name, self.target = frame_name, UP.copy()

    def compute_error(self, configuration):
        return configuration.get_transform_frame_to_world(self.frame_name).rotation.T @ UP - self.target

    def compute_jacobian(self, configuration):
        R = configuration.get_transform_frame_to_world(self.frame_name).rotation
        J = pin.getFrameJacobian(configuration.model, configuration.data,
                                 configuration.model.getFrameId(self.frame_name), pin.LOCAL)[3:]
        return pin.skew(R.T @ UP) @ J

    def __repr__(self):
        return f"GravityTask({self.frame_name})"


def estimate(shoulder, q, q_previous, cuffs, thorax=None, gravity=None):
    """One frame of model IK.

    cuffs            measured cuff poses (pin.SE3, upper arm and forearm)
    thorax           the 6 thorax coordinates of q (3 translations, 3 rotations), NaN where free, or None (all free)
    gravity          scapula IMU: the up direction in the scapula IMU frame, or None
    """
    s, model = shoulder, shoulder.model
    cuff_tasks = [FrameTask(name, position_cost=1.0, orientation_cost=W_ORIENTATION, lm_damping=1e-5)
                  for name in s.CUFFS]
    for task, target in zip(cuff_tasks, cuffs):
        task.set_target(target)
    cost = np.full(model.nv, W_POSTURE); cost[s.ROOT] = W_THORAX
    posture = PostureTask(cost=cost)
    q_reg = q_previous.copy()
    q_reg[s.ROOT] = 0.0
    posture.set_target(q_reg)
    tasks = cuff_tasks + [posture]
    if gravity is not None:
        gravity_task = GravityTask(s.IMU, W_GRAVITY)
        gravity_task.target = gravity / np.linalg.norm(gravity)
        tasks.append(gravity_task)
    constraints = [PointCoincidenceTask("ac", "acc")]

    # the thorax is free (wide limits), and its given coordinates are pinned through their limits
    saved = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    model.lowerPositionLimit[:6], model.upperPositionLimit[:6] = -10.0, 10.0
    pinned = [] if thorax is None else [i for i in range(6) if np.isfinite(thorax[i])]
    q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
    for i in pinned:
        q[i] = thorax[i]
        model.lowerPositionLimit[i], model.upperPositionLimit[i] = thorax[i] - 1e-7, thorax[i] + 1e-7
    try:
        configuration = pink.Configuration(model, s.data, q.copy())
        for _ in range(ITERATIONS):
            position_error = max(np.linalg.norm(t.compute_error(configuration)[:3]) for t in cuff_tasks)
            rotation_error = max(np.linalg.norm(t.compute_error(configuration)[3:]) for t in cuff_tasks)
            imu_error = np.linalg.norm(tasks[-1].compute_error(configuration)) if gravity is not None else 0.0
            if position_error < 0.002 and rotation_error < np.radians(0.5) and imu_error < 0.01:
                break
            velocity = pink.solve_ik(configuration, tasks, dt=DT, solver="quadprog", damping=DAMPING,
                                     constraints=constraints, safety_break=False)
            if not np.all(np.isfinite(velocity)):    # the QP can fail when the cuffs are out of reach
                break
            configuration.integrate_inplace(velocity, DT)
        q_new = configuration.q.copy()
    finally:
        model.lowerPositionLimit[:], model.upperPositionLimit[:] = saved
    for i in pinned:
        q_new[i] = thorax[i]
    return q_new if np.all(np.isfinite(q_new)) else q


def sensors(shoulder, q_true):
    """Ideal measurements from a true pose: the thorax coordinates (position: seated user; rotations: thorax IMU) and
    gravity in the scapula IMU frame."""
    s = shoulder
    s.update(q_true)
    return q_true[:6].copy(), s.frame(s.IMU).rotation.T @ UP


def estimate_case(shoulder, case, q, q_previous, cuffs, imu, thorax_start):
    """estimate() for one of CASES; imu = (thorax, gravity) from sensors(), thorax_start the thorax coordinates at the
    start (used by "scapula", which has no thorax IMU)."""
    thorax, gravity = imu
    thorax = {"robot": np.r_[thorax[:3], np.full(3, np.nan)], "scapula": thorax_start, "both": thorax}[case]
    return estimate(shoulder, q, q_previous, cuffs, thorax=thorax, gravity=None if case == "robot" else gravity)
