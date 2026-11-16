"""A generic arm exoskeleton, built around the shoulder model (not any particular device).

Joints, from the base:
    girdle_protraction   vertical axis through a pivot medial to the shoulder (positive: shoulder forward)
    girdle_elevation     horizontal axis through the same pivot, pointing backwards (positive: shoulder up)
    gh_1, gh_2, gh_3     three shoulder joints whose axes meet at the robot GH centre; the first is horizontal,
                         pointing back and out, so that the arm never lines up with it in the usual workspace
    elbow                the elbow axis of the human model
The robot holds the arm with two cuffs, on the upper arm and on the forearm, placed where the ideal cuffs of the
shoulder model are in the reference pose (arm down).
"""
import numpy as np
import pinocchio as pin

from .model import UP

JOINTS = ("girdle_protraction", "girdle_elevation", "gh_1", "gh_2", "gh_3", "elbow")
CUFFS = ("upper_arm_cuff", "forearm_cuff")


class Exoskeleton:
    def __init__(self, shoulder):
        s = shoulder
        s.update(s.q_ref)
        R_thorax = s.frame("/bodyset/thorax").rotation
        self.anterior, self.up, self.right = R_thorax[:, 0], R_thorax[:, 1], R_thorax[:, 2]
        gh = s.frame(s.GH_CENTRE).translation.copy()
        elbow = s.frame("centelbow").translation.copy()
        self.pivot = gh - 0.17 * self.right + 0.02 * self.up       # girdle pivot, medial to the shoulder (near the SC joint)

        arm = (elbow - gh) / np.linalg.norm(elbow - gh)
        # First shoulder axis: horizontal, pointing back and out. The three shoulder axes become singular when the
        # arm lines up with it, i.e. when the arm points back-lateral or forward-medial (across the chest), which
        # keeps the singularity out of the usual workspace (arm in front of or beside the body).
        a1 = (self.right - self.anterior) / np.linalg.norm(self.right - self.anterior)
        a2 = np.cross(a1, arm); a2 /= np.linalg.norm(a2)
        a3 = arm                                                                      # along the arm
        elbow_axis = pin.getJointJacobian(s.model, s.data, s.model.getJointId("elbow"),
                                          pin.LOCAL_WORLD_ALIGNED)[3:, s.dofs("elbow")[0]]

        m = pin.Model()
        revolute = lambda axis: pin.JointModelRevoluteUnaligned(*axis)
        j = m.addJoint(0, revolute(UP), pin.SE3(np.eye(3), self.pivot), JOINTS[0])
        j = m.addJoint(j, revolute(-self.anterior), pin.SE3.Identity(), JOINTS[1])   # positive: shoulder up
        j = m.addJoint(j, revolute(a1), pin.SE3(np.eye(3), gh - self.pivot), JOINTS[2])
        j = m.addJoint(j, revolute(a2), pin.SE3.Identity(), JOINTS[3])
        j = m.addJoint(j, revolute(a3), pin.SE3.Identity(), JOINTS[4])
        cuff = s.frame("upper_arm_cuff")
        m.addFrame(pin.Frame(CUFFS[0], j, 0, pin.SE3(cuff.rotation, cuff.translation - gh), pin.FrameType.OP_FRAME))
        j = m.addJoint(j, revolute(elbow_axis), pin.SE3(np.eye(3), elbow - gh), JOINTS[5])
        cuff = s.frame("forearm_cuff")
        m.addFrame(pin.Frame(CUFFS[1], j, 0, pin.SE3(cuff.rotation, cuff.translation - elbow), pin.FrameType.OP_FRAME))
        for k in range(1, m.njoints):
            m.appendBodyToJoint(k, pin.Inertia.Identity(), pin.SE3.Identity())

        self.model, self.data = m, m.createData()
        self.axes = [UP, -self.anterior, a1, a2, a3, elbow_axis]      # joint axes at the zero pose (world)
        self.q0 = np.zeros(m.nq)

    def cuffs(self, q):
        """Cuff poses measured by the robot (from its joint angles)."""
        pin.framesForwardKinematics(self.model, self.data, q)
        return [self.data.oMf[self.model.getFrameId(c)].copy() for c in CUFFS]

    def joint_positions(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        return [self.data.oMi[k].translation.copy() for k in range(1, self.model.njoints)]

    def joint_axes(self, q):
        """Joint axes in the world frame (unit vectors), one per joint."""
        pin.forwardKinematics(self.model, self.data, q)
        return [self.data.oMi[k].rotation @ a for k, a in zip(range(1, self.model.njoints), self.axes)]

    def gh_centre(self, q):
        """Robot GH centre: where the three shoulder axes meet."""
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[self.model.getJointId("gh_1")].translation.copy()

    def follow(self, q, targets, iterations=30):
        """Robot IK: move the cuffs onto target poses (e.g. the human arm), staying close to q.
        Returns the new joint angles and the largest cuff position error (m)."""
        q_start = q.copy()
        ids = [self.model.getFrameId(c) for c in CUFFS]
        for _ in range(iterations):
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            rows, residuals = [], []
            for i, T in zip(ids, targets):
                M = self.data.oMf[i]
                J = pin.getFrameJacobian(self.model, self.data, i, pin.LOCAL_WORLD_ALIGNED)
                rows += [J[:3], 0.3 * J[3:]]
                residuals += [M.translation - T.translation, 0.3 * pin.log3(M.rotation @ T.rotation.T)]
            rows.append(np.sqrt(1e-3) * np.eye(self.model.nv)); residuals.append(np.sqrt(1e-3) * (q - q_start))
            A, r = np.vstack(rows), np.concatenate(residuals)
            dq = -np.linalg.solve(A.T @ A + 1e-4 * np.eye(self.model.nv), A.T @ r)
            q = q + dq
            if np.linalg.norm(dq) < 1e-7:
                break
        pin.framesForwardKinematics(self.model, self.data, q)
        error = max(np.linalg.norm(self.data.oMf[i].translation - T.translation) for i, T in zip(ids, targets))
        return q, error
