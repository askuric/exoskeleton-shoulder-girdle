"""The thoracoscapular shoulder model of Seth et al., loaded in Pinocchio.

The model has a free thorax, a sternoclavicular joint (2 DOF), a scapulothoracic joint (4 DOF), a glenohumeral
joint (3 DOF), an elbow and a forearm joint. The scapula and the clavicle meet at the acromioclavicular (AC)
joint, which is kept closed as a constraint: the points "ac" (on the scapula) and "acc" (on the clavicle) coincide.

World axes: z up. Thorax axes (OpenSim): x anterior, y up, z right.
"""
from pathlib import Path

import numpy as np
import pin_actuation as pa
import pinocchio as pin

UP = np.array([0.0, 0.0, 1.0])

# coordinate names in the IK results of the dataset, per joint
MOT_COLUMNS = {
    "ground_thorax": ("ground_thorax_tx", "ground_thorax_ty", "ground_thorax_tz",
                      "ground_thorax_rot_x", "ground_thorax_rot_y", "ground_thorax_rot_z"),
    "sternoclavicular": ("clav_prot", "clav_elev"),
    "scapulothoracic": ("scapula_abduction", "scapula_elevation", "scapula_upward_rot", "scapula_winging"),
    "GlenoHumeral": ("plane_elv", "shoulder_elv", "axial_rot"),
    "elbow": ("elbow_flexion",),
    "radioulnar": ("pro_sup",),
}

# Bounds (deg) for the reconstruction, as in the paper. The model leaves the girdle coordinates practically unbounded
# (+-180 deg); these lie well beyond physiological motion and only exclude mirrored or wrapped-around scapula solutions.
BOUNDS = {
    ("sternoclavicular", 0): (-75, 45),     # clavicle protraction
    ("sternoclavicular", 1): (-45, 75),     # clavicle elevation
    ("scapulothoracic", 0): (-75, 75),      # scapula abduction
    ("scapulothoracic", 1): (-60, 60),      # scapula elevation
    ("scapulothoracic", 2): (-45, 135),     # scapula upward rotation
    ("scapulothoracic", 3): (-60, 60),      # scapula winging
    ("GlenoHumeral", 1): (-60, 160),        # GH elevation
}


class Shoulder:
    """Shoulder model with an ideal cuff on the upper arm and on the forearm and a scapula IMU frame."""

    CUFFS = ("upper_arm_cuff", "forearm_cuff")
    IMU = "scapula_imu"
    GH_CENTRE = "/bodyset/humerus"   # the humerus frame sits at the GH joint centre

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.model, self.visual_model, _ = pa.build_pinocchio_model_from_opensim(
            str(self.data_dir / "Model" / "ThoracoscapularShoulderModel.osim"),
            str(self.data_dir / "Model" / "Geometry"),
            build_visual=True, root_vertical="pinocchio")        
        for g in list(self.visual_model.geometryObjects):    # drop the thoracic surface ellipsoid: show only bones
            if "ellips" in g.name.lower() or type(g.geometry).__name__ == "Ellipsoid":
                self.visual_model.removeGeometryObject(g.name)
        for (joint, k), (lower, upper) in BOUNDS.items():
            i = self.model.joints[self.model.getJointId(joint)].idx_q + k
            self.model.lowerPositionLimit[i], self.model.upperPositionLimit[i] = np.radians(lower), np.radians(upper)
        self.neutral = pin.neutral(self.model)
        self._add_frames()
        self.data = self.model.createData()

        self.ROOT = self.dofs("ground_thorax")                                   # 3 translations, 3 rotations
        self.GIRDLE = self.dofs("sternoclavicular") + self.dofs("scapulothoracic")  # 6, of which 3 free with the AC loop closed
        self.GH = self.dofs("GlenoHumeral")
        self.GH_Q = [self.model.joints[self.model.getJointId("GlenoHumeral")].idx_q + k for k in range(3)]
        self.UPWARD_ROTATION = self.model.joints[self.model.getJointId("scapulothoracic")].idx_q + 2

        self.q_ref = self.close_loop(self.neutral.copy())   # arm down, thorax upright, AC loop closed

    # ---- setup ---------------------------------------------------------------------------------------
    def _add_frames(self):
        d = self.model.createData()
        pin.framesForwardKinematics(self.model, d, self.neutral)

        def local(body, point):            # position of a model point in a body frame
            return d.oMf[self.fid(body)].actInv(d.oMf[self.fid(point)].translation)

        def add(name, body, offset):
            f = self.model.frames[self.fid(body)]
            self.model.addFrame(pin.Frame(name, f.parentJoint, self.fid(body),
                                          f.placement * pin.SE3(np.eye(3), offset), pin.FrameType.OP_FRAME))

        towards = lambda body, point, length: length * local(body, point) / np.linalg.norm(local(body, point))
        add(self.IMU, "/bodyset/scapula", local("/bodyset/scapula", "ac"))                 # scapula orientation
        add("upper_arm_cuff", "/bodyset/humerus", towards("/bodyset/humerus", "centelbow", 0.15))  # 15 cm below GH
        add("forearm_cuff", "/bodyset/ulna", towards("/bodyset/ulna", "rc", 0.08))          # 8 cm below the elbow

    def fid(self, name):
        return self.model.getFrameId(name)

    def dofs(self, joint):
        j = self.model.joints[self.model.getJointId(joint)]
        return list(range(j.idx_v, j.idx_v + j.nv))

    # ---- kinematics ----------------------------------------------------------------------------------
    def update(self, q):
        """Forward kinematics with Jacobians for configuration q."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def frame(self, name):
        return self.data.oMf[self.fid(name)]

    def jacobian(self, name, reference=pin.LOCAL_WORLD_ALIGNED):
        return pin.getFrameJacobian(self.model, self.data, self.fid(name), reference)

    def ac_loop(self):
        """AC loop error (m) and its Jacobian; call update() first."""
        error = self.frame("ac").translation - self.frame("acc").translation
        J = self.jacobian("ac")[:3] - self.jacobian("acc")[:3]
        return error, J

    def close_loop(self, q, R_humerus=None):
        """Close the AC loop with the smallest girdle change, keeping the humerus orientation
        (the given one, or the current one)."""
        q_start = q.copy()
        self.update(q)
        if R_humerus is None:
            R_humerus = self.frame("/bodyset/humerus").rotation.copy()
        for _ in range(60):
            self.update(q)
            loop, J_loop = self.ac_loop()
            turn = pin.log3(R_humerus @ self.frame("/bodyset/humerus").rotation.T)
            if np.linalg.norm(loop) < 1e-8 and np.linalg.norm(turn) < 1e-8:
                break
            J_hum = self.jacobian("/bodyset/humerus")[3:]
            step = np.zeros(self.model.nv)
            step[self.GIRDLE] = -np.linalg.pinv(J_loop[:, self.GIRDLE]) @ loop
            # GH step, pulled towards the starting GH angles: with the arm hanging down the GH angles are
            # singular (plane of elevation and axial rotation turn about the same axis) and would drift
            J_gh, pull = J_hum[:, self.GH], 1e-3
            step[self.GH] = np.linalg.solve(J_gh.T @ J_gh + pull * np.eye(3),
                                            J_gh.T @ (turn - J_hum[:, self.GIRDLE] @ step[self.GIRDLE])
                                            + pull * (q_start[self.GH_Q] - q[self.GH_Q]))
            q = pin.integrate(self.model, q, step)
        return q

    # ---- outcomes ------------------------------------------------------------------------------------
    def in_thorax(self, q, point):
        """Position of a model point in the thorax frame."""
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.frame("/bodyset/thorax").actInv(self.frame(point).translation)

    def acromion_height(self, q):
        """Height of the acromion (ac point) along the thorax vertical axis (mm)."""
        return 1000 * self.in_thorax(q, "ac")[1]

    def upward_rotation(self, q):
        """Scapular upward rotation coordinate of the model (deg)."""
        return float(np.degrees(q[self.UPWARD_ROTATION]))

    def arm_elevation(self, q):
        """Humerothoracic elevation (deg): angle of the humerus from the thorax downward vertical."""
        axis = self.in_thorax(q, "centelbow") - self.in_thorax(q, self.GH_CENTRE)
        return float(np.degrees(np.arccos(-axis[1] / np.linalg.norm(axis))))

    # ---- dataset -------------------------------------------------------------------------------------
    def load_trial(self, trial, step=1):
        """Times and configurations of a trial of the dataset (Results/IK/<trial>_IK.mot), every `step`-th frame.

        The thorax is put at the origin (its translations are not used). The dataset poses leave the AC loop
        open by about 2 cm, so each pose is closed with the smallest girdle change."""
        lines = (self.data_dir / "Results" / "IK" / f"{trial}_IK.mot").read_text().splitlines()
        header_end = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
        columns = lines[header_end + 1].split()
        values = np.loadtxt(lines[header_end + 2:], ndmin=2)[::step]
        Q = np.tile(self.neutral, (len(values), 1))
        for joint, names in MOT_COLUMNS.items():
            start = self.model.joints[self.model.getJointId(joint)].idx_q
            for k, name in enumerate(names):
                if name in columns and not name.endswith(("_tx", "_ty", "_tz")):
                    Q[:, start + k] = np.radians(values[:, columns.index(name)])
        return values[:, 0], np.array([self.close_loop(q) for q in Q])
