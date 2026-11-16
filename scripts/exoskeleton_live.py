"""Reconstruct the shoulder girdle from the cuffs of a generic arm exoskeleton, live.

Two modes:

    python scripts/exoskeleton_live.py                 # move the robot joints with sliders
    python scripts/exoskeleton_live.py --trial abd21   # the robot follows a recorded movement of the dataset

The robot only knows its joint angles, and from them the poses of its two cuffs. From these, the shoulder model is
fitted (model IK) with the cuffs only, with a thorax IMU, or with a thorax and a scapula IMU.

With the sliders there is no true shoulder, so the IMU readings are set with sliders too: the thorax orientation
and the tilt of the scapula IMU (from its reading with the arm down). With --trial the true shoulder is shown in
white and the IMUs read it.
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path

import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so that `girdle` is found
from girdle import CASES, Exoskeleton, Shoulder, estimate_case, sensors  # noqa: E402
from girdle.robot import JOINTS  # noqa: E402

LABELS = {"robot": "cuffs only", "thorax": "+ thorax IMU", "both": "+ thorax and scapula IMU"}
COLOURS = {"robot": (0.38, 0.31, 0.84), "thorax": (0.92, 0.41, 0.20), "both": (0.16, 0.47, 0.84)}
GREEN, GREY = 0x1BAF7A, 0x8D929A
START_DEG = (0, 0, 30, 0, 0, 85)   # initial robot joint angles in the sliders mode (deg), in the order of JOINTS


def cylinder(a, b, radius):
    """Transform that stretches a unit meshcat cylinder (axis y, centred) from point a to point b."""
    axis = b - a
    y = axis / np.linalg.norm(axis)
    x = np.cross(y, [0.0, 0.0, 1.0]) if abs(y[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([radius * x, np.linalg.norm(axis) * y, radius * np.cross(x, y)])
    T[:3, 3] = (a + b) / 2
    return T


def on_axis(point, axis, half_length, radius):
    """A motor: short cylinder on a joint axis."""
    return cylinder(point - half_length * axis, point + half_length * axis, radius)


class Scene:
    """meshcat scene: the estimated shoulders (one per case), the true shoulder, the robot and the IMUs."""

    def __init__(self, shoulder, robot, show_truth, open_browser=True):
        s = shoulder
        self.shoulder, self.robot = shoulder, robot
        self.shoulders = {}
        viewer = None
        for name, colour in [*COLOURS.items(), ("truth", (0.95, 0.94, 0.90))]:
            if name == "truth" and not show_truth:
                continue
            viz = MeshcatVisualizer(s.model, s.visual_model, s.visual_model)
            viz.initViewer(viewer=viewer, open=open_browser and viewer is None)
            viz.loadViewerModel(rootNodeName=name, visual_color=[*colour, 1.0 if name == "truth" else 0.6])
            viewer = viz.viewer
            self.shoulders[name] = viz
        self.viewer = v = viewer
        for item in ("/Background", "/Grid", "/Axes"):     # plain white scene
            v[item].set_property("visible", False)

        metal = mg.MeshLambertMaterial(color=GREY)
        unit = mg.Cylinder(1.0, 1.0)
        for k in range(8):
            v[f"exo/link_{k}"].set_object(unit, metal)
        for k in range(len(JOINTS)):
            v[f"exo/motor_{k}"].set_object(unit, metal)
        v["exo/base"].set_object(mg.Box([0.10, 0.10, 0.03]), metal)
        v["exo/gh_centre"].set_object(mg.Sphere(0.011), mg.MeshLambertMaterial(color=GREEN))
        cuff = mg.MeshLambertMaterial(color=GREEN, opacity=0.45, transparent=True)
        v["exo/cuff_0"].set_object(mg.Cylinder(0.06, 0.052), cuff)
        v["exo/cuff_1"].set_object(mg.Cylinder(0.06, 0.042), cuff)
        v["imu/thorax"].set_object(mg.Box([0.012, 0.035, 0.035]), mg.MeshLambertMaterial(color=0xEB6834))
        v["imu/scapula"].set_object(mg.Box([0.035, 0.012, 0.035]), mg.MeshLambertMaterial(color=0x2A78D6))

        # the scapula IMU is drawn on top of the acromion, between the AC joint and the acromial angle
        s.update(s.q_ref)
        scapula = s.frame("/bodyset/scapula")
        self.scapula_imu_offset = scapula.actInv(0.5 * (s.frame("ac").translation + s.frame("aa").translation)
                                                 + 0.008 * robot.up)

    def show(self, q_robot, estimates, case, q_true=None):
        v = self.viewer
        for name, viz in self.shoulders.items():
            visible = name == case or name == "truth"
            v[name].set_property("visible", visible)
            if visible:
                viz.display(q_true if name == "truth" else estimates[name])
        self.show_robot(q_robot)
        self.show_imus(q_true if q_true is not None else estimates[case], case)

    def show_robot(self, q):
        """Draw the robot outside the body: each motor sits on its own joint axis, away from the joint centre,
        so the kinematics are those of the model. The robot hangs from a base plate on its first motor, above the user."""
        r, v = self.robot, self.viewer
        joints, axes, cuffs = r.joint_positions(q), r.joint_axes(q), r.cuffs(q)
        pivot, gh, elbow = joints[0], joints[2], joints[5]
        side = axes[3]                                        # links along the arm run on this side of it
        top = pivot + 0.45 * axes[0]                          # girdle protraction motor, on its axis above the head
        back = pivot + 0.25 * axes[1]                         # girdle elevation motor, on its axis behind the back
        m1, m2 = gh + 0.11 * axes[2], gh + 0.11 * axes[3]     # shoulder motors, around the GH centre
        m3 = gh + 0.07 * axes[4] + 0.07 * side
        upper_cuff = cuffs[0].translation + 0.055 * side
        elbow_motor = elbow + 0.07 * axes[5]
        forearm_cuff = cuffs[1].translation + 0.055 * axes[5]
        beam_end = top - 0.25 * r.anterior                    # from the top motor back, then down behind the head

        motors = [(top, axes[0], 0.030), (back, axes[1], 0.028), (m1, axes[2], 0.030), (m2, axes[3], 0.030),
                  (m3, axes[4], 0.030), (elbow_motor, axes[5], 0.026)]
        for k, (p, a, radius) in enumerate(motors):
            v[f"exo/motor_{k}"].set_transform(on_axis(p, a, 0.03, radius))
        links = [(back, m1), (m1, m2), (m2, m3), (m3, upper_cuff), (upper_cuff, elbow_motor),
                 (elbow_motor, forearm_cuff), (top, beam_end), (beam_end, back)]
        for k, (a, b) in enumerate(links):
            v[f"exo/link_{k}"].set_transform(cylinder(a, b, 0.012))
        T = np.eye(4); T[:3, 3] = top + 0.045 * axes[0]       # base plate right on top of the first motor
        v["exo/base"].set_transform(T)
        T = np.eye(4); T[:3, 3] = gh
        v["exo/gh_centre"].set_transform(T)
        for k, (a, b) in enumerate(((gh, elbow), (elbow, cuffs[1].translation))):
            T = cylinder(a, b, 1.0); T[:3, 1] /= np.linalg.norm(b - a); T[:3, 3] = cuffs[k].translation
            v[f"exo/cuff_{k}"].set_transform(T)             # ring around the arm segment, at the cuff

    def show_imus(self, q, case):
        """Thorax IMU on the sternum and scapula IMU on the acromion of the body they measure."""
        s, v = self.shoulder, self.viewer
        s.update(q)
        thorax = s.frame("/bodyset/thorax")
        T = np.eye(4); T[:3, :3] = thorax.rotation
        T[:3, 3] = 0.5 * (s.frame("ij").translation + s.frame("px").translation) + 0.02 * thorax.rotation[:, 0]
        v["imu/thorax"].set_transform(T)
        scapula = s.frame("/bodyset/scapula")
        T = np.eye(4); T[:3, :3] = scapula.rotation; T[:3, 3] = scapula.act(self.scapula_imu_offset)
        v["imu/scapula"].set_transform(T)
        v["imu/thorax"].set_property("visible", case in ("thorax", "both"))
        v["imu/scapula"].set_property("visible", case == "both")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct the shoulder girdle from the exoskeleton cuffs.")
    parser.add_argument("--data", default="ThoracoscapularShoulderPaperMaterials",
                        help="folder with the materials of Seth et al. (simtk.org/projects/thoracoscapular)")
    parser.add_argument("--trial", help="follow a dataset trial (e.g. abd21, flx21, shrug21) instead of the sliders")
    args = parser.parse_args()

    shoulder = Shoulder(args.data)
    robot = Exoskeleton(shoulder)
    scene = Scene(shoulder, robot, show_truth=args.trial is not None)
    if args.trial:
        times, Q = shoulder.load_trial(args.trial, step=2)
        cases = CASES
    else:
        cases = CASES
        thorax_ref, gravity_ref = sensors(shoulder, shoulder.q_ref)   # IMU readings of an upright user, arm down

    state = {"q_robot": robot.q0.copy(), "estimates": {c: shoulder.q_ref.copy() for c in CASES}, "playing": False}
    ac_ref = shoulder.acromion_height(shoulder.q_ref)

    window = tk.Tk(); window.title(f"Exoskeleton: {args.trial or 'move the joints'}")
    case_var = tk.StringVar(value=cases[-1])
    text = tk.StringVar()

    def refresh(*_):
        if args.trial:
            q_true = Q[frame_var.get()]
            shoulder.update(q_true)
            targets = [shoulder.frame(c).copy() for c in shoulder.CUFFS]
            state["q_robot"], reach_error = robot.follow(state["q_robot"], targets)
            imu = sensors(shoulder, q_true)
        else:
            q_true = None
            state["q_robot"] = np.radians([v.get() for v in joint_vars])
            # thorax IMU: rotation coordinates of the thorax; scapula IMU: its reading with the arm down, after
            # turning the scapula by the slider angles (about the thorax forward and lateral axes)
            thorax = thorax_ref + np.radians([v.get() for v in thorax_vars])
            turn = pin.exp3(np.radians(scapula_vars[0].get()) * -robot.anterior
                            + np.radians(scapula_vars[1].get()) * robot.right)
            shoulder.update(shoulder.q_ref)
            R_imu = turn @ shoulder.frame(shoulder.IMU).rotation
            imu = (thorax, R_imu.T @ np.array([0.0, 0.0, 1.0]))
        cuffs = robot.cuffs(state["q_robot"])
        for c in cases:                                    # a few steps, since the sliders can jump
            q = state["estimates"][c]
            for _ in range(3):
                q = estimate_case(shoulder, c, q, q, cuffs, imu)
            state["estimates"][c] = q

        case = case_var.get()
        q_est = state["estimates"][case]
        scene.show(state["q_robot"], state["estimates"], case, q_true)

        lines = [f"estimate: {LABELS[case]}", "",
                 "                       estimate" + ("     true" if args.trial else ""),
                 f"arm elevation (deg)    {shoulder.arm_elevation(q_est):8.1f}" + (f" {shoulder.arm_elevation(q_true):8.1f}" if args.trial else ""),
                 f"acromion rise (mm)     {shoulder.acromion_height(q_est) - ac_ref:8.1f}" + (f" {shoulder.acromion_height(q_true) - ac_ref:8.1f}" if args.trial else ""),
                 f"upward rotation (deg)  {shoulder.upward_rotation(q_est):8.1f}" + (f" {shoulder.upward_rotation(q_true):8.1f}" if args.trial else "")]
        if args.trial:
            lines += ["", f"time {times[frame_var.get()]:.2f} s" + ("" if reach_error < 0.01 else
                      f"   (out of reach: cuffs {1000 * reach_error:.0f} mm off the arm)")]
        text.set("\n".join(lines))

    if args.trial:
        frame_var = tk.IntVar(value=0)
        tk.Scale(window, label="time (frame)", from_=0, to=len(Q) - 1, orient="horizontal", length=420,
                 variable=frame_var, command=refresh).pack(padx=10, pady=4)

        def play():
            if state["playing"]:
                frame_var.set((frame_var.get() + 1) % len(Q)); refresh()
                window.after(40, play)

        def toggle():
            state["playing"] = not state["playing"]
            play_button.config(text="Pause" if state["playing"] else "Play")
            play()

        play_button = tk.Button(window, text="Play", command=toggle); play_button.pack(pady=4)
    else:
        joint_vars = []
        for name, start in zip(JOINTS, START_DEG):
            var = tk.DoubleVar(value=start); joint_vars.append(var)
            tk.Scale(window, label=f"{name} (deg)", from_=-90, to=90, resolution=1, orient="horizontal",
                     length=420, variable=var, command=refresh).pack(padx=10, pady=2)
        imus = tk.LabelFrame(window, text="IMU readings"); imus.pack(padx=10, pady=4, fill="x")
        thorax_vars, scapula_vars = [], []
        for label in ("thorax IMU: rot_x, lean sideways", "thorax IMU: rot_y, twist",
                      "thorax IMU: rot_z, lean back (+) / forward (-)"):
            var = tk.DoubleVar(value=0.0); thorax_vars.append(var)
            tk.Scale(imus, label=f"{label} (deg)", from_=-30, to=30, resolution=1, orient="horizontal",
                     length=400, variable=var, command=refresh).pack(padx=6, pady=1)
        for label in ("scapula IMU: upward rotation", "scapula IMU: tilt about the lateral axis"):
            var = tk.DoubleVar(value=0.0); scapula_vars.append(var)
            tk.Scale(imus, label=f"{label} (deg, from arm down)", from_=-30, to=60, resolution=1,
                     orient="horizontal", length=400, variable=var, command=refresh).pack(padx=6, pady=1)

    for c in cases:
        tk.Radiobutton(window, text=LABELS[c], value=c, variable=case_var, command=refresh).pack(anchor="w", padx=10)
    tk.Label(window, textvariable=text, justify="left", font=("TkFixedFont", 10)).pack(padx=10, pady=8, anchor="w")
    refresh()
    window.mainloop()


if __name__ == "__main__":
    main()
