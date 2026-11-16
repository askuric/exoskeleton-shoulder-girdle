"""Observability of the shoulder girdle through the cuffs of an arm exoskeleton.

Replays a trial of the Seth et al. dataset. At each pose it computes how far the GH centre (and with it the cuffs)
moves per degree of girdle motion, and moves the girdle along the least observable direction while the cuffs hold
the arm.

    python scripts/observability.py --trial flx21           # 3D viewer (browser) + sliders
    python scripts/observability.py --trial abd01 --plot    # singular values along the trial
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path

import meshcat.geometry as mg
import meshcat.transformations as mt
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so that `girdle` is found
from girdle import Shoulder, move_along, observability  # noqa: E402
from girdle.model import UP  # noqa: E402

GREEN, RED, BLUE, PURPLE = 0x1BAF7A, 0xE8221B, 0x2A78D6, 0x6250D6


def plot(shoulder, trial):
    import matplotlib.pyplot as plt
    times, Q = shoulder.load_trial(trial)
    elevation = [shoulder.arm_elevation(q) for q in Q]
    results = [observability(shoulder, q) for q in Q]
    sigma = np.array([r[0] for r in results]); sigma_imu = np.array([r[1] for r in results])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    top.plot(times, elevation, "k"); top.set_ylabel("arm elevation (deg)")
    bottom.plot(times, sigma[:, :2], color="0.7")
    bottom.plot(times, sigma[:, 2], color="#6250d6", label=r"$\sigma_3$, cuffs only")
    bottom.plot(times, sigma_imu[:, 2], color="#2a78d6", label=r"$\sigma_3$, + scapula IMU")
    bottom.set_ylabel("singular value (mm/deg)"); bottom.set_xlabel("time (s)"); bottom.set_ylim(0, None)
    bottom.legend(frameon=False)
    fig.suptitle(f"{trial}: how well the cuffs see the shoulder girdle"); fig.tight_layout(); plt.show()


def viewer(shoulder, trial):
    s = shoulder
    times, Q = s.load_trial(trial)

    viz = MeshcatVisualizer(s.model, s.visual_model, s.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel(rootNodeName="shoulder")
    for item in ("/Background", "/Grid", "/Axes"):     # plain white scene
        viz.viewer[item].set_property("visible", False)
    points = {"gh": (s.GH_CENTRE, 0.010, GREEN), "ac": ("ac", 0.007, RED), "ai": ("ai", 0.007, RED),
              "upper_arm_cuff": ("upper_arm_cuff", 0.012, GREEN), "forearm_cuff": ("forearm_cuff", 0.012, GREEN)}
    for name, (_, radius, colour) in points.items():
        viz.viewer[f"points/{name}"].set_object(mg.Sphere(radius), mg.MeshLambertMaterial(color=colour, opacity=0.8, transparent=True))
    viz.viewer["points/imu"].set_object(mg.Box([0.035, 0.012, 0.035]), mg.MeshLambertMaterial(color=BLUE))
    # the IMU is drawn flat on top of the acromion, between the AC joint and the acromial angle (as in exoskeleton_live)
    s.update(s.q_ref)
    scapula = s.frame("/bodyset/scapula")
    imu_offset = scapula.actInv(0.5 * (s.frame("ac").translation + s.frame("aa").translation) + 0.008 * UP)

    def show(q, imu_on):
        viz.display(q)
        s.update(q)
        for name, (frame, _, _) in points.items():
            viz.viewer[f"points/{name}"].set_transform(mt.translation_matrix(s.frame(frame).translation))
        scapula = s.frame("/bodyset/scapula")
        T = np.eye(4); T[:3, :3] = scapula.rotation; T[:3, 3] = scapula.act(imu_offset)
        viz.viewer["points/imu"].set_transform(T)
        viz.viewer["points/imu"].set_property("visible", bool(imu_on))

    def positions(q):
        s.update(q)
        return {name: s.frame(frame).translation.copy() for name, (frame, _, _) in points.items()}, s.frame(s.IMU).rotation.T @ UP

    window = tk.Tk(); window.title(f"Shoulder girdle observability: {trial}")
    frame_var, angle_var, imu_var = tk.IntVar(value=len(Q) // 3), tk.DoubleVar(value=0.0), tk.BooleanVar(value=True)
    text = tk.StringVar()
    state = {"rocking": False, "phase": 0.0}

    def refresh(*_):
        q0 = Q[frame_var.get()]
        sigma, sigma_imu, weakest = observability(s, q0)
        q = move_along(s, q0, weakest, angle_var.get()) if angle_var.get() else q0
        show(q, imu_var.get())
        p0, g0 = positions(q0); p, g = positions(q)
        moved = lambda name: 1000 * np.linalg.norm(p[name] - p0[name])
        lines = [f"time {times[frame_var.get()]:.2f} s, arm elevation {s.arm_elevation(q0):.0f} deg", "",
                 "singular values (mm of GH motion per deg of girdle motion):",
                 f"  sigma_1 {sigma[0]:.2f}   sigma_2 {sigma[1]:.2f}   sigma_3 {sigma[2]:.3f}"]
        if imu_var.get():
            lines.append(f"  with the scapula IMU: sigma_3 {sigma_imu[2]:.2f}")
        lines += ["", f"girdle moved {angle_var.get():+.0f} deg along the least observable direction:",
                  f"  inferior angle {moved('ai'):5.1f} mm",
                  f"  acromion       {moved('ac'):5.1f} mm",
                  f"  GH centre      {moved('gh'):5.1f} mm   (what the cuffs feel)"]
        if imu_var.get():
            lines.append(f"  scapula IMU tilts {np.degrees(np.arccos(np.clip(g @ g0, -1, 1))):.1f} deg")
        text.set("\n".join(lines))

    def rock():
        if state["rocking"]:
            state["phase"] += 0.12
            angle_var.set(round(15 * np.sin(state["phase"]), 1)); refresh()
            window.after(40, rock)

    def toggle():
        state["rocking"] = not state["rocking"]
        button.config(text="Stop" if state["rocking"] else "Move along the least observable direction")
        if state["rocking"]:
            rock()
        else:
            angle_var.set(0.0); refresh()

    tk.Scale(window, label="trial frame", from_=0, to=len(Q) - 1, orient="horizontal", length=420,
             variable=frame_var, command=refresh).pack(padx=10, pady=4)
    tk.Scale(window, label="girdle motion along the least observable direction (deg)", from_=-15, to=15,
             resolution=0.5, orient="horizontal", length=420, variable=angle_var, command=refresh).pack(padx=10, pady=4)
    button = tk.Button(window, text="Move along the least observable direction", command=toggle); button.pack(pady=4)
    tk.Checkbutton(window, text="scapula IMU (gravity only)", variable=imu_var, command=refresh).pack(pady=4)
    tk.Label(window, textvariable=text, justify="left", font=("TkFixedFont", 10)).pack(padx=10, pady=8, anchor="w")
    refresh()
    window.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Observability of the shoulder girdle through the exoskeleton cuffs.")
    parser.add_argument("--data", default="ThoracoscapularShoulderPaperMaterials",
                        help="folder with the materials of Seth et al. (simtk.org/projects/thoracoscapular)")
    parser.add_argument("--trial", default="flx21", help="dataset trial, e.g. abd01, flx21, shrug02")
    parser.add_argument("--plot", action="store_true", help="plot the singular values along the trial")
    args = parser.parse_args()

    shoulder = Shoulder(args.data)
    plot(shoulder, args.trial) if args.plot else viewer(shoulder, args.trial)
