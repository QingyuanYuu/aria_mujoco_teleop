#!/usr/bin/env python3
"""
Main script for Aria-MuJoCo teleoperation.

This script orchestrates hand tracking, stream reception, visualization,
and inverse kinematics to enable real-time robot control via hand movements.
"""
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_tracking import HandState, get_hand_state, HAND_EDGES_21
from stream_receiver import start_receiver
from visualization import draw_hand_markers, clear_markers
from ik_solver import ik_step_pos
from utils import apply_axis_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="MuJoCo XML path (e.g. franka_panda.xml)")
    ap.add_argument("--ee-site", type=str, default="end_effector", help="EE site name (default: end_effector)")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6768)

    # mapping / smoothing
    ap.add_argument("--scale", type=float, default=1.0, help="Scale device delta -> MJ meters")
    ap.add_argument("--alpha", type=float, default=0.2, help="EMA smoothing (0..1); smaller=more smooth")
    ap.add_argument("--conf", type=float, default=0.5, help="Min hand confidence to accept updates")
    ap.add_argument("--timeout", type=float, default=0.25, help="Seconds without update => hold target")
    ap.add_argument("--flip-x", action="store_true", help="Flip mapped X")
    ap.add_argument("--flip-y", action="store_true", help="Flip mapped Y")
    ap.add_argument("--flip-z", action="store_true", help="Flip mapped Z (common for up/down)")

    # IK params
    ap.add_argument("--ik-damping", type=float, default=1e-2)
    ap.add_argument("--ik-step-scale", type=float, default=0.4)
    ap.add_argument("--ik-max-dq", type=float, default=0.12)

    # hand visualization
    ap.add_argument("--show-hand", action="store_true", help="Visualize hand skeleton (3D) in MuJoCo viewer")
    ap.add_argument("--hand-point-size", type=float, default=0.008, help="Hand point sphere radius")
    ap.add_argument("--hand-edge-radius", type=float, default=0.004, help="Hand edge capsule radius")

    args = ap.parse_args()

    # 1) start receiver (bind 6768 once)
    start_receiver(args.host, args.port)

    # 2) load MuJoCo
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.ee_site)
    if site_id < 0:
        raise ValueError(f"Site '{args.ee_site}' not found. Use --ee-site end_effector or ee_site.")

    # 3) calibration: first confident hand sample defines origin mapping
    print("[Teleop] Waiting for first confident right-hand sample...")
    origin_dev = None
    origin_mj = data.site_xpos[site_id].copy()

    while origin_dev is None:
        h = get_hand_state()
        if h.valid and h.wrist_dev is not None and h.conf >= args.conf:
            origin_dev = h.wrist_dev.copy()
            origin_mj = data.site_xpos[site_id].copy()
            print(f"[Teleop] Calibrated origin.")
            print(f"         origin_dev={origin_dev}")
            print(f"         origin_mj ={origin_mj}")
        else:
            time.sleep(0.01)

    target = origin_mj.copy()
    target_smooth = target.copy()

    # 4) viewer loop
    print("[Teleop] Running. Move your right wrist; EE follows. Close viewer or Ctrl-C to exit.")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                now_ns = time.time_ns()
                h = get_hand_state()

                age_s = (now_ns - h.t_ns) / 1e9 if h.t_ns else 999.0

                # update EE target from wrist
                if h.valid and h.wrist_dev is not None and h.conf >= args.conf and age_s <= args.timeout:
                    delta = (h.wrist_dev - origin_dev) * args.scale
                    delta = apply_axis_ops(delta, args.flip_x, args.flip_y, args.flip_z)
                    target = origin_mj + delta
                    target_smooth = (1 - args.alpha) * target_smooth + args.alpha * target

                # IK step
                err = ik_step_pos(
                    model, data, site_id, target_smooth,
                    damping=args.ik_damping,
                    step_scale=args.ik_step_scale,
                    max_dq=args.ik_max_dq,
                )

                # hand skeleton visualization in same mapped space (optional)
                if args.show_hand:
                    pts_dev = None
                    edges = None

                    if h.landmarks_dev is not None and h.landmarks_dev.ndim == 2 and h.landmarks_dev.shape[1] == 3:
                        pts_dev = h.landmarks_dev
                        edges = HAND_EDGES_21 if pts_dev.shape[0] == 21 else None
                    else:
                        fallback = []
                        if h.wrist_dev is not None:
                            fallback.append(h.wrist_dev)
                        if h.palm_dev is not None:
                            fallback.append(h.palm_dev)
                        if len(fallback) >= 1:
                            pts_dev = np.stack(fallback, axis=0)
                            edges = [(0, 1)] if len(fallback) == 2 else None

                    if pts_dev is not None and origin_dev is not None:
                        pts_delta = (pts_dev - origin_dev[None, :]) * args.scale
                        # apply same axis flips as EE target
                        if args.flip_x:
                            pts_delta[:, 0] *= -1
                        if args.flip_y:
                            pts_delta[:, 1] *= -1
                        if args.flip_z:
                            pts_delta[:, 2] *= -1

                        pts_mj = origin_mj[None, :] + pts_delta
                        draw_hand_markers(
                            viewer,
                            pts_mj,
                            edges=edges,
                            sphere_size=args.hand_point_size,
                            edge_radius=args.hand_edge_radius,
                        )
                    else:
                        # clear markers if no data
                        clear_markers(viewer)

                # Some mujoco versions don't expose viewer.overlay on the passive handle.
                if hasattr(viewer, "overlay"):
                    try:
                        viewer.overlay(
                            mujoco.viewer.OverlayGrid.TOPLEFT,
                            "Aria streaming follow",
                            f"site={args.ee_site} conf={h.conf:.2f} age={age_s*1000:.0f}ms err={err:.3f}m",
                        )
                    except Exception:
                        pass

                viewer.sync()
                time.sleep(0.002)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
