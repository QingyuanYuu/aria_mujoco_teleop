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
from ik_solver import ik_step_pos, ik_step_pos_body, ik_step_pos_orient, ik_step_pos_orient_body, direction_to_quaternion
from utils import apply_axis_ops, quaternion_to_forward
from finger_control import FingerTeleop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="MuJoCo XML path (e.g. franka_panda.xml)")
    ap.add_argument("--ee-site", type=str, default="panda0_gripper", help="EE site/body name (default: panda0_gripper)")
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
    ap.add_argument("--enable-orient", action="store_true", default=True, help="Enable orientation control (wrist-to-palm direction aligns with gripper-to-fingers)")
    ap.add_argument("--disable-orient", action="store_false", dest="enable_orient", help="Disable orientation control")
    ap.add_argument("--ik-damping-orient", type=float, default=1e-1, help="Damping for orientation IK")

    # hand visualization
    ap.add_argument("--show-hand", action="store_true", help="Visualize hand skeleton (3D) in MuJoCo viewer")
    ap.add_argument("--hand-point-size", type=float, default=0.008, help="Hand point sphere radius")
    ap.add_argument("--hand-edge-radius", type=float, default=0.004, help="Hand edge capsule radius")
    
    # finger control
    ap.add_argument("--enable-fingers", action="store_true", help="Enable finger control (auto-calibrated pinch-to-gripper)")
    ap.add_argument("--finger-alpha", type=float, default=0.7, help="Finger control smoothing (0..1, higher = faster response)")
    ap.add_argument("--finger-verbose", action="store_true", help="Print finger control debug information")
    ap.add_argument("--test-fingers", action="store_true", help="Run finger oscillation test before starting teleoperation")

    args = ap.parse_args()

    # 1) start receiver (bind 6768 once)
    start_receiver(args.host, args.port)

    # 2) load MuJoCo
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Try to find site first, if not found, try body
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.ee_site)
    if site_id < 0:
        # Try as body instead
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.ee_site)
        if body_id < 0:
            raise ValueError(f"Site/body '{args.ee_site}' not found. Use --ee-site panda0_gripper or end_effector.")
        # Use body's position
        site_id = None
        use_body = True
        body_id_for_pos = body_id
    else:
        use_body = False
        body_id_for_pos = None
    
    # Find finger body to compute midpoint between gripper and finger
    finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda0_leftfinger")
    if finger_body_id < 0:
        finger_body_id = None
        print("[Warning] panda0_leftfinger not found, using gripper position directly")
    else:
        print(f"[Teleop] Using midpoint between gripper and finger for control")
    
    # Find finger joints to exclude from IK
    finger_joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda0_finger_joint1")
    finger_joint2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda0_finger_joint2")
    exclude_joints = []
    if finger_joint1_id >= 0:
        exclude_joints.append(finger_joint1_id)
    if finger_joint2_id >= 0:
        exclude_joints.append(finger_joint2_id)
    if exclude_joints:
        print(f"[Teleop] Excluding finger joints from IK: {exclude_joints}")

    # 3) calibration: first confident hand sample defines origin mapping
    print("[Teleop] Waiting for first confident right-hand sample...")
    origin_dev = None
    if use_body:
        gripper_pos = data.xpos[body_id_for_pos].copy()
    else:
        gripper_pos = data.site_xpos[site_id].copy()
    
    # Compute midpoint between gripper and finger if finger exists
    if finger_body_id is not None:
        finger_pos = data.xpos[finger_body_id].copy()
        origin_mj = (gripper_pos + finger_pos) / 2.0
        print(f"[Teleop] Using midpoint between gripper and finger as control target")
    else:
        origin_mj = gripper_pos.copy()
        print(f"[Teleop] Using '{args.ee_site}' as control target")

    while origin_dev is None:
        h = get_hand_state()
        if h.valid and h.wrist_dev is not None and h.conf >= args.conf:
            origin_dev = h.wrist_dev.copy()
            # Recompute midpoint for calibration
            if use_body:
                gripper_pos = data.xpos[body_id_for_pos].copy()
            else:
                gripper_pos = data.site_xpos[site_id].copy()
            
            if finger_body_id is not None:
                finger_pos = data.xpos[finger_body_id].copy()
                origin_mj = (gripper_pos + finger_pos) / 2.0
            else:
                origin_mj = gripper_pos.copy()
            
            print(f"[Teleop] Calibrated origin.")
            print(f"         origin_dev={origin_dev}")
            print(f"         origin_mj ={origin_mj}")
            if args.enable_orient:
                print(f"[Teleop] Orientation control enabled: wrist-to-palm direction -> gripper-to-fingers")
        else:
            time.sleep(0.01)

    target = origin_mj.copy()
    target_smooth = target.copy()
    
    # Initialize orientation target
    if use_body:
        target_quat = data.xquat[body_id_for_pos].copy()
    else:
        target_quat = data.site_xquat[site_id].copy()
    target_quat_smooth = target_quat.copy()

    # Initialize finger teleop controller (auto-calibrated)
    finger_teleop = None
    if args.enable_fingers:
        try:
            finger_teleop = FingerTeleop(
                model,
                alpha=args.finger_alpha,
                init_value=0.02,  # start half-open
            )
            print("[Teleop] Finger control enabled (auto-calibrated)")
        except Exception as e:
            print(f"[Teleop] WARNING: Failed to initialize finger control: {e}")
            print("[Teleop] Continuing without finger control...")
            args.enable_fingers = False

    # --- sanity test: oscillate finger joints for 2 seconds ---
    if args.test_fingers:
        print("[Test] Running finger oscillation test...")
        jid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda0_finger_joint1")
        jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda0_finger_joint2")
        print(f"[Test] finger joint ids: {jid1}, {jid2}")
        if jid1 >= 0 and jid2 >= 0:
            print(f"[Test] qposadr: {model.jnt_qposadr[jid1]}, {model.jnt_qposadr[jid2]}")
            print(f"[Test] ranges: {model.jnt_range[jid1]}, {model.jnt_range[jid2]}")
            
            q1 = model.jnt_qposadr[jid1]
            q2 = model.jnt_qposadr[jid2]
            
            for k in range(200):
                v = 0.02 + 0.02 * np.sin(2 * np.pi * k / 200)  # 0~0.04
                data.qpos[q1] = v
                data.qpos[q2] = v
                mujoco.mj_forward(model, data)
                time.sleep(0.01)
            print("[Test] Finger oscillation test completed.")
        else:
            print("[Test] WARNING: Finger joints not found, skipping test.")

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
                    
                    # Compute current gripper position
                    if use_body:
                        current_gripper_pos = data.xpos[body_id_for_pos].copy()
                    else:
                        current_gripper_pos = data.site_xpos[site_id].copy()
                    
                    # Compute target gripper position
                    target_gripper_pos = origin_mj + delta
                    
                    # If finger exists, compute midpoint between gripper and finger
                    if finger_body_id is not None:
                        current_finger_pos = data.xpos[finger_body_id].copy()
                        # Compute the offset from current finger position
                        finger_offset = current_finger_pos - current_gripper_pos
                        # Apply the same offset to target gripper position
                        target_finger_pos = target_gripper_pos + finger_offset
                        # Use midpoint as target
                        target = (target_gripper_pos + target_finger_pos) / 2.0
                    else:
                        target = target_gripper_pos
                    
                    # Update orientation target from wrist-to-palm direction
                    # This ensures gripper-to-fingers direction aligns with wrist-to-palm direction
                    if args.enable_orient and h.wrist_to_palm_dir_dev is not None:
                        # Transform direction to MuJoCo frame
                        dir_dev = h.wrist_to_palm_dir_dev.copy()
                        dir_dev = apply_axis_ops(dir_dev, args.flip_x, args.flip_y, args.flip_z)
                        
                        # Convert direction to quaternion
                        # The quaternion aligns the gripper's +Z axis (pointing toward fingers) 
                        # with the wrist-to-palm direction
                        target_quat = direction_to_quaternion(dir_dev)
                        target_quat_smooth = (1 - args.alpha) * target_quat_smooth + args.alpha * target_quat
                        # Normalize quaternion
                        target_quat_smooth = target_quat_smooth / np.linalg.norm(target_quat_smooth)
                    
                    target_smooth = (1 - args.alpha) * target_smooth + args.alpha * target

                # IK step
                # Note: target_smooth already includes the offset along gripper-to-fingers direction
                # Exclude finger joints from IK to prevent overwriting finger control
                if args.enable_orient:
                    # Position and orientation IK
                    if use_body:
                        err_pos, err_orient = ik_step_pos_orient_body(
                            model, data, body_id_for_pos, target_smooth, target_quat_smooth,
                            damping_pos=args.ik_damping,
                            damping_orient=args.ik_damping_orient,
                            step_scale=args.ik_step_scale,
                            max_dq=args.ik_max_dq,
                            exclude_joints=exclude_joints if exclude_joints else None,
                        )
                        err = err_pos
                    else:
                        err_pos, err_orient = ik_step_pos_orient(
                            model, data, site_id, target_smooth, target_quat_smooth,
                            damping_pos=args.ik_damping,
                            damping_orient=args.ik_damping_orient,
                            step_scale=args.ik_step_scale,
                            max_dq=args.ik_max_dq,
                            exclude_joints=exclude_joints if exclude_joints else None,
                        )
                        err = err_pos
                else:
                    # Position-only IK
                    if use_body:
                        err = ik_step_pos_body(
                            model, data, body_id_for_pos, target_smooth,
                            damping=args.ik_damping,
                            step_scale=args.ik_step_scale,
                            max_dq=args.ik_max_dq,
                            exclude_joints=exclude_joints if exclude_joints else None,
                        )
                    else:
                        err = ik_step_pos(
                            model, data, site_id, target_smooth,
                            damping=args.ik_damping,
                            step_scale=args.ik_step_scale,
                            max_dq=args.ik_max_dq,
                            exclude_joints=exclude_joints if exclude_joints else None,
                        )
                
                # --- finger control (after IK to prevent IK from overwriting) ---
                # Use auto-calibrated finger teleop controller
                if args.enable_fingers and finger_teleop is not None:
                    finger_teleop.update_from_landmarks(
                        data,
                        h.landmarks_dev if h.valid else None,
                        do_forward=True,  # Apply finger control and call mj_forward
                        verbose=args.finger_verbose,
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
