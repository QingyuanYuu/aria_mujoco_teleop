"""
Finger control module for gripper teleoperation.

This module handles control of robot gripper fingers based on hand tracking.
Distance between thumb and index finger tips controls both left and right fingers.
"""
import numpy as np
import mujoco


def _set_finger_qpos_and_forward(model, data, value: float, verbose: bool = False):
    """
    Mimic the simple test loop behavior:
    - set both finger qpos to value
    - mj_forward so viewer reflects it immediately
    """
    jid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda0_finger_joint1")
    jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda0_finger_joint2")
    if jid1 < 0 or jid2 < 0:
        if verbose:
            print("[Finger] joints not found")
        return

    q1 = model.jnt_qposadr[jid1]
    q2 = model.jnt_qposadr[jid2]

    lo1, hi1 = model.jnt_range[jid1]
    lo2, hi2 = model.jnt_range[jid2]
    lo = max(lo1, lo2)
    hi = min(hi1, hi2)

    v = float(np.clip(value, lo, hi))

    data.qpos[q1] = v
    data.qpos[q2] = v

    # Important: immediately propagate kinematics for rendering
    mujoco.mj_forward(model, data)

    if verbose:
        print(f"[Finger] set qpos[{q1}]={v:.4f} qpos[{q2}]={v:.4f}")


def pinch_to_gripper_value(
    thumb_pos_dev: np.ndarray,
    index_pos_dev: np.ndarray,
    *,
    min_distance: float = 0.015,
    max_distance: float = 0.065,
    min_gripper: float = 0.0,
    max_gripper: float = 0.04,
    alpha: float = 0.8,
    prev_value: float = 0.0,
    scale: float = 1.0,
    amplify: float = 1.0,
) -> float:
    """
    Compute gripper target from thumb-index distance (device frame).
    Returns a smoothed value in [0, 0.04].
    
    Args:
        amplify: Amplification factor for distance-to-gripper mapping.
                 > 1.0 makes small distance changes produce larger gripper value changes.
    """
    d = float(np.linalg.norm(thumb_pos_dev - index_pos_dev)) * scale
    d = float(np.clip(d, min_distance, max_distance))
    normalized = (d - min_distance) / (max_distance - min_distance)
    # Apply amplification: amplify the normalized value to make small changes more visible
    normalized = normalized * amplify
    normalized = float(np.clip(normalized, 0.0, 1.0))  # Clamp back to [0, 1]
    target = min_gripper + normalized * (max_gripper - min_gripper)
    smooth = (1 - alpha) * prev_value + alpha * target
    return float(np.clip(smooth, min_gripper, max_gripper))
