"""
Inverse Kinematics solver module.

This module implements position-based IK using Damped Least Squares (DLS) method.
"""
import numpy as np
import mujoco


def ik_step_pos(model, data, site_id, target_pos, damping=1e-2, step_scale=0.4, max_dq=0.12, exclude_joints=None):
    """
    Perform one step of position-based IK using Damped Least Squares.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        site_id: ID of the end-effector site
        target_pos: Target position (3,) in world frame
        damping: Damping factor for DLS (default: 1e-2)
        step_scale: Step scale for IK iterations (default: 0.4)
        max_dq: Maximum joint velocity change per step (default: 0.12)
        exclude_joints: List of joint IDs to exclude from IK (default: None)
    
    Returns:
        Position error magnitude (float)
    """
    ee = data.site_xpos[site_id].copy()
    err = (target_pos - ee).astype(np.float64)  # (3,)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    JJt = jacp @ jacp.T
    A = JJt + damping * np.eye(3)
    y = np.linalg.solve(A, err)
    dq = jacp.T @ y

    dq = np.clip(dq, -max_dq, max_dq)
    
    # Exclude finger joints from IK updates
    if exclude_joints is not None:
        for jid in exclude_joints:
            if jid >= 0:
                # Find the velocity dof index for this joint
                jnt_type = model.jnt_type[jid]
                jnt_dofadr = model.jnt_dofadr[jid]
                if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                    # Free joint has 6 velocity dofs (3 translation + 3 rotation)
                    for i in range(6):
                        if jnt_dofadr + i < model.nv:
                            dq[jnt_dofadr + i] = 0.0
                elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                    # Ball joint has 3 velocity dofs (rotation)
                    for i in range(3):
                        if jnt_dofadr + i < model.nv:
                            dq[jnt_dofadr + i] = 0.0
                else:
                    # Regular joint (hinge, slide, etc.) has 1 velocity dof
                    if jnt_dofadr < model.nv:
                        dq[jnt_dofadr] = 0.0
    
    mujoco.mj_integratePos(model, data.qpos, dq * step_scale, 1)
    mujoco.mj_forward(model, data)

    return float(np.linalg.norm(err))


def ik_step_pos_body(model, data, body_id, target_pos, damping=1e-2, step_scale=0.4, max_dq=0.12, exclude_joints=None):
    """
    Perform one step of position-based IK using Damped Least Squares for a body.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_id: ID of the target body
        target_pos: Target position (3,) in world frame
        damping: Damping factor for DLS (default: 1e-2)
        step_scale: Step scale for IK iterations (default: 0.4)
        max_dq: Maximum joint velocity change per step (default: 0.12)
        exclude_joints: List of joint IDs to exclude from IK (default: None)
    
    Returns:
        Position error magnitude (float)
    """
    ee = data.xpos[body_id].copy()
    err = (target_pos - ee).astype(np.float64)  # (3,)

    # Compute Jacobian for body position
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)

    JJt = jacp @ jacp.T
    A = JJt + damping * np.eye(3)
    y = np.linalg.solve(A, err)
    dq = jacp.T @ y

    dq = np.clip(dq, -max_dq, max_dq)
    
    # Exclude finger joints from IK updates
    if exclude_joints is not None:
        for jid in exclude_joints:
            if jid >= 0:
                jnt_type = model.jnt_type[jid]
                jnt_qposadr = model.jnt_qposadr[jid]
                if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                    for i in range(7):
                        if jnt_qposadr + i < model.nv:
                            dq[jnt_qposadr + i] = 0.0
                elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                    for i in range(4):
                        if jnt_qposadr + i < model.nv:
                            dq[jnt_qposadr + i] = 0.0
                else:
                    if jnt_qposadr < model.nv:
                        dq[jnt_qposadr] = 0.0
    
    mujoco.mj_integratePos(model, data.qpos, dq * step_scale, 1)
    mujoco.mj_forward(model, data)

    return float(np.linalg.norm(err))


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions. q1 * q2. Quaternions are [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def direction_to_quaternion(direction: np.ndarray, up_hint: np.ndarray = None) -> np.ndarray:
    """
    Convert a direction vector to a quaternion.
    The quaternion represents a rotation that aligns the +Z axis with the direction.
    
    Args:
        direction: Normalized direction vector (3,)
        up_hint: Optional hint for up direction to resolve ambiguity (3,)
    
    Returns:
        Quaternion [w, x, y, z]
    """
    direction = direction / np.linalg.norm(direction)
    
    # Default up direction
    if up_hint is None:
        up_hint = np.array([0, 0, 1], dtype=np.float64)
    else:
        up_hint = up_hint / np.linalg.norm(up_hint)
    
    # Compute right vector (cross product of up and direction)
    right = np.cross(up_hint, direction)
    if np.linalg.norm(right) < 1e-6:
        # If up and direction are parallel, use a different up
        right = np.cross(np.array([1, 0, 0]), direction)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(np.array([0, 1, 0]), direction)
    right = right / np.linalg.norm(right)
    
    # Compute actual up vector
    up = np.cross(direction, right)
    up = up / np.linalg.norm(up)
    
    # Build rotation matrix: columns are right, up, direction
    R = np.column_stack([right, up, direction])
    
    # Convert rotation matrix to quaternion
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    
    quat = np.array([w, x, y, z], dtype=np.float64)
    # Normalize
    quat = quat / np.linalg.norm(quat)
    return quat


def ik_step_pos_orient(model, data, site_id, target_pos, target_quat, 
                       damping_pos=1e-2, damping_orient=1e-1, 
                       step_scale=0.4, max_dq=0.12, exclude_joints=None):
    """
    Perform one step of position and orientation IK using Damped Least Squares.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        site_id: ID of the end-effector site
        target_pos: Target position (3,) in world frame
        target_quat: Target orientation as quaternion (4,) [w, x, y, z]
        damping_pos: Damping factor for position DLS (default: 1e-2)
        damping_orient: Damping factor for orientation DLS (default: 1e-1)
        step_scale: Step scale for IK iterations (default: 0.4)
        max_dq: Maximum joint velocity change per step (default: 0.12)
    
    Returns:
        Tuple of (position_error, orientation_error) magnitudes
    """
    ee_pos = data.site_xpos[site_id].copy()
    ee_quat = data.site_xquat[site_id].copy()  # [w, x, y, z]
    
    # Position error
    err_pos = (target_pos - ee_pos).astype(np.float64)  # (3,)
    
    # Orientation error (quaternion difference)
    # Compute relative rotation: target_quat * conjugate(ee_quat)
    ee_quat_conj = np.array([ee_quat[0], -ee_quat[1], -ee_quat[2], -ee_quat[3]])
    q_rel = _quat_multiply(target_quat, ee_quat_conj)
    
    # Convert quaternion error to axis-angle (approximate for small errors)
    # For small rotations: axis-angle ≈ 2 * [x, y, z] part of quaternion
    err_orient = 2.0 * q_rel[1:4]  # (3,) approximate orientation error
    
    # Get Jacobians
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    
    # Combine position and orientation errors
    err_combined = np.concatenate([err_pos, err_orient])  # (6,)
    
    # Combine Jacobians
    jac_combined = np.vstack([jacp, jacr])  # (6, nv)
    
    # Damped Least Squares
    JJt = jac_combined @ jac_combined.T
    damping = np.eye(6)
    damping[:3, :3] *= damping_pos
    damping[3:, 3:] *= damping_orient
    A = JJt + damping
    y = np.linalg.solve(A, err_combined)
    dq = jac_combined.T @ y
    
    dq = np.clip(dq, -max_dq, max_dq)
    
    # Exclude finger joints from IK updates
    if exclude_joints is not None:
        for jid in exclude_joints:
            if jid >= 0:
                jnt_type = model.jnt_type[jid]
                jnt_dofadr = model.jnt_dofadr[jid]
                if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                    for i in range(6):
                        if jnt_dofadr + i < model.nv:
                            dq[jnt_dofadr + i] = 0.0
                elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                    for i in range(3):
                        if jnt_dofadr + i < model.nv:
                            dq[jnt_dofadr + i] = 0.0
                else:
                    if jnt_dofadr < model.nv:
                        dq[jnt_dofadr] = 0.0
    
    mujoco.mj_integratePos(model, data.qpos, dq * step_scale, 1)
    mujoco.mj_forward(model, data)
    
    return float(np.linalg.norm(err_pos)), float(np.linalg.norm(err_orient))


def ik_step_pos_orient_body(model, data, body_id, target_pos, target_quat,
                            damping_pos=1e-2, damping_orient=1e-1,
                            step_scale=0.4, max_dq=0.12, exclude_joints=None):
    """
    Perform one step of position and orientation IK for a body.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_id: ID of the target body
        target_pos: Target position (3,) in world frame
        target_quat: Target orientation as quaternion (4,) [w, x, y, z]
        damping_pos: Damping factor for position DLS (default: 1e-2)
        damping_orient: Damping factor for orientation DLS (default: 1e-1)
        step_scale: Step scale for IK iterations (default: 0.4)
        max_dq: Maximum joint velocity change per step (default: 0.12)
    
    Returns:
        Tuple of (position_error, orientation_error) magnitudes
    """
    ee_pos = data.xpos[body_id].copy()
    ee_quat = data.xquat[body_id].copy()  # [w, x, y, z]
    
    # Position error
    err_pos = (target_pos - ee_pos).astype(np.float64)  # (3,)
    
    # Orientation error
    ee_quat_conj = np.array([ee_quat[0], -ee_quat[1], -ee_quat[2], -ee_quat[3]])
    q_rel = _quat_multiply(target_quat, ee_quat_conj)
    err_orient = 2.0 * q_rel[1:4]  # (3,)
    
    # Get Jacobians for body
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    
    # Combine errors and Jacobians
    err_combined = np.concatenate([err_pos, err_orient])  # (6,)
    jac_combined = np.vstack([jacp, jacr])  # (6, nv)
    
    # Damped Least Squares
    JJt = jac_combined @ jac_combined.T
    damping = np.eye(6)
    damping[:3, :3] *= damping_pos
    damping[3:, 3:] *= damping_orient
    A = JJt + damping
    y = np.linalg.solve(A, err_combined)
    dq = jac_combined.T @ y
    
    dq = np.clip(dq, -max_dq, max_dq)
    
    # Exclude finger joints from IK updates
    if exclude_joints is not None:
        for jid in exclude_joints:
            if jid >= 0:
                jnt_type = model.jnt_type[jid]
                jnt_dofadr = model.jnt_dofadr[jid]
                if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                    for i in range(6):
                        if jnt_dofadr + i < model.nv:
                            dq[jnt_dofadr + i] = 0.0
                elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                    for i in range(3):
                        if jnt_dofadr + i < model.nv:
                            dq[jnt_dofadr + i] = 0.0
                else:
                    if jnt_dofadr < model.nv:
                        dq[jnt_dofadr] = 0.0
    
    mujoco.mj_integratePos(model, data.qpos, dq * step_scale, 1)
    mujoco.mj_forward(model, data)
    
    return float(np.linalg.norm(err_pos)), float(np.linalg.norm(err_orient))
