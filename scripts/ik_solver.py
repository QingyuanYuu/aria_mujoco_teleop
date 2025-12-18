"""
Inverse Kinematics solver module.

This module implements position-based IK using Damped Least Squares (DLS) method.
"""
import numpy as np
import mujoco


def ik_step_pos(model, data, site_id, target_pos, damping=1e-2, step_scale=0.4, max_dq=0.12):
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
    mujoco.mj_integratePos(model, data.qpos, dq * step_scale, 1)
    mujoco.mj_forward(model, data)

    return float(np.linalg.norm(err))

