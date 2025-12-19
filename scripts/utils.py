"""
Utility functions for coordinate transformations and data processing.
"""
import numpy as np


def apply_axis_ops(v: np.ndarray, flip_x: bool, flip_y: bool, flip_z: bool) -> np.ndarray:
    """
    Apply axis flipping operations to a vector.
    
    Args:
        v: Input vector (3,)
        flip_x: Flip X axis if True
        flip_y: Flip Y axis if True
        flip_z: Flip Z axis if True
    
    Returns:
        Transformed vector (3,)
    """
    out = v.copy()
    if flip_x:
        out[0] *= -1
    if flip_y:
        out[1] *= -1
    if flip_z:
        out[2] *= -1
    return out


def quaternion_to_forward(quat: np.ndarray) -> np.ndarray:
    """
    Extract forward direction (Z axis) from quaternion.
    Quaternion format: [w, x, y, z]
    
    Args:
        quat: Quaternion [w, x, y, z]
    
    Returns:
        Forward direction vector (3,) in world frame
    """
    w, x, y, z = quat
    # Convert quaternion to rotation matrix
    # Rotation matrix columns are: right (X), up (Y), forward (Z)
    # For quaternion [w, x, y, z], the rotation matrix is:
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    # Extract Z axis (third column) - this is the forward direction
    forward = R[:, 2]
    # Normalize
    norm = np.linalg.norm(forward)
    if norm > 1e-6:
        forward = forward / norm
    return forward

