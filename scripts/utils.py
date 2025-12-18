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

