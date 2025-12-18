"""
Visualization module for MuJoCo viewer.

This module handles drawing hand skeleton and markers in the MuJoCo viewer.
"""
from typing import Optional, List, Tuple

import numpy as np
import mujoco
from hand_tracking import HAND_EDGES_21


def _safe_make_connector(geom, radius: float, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Try to create a capsule connector between p1 and p2.
    Some MuJoCo builds expose mjv_makeConnector; if not available, return False.
    """
    if not hasattr(mujoco, "mjv_makeConnector"):
        return False
    try:
        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            float(p1[0]), float(p1[1]), float(p1[2]),
            float(p2[0]), float(p2[1]), float(p2[2]),
        )
        return True
    except Exception:
        return False


def draw_hand_markers(viewer, points_mj: np.ndarray, edges: Optional[List[Tuple[int, int]]] = None,
                      sphere_size: float = 0.008, edge_radius: float = 0.004):
    """
    Draw hand points (and edges if possible) into viewer.user_scn.
    NOTE: We clear and redraw each frame (simple + robust).
    
    Args:
        viewer: MuJoCo viewer instance
        points_mj: Array of hand landmark positions in MuJoCo frame (N, 3)
        edges: List of edge connections as (i, j) tuples. If None, uses HAND_EDGES_21
               if points_mj has 21 points, otherwise no edges.
        sphere_size: Radius of hand point spheres
        edge_radius: Radius of hand edge capsules
    """
    scn = viewer.user_scn
    scn.ngeom = 0

    # Use default edges if not provided and we have 21 points
    if edges is None and points_mj.shape[0] == 21:
        edges = HAND_EDGES_21

    # points as spheres
    for p in points_mj:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([sphere_size, 0, 0], dtype=np.float64),
            p.astype(np.float64),
            np.eye(3).flatten(),
            np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),  # green
        )
        scn.ngeom += 1

    # edges as capsules (optional)
    if edges is None:
        return

    for i, j in edges:
        if i >= len(points_mj) or j >= len(points_mj):
            continue
        if scn.ngeom >= scn.maxgeom:
            break
        p1, p2 = points_mj[i], points_mj[j]
        g = scn.geoms[scn.ngeom]

        ok = _safe_make_connector(g, edge_radius, p1, p2)
        if not ok:
            # If connector isn't available, just skip edges (points still shown)
            return

        g.rgba[:] = np.array([0.1, 0.8, 1.0, 1.0], dtype=np.float32)  # cyan
        scn.ngeom += 1


def clear_markers(viewer):
    """Clear all markers from the viewer."""
    viewer.user_scn.ngeom = 0

