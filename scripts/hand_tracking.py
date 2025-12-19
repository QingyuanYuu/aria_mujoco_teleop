"""
Hand tracking module for Aria device.

This module handles hand pose tracking from Meta Aria glasses, including
state management and landmark extraction.
"""
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from projectaria_tools.core.mps import hand_tracking


# Hand skeleton edges (MediaPipe-style 21 landmarks)
HAND_EDGES_21: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]


@dataclass
class HandState:
    """State container for hand tracking data."""
    t_ns: int = 0
    valid: bool = False
    conf: float = 0.0
    wrist_dev: Optional[np.ndarray] = None    # (3,)
    palm_dev: Optional[np.ndarray] = None     # (3,)
    landmarks_dev: Optional[np.ndarray] = None # (N,3)
    wrist_to_palm_dir_dev: Optional[np.ndarray] = None  # (3,) normalized direction from wrist to palm


# Shared state (thread-safe)
_lock = threading.Lock()
_hand = HandState()


def _try_extract_landmarks_device(hand_obj) -> Optional[np.ndarray]:
    """
    Best-effort landmark extraction (device frame), because SDK versions differ.
    Returns array (N,3) in meters, or None.
    """
    candidates = [
        "get_landmark_positions_device",
        "get_landmarks_device",
        "landmark_positions_device",
        "landmarks_device",
        "keypoints_device",
        "joints_device",
    ]
    for name in candidates:
        if hasattr(hand_obj, name):
            v = getattr(hand_obj, name)
            try:
                pts = v() if callable(v) else v
                arr = np.array(pts, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 5:
                    return arr
            except Exception:
                pass
    return None


def hand_cb(ht: hand_tracking.HandTrackingResult):
    """Update right-hand info in DEVICE frame."""
    global _hand
    now_ns = time.time_ns()

    r = ht.right_hand
    if r is None:
        with _lock:
            _hand.t_ns = now_ns
            _hand.valid = False
        return

    wrist = np.array(r.get_wrist_position_device(), dtype=np.float64)
    conf = float(r.confidence)

    # palm (best-effort)
    palm = None
    wrist_to_palm_dir = None
    try:
        palm = np.array(r.get_palm_position_device(), dtype=np.float64)
        # Compute direction from wrist to palm
        if palm is not None:
            dir_vec = palm - wrist
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > 1e-6:  # Avoid division by zero
                wrist_to_palm_dir = dir_vec / dir_norm
    except Exception:
        palm = None
        wrist_to_palm_dir = None

    lm = _try_extract_landmarks_device(r)

    with _lock:
        _hand.t_ns = now_ns
        _hand.valid = True
        _hand.conf = conf
        _hand.wrist_dev = wrist
        _hand.palm_dev = palm
        _hand.landmarks_dev = lm
        _hand.wrist_to_palm_dir_dev = wrist_to_palm_dir


def get_hand_state() -> HandState:
    """Get a copy of the current hand state (thread-safe)."""
    with _lock:
        return HandState(**_hand.__dict__)

