#!/usr/bin/env python3
"""
Finger control module for gripper teleoperation (AUTO-CALIBRATED).

Goal:
- Map thumb-index pinch distance to Panda finger joints so output reliably spans [0.00, 0.04].
- Auto-calibrate distance range online (no manual min/max tuning).
- Apply by directly writing finger joint qpos and calling mj_forward (mimics your working test).

Usage in your teleop loop (example):

    from finger_control import FingerTeleop

    finger = FingerTeleop(model, alpha=0.7)

    # inside your viewer loop (each frame):
    if h.landmarks_dev is not None and h.landmarks_dev.shape[0] >= 21:
        finger.update_from_landmarks(
            data,
            h.landmarks_dev,
            verbose=False,
            do_forward=True,   # recommended when using qpos directly
        )

Notes:
- This module does NOT require coordinate transforms for pinch distance.
  Distance is invariant to rotations/translations; we only use device-frame distance.
- Auto-calibration learns d_min/d_max as you pinch closed and open during runtime.
- You can "reset" calibration when you want to re-center pinch range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

# MediaPipe 21 landmarks indices
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8


# -----------------------------
# Auto calibration helper
# -----------------------------
@dataclass
class PinchCalibrator:
    """
    Tracks observed pinch distance range and normalizes distance into [0, 1].

    d_min/d_max are updated online. To avoid exploding due to noise:
    - uses optional exponential smoothing on min/max updates (leaky update).
    - keeps a minimal range to prevent divide-by-zero.

    Parameters:
        init_min/init_max: initial guess for pinch distance range (meters)
        min_range: minimum allowed range between d_min and d_max (meters)
        leak: how quickly to adapt min/max (0..1). 1 means immediate update.
    """
    init_min: float = 0.02
    init_max: float = 0.06
    min_range: float = 1e-4
    leak: float = 1.0

    def __post_init__(self):
        self.d_min = float(self.init_min)
        self.d_max = float(self.init_max)
        self._initialized = False

    def reset(self, d: Optional[float] = None):
        """Reset calibration. If d provided, center around it."""
        self._initialized = False
        if d is None:
            self.d_min = float(self.init_min)
            self.d_max = float(self.init_max)
        else:
            d = float(d)
            # start with a small symmetric range around d
            self.d_min = d - 0.01
            self.d_max = d + 0.01

    def update(self, d: float):
        """Update observed min/max with optional leaky behavior."""
        d = float(d)

        # first sample: initialize around it
        if not self._initialized:
            self.d_min = d
            self.d_max = d + self.min_range
            self._initialized = True
            return

        # update min/max
        if d < self.d_min:
            if self.leak >= 1.0:
                self.d_min = d
            else:
                self.d_min = (1.0 - self.leak) * self.d_min + self.leak * d

        if d > self.d_max:
            if self.leak >= 1.0:
                self.d_max = d
            else:
                self.d_max = (1.0 - self.leak) * self.d_max + self.leak * d

        # ensure non-degenerate range
        if self.d_max - self.d_min < self.min_range:
            self.d_max = self.d_min + self.min_range

    def normalize(self, d: float) -> float:
        """Normalize distance into [0, 1] using current range."""
        d = float(d)
        denom = max(self.d_max - self.d_min, self.min_range)
        x = (d - self.d_min) / denom
        return float(np.clip(x, 0.0, 1.0))


# -----------------------------
# Panda finger qpos helper
# -----------------------------
@dataclass
class PandaFingerKinematics:
    joint1_name: str = "panda0_finger_joint1"
    joint2_name: str = "panda0_finger_joint2"

    jid1: int = -1
    jid2: int = -1
    qadr1: int = -1
    qadr2: int = -1
    lo: float = 0.0
    hi: float = 0.04

    def bind(self, model: mujoco.MjModel):
        self.jid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.joint1_name)
        self.jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.joint2_name)
        if self.jid1 < 0 or self.jid2 < 0:
            raise ValueError(
                f"Finger joints not found: {self.joint1_name} id={self.jid1}, {self.joint2_name} id={self.jid2}"
            )

        self.qadr1 = int(model.jnt_qposadr[self.jid1])
        self.qadr2 = int(model.jnt_qposadr[self.jid2])

        r1 = model.jnt_range[self.jid1]
        r2 = model.jnt_range[self.jid2]
        self.lo = float(max(r1[0], r2[0]))
        self.hi = float(min(r1[1], r2[1]))

    def set_qpos(self, model: mujoco.MjModel, data: mujoco.MjData, value: float, do_forward: bool = True):
        v = float(np.clip(value, self.lo, self.hi))
        data.qpos[self.qadr1] = v
        data.qpos[self.qadr2] = v
        if do_forward:
            mujoco.mj_forward(model, data)
        return v


# -----------------------------
# Main teleop wrapper
# -----------------------------
class FingerTeleop:
    """
    Auto-calibrated pinch-to-gripper controller.

    - Takes thumb/index tip landmarks (device frame) each frame.
    - Computes pinch distance.
    - Auto-calibrates d_min/d_max online.
    - Maps normalized distance -> [0.00, 0.04] gripper opening.
    - EMA smooth output (alpha).
    - Applies to MuJoCo by writing finger qpos and calling mj_forward.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        alpha: float = 0.7,                # 0..1, higher = faster response
        min_gripper: float = 0.0,
        max_gripper: float = 0.04,
        scale: float = 1.0,                # multiply distance (rarely needed)
        calibrator: Optional[PinchCalibrator] = None,
        init_value: float = 0.02,          # start half-open
    ):
        self.model = model
        self.alpha = float(alpha)
        self.min_gripper = float(min_gripper)
        self.max_gripper = float(max_gripper)
        self.scale = float(scale)

        self.cal = calibrator if calibrator is not None else PinchCalibrator()
        self.fk = PandaFingerKinematics()
        self.fk.bind(model)

        self.value = float(np.clip(init_value, self.min_gripper, self.max_gripper))

    def reset_calibration(self, current_landmarks: Optional[np.ndarray] = None):
        """
        Reset calibrator. Optionally pass current landmarks to seed around current distance.
        """
        if current_landmarks is not None and current_landmarks.ndim == 2 and current_landmarks.shape[0] >= 21:
            thumb = current_landmarks[THUMB_TIP_IDX]
            index = current_landmarks[INDEX_TIP_IDX]
            d = float(np.linalg.norm(thumb - index)) * self.scale
            self.cal.reset(d=d)
        else:
            self.cal.reset()

    def _compute_distance(self, landmarks_dev: np.ndarray) -> float:
        thumb = landmarks_dev[THUMB_TIP_IDX]
        index = landmarks_dev[INDEX_TIP_IDX]
        d = float(np.linalg.norm(thumb - index)) * self.scale
        return d

    def update_from_landmarks(
        self,
        data: mujoco.MjData,
        landmarks_dev: Optional[np.ndarray],
        *,
        do_forward: bool = True,
        verbose: bool = False,
    ) -> float:
        """
        Update controller from landmarks and apply to MuJoCo fingers.

        Returns:
            current gripper value (0..0.04)
        """
        # If no landmarks, hold last value (still apply if you want)
        if (
            landmarks_dev is None
            or not isinstance(landmarks_dev, np.ndarray)
            or landmarks_dev.ndim != 2
            or landmarks_dev.shape[1] != 3
            or landmarks_dev.shape[0] < 21
        ):
            v = self.fk.set_qpos(self.model, data, self.value, do_forward=do_forward)
            if verbose:
                print("[Finger] no landmarks; hold value:", v)
            return v

        d = self._compute_distance(landmarks_dev)
        self.cal.update(d)
        x = self.cal.normalize(d)  # 0..1

        target = self.min_gripper + x * (self.max_gripper - self.min_gripper)
        # EMA
        self.value = (1.0 - self.alpha) * self.value + self.alpha * target
        self.value = float(np.clip(self.value, self.min_gripper, self.max_gripper))

        v_applied = self.fk.set_qpos(self.model, data, self.value, do_forward=do_forward)

        if verbose:
            print(
                f"[Finger] d={d:.4f}  d_min={self.cal.d_min:.4f}  d_max={self.cal.d_max:.4f}  "
                f"x={x:.3f}  target={target:.4f}  applied={v_applied:.4f}"
            )

        return v_applied
