#!/usr/bin/env python3
"""
Simple test script for finger control.

This script loads a MuJoCo model and tests finger opening/closing without any streaming.
"""
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer


def main():
    ap = argparse.ArgumentParser(description="Test finger control without streaming")
    ap.add_argument("--model", type=str, required=True, help="MuJoCo XML path (e.g. franka_panda.xml)")
    ap.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds (default: 5.0)")
    ap.add_argument("--frequency", type=float, default=1.0, help="Oscillation frequency in Hz (default: 1.0)")
    
    args = ap.parse_args()
    
    # Load MuJoCo model
    print(f"[Test] Loading model: {args.model}")
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Find finger joints
    jid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda0_finger_joint1")
    jid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda0_finger_joint2")
    
    if jid1 < 0 or jid2 < 0:
        print(f"[Test] ERROR: Finger joints not found!")
        print(f"       panda0_finger_joint1: id={jid1}")
        print(f"       panda0_finger_joint2: id={jid2}")
        return
    
    # Get qpos indices
    q1 = model.jnt_qposadr[jid1]
    q2 = model.jnt_qposadr[jid2]
    
    # Get joint ranges
    range1 = model.jnt_range[jid1]
    range2 = model.jnt_range[jid2]
    
    print(f"[Test] Finger joint IDs: {jid1}, {jid2}")
    print(f"[Test] qpos indices: {q1}, {q2}")
    print(f"[Test] Joint ranges: {range1}, {range2}")
    print(f"[Test] Starting finger oscillation test for {args.duration} seconds...")
    print(f"[Test] Frequency: {args.frequency} Hz")
    print(f"[Test] Press Ctrl-C to stop early")
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            frame_count = 0
            
            while viewer.is_running() and (time.time() - start_time) < args.duration:
                elapsed = time.time() - start_time
                
                # Oscillate between closed (0.0) and open (0.04)
                # Using sine wave: 0.02 + 0.02 * sin(2*pi*freq*t)
                v = 0.02 + 0.02 * np.sin(2 * np.pi * args.frequency * elapsed)
                
                # Clamp to joint range
                v = np.clip(v, max(range1[0], range2[0]), min(range1[1], range2[1]))
                
                # Apply to both fingers
                data.qpos[q1] = v
                data.qpos[q2] = v
                
                # Forward dynamics
                mujoco.mj_forward(model, data)
                
                # Update viewer
                viewer.sync()
                
                # Print status every second
                if frame_count % 100 == 0:
                    print(f"[Test] t={elapsed:.2f}s, finger_value={v:.4f}")
                
                frame_count += 1
                time.sleep(0.01)
            
            print(f"[Test] Test completed. Total frames: {frame_count}")
            
    except KeyboardInterrupt:
        print("\n[Test] Test interrupted by user")
    except Exception as e:
        print(f"[Test] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

