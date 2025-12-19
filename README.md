# Aria MuJoCo Teleoperation

A real-time teleoperation system that enables control of a MuJoCo-simulated Franka Panda robot using hand tracking from Meta Aria glasses. The system streams hand pose data from the Aria device and uses inverse kinematics to make the robot's end-effector follow your hand movements in real-time.

## Overview

This project provides a seamless interface between Meta Aria's hand tracking capabilities and MuJoCo physics simulation. By tracking your right hand's position and orientation, the system translates your movements into robot control commands, allowing you to manipulate the simulated robot arm naturally.

### Key Features

- **Real-time Hand Tracking**: Streams hand pose data from Meta Aria glasses via HTTP server
- **Inverse Kinematics Control**: Uses damped least squares (DLS) IK solver to control the robot's end-effector
- **Orientation Control**: Aligns gripper orientation with wrist-to-palm direction
- **Auto-Calibrated Finger Control**: Automatically learns thumb-index pinch distance range and maps to gripper opening (0.00 to 0.04)
- **3D Hand Visualization**: Optional visualization of hand skeleton in the MuJoCo viewer
- **Configurable Mapping**: Adjustable axis flipping, scaling, and smoothing parameters
- **Robust Calibration**: Automatic origin calibration on first confident hand detection

## Requirements

### Hardware
- Meta Aria glasses with hand tracking enabled
- macOS (tested on macOS) or Ubuntu 24.04 (tested on Ubuntu 24.04)

### Software Dependencies
- Python 3.x
- MuJoCo Python bindings (`mujoco`)
- Meta Aria SDK (`aria.sdk_gen2`, `aria.stream_receiver`)
- Project Aria Tools (`projectaria_tools`)
- NumPy

## Installation

### macOS

1. **Install MuJoCo Python bindings:**
   ```bash
   pip install mujoco
   ```

2. **Install Meta Aria SDK and Project Aria Tools:**
   Follow the official Meta Aria SDK installation instructions to set up:
   - `aria.sdk_gen2`
   - `aria.stream_receiver`
   - `projectaria_tools`

3. **Install NumPy:**
   ```bash
   pip install numpy
   ```

### Ubuntu 24.04

1. **Install system dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-dev
   ```

2. **Install MuJoCo Python bindings:**
   ```bash
   pip3 install mujoco
   ```

3. **Install Meta Aria SDK and Project Aria Tools:**
   Follow the official Meta Aria SDK installation instructions to set up:
   - `aria.sdk_gen2`
   - `aria.stream_receiver`
   - `projectaria_tools`

4. **Install NumPy:**
   ```bash
   pip3 install numpy
   ```

**Note**: On Ubuntu, you may need to use `pip3` instead of `pip` depending on your Python installation. It's recommended to use a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install mujoco numpy
   ```

## Usage

### Basic Usage

#### macOS

On macOS, run the following command:

```bash
mjpython scripts/aria_follow_mujoco.py \
  --model mujoco_models/franka_sim/franka_panda.xml \
  --ee-site panda0_gripper \
  --show-hand \
  --flip-z \
  --enable-fingers
```

#### Ubuntu 24.04

On Ubuntu 24.04, run the following command:

```bash
python3 scripts/aria_follow_mujoco.py \
  --model mujoco_models/franka_sim/franka_panda.xml \
  --ee-site panda0_gripper \
  --show-hand \
  --flip-z \
  --enable-fingers
```

**Note**: On Ubuntu, you may need to use `python3` instead of `mjpython` depending on your MuJoCo installation. If you have `mjpython` installed, you can use it as well.

### Command Line Arguments

#### Required Arguments
- `--model`: Path to MuJoCo XML model file (e.g., `mujoco_models/franka_sim/franka_panda.xml`)
- `--ee-site`: Name of the end-effector site/body in the model (default: `panda0_gripper`)

#### Network Configuration
- `--host`: Server host address (default: `0.0.0.0`)
- `--port`: Server port number (default: `6768`)

#### Mapping and Smoothing
- `--scale`: Scale factor for device delta to MuJoCo meters (default: `1.0`)
- `--alpha`: EMA smoothing factor (0..1), smaller values = more smoothing (default: `0.2`)
- `--conf`: Minimum hand confidence threshold to accept updates (default: `0.5`)
- `--timeout`: Seconds without update before holding target position (default: `0.25`)
- `--flip-x`: Flip the mapped X axis
- `--flip-y`: Flip the mapped Y axis
- `--flip-z`: Flip the mapped Z axis (commonly used for up/down inversion)

#### Inverse Kinematics Parameters
- `--ik-damping`: Damping factor for DLS IK solver (default: `1e-2`)
- `--ik-step-scale`: Step scale for IK iterations (default: `0.4`)
- `--ik-max-dq`: Maximum joint velocity change per step (default: `0.12`)
- `--enable-orient`: Enable orientation control - aligns gripper orientation with wrist-to-palm direction (default: enabled)
- `--disable-orient`: Disable orientation control (position-only IK)
- `--ik-damping-orient`: Damping factor for orientation IK (default: `1e-1`)

#### Finger Control
- `--enable-fingers`: Enable auto-calibrated finger control (thumb-index pinch distance → gripper opening)
- `--finger-alpha`: Finger control smoothing factor (0..1, higher = faster response, default: `0.7`)
- `--finger-verbose`: Print finger control debug information (distance, calibration range, applied values)
- `--test-fingers`: Run finger oscillation test before starting teleoperation (for debugging)

**Note**: Finger control uses automatic calibration. The system learns your pinch distance range (closed to open) during runtime and maps it to gripper opening (0.00 to 0.04). No manual distance range configuration needed!

#### Hand Visualization
- `--show-hand`: Enable 3D hand skeleton visualization in MuJoCo viewer
- `--hand-point-size`: Radius of hand point spheres (default: `0.008`)
- `--hand-edge-radius`: Radius of hand edge capsules (default: `0.004`)

### Operation Flow

1. **Start the Script**: Launch the script with your desired parameters. The system will start an HTTP server listening for Aria device streams.

2. **Start Device Streaming**: Begin streaming from your Meta Aria glasses to the specified host and port.

3. **Calibration**: The system waits for the first confident right-hand detection to establish the origin mapping between your hand position and the robot's end-effector position.

4. **Control**: 
   - Move your right hand to control the robot. The end-effector will follow your hand movements in real-time.
   - If finger control is enabled (`--enable-fingers`), pinch your thumb and index finger together to close the gripper, and spread them apart to open it. The system automatically learns your pinch distance range during the first few seconds of operation.

5. **Exit**: Close the MuJoCo viewer window or press Ctrl-C to exit.

## Project Structure

```
aria_mujoco_teleop/
├── README.md                          # This file
├── scripts/                           # Main source code (modular architecture)
│   ├── __init__.py                   # Package initialization
│   ├── aria_follow_mujoco.py         # Main teleoperation script
│   ├── hand_tracking.py              # Hand tracking state and callbacks
│   ├── stream_receiver.py            # Aria stream receiver module
│   ├── visualization.py              # MuJoCo visualization helpers
│   ├── ik_solver.py                  # Inverse kinematics solver
│   ├── finger_control.py             # Auto-calibrated finger/gripper control
│   ├── utils.py                      # Utility functions
│   └── test_finger_control.py        # Standalone finger control test script
└── mujoco_models/
    └── franka_sim/                    # Franka Panda MuJoCo models
        ├── franka_panda.xml           # Standard Franka Panda model
        ├── franka_panda_teleop.xml    # Teleoperation-optimized model
        ├── bi-franka_panda.xml        # Bimanual Franka model
        ├── assets/                     # Model assets and actuators
        └── meshes/                     # Robot mesh files
```

### Module Architecture

The codebase is organized into modular components for better maintainability and extensibility:

- **`hand_tracking.py`**: Manages hand pose state, callbacks, and landmark extraction from Aria device
- **`stream_receiver.py`**: Handles HTTP server setup and stream reception from Meta Aria glasses
- **`visualization.py`**: Provides functions for drawing hand skeleton and markers in MuJoCo viewer
- **`ik_solver.py`**: Implements position and orientation inverse kinematics using Damped Least Squares (DLS)
- **`finger_control.py`**: Auto-calibrated finger/gripper control system that maps thumb-index pinch distance to gripper opening
- **`utils.py`**: Contains utility functions for coordinate transformations
- **`aria_follow_mujoco.py`**: Main orchestration script that coordinates all modules
- **`test_finger_control.py`**: Standalone test script for finger joint oscillation (for debugging)

## How It Works

The system follows a modular pipeline:

1. **Hand Tracking Stream** (`stream_receiver.py` + `hand_tracking.py`): 
   - The stream receiver module starts an HTTP server that receives hand tracking data from the Meta Aria device
   - The hand tracking callback (`hand_tracking.py`) updates a thread-safe shared state with wrist position, palm position, and 21-point hand landmarks

2. **Coordinate Mapping** (`utils.py`):
   - On first confident hand detection, the system establishes a mapping between the device coordinate frame and the MuJoCo world frame
   - This mapping includes:
     - Origin calibration (device wrist position → robot end-effector position)
     - Axis transformations (optional flipping via `--flip-x`, `--flip-y`, `--flip-z`)
     - Scaling (via `--scale` parameter)

3. **Target Smoothing**:
   - Hand position updates are smoothed using exponential moving average (EMA) to reduce jitter and provide stable control

4. **Inverse Kinematics** (`ik_solver.py`):
   - The system uses a damped least squares (DLS) IK solver to compute joint angles that position the end-effector at the target location
   - Supports both position-only and position+orientation IK
   - Orientation control aligns the gripper's forward direction (toward fingers) with the wrist-to-palm direction
   - Finger joints are excluded from IK to prevent conflicts with finger control
   - The IK runs iteratively each frame to track the moving target

5. **Finger Control** (`finger_control.py`):
   - Auto-calibrated system that learns your thumb-index pinch distance range during runtime
   - Computes pinch distance in device frame (invariant to coordinate transformations)
   - Maps normalized distance to gripper opening (0.00 = closed, 0.04 = fully open)
   - Uses exponential moving average (EMA) for smooth control
   - Applies directly to finger joint qpos and calls `mj_forward` for immediate effect

6. **Visualization** (`visualization.py`):
   - Optionally, the hand skeleton (21 landmarks with edges) can be visualized in the MuJoCo viewer
   - Hand markers are transformed to the same coordinate space as the robot for intuitive feedback

## Model Information

The project includes optimized Franka Panda models with:
- Adjusted joint positions for more natural arm posture
- Modified joint limits to avoid gimbal lock configurations
- Practical adjustments based on extensive hardware operation experience

See `mujoco_models/franka_sim/README.md` for detailed model information.

## Troubleshooting

- **No hand detected**: Ensure your Aria device is streaming and hand tracking is enabled. Check the confidence threshold with `--conf`.
- **Robot not moving**: Verify the `--ee-site` name matches a site or body in your model. Check that calibration completed successfully.
- **Inverted movements**: Use `--flip-x`, `--flip-y`, or `--flip-z` to correct axis orientation.
- **Jittery motion**: Reduce `--alpha` for more smoothing, or increase `--ik-damping` for more stable IK.
- **Fingers not moving**: 
  - Ensure `--enable-fingers` is set
  - Check that finger joints exist in your model (`panda0_finger_joint1`, `panda0_finger_joint2`)
  - Use `--finger-verbose` to see calibration status and applied values
  - Try `--test-fingers` to verify finger joints work independently
  - Make sure you're pinching your thumb and index finger - the system needs to see both landmarks
- **Finger range too small**: The auto-calibration learns from your actual pinch range. Try pinching more closed and more open during the first few seconds to expand the learned range.

## License

See individual model licenses in `mujoco_models/franka_sim/LICENSE`.

## Acknowledgments

- Franka Panda robot meshes and specifications from [franka_ros](https://github.com/frankaemika/franka_ros)
- MuJoCo physics simulator
- Meta Aria SDK and Project Aria Tools

