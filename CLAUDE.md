# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Assistive robotics system ("WallE") combining brain-computer interface (Muse 2 EEG), eye tracking, computer vision, and robotic arm control. The goal is gaze-directed, clench-triggered manipulation of a HiWonder xArm S1 robotic arm.

## Project Structure

```
robot-utility-test/
  utility/              # Reusable modules (importable package)
    __init__.py
    xarm_controller.py  # FK/IK solver, trajectory planning
    xarm_hid.py         # USB HID protocol driver
    home.py             # Move arm to home position
    connection.py       # USB HID diagnostics
    jaw_clench_detection.py  # Real-time EMG clench detection
    muse_connection.py  # Muse 2 BLE → LSL stream
    eye_track.py        # Gaze-controlled mouse cursor
    get_coordinate.py   # 3D object coordinates (YOLOv8 + Depth Pro)
    head_tracker.py     # Head yaw estimation
    xarm_config.json    # Servo calibration + DH parameters
  main.py               # Integration entry point (future)
  requirements.txt      # All dependencies
  CLAUDE.md
```

## Setup & Commands

```bash
# Environment
conda activate walle  # or use armTest/venv
pip install -r requirements.txt

# Depth Pro model weights (required for get_coordinate.py)
# Clone https://github.com/apple/ml-depth-pro.git, pip install -e ., then:
source get_pretrained_models.sh
```

### Running Modules Independently

Run from the project root:

```bash
# Robot arm
python -m utility.xarm_controller                          # interactive demo
python -m utility.xarm_controller fk 0 45 -90 45 0         # forward kinematics
python -m utility.xarm_controller ik 150 0 100              # inverse kinematics
python -m utility.xarm_controller goto 150 0 100            # move arm to XYZ
python -m utility.xarm_controller grip open                 # gripper control
python -m utility.home                                      # move arm to home
python -m utility.connection                                # USB HID diagnostics

# EEG (two terminals)
python -m utility.muse_connection                           # Terminal 1: BLE → LSL stream
python -m utility.jaw_clench_detection                      # Terminal 2: clench detection

# Vision
python -m utility.eye_track                                 # gaze-controlled mouse cursor
python -m utility.get_coordinate                            # 3D object coordinates (click UI)
python -m utility.head_tracker                              # head yaw estimation
```

### Importing in Integration Code

```python
from utility.jaw_clench_detection import JawClenchDetector
from utility.xarm_controller import XArmController
from utility.xarm_hid import XArmHID
```

## Architecture

### Data Flow

```
Muse 2 (BLE) → muse_connection.py (LSL stream) → jaw_clench_detection.py → clench callback
Webcam → eye_track.py / get_coordinate.py / head_tracker.py → coordinates/intent
Intent + coordinates → xarm_controller.py (FK/IK/trajectories) → xarm_hid.py (USB HID) → xArm S1
```

### Two-Layer Arm Control

- **xarm_controller.py** — High-level: FK/IK solver (roboticstoolbox-python), trajectory planning (trapezoidal blending, Cartesian line), servo calibration via `xarm_config.json`. Modified DH convention, 5-DOF. CLI with `fk`, `ik`, `goto`, `movej`, `line`, `grip` commands.
- **xarm_hid.py** — Low-level: USB HID protocol driver (VID `0x0483`, PID `0x5750`). Servo move/read, battery voltage, torque off. Context manager support.

### Jaw Clench Detection Pipeline

```
Raw EEG (256 Hz, TP9 + TP10 channels)
  → 60 Hz notch filter
  → Butterworth bandpass 15–120 Hz
  → Rectification + 75 ms moving-average envelope
  → Welford adaptive baseline (idle-only updates, 10s window)
  → Dual-channel amplitude gate (both channels > 70% of threshold)
  → Zero-crossing rate texture check (EMG confirmation at onset only)
  → State machine: WARMUP → IDLE → ONSET → ACTIVE → REFRACTORY
  → on_clench(duration_ms) callback
```

Key parameters: `THRESHOLD_K=3.0`, `MIN_HOLD_MS=30`, `MIN_THRESHOLD=20.0 µV`, `WARMUP_SEC=6.0`.

### Vision Components

- **get_coordinate.py** — YOLOv8 + Apple Depth Pro monocular depth. Threaded inference. Coordinate convention: x=forward (depth), y=horizontal (L+/R-), z=vertical (up+).
- **eye_track.py** — eyetrax 9-point calibration → Kalman+EMA smoothed gaze → pyautogui mouse control. Saves/loads `gaze_model.pkl`.
- **head_tracker.py** — MediaPipe face landmarks → geometric yaw → hysteresis state machine (Left/Center/Right).

## Key Config Files

- **utility/xarm_config.json** — Servo calibration (position↔angle mapping), DH link lengths (mm), joint limits, gripper config. Loaded automatically by `XArmController`.
- **requirements.txt** — Dependencies grouped by subsystem. No version pinning.

## Tech Stack

- **Robot:** HiWonder xArm S1 via `hidapi`, `roboticstoolbox-python` for kinematics
- **BCI:** Muse 2 via `muselsl`/`pylsl`, `scipy` for signal processing
- **Vision:** PyTorch, YOLOv8 (`ultralytics`), Apple Depth Pro, MediaPipe, OpenCV
- **Eye tracking:** `eyetrax`, `pyautogui`

## Platform Notes

- macOS (M2 Pro tested). Camera and Accessibility permissions required.
- Muse 2 connects via Bluetooth LE — can be flaky on macOS; toggle Bluetooth if stream drops.
- xArm connects via USB HID — run `python -m utility.connection` to diagnose.
