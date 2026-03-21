# Robot Utility Test

Assistive robotics project combining brain-computer interface, eye tracking, computer vision, and robotic arm control.

## Components

All scripts live in `armTest/`.

- **xarm_controller.py** — Controls an xArm S1 robotic arm over USB HID. Inverse/forward kinematics, calibration (saved to `xarm_config.json`), gripper control.
- **eye_track.py** — Gaze-controlled mouse cursor using `eyetrax` library. 9-point calibration → saved model (`gaze_model.pkl`) → real-time webcam tracking with Kalman+EMA smoothing.
- **get_coordinate.py** — Click-to-pick 3D coordinates from a webcam feed. YOLOv8 object detection + Apple Depth Pro monocular depth estimation. Threaded inference.
- **head_tracker.py** — Head pose tracking module.
- **muse_connection.py** — BLE connection to a Muse 2 EEG headband via `muselsl`. Starts an LSL stream for downstream consumers.
- **jaw_clench_detection.py** — Real-time jaw clench detection from Muse 2 EEG. Pipeline: 20–150 Hz Butterworth bandpass on TP9/TP10 → Welford adaptive baseline (idle-only updates) → dual-channel amplitude gate → Cooper zero-crossing rate validation → onset/hold/release state machine with refractory period. Fires a callback on confirmed clench.

## Tech Stack

- **Robotic arm**: xArm S1 via `hidapi` (USB HID)
- **EEG / BCI**: Muse 2 headband via `muselsl` + `pylsl` (BLE → LSL), `scipy` for signal processing
- **Eye tracking**: `eyetrax` with Kalman filtering
- **Computer vision**: OpenCV, YOLOv8 (`ultralytics`), Apple Depth Pro
- **Runtime**: Python, PyTorch
