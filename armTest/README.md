# WallE — xArm S1 Robotic Arm Project

A vision-assisted robotic arm system using a HiWonder xArm S1. The project combines eye tracking, monocular depth estimation, and inverse kinematics to enable gaze-controlled and camera-guided object manipulation.

## Tasks

### 1. Arm Control (`command.py`, `xarm_controller.py`)
Controls the xArm S1 via USB HID. Provides inverse kinematics, servo control, and a `pick_and_drop(x, y, z)` function that picks an object at given coordinates (mm) and drops it at a designated location.

**Libraries:** `hidapi` for USB HID communication.

### 2. Eye Tracking (`eye_track.py`)
Tracks the user's eye gaze and moves the macOS mouse cursor to the predicted screen position. Uses a 9-point calibration followed by a Kalman + EMA filter for smooth cursor movement.

**Libraries:** `eyetrax` (gaze estimation, calibration, Kalman filtering), `pyautogui` (mouse control), `opencv-python` (camera capture), `screeninfo` (screen resolution).

### 3. Object Coordinate Extraction (`object_coordinate.py`)
Captures a camera frame and runs Apple's Depth Pro model to estimate per-pixel depth. The user clicks on an object in the image to get its 3D coordinates (X, Y, Z in mm) relative to the camera. These coordinates can later be transformed to the arm's frame for automated pick-and-drop.

**Libraries:** `depth-pro` (monocular depth estimation), `torch` / `torchvision` (model inference), `opencv-python` (camera + UI), `Pillow`, `numpy`.

## Setup

### Prerequisites
- Python 3.9+
- macOS (tested on M2 Pro) or Linux
- USB connection to xArm S1 (for arm control tasks)
- Webcam

### Installation

```bash
# Clone and enter the project
cd WallE/armTest

# Create and activate a conda environment
conda create -n walle python=3.11
conda activate walle

# Install dependencies
pip install -r requirements.txt
```

### Depth Pro Model Weights

The object coordinate task requires the Depth Pro pretrained weights:

```bash
# Clone the Depth Pro repo (if not installed via pip)
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .

# Download the pretrained checkpoint (~500 MB)
source get_pretrained_models.sh
cd ..
```

The checkpoint will be saved to `ml-depth-pro/checkpoints/depth_pro.pt`. The script looks for this by default.

### macOS Permissions
- **Camera access**: grant to Terminal / IDE when prompted.
- **Accessibility** (for eye tracking mouse control): System Settings > Privacy & Security > Accessibility — enable your terminal app.

## Usage

### Arm Control
```bash
# Pick object at (260, 0, 70) mm and drop at the default location
python command.py 260 0 70

# With gripper angle
python command.py 260 0 70 -30
```

### Eye Tracking
```bash
python eye_track.py
# 1. Complete 9-point calibration (look at the dots)
# 2. Complete 3-point filter tuning
# 3. Your gaze now controls the mouse cursor
# Press ESC to quit
```

### Object Coordinate Extraction
```bash
python object_coordinate.py              # default camera
python object_coordinate.py --camera 1   # specify camera index

# Controls:
#   Left-click  — get 3D coordinates at that pixel
#   'c'         — capture a new frame
#   ESC         — quit
```
