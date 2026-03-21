# xArm S1 FK/IK Toolkit

Forward kinematics, inverse kinematics, and trajectory planning for the **HiWonder xArm 1S** robotic arm over USB HID.

## Quick Start

```bash
pip install -r requirements.txt
python connection.py        # verify USB connection
python calibrate.py         # one-time joint calibration
python xarm_controller.py   # FK/IK/trajectory demo
```

Requires the xArm 1S connected via USB. On macOS, grant **Input Monitoring** permission to your terminal (System Settings > Privacy & Security).

## Hardware Notes

- **5-DOF arm** (base, shoulder, elbow, wrist pitch, wrist rotation) + gripper
- **Base rotation**: ~240-degree working range (not a full 360 — jams at physical stops)
- **Communication**: USB HID, Vendor ID `0x0483`, Product ID `0x5750`
- **Servo positions**: signed 16-bit values, 0.24 degrees per unit

## Files

| File | Purpose |
|------|---------|
| `xarm_hid.py` | USB HID driver. `XArmHID` class handles all communication with the arm (move servos, read positions, battery voltage, torque off). Only file that imports `hidapi`. |
| `xarm_config.json` | Calibration data: servo home positions, min/max limits, direction, link lengths (mm), gripper settings. Written by `calibrate.py`, read by everything else. |
| `xarm_controller.py` | **Main controller.** FK, IK, and trajectory planning using [roboticstoolbox-python](https://github.com/petercorke/robotics-toolbox-python). Can drive the physical arm. |
| `xarm_model.py` | Standalone FK/IK using hand-coded Modified DH transforms and damped Jacobian pseudoinverse. No external robotics dependencies. |
| `calibrate.py` | Interactive wizard that records joint limits and link lengths, then writes `xarm_config.json`. |
| `connection.py` | USB connection diagnostic. Tests HID communication and reads battery voltage. |
| `home.py` | Moves all servos (including gripper) to their home positions from config. |
| `test.py` | Jogs each joint to its min and max positions to verify calibration visually. |
| `servo_range.py` | Live servo position reader. Disables torque on one servo and streams its position as you move it by hand. For finding min/max limits. |
| `backup.json` | Backup of a previous calibration run. |

## Usage

### Forward Kinematics

```bash
python xarm_controller.py fk 0 45 -90 45 0    # joint angles in degrees
```

### Inverse Kinematics

```bash
python xarm_controller.py ik 150 0 100         # target x y z in mm
```

### Move the Arm

```bash
python xarm_controller.py goto 150 0 100       # IK + smooth trajectory to target
python xarm_controller.py line 200 0 150 100 0 150  # straight-line path between two points
```

### Calibration & Testing

```bash
python connection.py       # step 1: verify USB connection
python calibrate.py        # step 2: record joint limits (interactive)
python home.py             # step 3: move to home pose
python test.py             # step 4: jog joints to verify limits
python servo_range.py 6    # utility: live position readout for servo 6 (base)
```
