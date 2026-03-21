# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt   # hidapi, numpy, roboticstoolbox-python
```

Requires a HiWonder xArm 1S connected via USB (Vendor ID: `0x0483`, Product ID: `0x5750`).

On macOS, grant **Input Monitoring** permission to your terminal in System Settings > Privacy & Security.

## Scripts

```bash
python connection.py              # USB connection diagnostic + battery/servo test
python calibrate.py               # Interactive wizard: record joint limits → xarm_config.json
python home.py                    # Move all servos to home positions from xarm_config.json
python test.py                    # Jog each joint forward/backward/home to verify calibration
python servo_range.py 6           # Live position readout for a single servo (by ID)
python xarm_controller.py         # FK/IK/trajectory demo (roboticstoolbox)
python xarm_controller.py fk 0 45 -90 45 0    # FK for given joint angles (degrees)
python xarm_controller.py ik 150 0 100        # IK to reach (x, y, z) mm
python xarm_controller.py goto 150 0 100      # Move physical arm to xyz
python xarm_controller.py line 200 0 150 100 0 150  # Straight-line trajectory
python xarm_model.py              # Standalone FK/IK (no roboticstoolbox dependency)
```

**Typical workflow**: `connection.py` → `calibrate.py` → `home.py` → `test.py` → `xarm_controller.py`

## Architecture

| Layer | File | Role |
|-------|------|------|
| HID driver | `xarm_hid.py` | `XArmHID` class — all USB HID communication |
| Config | `xarm_config.json` | Calibration data (written by `calibrate.py`, read by all others) |
| Controller | `xarm_controller.py` | `XArmController` — FK, IK, trajectory planning via `roboticstoolbox-python` |
| Kinematics (standalone) | `xarm_model.py` | `XArmModel` — hand-coded Modified DH FK + damped Jacobian IK (no rtb dependency) |
| Utilities | `calibrate.py`, `home.py`, `test.py`, `servo_range.py`, `connection.py` | Standalone scripts that import `xarm_hid.py` |

**`xarm_hid.py`** is the only file that touches `hidapi`. It exposes `XArmHID` as a context manager. Commands:
- `CMD_SERVO_MOVE` (0x03), `CMD_SERVO_OFF` (0x14), `CMD_SERVO_POS_READ` (0x15), `CMD_GET_BATTERY_VOLTAGE` (0x0F)

**Packet format**: 65 bytes — `[0x00 (report ID), 0x55, 0x55, length, cmd, ...data]`, zero-padded.

**Servo ID mapping** (servo 1 is gripper, excluded from kinematics):

| Servo ID | Joint | DH index |
|----------|-------|----------|
| 6 | Base | q1 |
| 5 | Shoulder | q2 |
| 4 | Elbow | q3 |
| 3 | Wrist Pitch | q4 |
| 2 | Wrist Rotation | q5 |
| 1 | Gripper | — |

**Base rotation**: The base (servo 6) has a ~240-degree working range, not a full 360. It will jam at the physical stops beyond that range.

**DH convention**: Modified DH (Craig). Both `xarm_controller.py` (via `RevoluteMDH`) and `xarm_model.py` use the same DH table. Link lengths are in mm.

**`xarm_config.json`** schema: `metadata`, `servos` (keyed by servo ID string), `links` (L_base/shoulder/elbow/wrist/tool in mm), `dh_joint_order`, `gripper`. The `servos` entries include `home_position`, `position_min/max`, `direction` (±1), `angle_per_unit` (0.24 deg/unit), and `angle_min/max_deg`.

**`xarm_controller.py`** uses `roboticstoolbox-python`:
- FK via `DHRobot.fkine()`
- IK via `DHRobot.ikine_LM()` with position-only mask `[1,1,1,0,0,0]` (5-DOF arm can't control full 6-DOF pose)
- Joint trajectory via `rtb.jtraj()` (quintic polynomial)
- Cartesian line trajectory via `rtb.ctraj()` + per-step IK
- Multi-waypoint trajectory via `rtb.mstraj()` (trapezoidal velocity blending)
- Physical arm execution via `execute()` which streams waypoints to `XArmHID`
