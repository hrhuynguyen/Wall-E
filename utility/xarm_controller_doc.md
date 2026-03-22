# xarm_controller.py — Command Reference

## Usage

```bash
python xarm_controller.py [command] [arguments]
```

## Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| *(none)* | — | Run interactive demo showing FK, IK, and trajectory examples |
| `fk` | `q1 q2 q3 q4 q5` (degrees) | Compute forward kinematics and print end-effector position (x, y, z in mm) for the given joint angles |
| `ik` | `x y z` (mm) | Solve inverse kinematics for a target position and print the joint angles and position error |
| `goto` | `x y z` (mm) | Solve IK for a target position, then smoothly move the physical arm there with a single servo command (3 s duration) |
| `movej` | `q1 q2 q3 q4 q5` (degrees) | Smoothly move the physical arm to the specified joint angles with a single servo command (3 s duration) |
| `line` | `x1 y1 z1 x2 y2 z2` (mm) | Move the arm to the start point, then execute a straight-line Cartesian trajectory to the end point |
| `grip` | `open` or `close` | Open or close the gripper (servo 1) |

## Examples

```bash
# Forward kinematics: arm with shoulder at 45°, elbow at -90°
python xarm_controller.py fk 0 45 -90 45 0

# Inverse kinematics: where should joints be to reach (150, 0, 100) mm?
python xarm_controller.py ik 150 0 100

# Move physical arm to xyz position
python xarm_controller.py goto 150 0 100

# Move physical arm to specific joint angles
python xarm_controller.py movej 0 45 -90 45 0

# Straight-line trajectory from (200, 0, 150) to (100, 0, 150) mm
python xarm_controller.py line 200 0 150 100 0 150

# Gripper control
python xarm_controller.py grip open
python xarm_controller.py grip close
```
