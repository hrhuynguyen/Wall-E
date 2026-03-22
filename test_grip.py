"""Interactive gripper test — find the best open/close positions.

Run: python test_grip.py

Controls:
  +/- or up/down arrows : adjust position by 10
  [/]                    : adjust position by 50
  o                      : jump to config open position
  c                      : jump to config closed position
  s                      : save current as new open/close (prompts)
  q                      : quit
"""

import json
import os
import sys
import time

from utility.xarm_hid import XArmHID

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "utility", "xarm_config.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)

grip_cfg = config["gripper"]
SERVO_ID = grip_cfg["servo_id"]
OPEN_POS = grip_cfg["open_position"]
CLOSED_POS = grip_cfg["closed_position"]

print("=" * 50)
print("  Gripper Test — Interactive Servo Control")
print("=" * 50)
print(f"  Servo ID:        {SERVO_ID}")
print(f"  Config open:     {OPEN_POS}")
print(f"  Config closed:   {CLOSED_POS}")
print()

arm = XArmHID()
arm.open()
voltage = arm.read_battery_voltage()
print(f"  Battery: {voltage} mV")

current_pos = OPEN_POS
arm.move_servos([(SERVO_ID, current_pos)], duration_ms=1000)
time.sleep(1.0)
print(f"\n  Starting at open position: {current_pos}")
print()
print("  +/-  : ±10    [/]  : ±50")
print("  o    : open    c   : close")
print("  s    : save    q   : quit")
print()

try:
    while True:
        cmd = input(f"  pos={current_pos} > ").strip().lower()

        delta = 0
        if cmd in ("+", "="):
            delta = 10
        elif cmd == "-":
            delta = -10
        elif cmd == "]":
            delta = 50
        elif cmd == "[":
            delta = -50
        elif cmd == "o":
            current_pos = OPEN_POS
            print(f"  → open ({current_pos})")
        elif cmd == "c":
            current_pos = CLOSED_POS
            print(f"  → closed ({current_pos})")
        elif cmd == "s":
            which = input("  Save as (o)pen or (c)lose? ").strip().lower()
            if which == "o":
                config["gripper"]["open_position"] = current_pos
                with open(CONFIG_PATH, "w") as f:
                    json.dump(config, f, indent=2)
                OPEN_POS = current_pos
                print(f"  Saved open_position = {current_pos}")
            elif which == "c":
                config["gripper"]["closed_position"] = current_pos
                with open(CONFIG_PATH, "w") as f:
                    json.dump(config, f, indent=2)
                CLOSED_POS = current_pos
                print(f"  Saved closed_position = {current_pos}")
            else:
                print("  Cancelled")
            continue
        elif cmd == "q":
            break
        elif cmd.isdigit() or (cmd.startswith("-") and cmd[1:].isdigit()):
            current_pos = int(cmd)
            print(f"  → jump to {current_pos}")
        else:
            print("  Unknown command")
            continue

        current_pos += delta
        current_pos = max(0, min(1000, current_pos))
        arm.move_servos([(SERVO_ID, current_pos)], duration_ms=500)
        time.sleep(0.5)

except KeyboardInterrupt:
    print()
finally:
    arm.close()
    print("  Done.")
