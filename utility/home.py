"""Move all servos (including gripper) to their home positions."""

import json
import time
import os
from utility.xarm_hid import XArmHID

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "xarm_config.json")

def main():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    servos_cfg = cfg["servos"]
    targets = [(int(sid), s["home_position"]) for sid, s in servos_cfg.items()]

    # Gripper home = open position
    if "gripper" in cfg:
        grip = cfg["gripper"]
        targets.append((grip["servo_id"], grip.get("open_position", 200)))

    with XArmHID() as arm:
        voltage = arm.read_battery_voltage()
        print(f"Battery: {voltage} mV")

        print("Moving all servos to home...")
        arm.move_servos(targets, duration_ms=3000)
        time.sleep(3.5)
        print("Done.")

if __name__ == "__main__":
    main()
