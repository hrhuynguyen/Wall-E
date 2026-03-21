"""
HiWonder xArm S1 Controller
USB HID communication, inverse kinematics, and motion control.
"""

import hid
import time
import math
import json
import os

HIWONDER_VENDOR_ID = 0x0483
XARM_PRODUCT_ID = 0x5750

HEADER_1 = 0x55
HEADER_2 = 0x55
CMD_SERVO_MOVE = 0x03
CMD_MULT_SERVO_POS_READ = 0x15

CONFIG_FILE = "xarm_config.json"


class XArmController:
    """
    Controller for HiWonder xArm S1 via USB HID.

    Coordinate system (origin = base center at table level):
        X = forward, Y = left, Z = up  (all in mm)

    Servo layout:
        6 = base rotation, 5 = shoulder, 4 = elbow,
        3 = wrist pitch, 2 = wrist rotation (fixed), 1 = gripper
    """

    # --- Arm dimensions in mm (measured) ---
    L1 = 69    # base height (table to shoulder pivot)
    L2 = 100   # upper arm (shoulder to elbow)
    L3 = 98    # forearm (elbow to wrist)
    L4 = 145   # hand (wrist to gripper tip)

    # --- Servo calibration ---
    SCALE = 4.267  # servo units per degree (384 units / 90°)

    S5_OFFSET = 114  # servo 5 value at shoulder_angle = 0° (horizontal)
    S4_ZERO = 502    # servo 4 at elbow straight
    S3_ZERO = 502    # servo 3 at wrist straight
    S6_ZERO = 498    # servo 6 facing forward

    S2_HOME = 495    # servo 2 wrist rotation (always fixed)
    GRIPPER_OPEN = 139
    GRIPPER_CLOSED = 450

    def __init__(self, auto_calibrate=True):
        self.device = hid.device()
        self.device.open(HIWONDER_VENDOR_ID, XARM_PRODUCT_ID)
        self.device.set_nonblocking(False)

        # Servo direction signs (angle → servo value mapping)
        # +1 = increasing angle → increasing servo value
        self.dir_5 = 1    # confirmed: + servo = shoulder backward/up
        self.dir_4 = -1   # confirmed: + servo = elbow forward/down (opposite of S5)
        self.dir_3 = -1   # confirmed: + servo = wrist forward/down (same as S4)
        self.dir_6 = 1    # default: + servo = base rotates left

        config_loaded = self._load_config()
        if not config_loaded and auto_calibrate:
            print("No calibration file found. Running first-time calibration...\n")
            self.calibrate()

    # ------------------------------------------------------------------ #
    #  Low-level HID communication
    # ------------------------------------------------------------------ #

    def _build_packet(self, cmd: int, data: bytes = b"") -> bytes:
        length = len(data) + 2
        packet = bytes([HEADER_1, HEADER_2, length, cmd]) + data
        return packet.ljust(64, b"\x00")

    def _flush_input(self):
        self.device.set_nonblocking(True)
        while self.device.read(64):
            pass
        self.device.set_nonblocking(False)

    def read_servo(self, servo_id: int, timeout_ms: int = 200) -> int | None:
        """Read current position of a single servo. Returns 0–1000 or None."""
        self._flush_input()
        packet = self._build_packet(CMD_MULT_SERVO_POS_READ, bytes([1, servo_id]))
        self.device.write(packet)

        self.device.set_nonblocking(True)
        deadline = time.time() + timeout_ms / 1000.0
        response = None
        while time.time() < deadline:
            response = self.device.read(64)
            if response:
                break
            time.sleep(0.01)
        self.device.set_nonblocking(False)

        if (response and len(response) >= 8
                and response[0] == HEADER_1 and response[1] == HEADER_2
                and response[3] == CMD_MULT_SERVO_POS_READ
                and response[5] == servo_id):
            pos = response[6] | (response[7] << 8)
            if 0 <= pos <= 1500:
                return pos
        return None

    def read_all_servos(self) -> dict:
        """Read positions of servos 1–6."""
        positions = {}
        for sid in range(1, 7):
            pos = self.read_servo(sid)
            if pos is not None:
                positions[sid] = pos
        return positions

    def move_servos(self, positions: dict, duration_ms: int = 1000):
        """Move multiple servos simultaneously. positions = {servo_id: value}."""
        count = len(positions)
        data = count.to_bytes(1, "little") + duration_ms.to_bytes(2, "little")
        for sid, pos in sorted(positions.items()):
            pos = max(0, min(1000, int(round(pos))))
            data += sid.to_bytes(1, "little") + pos.to_bytes(2, "little")
        packet = self._build_packet(CMD_SERVO_MOVE, data)
        self.device.write(packet)

    # ------------------------------------------------------------------ #
    #  Calibration
    # ------------------------------------------------------------------ #

    def _config_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)

    def _load_config(self) -> bool:
        path = self._config_path()
        if not os.path.exists(path):
            return False
        with open(path) as f:
            cfg = json.load(f)
        self.dir_4 = cfg.get("dir_4", self.dir_4)
        self.dir_3 = cfg.get("dir_3", self.dir_3)
        self.dir_6 = cfg.get("dir_6", self.dir_6)
        print(f"Loaded calibration: dir_4={self.dir_4}, dir_3={self.dir_3}, dir_6={self.dir_6}")
        return True

    def _save_config(self):
        path = self._config_path()
        cfg = {"dir_4": self.dir_4, "dir_3": self.dir_3, "dir_6": self.dir_6}
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Calibration saved to {CONFIG_FILE}")

    def calibrate(self):
        """
        Interactive servo direction calibration.
        Moves each joint slightly and asks the user which way it went.
        Run once — results are saved to xarm_config.json.
        """
        print("=" * 50)
        print("  Servo Direction Calibration")
        print("=" * 50)
        print("\nThe arm will make small movements.")
        print("Answer each question based on what you see.\n")

        # Home position: straight up
        home_s5 = self.S5_OFFSET + int(90 * self.SCALE)  # ~498
        print("Moving to home (straight up)...")
        self.move_servos({
            1: self.GRIPPER_OPEN, 2: self.S2_HOME,
            3: self.S3_ZERO, 4: self.S4_ZERO,
            5: home_s5, 6: self.S6_ZERO,
        }, duration_ms=1500)
        time.sleep(2)

        # --- Base (servo 6) ---
        print("\n[Base rotation — servo 6]")
        print("  Increasing servo 6 by +100 units...")
        self.move_servos({6: self.S6_ZERO + 100}, duration_ms=600)
        time.sleep(1)
        ans = input("  Facing the arm, did it rotate to YOUR LEFT? [Y/n]: ").strip().lower()
        self.dir_6 = 1 if ans != "n" else -1
        self.move_servos({6: self.S6_ZERO}, duration_ms=600)
        time.sleep(0.7)

        # Tilt shoulder to ~45° so elbow/wrist movement is clearly visible
        shoulder_45 = self.S5_OFFSET + int(45 * self.SCALE)  # ~306
        print("\n  Tilting shoulder to 45° for visibility...")
        self.move_servos({5: shoulder_45}, duration_ms=800)
        time.sleep(1)

        # --- Elbow (servo 4) ---
        print("\n[Elbow — servo 4]")
        print("  Increasing servo 4 by +100 units...")
        self.move_servos({4: self.S4_ZERO + 100}, duration_ms=600)
        time.sleep(1)
        ans = input("  Did the forearm swing UPWARD (toward the sky)? [Y/n]: ").strip().lower()
        self.dir_4 = 1 if ans != "n" else -1
        self.move_servos({4: self.S4_ZERO}, duration_ms=600)
        time.sleep(0.7)

        # --- Wrist (servo 3) ---
        print("\n[Wrist — servo 3]")
        print("  Increasing servo 3 by +100 units...")
        self.move_servos({3: self.S3_ZERO + 100}, duration_ms=600)
        time.sleep(1)
        ans = input("  Did the gripper tilt UPWARD? [Y/n]: ").strip().lower()
        self.dir_3 = 1 if ans != "n" else -1
        self.move_servos({3: self.S3_ZERO}, duration_ms=600)
        time.sleep(0.7)

        # Return to home
        print("\nReturning to home...")
        self.move_servos({
            3: self.S3_ZERO, 4: self.S4_ZERO,
            5: home_s5, 6: self.S6_ZERO,
        }, duration_ms=1000)
        time.sleep(1.2)

        self._save_config()
        print(f"\nCalibration complete!")
        print(f"  Base  (S6): dir = {self.dir_6:+d}")
        print(f"  Elbow (S4): dir = {self.dir_4:+d}")
        print(f"  Wrist (S3): dir = {self.dir_3:+d}")
        print()

    # ------------------------------------------------------------------ #
    #  Inverse Kinematics
    # ------------------------------------------------------------------ #

    def inverse_kinematics(self, x: float, y: float, z: float,
                            gripper_angle_deg: float = 0.0) -> dict:
        """
        Compute servo positions for the gripper tip at (x, y, z) in mm.

        gripper_angle_deg: angle of the gripper from horizontal.
            0 = horizontal (default), negative = tilted downward.
            Use -20 to -40 for closer/lower targets.

        Tries elbow-up first (more natural), falls back to elbow-down
        if elbow-up puts any servo out of range.

        Returns {3: val, 4: val, 5: val, 6: val}.
        Raises ValueError if the position is unreachable.
        """
        base_angle_deg = math.degrees(math.atan2(y, x))
        r = math.sqrt(x ** 2 + y ** 2)

        gripper_rad = math.radians(gripper_angle_deg)

        # Wrist position: tip is at (r, z), gripper extends L4 at gripper_angle
        wr = r - self.L4 * math.cos(gripper_rad)
        wz = z - self.L4 * math.sin(gripper_rad) - self.L1

        D = math.sqrt(wr ** 2 + wz ** 2)
        max_reach = self.L2 + self.L3 - 1
        min_reach = abs(self.L2 - self.L3) + 1

        if D > max_reach:
            raise ValueError(
                f"Target too far: wrist distance {D:.1f}mm > max {max_reach:.1f}mm. "
                f"Max gripper reach is ~{self.L2 + self.L3 + self.L4:.0f}mm from base."
            )
        if D < min_reach:
            raise ValueError(
                f"Target too close: wrist distance {D:.1f}mm < min {min_reach:.1f}mm. "
                f"Try a larger gripper_angle_deg (e.g. -30) to reach closer targets."
            )

        cos_gamma = (self.L2 ** 2 + self.L3 ** 2 - D ** 2) / (2 * self.L2 * self.L3)
        gamma = math.acos(max(-1.0, min(1.0, cos_gamma)))
        phi = math.atan2(wz, wr)
        cos_alpha = (self.L2 ** 2 + D ** 2 - self.L3 ** 2) / (2 * self.L2 * D)
        alpha = math.acos(max(-1.0, min(1.0, cos_alpha)))

        bend = math.pi - gamma

        candidates = [
            (phi + alpha, -bend),   # elbow-up: shoulder high, elbow bends down
            (phi - alpha,  bend),   # elbow-down: shoulder low, elbow bends up
        ]

        best = None
        best_margin = -999

        for shoulder_rad, elbow_bend_rad in candidates:
            shoulder_deg = math.degrees(shoulder_rad)
            elbow_bend_deg = math.degrees(elbow_bend_rad)
            forearm_deg = shoulder_deg + elbow_bend_deg
            wrist_bend_deg = gripper_angle_deg - forearm_deg

            s6 = self.S6_ZERO + base_angle_deg * self.SCALE * self.dir_6
            s5 = self.S5_OFFSET + shoulder_deg * self.SCALE * self.dir_5
            s4 = self.S4_ZERO + elbow_bend_deg * self.SCALE * self.dir_4
            s3 = self.S3_ZERO + wrist_bend_deg * self.SCALE * self.dir_3

            vals = {6: s6, 5: s5, 4: s4, 3: s3}
            if all(0 <= v <= 1000 for v in vals.values()):
                margin = min(min(v, 1000 - v) for v in vals.values())
                if margin > best_margin:
                    best = {sid: int(round(v)) for sid, v in vals.items()}
                    best_margin = margin

        if best is not None:
            return best

        raise ValueError(
            f"Target ({x:.0f}, {y:.0f}, {z:.0f}) with gripper angle {gripper_angle_deg:.0f}° "
            f"requires servos outside 0–1000. "
            f"Try: farther position, different height, or gripper_angle_deg=-30."
        )

    # ------------------------------------------------------------------ #
    #  Forward Kinematics (for verification)
    # ------------------------------------------------------------------ #

    def forward_kinematics(self, servo_positions: dict = None) -> tuple:
        """
        Compute gripper tip (x, y, z) and gripper angle from servo positions.
        If servo_positions is None, reads current positions from the arm.
        Returns (x, y, z, gripper_angle_deg).
        """
        if servo_positions is None:
            servo_positions = {}
            for sid in [3, 4, 5, 6]:
                pos = self.read_servo(sid)
                if pos is None:
                    raise RuntimeError(f"Cannot read servo {sid}")
                servo_positions[sid] = pos

        base_rad = math.radians(
            (servo_positions[6] - self.S6_ZERO) * self.dir_6 / self.SCALE
        )
        shoulder_rad = math.radians(
            (servo_positions[5] - self.S5_OFFSET) * self.dir_5 / self.SCALE
        )
        elbow_bend_rad = math.radians(
            (servo_positions[4] - self.S4_ZERO) * self.dir_4 / self.SCALE
        )
        wrist_bend_rad = math.radians(
            (servo_positions[3] - self.S3_ZERO) * self.dir_3 / self.SCALE
        )

        forearm_rad = shoulder_rad + elbow_bend_rad
        hand_rad = forearm_rad + wrist_bend_rad

        r_tip = (self.L2 * math.cos(shoulder_rad)
                 + self.L3 * math.cos(forearm_rad)
                 + self.L4 * math.cos(hand_rad))
        z_tip = (self.L1
                 + self.L2 * math.sin(shoulder_rad)
                 + self.L3 * math.sin(forearm_rad)
                 + self.L4 * math.sin(hand_rad))

        x = r_tip * math.cos(base_rad)
        y = r_tip * math.sin(base_rad)
        return (x, y, z_tip, math.degrees(hand_rad))

    # ------------------------------------------------------------------ #
    #  High-level motion commands
    # ------------------------------------------------------------------ #

    def move_to(self, x: float, y: float, z: float,
                duration_ms: int = 1000, gripper_angle_deg: float = 0.0):
        """Move gripper tip to (x, y, z) mm. gripper_angle_deg: 0=horizontal."""
        servos = self.inverse_kinematics(x, y, z, gripper_angle_deg)
        servos[2] = self.S2_HOME
        self.move_servos(servos, duration_ms)
        return servos

    def open_gripper(self, duration_ms: int = 500):
        """Open the gripper. Sends to full open (0) first to ensure direction change works."""
        self._flush_input()
        self.move_servos({1: 0}, duration_ms)
        time.sleep(duration_ms / 1000.0 + 0.3)
        if self.GRIPPER_OPEN > 10:
            self._flush_input()
            self.move_servos({1: self.GRIPPER_OPEN}, duration_ms)

    def close_gripper(self, duration_ms: int = 500):
        """Close the gripper gently to gripping position."""
        overshoot = min(self.GRIPPER_CLOSED + 100, 600)
        self._flush_input()
        self.move_servos({1: overshoot}, duration_ms)
        time.sleep(duration_ms / 1000.0 + 0.3)
        self._flush_input()
        self.move_servos({1: self.GRIPPER_CLOSED}, duration_ms)

    def home(self, duration_ms: int = 1500):
        """Move to safe home position (straight up, gripper open)."""
        home_s5 = self.S5_OFFSET + int(90 * self.SCALE)
        self.move_servos({
            1: self.GRIPPER_OPEN,
            2: self.S2_HOME,
            3: self.S3_ZERO,
            4: self.S4_ZERO,
            5: home_s5,
            6: self.S6_ZERO,
        }, duration_ms)

    def close(self):
        """Close the HID connection."""
        self.device.close()

    # ------------------------------------------------------------------ #
    #  Workspace info
    # ------------------------------------------------------------------ #

    def workspace_info(self) -> str:
        """Return a summary of the arm's reachable workspace."""
        max_r = self.L2 + self.L3 + self.L4
        max_z = self.L1 + self.L2 + self.L3
        max_elbow_bend = 502 / self.SCALE
        return (
            f"Workspace:\n"
            f"  Max horizontal reach:  {max_r:.0f}mm from base center\n"
            f"  Comfortable range:     ~245–340mm (gripper horizontal)\n"
            f"  Closer targets:        use gripper_angle_deg=-30 to reach ~170mm+\n"
            f"  Max height:            {max_z:.0f}mm above table\n"
            f"  Max elbow bend:        {max_elbow_bend:.0f}° (servo limit)\n"
            f"  Base rotation:         ~±110° from forward"
        )
