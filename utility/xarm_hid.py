"""
Reusable HID protocol driver for the HiWonder xArm S1 robotic arm.
"""

import sys
import time

try:
    import hid
except ImportError:
    print("ERROR: 'hidapi' package not installed.")
    print("Install it with: pip install hidapi")
    sys.exit(1)

HIWONDER_VENDOR_ID = 0x0483
XARM_PRODUCT_ID = 0x5750

HEADER_1 = 0x55
HEADER_2 = 0x55

CMD_SERVO_MOVE = 0x03
CMD_GET_BATTERY_VOLTAGE = 0x0F
CMD_SERVO_OFF = 0x14
CMD_SERVO_POS_READ = 0x15
CMD_BEEP = 0x3F

ALL_SERVO_IDS = [1, 2, 3, 4, 5, 6]


def _build_packet(cmd: int, data: bytes = b"") -> bytes:
    """Build an xArm HID command packet (65 bytes: 1 report ID + 64 payload)."""
    length = len(data) + 2  # data length + cmd byte + length byte
    packet = bytes([0x00, HEADER_1, HEADER_2, length, cmd]) + data
    packet = packet.ljust(65, b"\x00")
    return packet


class XArmHID:
    """HID interface for the HiWonder xArm S1."""

    def __init__(self):
        self.device = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        """Open HID connection to the xArm."""
        self.device = hid.device()
        self.device.open(HIWONDER_VENDOR_ID, XARM_PRODUCT_ID)
        self.device.set_nonblocking(True)

    def close(self):
        """Close the HID connection."""
        if self.device:
            self.device.close()
            self.device = None

    def _write_and_read(self, packet: bytes, timeout_ms: int = 500) -> list | None:
        """Send a packet and poll for a response."""
        self.device.write(packet)
        polls = timeout_ms // 10
        for _ in range(polls):
            time.sleep(0.01)
            response = self.device.read(64)
            if response:
                return response
        return None

    def read_battery_voltage(self) -> int:
        """Read battery voltage in millivolts."""
        packet = _build_packet(CMD_GET_BATTERY_VOLTAGE)
        response = self._write_and_read(packet)
        if response and len(response) >= 6 and response[0] == HEADER_1 and response[1] == HEADER_2:
            return response[4] | (response[5] << 8)
        return 0

    def move_servo(self, servo_id: int, position: int, duration_ms: int = 3000):
        """Move a single servo to a target position."""
        self.move_servos([(servo_id, position)], duration_ms)

    def move_servos(self, targets: list[tuple[int, int]], duration_ms: int = 3000):
        """Move multiple servos. targets = [(id, position), ...]"""
        count = len(targets)
        data = count.to_bytes(1, "little") + duration_ms.to_bytes(2, "little")
        for servo_id, position in targets:
            position = max(-500, min(1500, position))
            data += servo_id.to_bytes(1, "little") + (position & 0xFFFF).to_bytes(2, "little")
        packet = _build_packet(CMD_SERVO_MOVE, data)
        written = self.device.write(packet)
        if written < 0:
            raise IOError(f"device.write() returned {written}")

    def servos_off(self, servo_ids: list[int] | None = None):
        """Disable torque on servos so they can be moved by hand."""
        if servo_ids is None:
            servo_ids = ALL_SERVO_IDS
        count = len(servo_ids)
        data = bytes([count]) + bytes(servo_ids)
        packet = _build_packet(CMD_SERVO_OFF, data)
        written = self.device.write(packet)
        if written < 0:
            raise IOError(f"device.write() returned {written}")

    def read_positions(self, servo_ids: list[int] | None = None) -> dict[int, int]:
        """Read current positions of servos. Returns {servo_id: position}."""
        if servo_ids is None:
            servo_ids = ALL_SERVO_IDS
        count = len(servo_ids)
        data = bytes([count]) + bytes(servo_ids)
        packet = _build_packet(CMD_SERVO_POS_READ, data)
        response = self._write_and_read(packet, timeout_ms=500)
        if not response:
            return {}
        # Response format: [0x55, 0x55, length, CMD_SERVO_POS_READ, count, (id, pos_lo, pos_hi) * count]
        if len(response) < 5:
            return {}
        resp_count = response[4]
        positions = {}
        for i in range(resp_count):
            offset = 5 + 3 * i
            if offset + 2 >= len(response):
                break
            sid = response[offset]
            raw = response[offset + 1] | (response[offset + 2] << 8)
            pos = raw if raw < 0x8000 else raw - 0x10000
            positions[sid] = pos
        return positions
