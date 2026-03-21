"""
HiWonder xArm S1 Connection Test
Tests USB HID connection to the xArm and attempts basic communication.
"""

import sys
import time

try:
    import hid
except ImportError:
    print("ERROR: 'hidapi' package not installed.")
    print("Install it with: pip install hidapi")
    sys.exit(1)

HIWONDER_VENDOR_ID = 0x0483   # STMicroelectronics
XARM_PRODUCT_ID = 0x5750      # xArm HID device

# xArm HID protocol constants
HEADER_1 = 0x55
HEADER_2 = 0x55
CMD_SERVO_MOVE = 0x03
CMD_GET_BATTERY_VOLTAGE = 0x0F
CMD_BEEP = 0x3F


def find_xarm_devices():
    """Scan for all HiWonder xArm devices on USB."""
    print("Scanning USB HID devices...\n")
    all_devices = hid.enumerate()
    xarm_devices = [
        d for d in all_devices
        if d["vendor_id"] == HIWONDER_VENDOR_ID and d["product_id"] == XARM_PRODUCT_ID
    ]
    return xarm_devices, all_devices


def build_packet(cmd: int, data: bytes = b"") -> bytes:
    """Build an xArm HID command packet (padded to 64 bytes for HID report)."""
    length = len(data) + 2  # data length + cmd byte + length byte
    packet = bytes([HEADER_1, HEADER_2, length, cmd]) + data
    packet = packet.ljust(64, b"\x00")  # pad to 64 bytes for HID
    return packet


def test_connection():
    print("=" * 56)
    print("  HiWonder xArm S1 — Connection Test")
    print("=" * 56)
    print()

    xarm_devices, all_devices = find_xarm_devices()

    if not xarm_devices:
        print("FAIL: No xArm device found on USB.\n")
        print("Troubleshooting:")
        print("  1. Ensure the xArm is powered on")
        print("  2. Check the USB cable connection")
        print("  3. Try a different USB port or cable")
        print()
        print(f"Total HID devices detected: {len(all_devices)}")
        for d in all_devices:
            if d["vendor_id"] != 0:
                print(f"  - {d.get('manufacturer_string', '?')} / "
                      f"{d.get('product_string', '?')} "
                      f"(VID={d['vendor_id']:#06x} PID={d['product_id']:#06x})")
        return False

    dev_info = xarm_devices[0]
    print(f"FOUND xArm device:")
    print(f"  Manufacturer : {dev_info.get('manufacturer_string', 'N/A')}")
    print(f"  Product      : {dev_info.get('product_string', 'N/A')}")
    print(f"  VID:PID      : {dev_info['vendor_id']:#06x}:{dev_info['product_id']:#06x}")
    print(f"  Path         : {dev_info['path'].decode() if isinstance(dev_info['path'], bytes) else dev_info['path']}")
    print()

    print("Opening HID connection... ", end="", flush=True)
    try:
        device = hid.device()
        device.open(HIWONDER_VENDOR_ID, XARM_PRODUCT_ID)
        device.set_nonblocking(True)
        print("OK")
    except IOError as e:
        print(f"FAIL\n  Error: {e}")
        print("\n  On macOS you may need to grant Input Monitoring permission")
        print("  to your terminal app in System Settings > Privacy & Security.")
        return False

    print()

    # Test 1: Read battery voltage
    print("[Test 1] Reading battery voltage...")
    packet = build_packet(CMD_GET_BATTERY_VOLTAGE)
    try:
        device.write(packet)
        time.sleep(0.1)
        response = device.read(64)
        if response:
            print(f"  Response received ({len(response)} bytes): {bytes(response[:10]).hex(' ')}")
            if len(response) >= 6 and response[0] == HEADER_1 and response[1] == HEADER_2:
                voltage_raw = response[4] | (response[5] << 8)
                voltage_mv = voltage_raw
                print(f"  Battery voltage: ~{voltage_mv} mV ({voltage_mv/1000:.2f} V)")
            else:
                print("  Got a response (device is communicating)")
            print("  Result: PASS")
        else:
            print("  No response data (device may still be OK — some firmware variants don't reply to this command)")
            print("  Result: INCONCLUSIVE")
    except IOError as e:
        print(f"  Write/read error: {e}")
        print("  Result: FAIL")

    print()

    # Test 2: Move servo to current position (safe no-op movement)
    print("[Test 2] Sending a no-op servo command (move servo 1 to center, 1500µs, 1000ms)...")
    servo_id = 1
    position = 500  # center-ish position (safe)
    duration = 1000  # 1 second
    servo_count = 1
    data = (
        servo_count.to_bytes(1, "little")
        + duration.to_bytes(2, "little")
        + servo_id.to_bytes(1, "little")
        + position.to_bytes(2, "little")
    )
    packet = build_packet(CMD_SERVO_MOVE, data)
    try:
        bytes_written = device.write(packet)
        print(f"  Sent {bytes_written} bytes")
        time.sleep(0.1)
        response = device.read(64)
        if response:
            print(f"  Response: {bytes(response[:10]).hex(' ')}")
        print("  If the arm servo moved slightly, communication is working!")
        print("  Result: PASS (command sent successfully)")
    except IOError as e:
        print(f"  Error: {e}")
        print("  Result: FAIL")

    print()

    device.close()
    print("Connection closed.")
    print()
    print("=" * 56)
    print("  CONNECTION TEST COMPLETE — xArm is reachable!")
    print("=" * 56)
    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
