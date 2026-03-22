"""Background worker thread that owns the xArm HID connection.

All blocking arm operations run here so they never stall the UI.
The main thread enqueues commands; the worker executes them and posts
completion events to the EventBus.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.events import (
    ARM_ERROR,
    ARM_MOVE_DONE,
    DROP_DONE,
    HOME_REACHED,
    LIFT_DONE,
    PICK_DONE,
    XARM_CONNECTED,
    XARM_DISCONNECTED,
    Event,
    EventBus,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "utility", "xarm_config.json"
)

# Sentinel that tells the worker thread to exit
_STOP = object()


@dataclass
class _Cmd:
    name: str
    data: Any = None


class ArmWorker:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._arm = None  # XArmHID — only touched by worker thread
        self._ctrl = None  # XArmController — only touched by worker thread
        self._config: dict | None = None
        self._hid_lock = threading.Lock()  # for direct servo writes (head tracking)
        self._last_xyz: list[float] | None = None  # last target in mm (for lift)
        # Base rotation state (servo 6) — initialized after connect
        self._base_pos: float | None = None
        self._base_min: int = 0
        self._base_max: int = 1000
        self._base_direction: int = 1
        self._base_angle_per_unit: float = 0.24

    @property
    def connected(self) -> bool:
        return self._arm is not None

    # ------------------------------------------------------------------
    # Public API (called from main thread, enqueues commands)
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="arm-worker")
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def connect(self) -> None:
        self._queue.put(_Cmd("connect"))

    def disconnect(self) -> None:
        self._queue.put(_Cmd("disconnect"))

    def move_home(self) -> None:
        self._queue.put(_Cmd("home"))

    def move_to_xyz(self, xyz: list[float]) -> None:
        self._queue.put(_Cmd("move_to_xyz", xyz))

    def pick(self) -> None:
        self._queue.put(_Cmd("pick"))

    def lift(self, height_mm: float) -> None:
        self._queue.put(_Cmd("lift", height_mm))

    def drop(self, height_mm: float = 20.0) -> None:
        self._queue.put(_Cmd("drop", height_mm))

    def rotate_servo(self, servo_id: int, position: int, duration_ms: int = 100) -> None:
        """Direct servo write for head-tracking rotation (bypasses queue)."""
        with self._hid_lock:
            if self._arm is not None:
                try:
                    self._arm.move_servos([(servo_id, position)], duration_ms=duration_ms)
                except Exception:
                    pass  # non-critical, next frame will retry

    def init_base_rotation(self) -> None:
        """Initialize base rotation tracking from servo 6 config."""
        if self._config is None:
            return
        servo_cfg = self._config["servos"]["6"]
        self._base_pos = float(servo_cfg["home_position"])
        self._base_min = servo_cfg["position_min"]
        self._base_max = servo_cfg["position_max"]
        self._base_direction = servo_cfg["direction"]
        self._base_angle_per_unit = servo_cfg["angle_per_unit"]
        print(f"[ArmWorker] base rotation initialized at pos={self._base_pos}")

    def rotate_base_incremental(self, angle_delta_deg: float) -> None:
        """Rotate base servo by angle_delta degrees (non-blocking, direct write)."""
        if self._base_pos is None:
            return
        pos_delta = angle_delta_deg / (self._base_direction * self._base_angle_per_unit)
        self._base_pos += pos_delta
        self._base_pos = max(self._base_min, min(self._base_max, self._base_pos))
        target = int(round(self._base_pos))
        self.rotate_servo(6, target, duration_ms=100)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._do_disconnect()
                break
            if isinstance(item, _Cmd):
                self._dispatch(item)

    def _dispatch(self, cmd: _Cmd) -> None:
        try:
            handler = getattr(self, f"_do_{cmd.name}", None)
            if handler:
                handler(cmd.data)
            else:
                print(f"[ArmWorker] unknown command: {cmd.name}")
        except Exception as e:
            print(f"[ArmWorker] error in {cmd.name}: {e}")
            self._bus.post(Event(ARM_ERROR, str(e)))

    # ------------------------------------------------------------------
    # Command handlers (run on worker thread)
    # ------------------------------------------------------------------

    def _do_connect(self, _data: Any) -> None:
        from utility.xarm_controller import XArmController, load_config
        from utility.xarm_hid import XArmHID

        try:
            self._config = load_config(CONFIG_PATH)
            arm = XArmHID()
            arm.open()
            # Quick health check
            voltage = arm.read_battery_voltage()
            print(f"[ArmWorker] xArm connected, battery: {voltage} mV")
            self._arm = arm
            self._ctrl = XArmController(self._config)
            self._bus.post(Event(XARM_CONNECTED))
        except Exception as e:
            print(f"[ArmWorker] connection failed: {e}")
            self._bus.post(Event(XARM_DISCONNECTED))

    def _do_disconnect(self, _data: Any = None) -> None:
        with self._hid_lock:
            if self._arm is not None:
                try:
                    self._arm.close()
                except Exception:
                    pass
                self._arm = None
                self._ctrl = None
                print("[ArmWorker] xArm disconnected")

    def _do_home(self, _data: Any) -> None:
        if self._arm is None or self._config is None:
            self._bus.post(Event(ARM_ERROR, "xArm not connected"))
            return

        servos_cfg = self._config["servos"]
        targets = [(int(sid), s["home_position"]) for sid, s in servos_cfg.items()]

        # Gripper home = open position
        if "gripper" in self._config:
            grip = self._config["gripper"]
            targets.append((grip["servo_id"], grip.get("open_position", 200)))

        with self._hid_lock:
            self._arm.move_servos(targets, duration_ms=3000)
        time.sleep(3.5)
        print("[ArmWorker] home reached")
        self._bus.post(Event(HOME_REACHED))

    def _ik_with_fallback(self, xyz: list) -> tuple[list | None, list]:
        """Try IK; on failure fall back to [260, y, 0] keeping original y."""
        result = self._ctrl.ik(xyz)
        if result is not None:
            return result, xyz
        fallback = [230.0, xyz[1], 0.0]
        print(f"[ArmWorker] IK failed for {xyz}, falling back to {fallback}")
        result = self._ctrl.ik(fallback)
        return result, fallback

    def _do_move_to_xyz(self, xyz: Any) -> None:
        if self._arm is None or self._ctrl is None:
            self._bus.post(Event(ARM_ERROR, "xArm not connected"))
            return

        result, target = self._ik_with_fallback(list(xyz))
        if result is None:
            self._bus.post(Event(ARM_ERROR, f"IK failed for {xyz} and fallback"))
            return

        with self._hid_lock:
            self._ctrl.move_to(result, self._arm)
        self._last_xyz = list(target)
        print(f"[ArmWorker] moved to {target}")
        self._bus.post(Event(ARM_MOVE_DONE))

    def _do_pick(self, _data: Any) -> None:
        if self._arm is None or self._ctrl is None:
            self._bus.post(Event(ARM_ERROR, "xArm not connected"))
            return

        with self._hid_lock:
            # Open gripper to prepare for grab
            print("[ArmWorker] opening gripper...")
            self._ctrl.open_gripper(self._arm)
            # Close gripper to grab the object
            print("[ArmWorker] closing gripper...")
            self._ctrl.close_gripper(self._arm)
        print("[ArmWorker] pick done")
        self._bus.post(Event(PICK_DONE))

    def _do_lift(self, height_mm: Any) -> None:
        if self._arm is None or self._ctrl is None:
            self._bus.post(Event(ARM_ERROR, "xArm not connected"))
            return

        if self._last_xyz is None:
            self._bus.post(Event(ARM_ERROR, "No known position for lift"))
            return

        # Lift: same x, y but raise z by height_mm
        lifted_xyz = [
            self._last_xyz[0],
            self._last_xyz[1],
            self._last_xyz[2] + float(height_mm),
        ]
        print(f"[ArmWorker] lifting from z={self._last_xyz[2]:.1f} to z={lifted_xyz[2]:.1f} mm")

        result, target = self._ik_with_fallback(lifted_xyz)
        if result is None:
            self._bus.post(Event(ARM_ERROR, f"IK failed for lift and fallback"))
            return

        with self._hid_lock:
            self._ctrl.move_to(result, self._arm)
        self._last_xyz = target
        print(f"[ArmWorker] lift {height_mm}mm done")
        self._bus.post(Event(LIFT_DONE))

    def _do_drop(self, height_mm: Any) -> None:
        if self._arm is None or self._ctrl is None:
            self._bus.post(Event(ARM_ERROR, "xArm not connected"))
            return

        # Lower arm by height_mm before releasing
        if self._last_xyz is not None:
            lowered_xyz = [
                self._last_xyz[0],
                self._last_xyz[1],
                self._last_xyz[2] - float(height_mm),
            ]
            print(f"[ArmWorker] lowering from z={self._last_xyz[2]:.1f} to z={lowered_xyz[2]:.1f} mm")
            result, target = self._ik_with_fallback(lowered_xyz)
            if result is not None:
                with self._hid_lock:
                    self._ctrl.move_to(result, self._arm)
                self._last_xyz = target
            else:
                print(f"[ArmWorker] IK failed for lower and fallback, dropping in place")

        # Open gripper to release object
        with self._hid_lock:
            print("[ArmWorker] opening gripper...")
            self._ctrl.open_gripper(self._arm)
        self._last_xyz = None
        print("[ArmWorker] drop done")
        self._bus.post(Event(DROP_DONE))

    # ------------------------------------------------------------------
    # Health check (called by heartbeat thread)
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """Return True if the arm responds to a battery read."""
        with self._hid_lock:
            if self._arm is None:
                return False
            try:
                v = self._arm.read_battery_voltage()
                return v > 0
            except Exception:
                return False
