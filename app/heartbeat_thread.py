"""Periodic health checks for connected devices.

Runs every HEARTBEAT_INTERVAL_S seconds.  If a device stops responding,
posts HEARTBEAT_FAIL so the state machine can surface an error.
"""

from __future__ import annotations

import threading
import time

from app.config import HEARTBEAT_INTERVAL_S
from app.events import HEARTBEAT_FAIL, Event, EventBus


class HeartbeatThread:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._running = False
        self._thread: threading.Thread | None = None
        self._arm_worker = None  # set via set_arm_worker
        self._check_muse = False

    def set_arm_worker(self, arm_worker: object) -> None:
        """Provide the ArmWorker so we can call check_health()."""
        self._arm_worker = arm_worker

    def set_check_muse(self, enabled: bool) -> None:
        """Enable/disable Muse LSL health checks."""
        self._check_muse = enabled

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="heartbeat",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=HEARTBEAT_INTERVAL_S + 1)
            self._thread = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL_S)
            if not self._running:
                break
            self._check_devices()

    def _check_devices(self) -> None:
        # xArm
        if self._arm_worker is not None:
            try:
                if not self._arm_worker.check_health():
                    print("[Heartbeat] xArm health check failed")
                    self._bus.post(Event(HEARTBEAT_FAIL, "xArm"))
            except Exception as e:
                print(f"[Heartbeat] xArm check error: {e}")
                self._bus.post(Event(HEARTBEAT_FAIL, "xArm"))

        # Muse LSL — check that the stream is still resolvable
        if self._check_muse:
            try:
                from pylsl import resolve_byprop

                streams = resolve_byprop("type", "EEG", timeout=2.0)
                if not streams:
                    print("[Heartbeat] Muse LSL stream lost")
                    self._bus.post(Event(HEARTBEAT_FAIL, "Muse"))
            except Exception as e:
                print(f"[Heartbeat] Muse check error: {e}")
                self._bus.post(Event(HEARTBEAT_FAIL, "Muse"))
