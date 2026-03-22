"""Event system for inter-thread communication."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Event types (string constants)
# ---------------------------------------------------------------------------

# Device connection
XARM_CONNECTED = "xarm_connected"
XARM_DISCONNECTED = "xarm_disconnected"
MUSE_CONNECTED = "muse_connected"
MUSE_DISCONNECTED = "muse_disconnected"

# Setup
HOME_REACHED = "home_reached"
EYE_CALIBRATION_DONE = "eye_calibration_done"
EEG_WARMUP_DONE = "eeg_warmup_done"

# Vision
OBJECTS_DETECTED = "objects_detected"       # data: list[dict]
GAZE_UPDATE = "gaze_update"                 # data: (sx, sy)
HEAD_UPDATE = "head_update"                 # data: {"yaw": float, "direction": str}

# BCI
JAW_CLENCH = "jaw_clench"                   # data: duration_ms

# Arm
ARM_MOVE_DONE = "arm_move_done"
PICK_DONE = "pick_done"
DROP_DONE = "drop_done"
LIFT_DONE = "lift_done"

# Errors
ARM_ERROR = "arm_error"                     # data: str (error message)
HEARTBEAT_FAIL = "heartbeat_fail"           # data: str (which device)

# User actions
REQUEST_START_SESSION = "request_start_session"
REQUEST_END_SESSION = "request_end_session"

# Selection
OBJECT_SELECTED = "object_selected"         # data: dict (object info)


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class Event:
    type: str
    data: Any = None
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# EventBus — thread-safe event queue
# ---------------------------------------------------------------------------

class EventBus:
    def __init__(self) -> None:
        self._queue: queue.Queue[Event] = queue.Queue()

    def post(self, event: Event) -> None:
        self._queue.put(event)

    def poll(self) -> Event | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> list[Event]:
        events = []
        while True:
            event = self.poll()
            if event is None:
                break
            events.append(event)
        return events
