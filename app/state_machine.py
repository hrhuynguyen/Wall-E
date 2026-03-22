"""Central state machine for the WallE assistive arm system.

Pure logic — no hardware calls. Returns Command objects that the
orchestrator (app_window.py) dispatches to the appropriate thread/worker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from app.config import CENTER_HOLD_MS
from app.events import (
    ARM_ERROR,
    ARM_MOVE_DONE,
    DROP_DONE,
    EEG_WARMUP_DONE,
    EYE_CALIBRATION_DONE,
    GAZE_UPDATE,
    HEAD_UPDATE,
    HEARTBEAT_FAIL,
    HOME_REACHED,
    JAW_CLENCH,
    LIFT_DONE,
    MUSE_CONNECTED,
    MUSE_DISCONNECTED,
    OBJECT_SELECTED,
    OBJECTS_DETECTED,
    PICK_DONE,
    REQUEST_END_SESSION,
    REQUEST_START_SESSION,
    XARM_CONNECTED,
    XARM_DISCONNECTED,
    Event,
)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(Enum):
    LANDING = auto()
    CONNECTING_XARM = auto()
    CONNECTING_MUSE = auto()
    HOMING = auto()
    CALIBRATING_EYES = auto()
    CALIBRATING_EEG = auto()
    SCANNING = auto()
    OBJECT_HIGHLIGHTED = auto()
    ARM_MOVING_TO_OBJ = auto()
    PICKING = auto()
    LIFTING = auto()
    HEAD_TRACKING = auto()
    DROPPING = auto()
    RETURNING_HOME = auto()
    ENDING_SESSION = auto()
    ERROR = auto()


# ---------------------------------------------------------------------------
# Command — side-effect descriptor returned to the orchestrator
# ---------------------------------------------------------------------------

@dataclass
class Command:
    name: str
    data: Any = None


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class StateMachine:
    def __init__(self) -> None:
        self._state = State.LANDING
        self._highlighted_object: dict | None = None
        self._selected_object: dict | None = None
        self._head_center_since: float | None = None
        self._eeg_enabled: bool = False

    @property
    def state(self) -> State:
        return self._state

    @property
    def highlighted_object(self) -> dict | None:
        return self._highlighted_object

    @property
    def selected_object(self) -> dict | None:
        return self._selected_object

    @property
    def eeg_enabled(self) -> bool:
        return self._eeg_enabled

    def handle_event(self, event: Event) -> list[Command]:
        """Process an event and return commands for the orchestrator."""

        # End session from any state
        if event.type == REQUEST_END_SESSION:
            self._state = State.ENDING_SESSION
            return [Command("stop_all"), Command("move_home")]

        # Error handling from any state
        if event.type == ARM_ERROR:
            self._state = State.ERROR
            return [Command("show_error", event.data)]

        if event.type == HEARTBEAT_FAIL:
            self._state = State.ERROR
            return [Command("show_error", f"Device lost: {event.data}")]

        # State-specific transitions
        handler = getattr(self, f"_on_{self._state.name.lower()}", None)
        if handler:
            return handler(event)
        return []

    # -------------------------------------------------------------------
    # Per-state handlers
    # -------------------------------------------------------------------

    def _on_landing(self, event: Event) -> list[Command]:
        if event.type == REQUEST_START_SESSION:
            self._state = State.CONNECTING_XARM
            return [Command("connect_xarm")]
        return []

    def _on_connecting_xarm(self, event: Event) -> list[Command]:
        if event.type == XARM_CONNECTED:
            self._state = State.CONNECTING_MUSE
            return [Command("connect_muse")]
        if event.type == XARM_DISCONNECTED:
            self._state = State.LANDING
            return [Command("show_error", "xArm connection failed")]
        return []

    def _on_connecting_muse(self, event: Event) -> list[Command]:
        if event.type == MUSE_CONNECTED:
            self._state = State.HOMING
            return [Command("move_home"), Command("start_heartbeat")]
        if event.type == MUSE_DISCONNECTED:
            self._state = State.LANDING
            return [Command("show_error", "Muse connection failed"), Command("disconnect_xarm")]
        return []

    def _on_homing(self, event: Event) -> list[Command]:
        if event.type == HOME_REACHED:
            self._state = State.CALIBRATING_EYES
            return [Command("start_eye_calibration")]
        return []

    def _on_calibrating_eyes(self, event: Event) -> list[Command]:
        if event.type == EYE_CALIBRATION_DONE:
            self._state = State.CALIBRATING_EEG
            return [Command("start_eeg_calibration")]
        return []

    def _on_calibrating_eeg(self, event: Event) -> list[Command]:
        if event.type == EEG_WARMUP_DONE:
            self._state = State.SCANNING
            self._eeg_enabled = True
            return [Command("start_scanning")]
        return []

    def _on_scanning(self, event: Event) -> list[Command]:
        if event.type == JAW_CLENCH:
            # Clench at any gaze position — orchestrator computes 3D from depth
            self._highlighted_object = None
            self._state = State.ARM_MOVING_TO_OBJ
            return [Command("move_to_gaze_point")]

        if event.type == GAZE_UPDATE:
            # Gaze highlighting is handled by the orchestrator's vision loop.
            pass

        return []

    def _on_object_highlighted(self, event: Event) -> list[Command]:
        if event.type == JAW_CLENCH:
            # Clench at any gaze position — orchestrator computes 3D from depth
            self._highlighted_object = None
            self._state = State.ARM_MOVING_TO_OBJ
            return [Command("move_to_gaze_point")]

        if event.type == GAZE_UPDATE:
            # Orchestrator will call set_highlighted / clear_highlighted
            pass

        return []

    def _on_arm_moving_to_obj(self, event: Event) -> list[Command]:
        if event.type == ARM_MOVE_DONE:
            self._state = State.PICKING
            return [Command("start_pick_sequence")]
        return []

    def _on_picking(self, event: Event) -> list[Command]:
        if event.type == PICK_DONE:
            self._state = State.LIFTING
            return [Command("lift_object")]
        return []

    def _on_lifting(self, event: Event) -> list[Command]:
        if event.type == LIFT_DONE:
            self._state = State.HEAD_TRACKING
            self._eeg_enabled = False
            self._head_center_since = None
            return [Command("enable_head_tracking")]
        return []

    def _on_head_tracking(self, event: Event) -> list[Command]:
        commands: list[Command] = []

        if event.type == HEAD_UPDATE:
            direction = event.data.get("direction", "Center")
            yaw = event.data.get("yaw", 0.0)

            # Rotate arm base with head yaw
            if direction != "Center":
                commands.append(Command("rotate_base", yaw))
                self._head_center_since = None
                self._eeg_enabled = False
            else:
                # Track how long head has been centered
                if self._head_center_since is None:
                    self._head_center_since = time.monotonic()

                held_ms = (time.monotonic() - self._head_center_since) * 1000
                if held_ms >= CENTER_HOLD_MS and not self._eeg_enabled:
                    self._eeg_enabled = True
                    commands.append(Command("enable_eeg"))

        if event.type == JAW_CLENCH and self._eeg_enabled:
            self._state = State.DROPPING
            self._eeg_enabled = False
            return [Command("start_drop_sequence")]

        return commands

    def _on_dropping(self, event: Event) -> list[Command]:
        if event.type == DROP_DONE:
            self._state = State.RETURNING_HOME
            return [Command("move_home")]
        return []

    def _on_returning_home(self, event: Event) -> list[Command]:
        if event.type == HOME_REACHED:
            self._state = State.SCANNING
            self._selected_object = None
            self._eeg_enabled = True
            return [Command("start_scanning")]
        return []

    def _on_ending_session(self, event: Event) -> list[Command]:
        if event.type == HOME_REACHED:
            self._state = State.LANDING
            return [Command("disconnect_all"), Command("show_landing")]
        return []

    def _on_error(self, event: Event) -> list[Command]:
        if event.type == REQUEST_START_SESSION:
            self._state = State.CONNECTING_XARM
            return [Command("connect_xarm")]
        if event.type == REQUEST_END_SESSION:
            self._state = State.LANDING
            return [Command("disconnect_all"), Command("show_landing")]
        return []

    # -------------------------------------------------------------------
    # Helpers for orchestrator to update highlight state
    # -------------------------------------------------------------------

    def set_highlighted(self, obj: dict) -> None:
        self._highlighted_object = obj
        if self._state == State.SCANNING:
            self._state = State.OBJECT_HIGHLIGHTED

    def clear_highlighted(self) -> None:
        self._highlighted_object = None
        if self._state == State.OBJECT_HIGHLIGHTED:
            self._state = State.SCANNING
