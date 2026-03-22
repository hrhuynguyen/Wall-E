"""Root Tk window — event pump, page switching, command dispatch."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from app.arm_worker import ArmWorker
from app.camera_thread import CameraThread
from app.config import (
    CAM_OFFSET_X_MM,
    CAM_OFFSET_Y_MM,
    CAM_Z_FLOOR_MM,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    GAZE_HIGHLIGHT_RADIUS_PX,
    HEAD_ROTATION_SPEED_DEG_S,
    LIFT_HEIGHT_MM,
    VISION_FPS,
)
from app.eeg_thread import EEGThread
from app.heartbeat_thread import HeartbeatThread
from app.events import (
    ARM_ERROR,
    EEG_WARMUP_DONE,
    EYE_CALIBRATION_DONE,
    GAZE_UPDATE,
    HEAD_UPDATE,
    JAW_CLENCH,
    MUSE_CONNECTED,
    MUSE_DISCONNECTED,
    REQUEST_END_SESSION,
    REQUEST_START_SESSION,
    XARM_CONNECTED,
    XARM_DISCONNECTED,
    Event,
    EventBus,
)
from app.vision import GazeTracker, HeadTracker, ObjectDetector, run_eye_calibration
from utility.head_tracker import MAX_YAW_INPUT
from app.state_machine import Command, State, StateMachine
from app.ui.landing_page import LandingPage
from app.ui.session_page import SessionPage

# How often (ms) the main thread drains the event bus
POLL_INTERVAL_MS = 16
# How often (ms) to refresh the camera feed on screen
FEED_INTERVAL_MS = 33  # ~30 fps display


class AppWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("WallE Assistive Arm")
        self.geometry("1560x1000")
        self.configure(bg="#1e1e2e")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Core
        self.bus = EventBus()
        self.sm = StateMachine()
        self._camera = CameraThread()
        self._arm_worker = ArmWorker(self.bus)
        self._arm_worker.start()
        self._eeg_thread = EEGThread(self.bus)
        self._heartbeat = HeartbeatThread(self.bus)
        self._heartbeat.set_arm_worker(self._arm_worker)
        self._gaze_tracker = GazeTracker()
        self._object_detector = ObjectDetector(self._camera)
        self._head_tracker = HeadTracker()
        self._gaze_polling = False
        self._head_polling = False
        self._last_head_time: float | None = None
        self._muse_stream_thread: threading.Thread | None = None
        self._muse_proc: subprocess.Popen | None = None
        self._muse_stream_ready = False
        self._photo_image: ImageTk.PhotoImage | None = None  # prevent GC

        # Container for page switching
        self._container = tk.Frame(self, bg="#1e1e2e")
        self._container.pack(fill="both", expand=True)
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)

        # Pages
        self._pages: dict[str, tk.Frame] = {}
        self._current_page: str | None = None
        self._init_pages()
        self._show_page("landing")

        # Start event pump
        self.after(POLL_INTERVAL_MS, self._poll_events)

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _init_pages(self) -> None:
        landing = LandingPage(
            self._container,
            on_start=self._request_start,
            on_connect_muse=self._connect_muse_landing,
        )
        landing.grid(row=0, column=0, sticky="nsew")
        self._pages["landing"] = landing

        self._session_page = SessionPage(
            self._container,
            on_end=self._request_end,
            on_connect_muse=self._connect_muse_landing,
        )
        self._session_page.grid(row=0, column=0, sticky="nsew")
        self._pages["session"] = self._session_page

    def _show_page(self, name: str) -> None:
        self._pages[name].tkraise()
        self._current_page = name

    # ------------------------------------------------------------------
    # Camera feed
    # ------------------------------------------------------------------

    def _start_camera(self) -> None:
        if not self._camera.running:
            self._camera.start()
            self._update_feed()

    def _stop_camera(self) -> None:
        self._camera.stop()

    def _update_feed(self) -> None:
        if not self._camera.running:
            return
        frame = self._camera.get_frame()
        if frame is not None:
            # Show EEG calibration instruction overlay
            if self.sm.state == State.CALIBRATING_EEG:
                H, W = frame.shape[:2]
                cv2.putText(
                    frame, "Sit still with jaw RELAXED", (20, H // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, "Calibrating EEG baseline... (6s)", (20, H // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                )

            if self._object_detector.running:
                if self._object_detector.ready:
                    highlighted = self.sm.highlighted_object
                    hl_bbox = highlighted["bbox"] if highlighted else None
                    self._draw_detections(
                        frame, self._object_detector.get_detections(), hl_bbox,
                    )
                else:
                    # Show loading status on camera feed
                    status = self._object_detector.get_status()
                    H, W = frame.shape[:2]
                    cv2.putText(
                        frame, status, (20, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                    )

            # Scale up to display size
            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # BGR → RGB for PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._photo_image = ImageTk.PhotoImage(image=img)
            canvas = self._session_page.camera_canvas
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=self._photo_image)
        self.after(FEED_INTERVAL_MS, self._update_feed)

    @staticmethod
    def _draw_detections(
        frame: "np.ndarray",
        detections: list[dict],
        highlighted_bbox: tuple | None = None,
    ) -> None:
        """Draw bounding boxes and labels on the camera frame (BGR, in-place)."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            xyz = det.get("xyz")

            is_highlighted = (highlighted_bbox == (x1, y1, x2, y2))
            color = (0, 255, 0) if is_highlighted else (0, 255, 255)  # green / yellow
            thickness = 3 if is_highlighted else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if xyz is not None:
                d, y_3d, z_3d = xyz
                text = f"{label} x={d*1000:.0f} y={y_3d*1000:.0f} z={z_3d*1000:.0f}mm"
            else:
                text = f"{label} {det['confidence']:.0%}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, text, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Muse BLE→LSL stream daemon (shared by landing button + session)
    # ------------------------------------------------------------------

    def _start_muse_stream(self) -> None:
        """Launch the Muse BLE→LSL bridge if not already running."""
        if self._muse_stream_thread is not None and self._muse_stream_thread.is_alive():
            return
        self._muse_stream_ready = False
        self._muse_stream_thread = threading.Thread(
            target=self._run_muse_stream, daemon=True, name="muse-stream",
        )
        self._muse_stream_thread.start()

    def _run_muse_stream(self) -> None:
        """Run muse_connection.py as a subprocess (macOS CoreBluetooth
        requires its own process event loop for BLE scanning)."""
        try:
            self._muse_proc = subprocess.Popen(
                [sys.executable, "-m", "utility.muse_connection"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            muse_name = "Muse"
            for line in self._muse_proc.stdout:
                line = line.rstrip()
                if line:
                    print(f"[MuseStream] {line}")
                if "Found Muse:" in line:
                    # Extract name: "Found Muse: Muse-36F0 at MAC ..."
                    parts = line.split("Found Muse:")
                    if len(parts) > 1:
                        muse_name = parts[1].split(" at ")[0].strip()
                    self.after(0, self._muse_stream_scanning_ok, muse_name)
                elif "Streaming EEG" in line or "Connected" in line:
                    self._muse_stream_ready = True
                    self.after(0, self._muse_stream_ok, muse_name)
                elif "No Muse devices found" in line:
                    self.after(0, self._muse_stream_fail)
                    return

            # Process ended unexpectedly
            if not self._muse_stream_ready:
                self.after(0, self._muse_stream_fail)
        except Exception as e:
            print(f"[MuseStream] Error: {e}")
            self.after(0, self._muse_stream_fail)

    def _set_muse_ui(self, text: str, color: str, btn_enabled: bool | None = None) -> None:
        """Push Muse status to both landing page and session sidebar."""
        landing = self._pages.get("landing")
        if isinstance(landing, LandingPage):
            landing.set_muse_status(text, color=color)
            if btn_enabled is not None:
                landing.set_muse_button_enabled(btn_enabled)
        self._session_page.set_muse_status(text, color=color)
        if btn_enabled is not None:
            self._session_page.set_muse_button_enabled(btn_enabled)

    def _muse_stream_scanning_ok(self, name: str = "") -> None:
        self._set_muse_ui(f"Found {name}, connecting...", "#f9e2af")

    def _muse_stream_ok(self, name: str = "") -> None:
        self._set_muse_ui(f"{name} streaming", "#a6e3a1", btn_enabled=False)

    def _muse_stream_fail(self) -> None:
        self._set_muse_ui("Not found — check Bluetooth", "#f38ba8", btn_enabled=True)

    def _connect_muse_landing(self) -> None:
        """'Connect to Muse 2' button handler (shared by landing + session)."""
        self._set_muse_ui("Scanning...", "#f9e2af", btn_enabled=False)
        self._start_muse_stream()

    # ------------------------------------------------------------------
    # User actions → events
    # ------------------------------------------------------------------

    def _request_start(self) -> None:
        self._start_camera()
        self._object_detector.start()  # Start loading Depth Pro + YOLO early
        self.bus.post(Event(REQUEST_START_SESSION))

    def _request_end(self) -> None:
        self.bus.post(Event(REQUEST_END_SESSION))

    # ------------------------------------------------------------------
    # Event pump (runs on main thread via after())
    # ------------------------------------------------------------------

    def _poll_events(self) -> None:
        for event in self.bus.drain():
            self._on_event_ui(event)
            commands = self.sm.handle_event(event)
            for cmd in commands:
                self._dispatch(cmd)

        # Update UI based on current state
        self._sync_ui()

        self.after(POLL_INTERVAL_MS, self._poll_events)

    def _on_event_ui(self, event: Event) -> None:
        """Update sidebar from events (before state machine processes them)."""
        if event.type == XARM_CONNECTED:
            self._session_page.update_xarm(True)
        elif event.type == XARM_DISCONNECTED:
            self._session_page.update_xarm(False)
        elif event.type == MUSE_CONNECTED:
            self._session_page.update_muse(True)
        elif event.type == MUSE_DISCONNECTED:
            self._session_page.update_muse(False)
        elif event.type == EEG_WARMUP_DONE:
            self._session_page.update_eeg("Active")
        elif event.type == JAW_CLENCH:
            self._session_page.flash_clench(event.data or 0)

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, cmd: Command) -> None:
        handler = getattr(self, f"_cmd_{cmd.name}", None)
        if handler:
            handler(cmd.data)
        else:
            print(f"[AppWindow] unhandled command: {cmd.name} {cmd.data}")

    # -- Implemented commands --

    def _cmd_show_landing(self, _data: object) -> None:
        self._stop_camera()
        self._show_page("landing")

    def _cmd_show_error(self, msg: object) -> None:
        landing = self._pages["landing"]
        if isinstance(landing, LandingPage):
            landing.set_status(str(msg))
        self._show_page("landing")

    # -- Arm commands (Step 5) --

    def _cmd_connect_xarm(self, _data: object) -> None:
        self._arm_worker.connect()

    def _cmd_move_home(self, _data: object) -> None:
        self._arm_worker.move_home()

    def _cmd_move_to_object(self, data: object) -> None:
        if isinstance(data, dict) and "xyz" in data:
            self._move_arm_to_vision_xyz(data["xyz"])
        else:
            print(f"[AppWindow] move_to_object: bad data {data}")

    def _cmd_move_to_gaze_point(self, _data: object) -> None:
        """Compute 3D coordinate at current gaze pixel from depth map."""
        gaze = self._gaze_tracker.get_gaze()
        if gaze is None:
            print("[AppWindow] no gaze data — cannot select point")
            self.bus.post(Event(ARM_ERROR, "No gaze data"))
            return

        depth_map, f_px = self._object_detector.get_depth()
        if depth_map is None or f_px is None:
            print("[AppWindow] no depth data — cannot select point")
            self.bus.post(Event(ARM_ERROR, "No depth data"))
            return

        # Map screen gaze → camera pixel coordinates
        sx, sy = gaze
        canvas = self._session_page.camera_canvas
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 0 or ch <= 0:
            self.bus.post(Event(ARM_ERROR, "Canvas not ready"))
            return

        H, W = depth_map.shape
        cam_u = int((sx - cx) * W / cw)
        cam_v = int((sy - cy) * H / ch)
        cam_u = max(0, min(W - 1, cam_u))
        cam_v = max(0, min(H - 1, cam_v))

        # Compute 3D from depth map (same convention as vision.py)
        d = float(depth_map[cam_v, cam_u])
        cx_img, cy_img = W / 2.0, H / 2.0
        y_3d = (cam_u - cx_img) * d / f_px       # left=neg, right=pos
        z_3d = -((cam_v - cy_img) * d / f_px)     # up=pos

        xyz = (d, y_3d, z_3d)
        print(f"[AppWindow] gaze point ({cam_u}, {cam_v}) → x={d:.3f}m y={y_3d:.3f}m z={z_3d:.3f}m")

        # Store as selected object for sidebar display
        self.sm._selected_object = {"label": f"Point ({cam_u},{cam_v})", "xyz": xyz}
        self._move_arm_to_vision_xyz(xyz)

    def _move_arm_to_vision_xyz(self, xyz: tuple) -> None:
        """Convert vision xyz (meters) to arm xyz (mm) with offsets, send to arm."""
        d_m, y_m, z_m = xyz
        y_mm = y_m * 1000 + CAM_OFFSET_Y_MM
        # Shrink y toward zero by 1mm (camera-to-arm y bias)
        if abs(y_mm) > 1.0:
            y_mm -= 1.0 if y_mm > 0 else -1.0
        else:
            y_mm = 0.0
        xyz_mm = [
            d_m * 1000 + CAM_OFFSET_X_MM,
            y_mm,
            max(z_m * 1000, CAM_Z_FLOOR_MM),
        ]
        print(f"[AppWindow] moving arm to {xyz_mm[0]:.0f}, {xyz_mm[1]:.0f}, {xyz_mm[2]:.0f} mm")
        self._arm_worker.move_to_xyz(xyz_mm)

    def _cmd_start_pick_sequence(self, _data: object) -> None:
        self._arm_worker.pick()

    def _cmd_lift_object(self, _data: object) -> None:
        self._arm_worker.lift(LIFT_HEIGHT_MM)

    def _cmd_start_drop_sequence(self, _data: object) -> None:
        self._stop_head_polling()
        self._head_tracker.stop()
        self._arm_worker.drop(LIFT_HEIGHT_MM)

    def _cmd_disconnect_xarm(self, _data: object) -> None:
        self._arm_worker.disconnect()

    def _cmd_stop_all(self, _data: object) -> None:
        self._stop_gaze_polling()
        self._stop_head_polling()
        self._gaze_tracker.stop()
        self._head_tracker.stop()
        self._object_detector.stop()
        self._heartbeat.stop()
        self._eeg_thread.stop()
        self._arm_worker.disconnect()

    def _cmd_disconnect_all(self, _data: object) -> None:
        self._stop_gaze_polling()
        self._stop_head_polling()
        self._gaze_tracker.stop()
        self._head_tracker.stop()
        self._object_detector.stop()
        self._heartbeat.stop()
        self._eeg_thread.stop()
        self._arm_worker.disconnect()

    # -- EEG commands (Step 6) --

    def _cmd_connect_muse(self, _data: object) -> None:
        self._start_muse_stream()  # no-op if already running from landing button
        self._eeg_thread.start()

    def _cmd_start_eeg_calibration(self, _data: object) -> None:
        self._session_page.update_eeg("Warmup")
        self._eeg_thread.start_calibration()

    # -- Stubs (will be implemented in later steps) --

    def _cmd_start_heartbeat(self, _data: object) -> None:
        self._heartbeat.set_check_muse(True)
        self._heartbeat.start()

    def _cmd_start_eye_calibration(self, _data: object) -> None:
        """Schedule blocking 9-point calibration + smoother fine-tune."""
        self.after(200, self._do_eye_calibration)

    def _do_eye_calibration(self) -> None:
        """Run eye calibration on main thread (fullscreen OpenCV windows)."""
        self.withdraw()  # hide Tk so calibration windows are visible
        try:
            success = run_eye_calibration()
        except Exception as e:
            print(f"[AppWindow] Eye calibration error: {e}")
            success = False
        finally:
            self.deiconify()

        if success:
            # Start the gaze tracker now that calibration is done
            self._gaze_tracker.start()
            self.bus.post(Event(EYE_CALIBRATION_DONE))
        else:
            self.bus.post(Event(ARM_ERROR, "Eye calibration failed or cancelled"))

    def _cmd_start_scanning(self, _data: object) -> None:
        self._show_page("session")
        # Stop head tracking if returning from HEAD_TRACKING state
        self._stop_head_polling()
        self._head_tracker.stop()
        # Start/resume gaze tracker + object detector
        self._gaze_tracker.start()
        self._object_detector.start()
        self._start_gaze_polling()

    def _cmd_enable_head_tracking(self, _data: object) -> None:
        # Stop gaze (frees webcam for head tracker)
        self._stop_gaze_polling()
        self._gaze_tracker.stop()
        # Start head tracker
        self._head_tracker.start()
        self._arm_worker.init_base_rotation()
        self._last_head_time = None
        self._start_head_polling()
        self._session_page.update_eeg("Paused")
        print("[AppWindow] head tracking enabled")

    def _cmd_rotate_base(self, yaw: object) -> None:
        """Velocity-based base rotation — mirrors BaseRotator.update() logic."""
        yaw_val = float(yaw)
        now = time.monotonic()
        dt = now - self._last_head_time if self._last_head_time else 0.033
        self._last_head_time = now

        fraction = min(abs(yaw_val) / MAX_YAW_INPUT, 1.0)
        speed = HEAD_ROTATION_SPEED_DEG_S * fraction
        sign = 1.0 if yaw_val > 0 else -1.0
        angle_delta = sign * speed * dt
        self._arm_worker.rotate_base_incremental(angle_delta)

    def _cmd_enable_eeg(self, _data: object) -> None:
        self._session_page.update_eeg("Active")

    # ------------------------------------------------------------------
    # Gaze polling (runs on main thread via after())
    # ------------------------------------------------------------------

    def _start_gaze_polling(self) -> None:
        self._gaze_polling = True
        interval = max(1, 1000 // VISION_FPS)
        self._process_gaze(interval)

    def _stop_gaze_polling(self) -> None:
        self._gaze_polling = False

    def _process_gaze(self, interval_ms: int) -> None:
        if not self._gaze_polling:
            return
        gaze = self._gaze_tracker.get_gaze()
        if gaze is not None:
            sx, sy = gaze
            self.bus.post(Event(GAZE_UPDATE, gaze))

            # Gaze-to-object matching during scanning
            if self._object_detector.ready and self.sm.state in (
                State.SCANNING, State.OBJECT_HIGHLIGHTED,
            ):
                obj = self._match_gaze_to_object(sx, sy)
                if obj is not None:
                    self.sm.set_highlighted(obj)
                    self._session_page.update_highlight(obj["label"])
                else:
                    self.sm.clear_highlighted()
                    self._session_page.update_highlight(f"({sx}, {sy})")
            else:
                self._session_page.update_highlight(f"({sx}, {sy})")

        self.after(interval_ms, self._process_gaze, interval_ms)

    # ------------------------------------------------------------------
    # Head tracking polling (runs on main thread via after())
    # ------------------------------------------------------------------

    def _start_head_polling(self) -> None:
        self._head_polling = True
        self._process_head()

    def _stop_head_polling(self) -> None:
        self._head_polling = False

    def _process_head(self) -> None:
        if not self._head_polling:
            return
        yaw, direction = self._head_tracker.get_state()
        self.bus.post(Event(HEAD_UPDATE, {"yaw": yaw, "direction": direction}))
        self._session_page.update_highlight(f"Head: {direction} ({yaw:+.0f}\u00b0)")
        self.after(FEED_INTERVAL_MS, self._process_head)

    def _match_gaze_to_object(self, sx: int, sy: int) -> dict | None:
        """Map gaze screen coordinates to camera pixels, find matching detection."""
        canvas = self._session_page.camera_canvas
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        if cw <= 0 or ch <= 0:
            return None

        # Check if gaze falls on the canvas
        if not (cx <= sx <= cx + cw and cy <= sy <= cy + ch):
            return None

        # Map screen → camera pixel coordinates
        cam_x = (sx - cx) * CAMERA_WIDTH / cw
        cam_y = (sy - cy) * CAMERA_HEIGHT / ch

        detections = self._object_detector.get_detections()
        best: dict | None = None
        best_dist = float("inf")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cu, cv_pt = det["pixel_center"]

            # Check if gaze is inside the bounding box
            if x1 <= cam_x <= x2 and y1 <= cam_y <= y2:
                dist = ((cam_x - cu) ** 2 + (cam_y - cv_pt) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = det

        # Fallback: check proximity to center within radius
        if best is None:
            for det in detections:
                cu, cv_pt = det["pixel_center"]
                dist = ((cam_x - cu) ** 2 + (cam_y - cv_pt) ** 2) ** 0.5
                if dist < GAZE_HIGHLIGHT_RADIUS_PX and dist < best_dist:
                    best_dist = dist
                    best = det

        return best

    # ------------------------------------------------------------------
    # UI sync
    # ------------------------------------------------------------------

    def _sync_ui(self) -> None:
        state = self.sm.state
        if state == State.LANDING:
            if self._current_page != "landing":
                self._show_page("landing")
        elif state in (State.ENDING_SESSION, State.ERROR):
            pass  # handled by commands
        else:
            if self._current_page != "session":
                self._show_page("session")

        # Keep sidebar in sync with state machine
        self._session_page.update_state(state.name.replace("_", " ").title())

        selected = self.sm.selected_object
        self._session_page.update_selected(selected["label"] if selected else None)

        # EEG status during head tracking
        if state == State.HEAD_TRACKING:
            if self.sm.eeg_enabled:
                self._session_page.update_eeg("Active — clench to drop")
            else:
                self._session_page.update_eeg("Paused (look center)")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self._stop_gaze_polling()
        self._stop_head_polling()
        self._gaze_tracker.stop()
        self._head_tracker.stop()
        self._object_detector.stop()
        self._stop_camera()
        self._heartbeat.stop()
        self._eeg_thread.stop()
        self._arm_worker.stop()
        if self._muse_proc is not None:
            self._muse_proc.terminate()
            self._muse_proc = None
        self.destroy()
