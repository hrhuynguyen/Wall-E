"""Vision wrappers — gaze tracking, object detection, head tracking.

Each wrapper runs heavy inference in a background thread and exposes
a simple get_*() method for the main thread to read the latest result.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import pyautogui

from app.config import WEBCAM_INDEX

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "gaze_model.pkl"
)
SMOOTHER_NOISE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "gaze_smoother_noise.npy"
)


# ---------------------------------------------------------------------------
# Eye calibration (blocking — must run on main thread for macOS OpenCV)
# ---------------------------------------------------------------------------

def run_eye_calibration(
    webcam_index: int = WEBCAM_INDEX,
    model_path: str = MODEL_PATH,
) -> bool:
    """Run 9-point calibration + smoother fine-tune.

    Opens fullscreen OpenCV windows (blocking).  Returns True on success.
    """
    from eyetrax import GazeEstimator, run_9_point_calibration
    from eyetrax.filters import KalmanEMASmoother, make_kalman

    # Phase 1: 9-point calibration
    estimator = GazeEstimator()
    run_9_point_calibration(estimator, camera_index=webcam_index)

    try:
        estimator.save_model(model_path)
        print(f"[EyeCal] Saved model: {model_path}")
    except Exception as e:
        print(f"[EyeCal] Calibration failed/cancelled: {e}")
        return False

    # Phase 2: smoother fine-tune (3-point)
    estimator = GazeEstimator()
    estimator.load_model(model_path)

    kalman = make_kalman()
    smoother = KalmanEMASmoother(kalman, ema_alpha=0.5)
    smoother.tune(estimator, camera_index=webcam_index)

    # Save tuned noise covariance for GazeTracker to load later
    noise_cov = np.array(kalman.measurementNoiseCov)
    np.save(SMOOTHER_NOISE_PATH, noise_cov)
    print(f"[EyeCal] Saved smoother noise: {SMOOTHER_NOISE_PATH}")

    return True


# ---------------------------------------------------------------------------
# Gaze Tracker (Step 8)
# ---------------------------------------------------------------------------

class GazeTracker:
    """Background thread that captures the webcam, runs eyetrax gaze
    estimation, and stores the latest smoothed screen coordinates."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        webcam_index: int = WEBCAM_INDEX,
    ) -> None:
        self._model_path = model_path
        self._webcam_index = webcam_index
        self._gaze: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="gaze-tracker",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._gaze = None

    def get_gaze(self) -> tuple[int, int] | None:
        """Return latest (screen_x, screen_y) or None."""
        with self._lock:
            return self._gaze

    def _run(self) -> None:
        try:
            from eyetrax import GazeEstimator
            from eyetrax.filters import KalmanEMASmoother, make_kalman
            from eyetrax.utils.screen import get_screen_size

            # Load pre-calibrated model
            estimator = GazeEstimator()
            estimator.load_model(self._model_path)
            print(f"[GazeTracker] Loaded model: {self._model_path}")

            sw, sh = get_screen_size()

            # Kalman + EMA smoother
            kalman = make_kalman()
            smoother = KalmanEMASmoother(kalman, ema_alpha=0.5)

            # Load tuned noise covariance if available
            if os.path.exists(SMOOTHER_NOISE_PATH):
                kalman.measurementNoiseCov = np.load(SMOOTHER_NOISE_PATH)
                print("[GazeTracker] Loaded tuned noise covariance")

            cap = cv2.VideoCapture(self._webcam_index)
            if not cap.isOpened():
                print(f"[GazeTracker] Failed to open webcam index {self._webcam_index}")
                self._running = False
                return

            print(f"[GazeTracker] Running on webcam {self._webcam_index}")

            try:
                while self._running:
                    ok, frame = cap.read()
                    if not ok:
                        continue

                    features, blink = estimator.extract_features(frame)
                    if features is not None and not blink:
                        x, y = estimator.predict(np.array([features]))[0]
                        sx, sy = smoother.step(int(x), int(y))
                        cx = max(0, min(int(sx), sw - 1))
                        cy = max(0, min(int(sy), sh - 1))
                        pyautogui.moveTo(cx, cy)
                        with self._lock:
                            self._gaze = (cx, cy)
                    # If blink or no features, keep last known gaze
            finally:
                cap.release()

        except Exception as e:
            print(f"[GazeTracker] Error: {e}")
            self._running = False


# ---------------------------------------------------------------------------
# Object Detector (Step 9) — YOLOv8 + Depth Pro
# ---------------------------------------------------------------------------

DEPTH_CKPT_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "utility", "checkpoints", "depth_pro.pt",
)
DETECTION_CONFIDENCE = 0.5


class ObjectDetector:
    """YOLOv8 + Depth Pro object detection with 3D coordinates.

    Reads frames from a CameraThread (doesn't own a camera).
    Runs YOLO in its own thread and Depth Pro in a sub-thread.
    """

    def __init__(self, camera_thread: object) -> None:
        self._camera = camera_thread
        self._detections: list[dict] = []
        self._depth_map: np.ndarray | None = None
        self._f_px: float | None = None
        self._lock = threading.Lock()
        self._running = False
        self._ready = False
        self._status = ""
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def ready(self) -> bool:
        return self._ready

    def get_status(self) -> str:
        return self._status

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="object-detector",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def get_detections(self) -> list[dict]:
        """Return latest detections.

        Each dict: label, confidence, bbox (x1,y1,x2,y2),
        pixel_center (u,v), xyz (depth_m, y_m, z_m) or None.
        """
        with self._lock:
            return list(self._detections)

    def get_depth(self) -> tuple[np.ndarray | None, float | None]:
        """Return (depth_map, f_px) at camera resolution, or (None, None)."""
        with self._lock:
            return self._depth_map, self._f_px

    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            import torch
            from ultralytics import YOLO
            import depth_pro
            from utility.get_coordinate import (
                DepthInferenceThread,
                get_object_depth,
                get_torch_device,
            )

            # --- Load Depth Pro ---
            if not os.path.exists(DEPTH_CKPT_PATH):
                self._status = "Depth checkpoint not found"
                print(f"[ObjectDetector] {self._status}: {DEPTH_CKPT_PATH}")
                self._running = False
                return

            device = get_torch_device()
            self._status = f"Loading Depth Pro ({device})..."
            print(f"[ObjectDetector] {self._status}")

            depth_cfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
            depth_cfg.checkpoint_uri = DEPTH_CKPT_PATH
            depth_model, transform = depth_pro.create_model_and_transforms(
                config=depth_cfg, device=device, precision=torch.half,
            )
            depth_model.eval()
            print("[ObjectDetector] Depth Pro loaded")

            # --- Load YOLO ---
            self._status = "Loading YOLOv8..."
            print(f"[ObjectDetector] {self._status}")
            yolo = YOLO("yolov8n.pt")
            print("[ObjectDetector] YOLOv8 nano loaded")

            # --- Background depth inference thread ---
            self._status = "Models ready — detecting..."
            self._ready = True
            print(f"[ObjectDetector] {self._status}")
            inferencer = DepthInferenceThread(depth_model, transform)
            depth_logged = False

            try:
                while self._running:
                    frame = self._camera.get_frame()
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    H, W = frame.shape[:2]

                    # Submit to depth (non-blocking)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inferencer.submit_frame(rgb)

                    # Run YOLO (fast)
                    results = yolo(frame, verbose=False)

                    # Latest depth result (may lag a frame)
                    depth_map, f_px, _, _ = inferencer.get_result()

                    # Check if depth thread is still alive
                    if not inferencer._thread.is_alive() and depth_map is None:
                        print("[ObjectDetector] Depth thread died — no depth data")
                        self._status = "Depth inference failed"
                        break

                    # Resize depth map to camera resolution if needed
                    f_px_scaled = f_px
                    if depth_map is not None and f_px is not None:
                        dh, dw = depth_map.shape
                        if (dh, dw) != (H, W):
                            if not depth_logged:
                                print(
                                    f"[ObjectDetector] Depth map {dw}x{dh} "
                                    f"→ resizing to {W}x{H}"
                                )
                            depth_map = cv2.resize(
                                depth_map, (W, H),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            f_px_scaled = f_px * W / dw
                        if not depth_logged:
                            print(
                                f"[ObjectDetector] First depth result: "
                                f"f_px={f_px:.1f} shape={depth_map.shape}"
                            )
                            depth_logged = True

                        # Store for get_depth()
                        with self._lock:
                            self._depth_map = depth_map.copy()
                            self._f_px = f_px_scaled

                    # Build detection list
                    dets: list[dict] = []
                    if len(results) > 0:
                        for box in results[0].boxes:
                            conf = float(box.conf[0])
                            if conf < DETECTION_CONFIDENCE:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cls_id = int(box.cls[0])
                            label = yolo.names[cls_id]
                            cu = int((x1 + x2) / 2)
                            cv_pt = int((y1 + y2) / 2)

                            xyz = None
                            if depth_map is not None and f_px_scaled is not None:
                                depth_val = get_object_depth(
                                    x1, y1, x2, y2, depth_map,
                                )
                                cx_img, cy_img = W / 2.0, H / 2.0
                                y_3d = (cu - cx_img) * depth_val / f_px_scaled   # left=neg, right=pos
                                z_3d = -((cv_pt - cy_img) * depth_val / f_px_scaled)
                                xyz = (depth_val, y_3d, z_3d)

                            dets.append({
                                "label": label,
                                "confidence": conf,
                                "bbox": (x1, y1, x2, y2),
                                "pixel_center": (cu, cv_pt),
                                "xyz": xyz,
                            })

                    with self._lock:
                        self._detections = dets

            finally:
                inferencer.stop()

        except Exception as e:
            print(f"[ObjectDetector] Error: {e}")
            self._running = False


# ---------------------------------------------------------------------------
# Head Tracker (Step 12) — MediaPipe face mesh → yaw + direction
# ---------------------------------------------------------------------------

class HeadTracker:
    """Background thread that captures the webcam, runs MediaPipe face mesh,
    and stores the latest smoothed yaw + direction label."""

    def __init__(self, webcam_index: int = WEBCAM_INDEX) -> None:
        self._webcam_index = webcam_index
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._yaw: float = 0.0
        self._direction: str = "Center"

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="head-tracker",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_state(self) -> tuple[float, str]:
        """Return (smoothed_yaw, direction)."""
        with self._lock:
            return self._yaw, self._direction

    def _run(self) -> None:
        try:
            import collections

            import mediapipe as mp
            from utility.head_tracker import (
                SMOOTH_WINDOW,
                direction_label,
                estimate_yaw,
            )

            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            cap = cv2.VideoCapture(self._webcam_index)
            if not cap.isOpened():
                print(f"[HeadTracker] Cannot open webcam {self._webcam_index}")
                self._running = False
                return

            print(f"[HeadTracker] Running on webcam {self._webcam_index}")
            yaw_buffer = collections.deque(maxlen=SMOOTH_WINDOW)
            current_dir = "Center"

            try:
                while self._running:
                    ok, frame = cap.read()
                    if not ok:
                        continue

                    frame = cv2.flip(frame, 1)  # mirror for natural interaction
                    h, w = frame.shape[:2]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        yaw = estimate_yaw(landmarks, w, h)
                        yaw_buffer.append(yaw)
                        smooth_yaw = float(np.mean(yaw_buffer))
                        current_dir = direction_label(smooth_yaw, current_dir)
                        with self._lock:
                            self._yaw = smooth_yaw
                            self._direction = current_dir
            finally:
                cap.release()
                face_mesh.close()

        except Exception as e:
            print(f"[HeadTracker] Error: {e}")
            self._running = False
