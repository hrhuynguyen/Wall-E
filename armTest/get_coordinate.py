#!/usr/bin/env python3
"""Live camera depth estimation + object detection with Depth Pro and YOLOv8.

Captures frames from the MacBook camera, detects objects with YOLOv8 (real-time),
runs Depth Pro depth inference in a background thread, and displays RGB + depth
side-by-side with auto-computed 3D coordinates for each detected object.

Requires: pip install opencv-python ultralytics
Controls: click = show 3D coord, 'c' = clear marker, 'q'/ESC = quit
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import depth_pro


def get_torch_device() -> torch.device:
    """Get the best available Torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DepthInferenceThread:
    """Runs Depth Pro inference in a background thread on the latest frame."""

    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

        self._lock = threading.Lock()
        self._pending_frame = None
        self._result_depth = None
        self._result_f_px = None
        self._result_depth_colored = None
        self._running = True
        self._infer_fps = 0.0

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit_frame(self, frame_rgb: np.ndarray):
        """Submit a new frame for inference (drops old pending frames)."""
        with self._lock:
            self._pending_frame = frame_rgb

    def get_result(self):
        """Get the latest inference result."""
        with self._lock:
            return (
                self._result_depth,
                self._result_f_px,
                self._result_depth_colored,
                self._infer_fps,
            )

    def stop(self):
        self._running = False
        self._thread.join(timeout=5)

    def _worker(self):
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.time()

            image_tensor = self.transform(frame)
            prediction = self.model.infer(image_tensor, f_px=None)

            depth = prediction["depth"].detach().cpu().numpy().squeeze()
            f_px = prediction["focallength_px"].detach().cpu().item()

            # Colorize depth (inverse depth + turbo)
            inverse_depth = 1.0 / np.clip(depth, 0.1, 250.0)
            inv_min, inv_max = inverse_depth.min(), inverse_depth.max()
            inv_normalized = (inverse_depth - inv_min) / (inv_max - inv_min + 1e-8)
            depth_colored = cv2.applyColorMap(
                (inv_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )

            infer_time = time.time() - t0

            with self._lock:
                self._result_depth = depth
                self._result_f_px = f_px
                self._result_depth_colored = depth_colored
                self._infer_fps = 1.0 / max(infer_time, 1e-6)


def compute_3d(u, v, depth_map, f_px):
    """Compute 3D coordinate (x, y, z) in meters for pixel (u, v).

    x = distance from camera (depth)
    y = horizontal: positive = left, negative = right
    z = vertical: positive = up (height)
    """
    H, W = depth_map.shape
    d = float(depth_map[v, u])
    cx, cy = W / 2.0, H / 2.0
    y_3d = -((u - cx) * d / f_px)   # left=positive, right=negative
    z_3d = -((v - cy) * d / f_px)   # up=positive (height)
    return d, y_3d, z_3d


def get_object_depth(x1, y1, x2, y2, depth_map):
    """Get robust depth for a bounding box using the median of the central region."""
    H, W = depth_map.shape
    # Use the central 50% of the bounding box to avoid edge noise
    cx1 = int(x1 + (x2 - x1) * 0.25)
    cy1 = int(y1 + (y2 - y1) * 0.25)
    cx2 = int(x1 + (x2 - x1) * 0.75)
    cy2 = int(y1 + (y2 - y1) * 0.75)
    # Clamp to image bounds
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(W, cx2), min(H, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        # Fallback to center pixel
        cu, cv = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cu, cv = min(max(cu, 0), W - 1), min(max(cv, 0), H - 1)
        return float(depth_map[cv, cu])
    region = depth_map[cy1:cy2, cx1:cx2]
    return float(np.median(region))


def on_mouse(event, x, y, flags, param):
    """Mouse callback: compute 3D coordinates on left click."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    state = param
    depth_map = state["depth_map"]
    f_px = state["f_px"]
    if depth_map is None or f_px is None:
        return

    H, W = depth_map.shape
    u = x - W if x >= W else x
    v = y

    if not (0 <= u < W and 0 <= v < H):
        return

    x_3d, y_3d, z_3d = compute_3d(u, v, depth_map, f_px)
    state["pixel"] = (u, v)
    state["xyz"] = (x_3d, y_3d, z_3d)


def draw_detection(display, x1, y1, x2, y2, label, xyz, W_frame):
    """Draw a detection box with 3D coordinate on both RGB and depth panels."""
    color = (0, 255, 255)  # yellow

    # Draw on RGB panel (left)
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    # Draw on depth panel (right, offset by W_frame)
    cv2.rectangle(display, (x1 + W_frame, y1), (x2 + W_frame, y2), color, 2)

    # Label with class name and depth
    if xyz is not None:
        x_3d, y_3d, z_3d = xyz
        text = f"{label} x={x_3d*1000:.0f} y={y_3d*1000:.0f} z={z_3d*1000:.0f}mm"
    else:
        text = label

    # Label above box on RGB panel
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        display, text, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )

    # Label above box on depth panel
    cv2.rectangle(
        display, (x1 + W_frame, y1 - th - 8), (x1 + W_frame + tw + 4, y1), color, -1
    )
    cv2.putText(
        display, text, (x1 + W_frame + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def main():
    # --- Check checkpoint ---
    ckpt = Path(__file__).parent / "checkpoints" / "depth_pro.pt"
    if not ckpt.exists():
        print("Checkpoint not found. Run: source get_pretrained_models.sh")
        return

    # --- Load models ---
    device = get_torch_device()
    print(f"Using device: {device}")

    depth_config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    depth_config.checkpoint_uri = str(ckpt)
    depth_model, transform = depth_pro.create_model_and_transforms(
        config=depth_config,
        device=device,
        precision=torch.half,
    )
    depth_model.eval()
    print("Depth Pro loaded.")

    yolo_model = YOLO("yolov8n.pt")
    print("YOLOv8 nano loaded.")

    # --- Open camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Check macOS Privacy > Camera permissions.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- Setup window and mouse callback ---
    click_state = {"pixel": None, "xyz": None, "depth_map": None, "f_px": None}
    window_name = "Depth Pro + YOLO - Live Camera"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse, click_state)

    # --- Start background depth inference ---
    inferencer = DepthInferenceThread(depth_model, transform)

    display_fps_start = time.time()
    display_fps_count = 0
    display_fps = 0.0
    last_depth_colored = None

    print("Running... Click for manual 3D coords. Press 'q'/ESC to quit, 'c' to clear.")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            H_frame, W_frame = frame_bgr.shape[:2]

            # Submit frame for depth inference (background, non-blocking)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inferencer.submit_frame(frame_rgb)

            # --- Run YOLO detection (fast, main thread) ---
            yolo_results = yolo_model(frame_bgr, verbose=False)
            detections = []
            if len(yolo_results) > 0:
                boxes = yolo_results[0].boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    detections.append((x1, y1, x2, y2, label, conf))

            # Get latest depth result (may be from a previous frame)
            depth_map, f_px, depth_colored, infer_fps = inferencer.get_result()

            if depth_colored is not None:
                last_depth_colored = depth_colored
                click_state["depth_map"] = depth_map
                click_state["f_px"] = f_px

            # --- Build side-by-side display ---
            if last_depth_colored is not None:
                dh, dw = last_depth_colored.shape[:2]
                if (dh, dw) != (H_frame, W_frame):
                    depth_display = cv2.resize(last_depth_colored, (W_frame, H_frame))
                else:
                    depth_display = last_depth_colored
                display = np.hstack([frame_bgr, depth_display])
            else:
                placeholder = np.zeros_like(frame_bgr)
                cv2.putText(
                    placeholder, "Warming up...", (50, H_frame // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
                )
                display = np.hstack([frame_bgr, placeholder])

            # --- Draw detections with 3D coordinates ---
            for x1, y1, x2, y2, label, conf in detections:
                xyz = None
                if depth_map is not None and f_px is not None:
                    # Use median depth of the central bbox region for robustness
                    depth = get_object_depth(x1, y1, x2, y2, depth_map)
                    # Compute 3D position at bbox center
                    cu = int((x1 + x2) / 2)
                    cv_pt = int((y1 + y2) / 2)
                    cx, cy = W_frame / 2.0, H_frame / 2.0
                    y_3d = -((cu - cx) * depth / f_px)   # left=positive, right=negative
                    z_3d = -((cv_pt - cy) * depth / f_px)  # up=positive (height)
                    xyz = (depth, y_3d, z_3d)
                draw_detection(display, x1, y1, x2, y2, label, xyz, W_frame)

            # --- FPS counters ---
            display_fps_count += 1
            elapsed = time.time() - display_fps_start
            if elapsed >= 1.0:
                display_fps = display_fps_count / elapsed
                display_fps_count = 0
                display_fps_start = time.time()

            cv2.putText(
                display, f"Display: {display_fps:.0f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                display, f"Depth: {infer_fps:.1f} FPS" if infer_fps else "Depth: ...",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                display, f"Objects: {len(detections)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )

            # --- Overlay manual click marker ---
            if click_state["pixel"] is not None and click_state["xyz"] is not None:
                u, v = click_state["pixel"]
                x_3d, y_3d, z_3d = click_state["xyz"]

                cv2.drawMarker(display, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.drawMarker(
                    display, (u + W_frame, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2
                )
                cv2.putText(
                    display, f"x={x_3d:.2f}m", (u + 15, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )

                coord_text = f"Click: x={x_3d:.2f}m (dist)  y={y_3d:.2f}m (L+/R-)  z={z_3d:.2f}m (height)"
                text_y = display.shape[0] - 20
                cv2.putText(
                    display, coord_text, (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    display, coord_text, (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
                )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                click_state["pixel"] = None
                click_state["xyz"] = None

    finally:
        inferencer.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
