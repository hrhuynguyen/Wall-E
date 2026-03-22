"""Standalone test for Step 9 — ObjectDetector + CameraThread.

Run: python3 test_step9.py

Shows the camera feed with YOLO detections + Depth Pro coordinates.
Click = show 3D coordinate at that pixel.  'c' = clear marker.
Press 'q' or ESC to quit.
"""

import cv2
import time
from app.camera_thread import CameraThread
from app.vision import ObjectDetector
from utility.get_coordinate import compute_3d

# --- Mouse click state (same pattern as get_coordinate.py) ---
click_state = {"pixel": None, "xyz": None}


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    depth_map, f_px = det.get_depth()
    if depth_map is None or f_px is None:
        return
    H, W = depth_map.shape
    u, v = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
    x_3d, y_3d, z_3d = compute_3d(u, v, depth_map, f_px)
    y_3d = -y_3d  # invert: left=negative, right=positive
    click_state["pixel"] = (u, v)
    click_state["xyz"] = (x_3d, y_3d, z_3d)
    print(f"Click ({u}, {v}): x={x_3d:.3f}m  y={y_3d:.3f}m  z={z_3d:.3f}m")


# --- Start camera + detector ---
cam = CameraThread()
cam.start()

det = ObjectDetector(cam)
det.start()

print("Waiting for models to load...")
while not det.ready and det.running:
    status = det.get_status()
    print(f"  {status}")
    time.sleep(2)

if not det.running:
    print("ObjectDetector failed to start. Check console errors above.")
    cam.stop()
    exit(1)

print("Detecting — click for 3D coords, 'c' to clear, 'q'/ESC to quit.\n")

window_name = "Step 9 Test — ObjectDetector"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, on_mouse)

try:
    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        detections = det.get_detections()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = d["label"]
            xyz = d.get("xyz")

            color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if xyz is not None:
                x_val, y_val, z_val = xyz
                # Negate y: left=negative, right=positive
                text = f"{label} x={x_val*1000:.0f} y={-y_val*1000:.0f} z={z_val*1000:.0f}mm"
            else:
                text = f"{label} {d['confidence']:.0%} (no depth yet)"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, text, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

        cv2.putText(
            frame, f"Objects: {len(detections)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )

        # --- Draw click marker (same as get_coordinate.py) ---
        if click_state["pixel"] is not None and click_state["xyz"] is not None:
            u, v = click_state["pixel"]
            x_3d, y_3d, z_3d = click_state["xyz"]

            cv2.drawMarker(frame, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            coord_text = f"x={x_3d:.2f}m  y={y_3d:.2f}m  z={z_3d:.2f}m"
            cv2.putText(
                frame, coord_text, (u + 15, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

            # Bottom bar
            bar_text = f"Click: x={x_3d:.3f}m  y={y_3d:.3f}m  z={z_3d:.3f}m"
            text_y = frame.shape[0] - 20
            cv2.putText(
                frame, bar_text, (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                frame, bar_text, (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            click_state["pixel"] = None
            click_state["xyz"] = None

finally:
    det.stop()
    cam.stop()
    cv2.destroyAllWindows()
    print("Done.")
