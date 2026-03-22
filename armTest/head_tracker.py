import argparse
import collections
import json
import os
import time

import cv2
import numpy as np
import mediapipe as mp


# Landmark indices
NOSE = 1
LEFT_EYE = 33
RIGHT_EYE = 263
LEFT_EDGE = 234    # left face silhouette
RIGHT_EDGE = 454   # right face silhouette

# Hysteresis thresholds (in approximate degrees)
YAW_ENTER = 20     # must exceed to trigger Left/Right
YAW_EXIT = 8       # must drop below to return to Center

# Moving-average window size (frames)
SMOOTH_WINDOW = 7

# Arm rotation control
ROTATION_SPEED_DEG_S = 30.0   # servo degrees per second at max yaw
MAX_YAW_INPUT = 35.0           # head yaw that maps to max rotation speed
SERVO_MOVE_DURATION_MS = 100   # short duration for smooth incremental moves


def estimate_yaw(landmarks, w, h):
    """Estimate head yaw from landmark geometry — no solvePnP needed.

    Compares nose-to-left-edge vs nose-to-right-edge distances.
    When the head turns right, the nose gets closer to the right edge
    and farther from the left edge (and vice versa).

    Returns approximate yaw in degrees (positive = right, negative = left).
    """
    nose_x = landmarks[NOSE].x * w
    left_x = landmarks[LEFT_EDGE].x * w
    right_x = landmarks[RIGHT_EDGE].x * w

    face_width = right_x - left_x
    if abs(face_width) < 1:
        return 0.0

    dist_left = nose_x - left_x
    dist_right = right_x - nose_x

    # ratio: 0 when centred, positive when turning right, negative when left
    ratio = (dist_left - dist_right) / (dist_left + dist_right)

    # Scale to approximate degrees (~70 maps the usable ratio range to ±35°)
    return ratio * 70.0


def direction_label(yaw, prev):
    """Hysteresis to prevent flickering at the threshold boundary."""
    if prev == "Left":
        return "Left" if yaw < -YAW_EXIT else "Center"
    elif prev == "Right":
        return "Right" if yaw > YAW_EXIT else "Center"
    else:
        if yaw < -YAW_ENTER:
            return "Left"
        elif yaw > YAW_ENTER:
            return "Right"
        return "Center"


def draw_hud(frame, yaw, direction):
    """Draw yaw angle and direction on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Yaw: {yaw:+.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Direction label – large, centred at the bottom
    color = (0, 255, 255) if direction != "Center" else (200, 200, 200)
    text_size = cv2.getTextSize(direction, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    tx = (w - text_size[0]) // 2
    ty = h - 30
    cv2.putText(frame, direction, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)


class BaseRotator:
    """Incremental velocity control for servo 6 (base) via head yaw."""

    def __init__(self):
        from utility.xarm_hid import XArmHID

        config_path = os.path.join(os.path.dirname(__file__), "xarm_config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        servo_cfg = cfg["servos"]["6"]
        self._home = servo_cfg["home_position"]
        self._pos_min = servo_cfg["position_min"]
        self._pos_max = servo_cfg["position_max"]
        self._direction = servo_cfg["direction"]
        self._angle_per_unit = servo_cfg["angle_per_unit"]

        self._arm = XArmHID()
        self._arm.open()
        self._current_pos = float(self._home)
        self._last_time = time.monotonic()

        # Move to home first
        self._arm.move_servo(6, self._home, duration_ms=1000)
        time.sleep(1.0)
        print(f"[BaseRotator] Connected. Servo 6 homed to {self._home}")

    def angle_to_pos(self, angle_deg: float) -> int:
        raw = self._home + angle_deg / (self._direction * self._angle_per_unit)
        return int(round(max(self._pos_min, min(self._pos_max, raw))))

    def update(self, yaw: float, direction: str) -> None:
        """Call each frame. Rotates base when head is Left/Right."""
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now

        if direction == "Center":
            return

        # Proportional speed: larger head turn → faster rotation
        fraction = min(abs(yaw) / MAX_YAW_INPUT, 1.0)
        speed = ROTATION_SPEED_DEG_S * fraction  # degrees per second

        # Convert to position delta
        # Positive yaw (head right) → we want base to rotate right
        sign = 1.0 if yaw > 0 else -1.0
        angle_delta = sign * speed * dt
        pos_delta = angle_delta / (self._direction * self._angle_per_unit)

        self._current_pos += pos_delta
        self._current_pos = max(self._pos_min, min(self._pos_max, self._current_pos))

        target = int(round(self._current_pos))
        self._arm.move_servo(6, target, duration_ms=SERVO_MOVE_DURATION_MS)

    def close(self) -> None:
        # Return to home before closing
        self._arm.move_servo(6, self._home, duration_ms=1000)
        time.sleep(1.0)
        self._arm.close()
        print("[BaseRotator] Closed, servo 6 returned to home")


def list_cameras(max_index: int = 5) -> None:
    """Probe camera indices and print name/resolution for each."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            backend = cap.getBackendName()
            print(f"  Index {i}: {w}x{h}  backend={backend}")
            cap.release()
        else:
            break


def main():
    parser = argparse.ArgumentParser(description="Head yaw tracker")
    parser.add_argument("--arm", action="store_true",
                        help="Enable xArm servo 6 rotation control")
    parser.add_argument("--camera", type=int, default=1,
                        help="Camera index (default 1 = built-in webcam)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List available cameras and exit")
    args = parser.parse_args()

    if args.list_cameras:
        print("Available cameras:")
        list_cameras()
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rotator = None
    if args.arm:
        try:
            rotator = BaseRotator()
        except Exception as e:
            print(f"[WARNING] Could not connect to xArm: {e}")
            print("Running without arm control.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {args.camera}")
        print("Run with --list-cameras to see available cameras.")
        return
    print(f"Using camera index {args.camera}")

    yaw_buffer = collections.deque(maxlen=SMOOTH_WINDOW)
    current_dir = "Center"

    mode = "Head Tracker + Arm" if rotator else "Head Tracker"
    print(f"{mode} running – press 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                yaw = estimate_yaw(landmarks, w, h)

                yaw_buffer.append(yaw)
                smooth_yaw = np.mean(yaw_buffer)
                current_dir = direction_label(smooth_yaw, current_dir)
                draw_hud(frame, smooth_yaw, current_dir)

                if rotator:
                    rotator.update(smooth_yaw, current_dir)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(mode, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        if rotator:
            rotator.close()


if __name__ == "__main__":
    main()
