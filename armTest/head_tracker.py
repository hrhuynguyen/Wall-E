import collections

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


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    yaw_buffer = collections.deque(maxlen=SMOOTH_WINDOW)
    current_dir = "Center"

    print("Head Tracker running – press 'q' to quit")

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
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Head Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
