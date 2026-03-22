from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.filters import KalmanEMASmoother, make_kalman
from eyetrax.utils.screen import get_screen_size
import cv2
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

WEBCAM_INDEX = 1  # Built-in MacBook webcam

estimator = GazeEstimator()
run_9_point_calibration(estimator, camera_index=WEBCAM_INDEX)

estimator.save_model("gaze_model.pkl")

estimator = GazeEstimator()
estimator.load_model("gaze_model.pkl")

sw, sh = get_screen_size()

kalman = make_kalman()
smoother = KalmanEMASmoother(kalman, ema_alpha=0.5)
smoother.tune(estimator, camera_index=WEBCAM_INDEX)

cap = cv2.VideoCapture(1)  # Built-in MacBook webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    features, blink = estimator.extract_features(frame)

    if features is not None and not blink:
        x, y = estimator.predict(np.array([features]))[0]
        sx, sy = smoother.step(int(x), int(y))

        cx = max(0, min(sx, sw - 1))
        cy = max(0, min(sy, sh - 1))

        pyautogui.moveTo(cx, cy)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()