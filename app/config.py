"""Tunable constants for the WallE application."""

# Arm motion
LIFT_HEIGHT_MM: float = 20.0
ARM_MOVE_DURATION_MS: int = 3000
GRIPPER_DURATION_MS: int = 1000

# Head tracking
HEAD_ROTATION_SPEED_DEG_S: float = 15.0
CENTER_HOLD_MS: float = 500.0

# Gaze
GAZE_HIGHLIGHT_RADIUS_PX: int = 80

# Heartbeat
HEARTBEAT_INTERVAL_S: float = 5.0

# Vision
VISION_FPS: int = 15

# Camera indices
WEBCAM_INDEX: int = 1         # Built-in MacBook webcam (head tracking, eye gaze)
INNOMAKER_INDEX: int = 0      # Innomaker USB camera (object detection, depth)
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
DISPLAY_WIDTH: int = 1280     # 2x capture = 4x area
DISPLAY_HEIGHT: int = 960
