"""Background thread that owns a VideoCapture and serves the latest frame.

The main thread calls get_frame() to grab the most recent capture.
Only one CameraThread should own a given camera index at a time.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from app.config import CAMERA_HEIGHT, CAMERA_WIDTH, INNOMAKER_INDEX


class CameraThread:
    def __init__(
        self,
        index: int = INNOMAKER_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._frame = None

    def get_frame(self) -> np.ndarray | None:
        """Return a copy of the latest frame, or None if not available."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _run(self) -> None:
        cap = cv2.VideoCapture(self._index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        if not cap.isOpened():
            print(f"[CameraThread] failed to open camera index {self._index}")
            self._running = False
            cap.release()
            return

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    continue
                with self._lock:
                    self._frame = frame
        finally:
            cap.release()
