"""EEG thread — LSL consumer + jaw clench detection.

Assumes the Muse BLE→LSL stream is already running (launched by
AppWindow either from the landing page button or during session start).
This thread resolves the LSL stream, consumes samples, and runs the
JawClenchDetector.
"""

from __future__ import annotations

import threading

from app.events import (
    EEG_WARMUP_DONE,
    JAW_CLENCH,
    MUSE_CONNECTED,
    MUSE_DISCONNECTED,
    Event,
    EventBus,
)


class EEGThread:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._running = False
        self._thread: threading.Thread | None = None
        self._detector = None  # JawClenchDetector
        self._calibration_requested = threading.Event()

    # ------------------------------------------------------------------
    # Public API (called from main thread)
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._calibration_requested.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="eeg",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._calibration_requested.set()  # unblock if waiting
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def start_calibration(self) -> None:
        """Signal the consumer to begin the warmup / detector."""
        self._calibration_requested.set()

    # ------------------------------------------------------------------
    # LSL consumer thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            from pylsl import StreamInlet, resolve_byprop

            from utility.jaw_clench_detection import (
                CH_TP9,
                CH_TP10,
                JawClenchDetector,
            )

            # Resolve the LSL EEG stream (the BLE→LSL bridge should already
            # be running — launched from the landing page or _cmd_connect_muse)
            print("[EEGThread] Resolving LSL EEG stream...")
            streams = resolve_byprop("type", "EEG", timeout=30.0)
            if not streams:
                print("[EEGThread] LSL stream not found (timeout)")
                self._bus.post(Event(MUSE_DISCONNECTED))
                return

            inlet = StreamInlet(streams[0], max_buflen=60)
            info = inlet.info()
            print(
                f"[EEGThread] LSL connected: {info.name()} "
                f"fs={info.nominal_srate():.0f} Hz"
            )
            self._bus.post(Event(MUSE_CONNECTED))

            # Drain samples until calibration is requested (keep buffer clear)
            while self._running and not self._calibration_requested.is_set():
                inlet.pull_sample(timeout=0.5)

            if not self._running:
                return

            # Start the detector — warmup begins now
            print("[EEGThread] Starting jaw clench detector (warmup)...")
            self._detector = JawClenchDetector(
                on_clench=self._on_clench,
                on_warmup_done=self._on_warmup_done,
                verbose=True,
            )

            while self._running:
                sample, _ts = inlet.pull_sample(timeout=1.0)
                if sample is not None:
                    self._detector.push_sample(sample[CH_TP9], sample[CH_TP10])

        except Exception as e:
            print(f"[EEGThread] error: {e}")
            self._bus.post(Event(MUSE_DISCONNECTED))

    # ------------------------------------------------------------------
    # Callbacks (called on consumer thread, post to bus)
    # ------------------------------------------------------------------

    def _on_clench(self, duration_ms: float) -> None:
        self._bus.post(Event(JAW_CLENCH, duration_ms))

    def _on_warmup_done(self) -> None:
        self._bus.post(Event(EEG_WARMUP_DONE))
