"""
Adaptive Baseline Jaw Clench Detector — Muse 2
================================================
Streams raw EEG from muselsl (LSL), filters TP9 + TP10
to isolate jaw EMG, and fires a callback whenever a deliberate
clench is detected.

Pipeline
--------
Raw EEG (256 Hz)
  → 60 Hz notch filter              (removes power line interference)
  → Butterworth bandpass 20–100 Hz   (isolates EMG, removes EEG + drift)
  → Rectification + envelope smooth  (stable amplitude from oscillating EMG)
  → Welford adaptive baseline        (idle-only updates, never learns clenches)
  → Dual-channel amplitude gate      (TP9 AND TP10 must exceed threshold)
  → Zero-crossing rate check         (confirms EMG texture at onset only)
  → State machine with debounce      (onset / hold / release / refractory)
  → on_clench() callback
"""

from __future__ import annotations

import time
import math
from collections import deque
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch
from pylsl import StreamInlet, resolve_byprop


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS: int = 256

# Muse 2 LSL channel order: TP9=0, AF7=1, AF8=2, TP10=3
CH_TP9: int = 0
CH_TP10: int = 3

# 60 Hz notch filter for power line noise
NOTCH_FREQ: float = 60.0
NOTCH_Q: float = 30.0

# Bandpass for jaw EMG (above EEG drift, below Nyquist)
BP_LO: float = 15.0
BP_HI: float = 120.0
BP_ORDER: int = 4

# Envelope smoothing: moving average after rectification
# 75 ms ≈ 19 samples — smooths rapid EMG oscillation into stable envelope
ENVELOPE_WINDOW: int = int(0.075 * FS)

# Adaptive threshold: mean + K * std
THRESHOLD_K: float = 3.0

# Seconds of quiet data before threshold is valid
WARMUP_SEC: float = 6.0

# Zero-crossing rate: min crossings per 15-sample block
ZCR_BLOCK: int = 15
ZCR_MIN: int = 6

# Minimum threshold floor (µV)
MIN_THRESHOLD: float = 20.0

# State machine timing (ms)
MIN_HOLD_MS: float = 30.0
REFRACTORY_MS: float = 500.0
MAX_CLENCH_MS: float = 2000.0

# Rolling baseline window (10s at 256 Hz = 2560 samples)
BASELINE_WINDOW: int = int(10.0 * FS)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class OnlineFilter:
    """Sample-by-sample IIR filter using second-order sections."""

    def __init__(self, sos: np.ndarray) -> None:
        self._sos = sos
        self._zi = np.zeros((sos.shape[0], 2))

    def push(self, x: float) -> float:
        out, self._zi = sosfilt(self._sos, [x], zi=self._zi)
        return float(out[0])


def make_bandpass(lo: float = BP_LO, hi: float = BP_HI,
                  fs: int = FS, order: int = BP_ORDER) -> OnlineFilter:
    nyq = fs / 2.0
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return OnlineFilter(sos)


def make_notch(freq: float = NOTCH_FREQ, q: float = NOTCH_Q,
               fs: int = FS) -> OnlineFilter:
    b, a = iirnotch(freq, q, fs)
    sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])
    return OnlineFilter(sos)


class EnvelopeSmoother:
    """Moving average over rectified signal to extract a stable EMG envelope."""

    def __init__(self, window: int = ENVELOPE_WINDOW) -> None:
        self._window = max(window, 1)
        self._buf = np.zeros(self._window)
        self._idx = 0
        self._sum = 0.0

    def push(self, rectified: float) -> float:
        self._sum -= self._buf[self._idx]
        self._buf[self._idx] = rectified
        self._sum += rectified
        self._idx = (self._idx + 1) % self._window
        return self._sum / self._window


class WelfordBaseline:
    """Online mean + std over a sliding window using Welford's algorithm.

    Only updated during idle periods so clenches never corrupt the baseline.
    """

    def __init__(self, window: int = BASELINE_WINDOW) -> None:
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)
        self._mean = 0.0
        self._m2 = 0.0
        self._count = 0

    def _add(self, x: float) -> None:
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (x - self._mean)

    def _remove(self, x: float) -> None:
        if self._count <= 1:
            self._mean = 0.0
            self._m2 = 0.0
            self._count = 0
            return
        self._count -= 1
        delta = x - self._mean
        self._mean -= delta / self._count
        self._m2 -= delta * (x - self._mean)
        self._m2 = max(self._m2, 0.0)

    def update(self, x: float) -> None:
        if len(self._buf) == self._window:
            self._remove(self._buf[0])
        self._buf.append(x)
        self._add(x)

    @property
    def ready(self) -> bool:
        return self._count >= int(WARMUP_SEC * FS)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._count < 2:
            return 0.0
        return math.sqrt(self._m2 / (self._count - 1))

    def threshold(self, k: float = THRESHOLD_K) -> float:
        if not self.ready:
            return float("inf")
        return max(self.mean + k * self.std, MIN_THRESHOLD)


class ZeroCrossingCheck:
    """Confirm EMG texture by counting zero crossings in a short block.

    Real jaw EMG oscillates rapidly (high ZCR). Noise spikes and
    motion artifacts are slow deflections (low ZCR).
    """

    def __init__(self, block: int = ZCR_BLOCK, minimum: int = ZCR_MIN) -> None:
        self._min = minimum
        self._buf: deque[float] = deque(maxlen=block)

    def push(self, x: float) -> bool:
        self._buf.append(x)
        if len(self._buf) < self._buf.maxlen:
            return False
        arr = np.array(self._buf)
        signs = np.sign(arr)
        zcr = int(np.sum(np.abs(np.diff(signs)) > 0))
        return zcr >= self._min


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class _State:
    WARMUP = "WARMUP"
    IDLE = "IDLE"
    ONSET = "ONSET"
    ACTIVE = "ACTIVE"
    REFRACTORY = "REFRACTORY"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class JawClenchDetector:
    """Adaptive-baseline jaw clench detector for Muse 2.

    Parameters
    ----------
    on_clench : callable, optional
        Called with (duration_ms: float) on each confirmed clench.
    on_warmup_done : callable, optional
        Called once when baseline calibration finishes.
    threshold_k : float
        Sensitivity multiplier (higher = fewer false positives).
    min_hold_ms : float
        Minimum ms above threshold to confirm a clench.
    refractory_ms : float
        Lockout ms after each confirmed clench.
    verbose : bool
        Print state transitions to stdout.
    """

    def __init__(
        self,
        on_clench: Optional[Callable[[float], None]] = None,
        on_warmup_done: Optional[Callable[[], None]] = None,
        threshold_k: float = THRESHOLD_K,
        min_hold_ms: float = MIN_HOLD_MS,
        refractory_ms: float = REFRACTORY_MS,
        verbose: bool = True,
    ) -> None:
        self._on_clench = on_clench or (lambda ms: None)
        self._on_warmup_done = on_warmup_done or (lambda: None)
        self._k = threshold_k
        self._min_hold_ms = min_hold_ms
        self._refractory_ms = refractory_ms
        self._verbose = verbose

        # Per-channel: notch → bandpass → envelope → ZCR
        self._notch_tp9 = make_notch()
        self._notch_tp10 = make_notch()
        self._bp_tp9 = make_bandpass()
        self._bp_tp10 = make_bandpass()
        self._env_tp9 = EnvelopeSmoother()
        self._env_tp10 = EnvelopeSmoother()
        self._zcr_tp9 = ZeroCrossingCheck()
        self._zcr_tp10 = ZeroCrossingCheck()

        self._baseline = WelfordBaseline()

        # State machine
        self._state = _State.WARMUP
        self._onset_ts = 0.0
        self._state_ts = 0.0
        self._clench_count = 0

    # -------------------------------------------------------------------
    # Core per-sample logic
    # -------------------------------------------------------------------

    def push_sample(self, raw_tp9: float, raw_tp10: float) -> bool:
        """Feed one raw EEG sample pair from TP9 and TP10.

        Returns True on the sample that fires a confirmed clench.
        """
        now_ms = time.monotonic() * 1000.0

        # 1. Notch filter (remove 60 Hz power line)
        n9 = self._notch_tp9.push(raw_tp9)
        n10 = self._notch_tp10.push(raw_tp10)

        # 2. Bandpass filter (isolate EMG)
        f9 = self._bp_tp9.push(n9)
        f10 = self._bp_tp10.push(n10)

        # 3. Rectify + envelope smooth
        amp9 = self._env_tp9.push(abs(f9))
        amp10 = self._env_tp10.push(abs(f10))
        amp = max(amp9, amp10)

        # 4. ZCR texture check (on raw filtered signal, not envelope)
        zcr_ok = self._zcr_tp9.push(f9) and self._zcr_tp10.push(f10)

        # 5. Adaptive threshold + dual-channel gate
        thr = self._baseline.threshold(self._k)
        dual_gate = (amp9 > thr * 0.70) and (amp10 > thr * 0.70)
        amp_above = (amp > thr) and dual_gate

        # 6. State machine (ZCR only required at ONSET entry)
        fired = self._transition(amp_above, zcr_ok, now_ms)

        # 7. Update baseline only when idle
        if self._state in (_State.WARMUP, _State.IDLE):
            self._baseline.update(amp)

        # 8. Warmup → idle
        if self._state == _State.WARMUP and self._baseline.ready:
            self._state = _State.IDLE
            if self._verbose:
                print(
                    f"[WARMUP DONE] baseline mean={self._baseline.mean:.1f} µV  "
                    f"std={self._baseline.std:.1f} µV  "
                    f"threshold={self._baseline.threshold(self._k):.1f} µV"
                )
            self._on_warmup_done()

        return fired

    def _transition(self, amp_above: bool, zcr_ok: bool, now_ms: float) -> bool:
        if self._state == _State.WARMUP:
            return False

        if self._state == _State.IDLE:
            if amp_above and zcr_ok:
                self._state = _State.ONSET
                self._onset_ts = now_ms
                self._state_ts = now_ms
            return False

        if self._state == _State.ONSET:
            if not amp_above:
                self._state = _State.IDLE
                return False
            if now_ms - self._state_ts >= self._min_hold_ms:
                self._state = _State.ACTIVE
                self._state_ts = now_ms
            return False

        if self._state == _State.ACTIVE:
            elapsed = now_ms - self._state_ts
            if not amp_above or elapsed > MAX_CLENCH_MS:
                duration = now_ms - self._onset_ts
                self._state = _State.REFRACTORY
                self._state_ts = now_ms
                self._clench_count += 1
                if self._verbose:
                    print(f"  [CLENCH #{self._clench_count}] duration ≈ {duration:.0f} ms")
                self._on_clench(duration)
                return True
            return False

        if self._state == _State.REFRACTORY:
            if now_ms - self._state_ts >= self._refractory_ms:
                self._state = _State.IDLE
            return False

        return False

    # -------------------------------------------------------------------
    # Blocking LSL loop
    # -------------------------------------------------------------------

    def run(self, timeout: float = 10.0) -> None:
        """Resolve the muselsl EEG stream and run detection forever."""
        print("Resolving LSL EEG stream...")
        streams = resolve_byprop("type", "EEG", timeout=timeout)
        if not streams:
            raise RuntimeError(
                "No EEG LSL stream found.\n"
                "  1. Pair your Muse 2 via Bluetooth\n"
                "  2. Run: python muse_connection.py\n"
                "  3. Re-run this script"
            )

        inlet = StreamInlet(streams[0], max_buflen=60)
        info = inlet.info()
        print(f"Connected: {info.name()}  fs={info.nominal_srate():.0f} Hz  ch={info.channel_count()}")
        print(f"\nSit still with jaw RELAXED for {WARMUP_SEC:.0f}s while baseline calibrates...\n")

        try:
            while True:
                sample, _ts = inlet.pull_sample()
                self.push_sample(sample[CH_TP9], sample[CH_TP10])
        except KeyboardInterrupt:
            print(f"\nStopped. Total clenches detected: {self._clench_count}")

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._baseline.threshold(self._k)

    @property
    def baseline_mean(self) -> float:
        return self._baseline.mean

    @property
    def baseline_std(self) -> float:
        return self._baseline.std

    @property
    def clench_count(self) -> int:
        return self._clench_count

    @property
    def state(self) -> str:
        return self._state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _default_callback(duration_ms: float) -> None:
    print(f"\n*** JAW CLENCH ({duration_ms:.0f} ms) ***\n")


if __name__ == "__main__":
    detector = JawClenchDetector(
        on_clench=_default_callback,
        threshold_k=THRESHOLD_K,
        verbose=True,
    )
    detector.run()
