"""Standalone test for Step 10 — Jaw clench detection via Muse 2.

Prerequisites:
  Terminal 1: python -m utility.muse_connection   (BLE → LSL stream)
  Terminal 2: python test_step10.py

Shows real-time detector state, baseline stats, and clench events.
Press Ctrl+C to quit.
"""

import time
from utility.jaw_clench_detection import (
    JawClenchDetector,
    THRESHOLD_K,
    WARMUP_SEC,
)

clench_log: list[tuple[float, float]] = []


def on_clench(duration_ms: float) -> None:
    clench_log.append((time.monotonic(), duration_ms))
    n = len(clench_log)
    print(f"\n*** JAW CLENCH #{n}  ({duration_ms:.0f} ms) ***\n")


def on_warmup_done() -> None:
    print("\n=== WARMUP COMPLETE — Clench detection ACTIVE ===")
    print(f"    Baseline mean: {detector.baseline_mean:.1f} µV")
    print(f"    Baseline std:  {detector.baseline_std:.1f} µV")
    print(f"    Threshold:     {detector.baseline_mean + THRESHOLD_K * detector.baseline_std:.1f} µV")
    print("    Clench your jaw to test.\n")


print("=" * 50)
print("  Step 10 Test — Jaw Clench Detection")
print("=" * 50)
print(f"\nSit still with jaw RELAXED for {WARMUP_SEC:.0f}s while baseline calibrates...")
print(f"Threshold K = {THRESHOLD_K}\n")

detector = JawClenchDetector(
    on_clench=on_clench,
    on_warmup_done=on_warmup_done,
    threshold_k=THRESHOLD_K,
    verbose=True,
)

try:
    detector.run()
except KeyboardInterrupt:
    print(f"\n\nStopped. Total clenches detected: {len(clench_log)}")
    for i, (t, dur) in enumerate(clench_log, 1):
        print(f"  #{i}: {dur:.0f} ms")
