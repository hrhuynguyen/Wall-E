# Build Progress — State Machine + Tkinter UI

Building the central state machine + Tkinter UI for the assistive arm system.
Full plan at: `~/.claude/plans/snoopy-snuggling-whale.md`

## Build Steps

- [x] **Step 1**: Events + state machine skeleton — `app/events.py`, `app/state_machine.py`, `app/config.py` — DONE
- [x] **Step 2**: Tkinter shell + landing page — `main.py`, `app/ui/app_window.py`, `app/ui/landing_page.py` — DONE
- [x] **Step 3**: Camera thread + live feed — `app/camera_thread.py` — DONE
- [x] **Step 4**: Session page layout — `app/ui/session_page.py` — DONE
- [x] **Step 5**: Arm worker + connect + home — `app/arm_worker.py` — DONE
- [x] **Step 6**: EEG thread + warmup — `app/eeg_thread.py` — DONE
- [x] **Step 7**: Heartbeat thread — `app/heartbeat_thread.py` — DONE
- [x] **Step 8**: Gaze tracking — `app/vision.py` (GazeTracker), wired into `app_window.py` — DONE
- [x] **Step 9**: Object detection + depth — `app/vision.py` (ObjectDetector), wired into `app_window.py` — DONE
- [x] **Step 10**: Object selection (clench) — gaze-to-object matching + jaw clench selection — DONE
- [x] **Step 11**: Pick sequence — wire arm_worker. Converts vision coords (m) → arm coords (mm). Pick: open gripper → close gripper. Lift: z + 20mm via IK — DONE
- [x] **Step 12**: Head tracking + rotation — HeadTracker in `app/vision.py`, velocity-based base rotation via arm_worker, head polling loop in app_window — DONE
- [x] **Step 13**: Drop + return home — lower z by 20mm, open gripper, move home, resume scanning. Full cycle complete — DONE

## File Structure

```
robot-utility-test/
  main.py                      # Entry point (Step 2)
  app/
    __init__.py                # Created
    config.py                  # Created (Step 1)
    events.py                  # Created (Step 1)
    state_machine.py           # Created (Step 1)
    camera_thread.py           # Step 3
    eeg_thread.py              # Step 6
    arm_worker.py              # Step 5
    heartbeat_thread.py        # Step 7
    vision.py                  # Steps 8, 9, 12
    ui/
      __init__.py              # Created
      app_window.py            # Step 2
      landing_page.py          # Step 2
      session_page.py          # Step 4
      calibration_page.py      # Step 8
  utility/                     # Reusable modules
    head_tracker.py            # Head yaw detection + optional servo 6 rotation (--arm flag)
```

## Threading Model

- Main thread: Tkinter event loop + vision processing
- CameraThread: owns VideoCapture(0)
- EEGThread: LSL consumer + JawClenchDetector
- ArmWorker: owns XArmHID, executes blocking moves
- HeartbeatThread: periodic device health checks
- Muse stream: daemon thread (auto-launched, no separate terminal)

## Key Design Decisions

- Muse BLE→LSL stream launched as daemon thread inside app (no separate terminal)
- Arm commands queued to ArmWorker thread (never block the UI)
- All threads communicate via EventBus (queue.Queue wrapper)
- State machine is pure logic — returns Command objects, never calls hardware
- Head tracker (`utility/head_tracker.py`) detects head left/right/center via MediaPipe face landmarks + hysteresis, and can rotate servo 6 (base) with `--arm` flag using incremental velocity control (proportional to yaw, up to 30°/s)
