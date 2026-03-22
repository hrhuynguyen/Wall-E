Setup Phase
1. Landing page
2. Heartbeat thread starts (monitors xArm + Muse 2 throughout session)
3. Connect xArm → confirm
4. Connect Muse 2 → confirm
5. Move arm to home position

Calibration Phase
6. Eye calibration (eye_tracker.py) — user in fixed position
7. EEG calibration (eeg.py) — user stays in same position

Main Loop
8.  coordinate.py scans scene → displays valid objects and its coordinate.
9.  User gazes at object → closest object to gaze point highlights
10. User jaw clenches once → highlighted object is selected and locked
11. IK solve with xarm_controller.py go → arm moves to object
12. Arrival confirmation → gripper opens → arm lowers → gripper closes -> pick the object
13. Arm lifts by configurable height (xarm_config.json, default 20mm)
14. Head tracker enables → yaw maps to servo 6
    (rotation speed configurable, default 15°/s)
15. EEG active only when head central ≥ 500ms
16. User rotates head to move arm to drop location
17. User returns head to center → holds ≥ 500ms → EEG re-enables
18. jaw_detection = True AND head_in_central = True → drop triggered
    → arm lowers → gripper opens → object released
19. Head tracker disables, EEG disables
20. Arm returns to home position.
21. Return to step 8, or end session and return to homepage.


End Session
22. Arm moves to home position (home.py)
23. Disconnect xArm and Muse 2 gracefully
24. Return to landing page
