"""
xArm S1 controller — FK, IK, and trajectory planning.

Built on roboticstoolbox-python (Peter Corke) with calibration from
xarm_config.json.  Uses Modified DH convention.

DH table (Modified DH / Craig convention):
  Link │ α_{i-1}  │ a_{i-1}           │ d_i    │ θ_i          │ offset
  ─────┼──────────┼───────────────────┼────────┼──────────────┼────────
    1  │    0°    │     0             │ L_base │  q1  Base    │   0°
    2  │  -90°    │     0             │   0    │  q2  Shoulder│ -90°
    3  │    0°    │ L_shoulder        │   0    │  q3  Elbow   │   0°
    4  │    0°    │ L_elbow           │   0    │  q4  Wrist   │   0°
    5  │   90°    │ L_wrist + L_tool  │   0    │  q5  WristRot│   0°

Joint angles are relative to physical home (arm straight up).
The shoulder offset of -90° maps physical-home to the DH zero where
the arm would be horizontal.  The tool length is along a₄ (forearm
direction) so the end-effector stays on the forearm axis.

All link lengths in mm.  Joint angles in degrees at the interface,
radians internally (roboticstoolbox convention).

Usage:
  python xarm_controller.py                          # demo
  python xarm_controller.py fk 0 45 -90 45 0         # forward kinematics
  python xarm_controller.py ik 150 0 100              # inverse kinematics
  python xarm_controller.py goto 150 0 100            # move arm to xyz (mm)
  python xarm_controller.py line 200 0 150 100 0 150  # straight-line trajectory
"""

import json
import sys
import time
from math import atan2, pi, sqrt

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

CONFIG_PATH = "xarm_config.json"
JOINT_IDS = [6, 5, 4, 3, 2]  # base → wrist, matches dh_joint_order


# ── Servo calibration ────────────────────────────────────────────────────────


class JointCalib:
    """Servo position ↔ joint angle conversion from xarm_config.json."""

    def __init__(self, data: dict):
        self.name = data["name"]
        self.home = data["home_position"]
        self.pos_min = data["position_min"]
        self.pos_max = data["position_max"]
        self.direction = data["direction"]
        self.angle_per_unit = data["angle_per_unit"]
        self.lo_deg = min(data["angle_min_deg"], data["angle_max_deg"])
        self.hi_deg = max(data["angle_min_deg"], data["angle_max_deg"])

    def pos_to_angle(self, pos: int) -> float:
        """Servo position → angle in degrees relative to home."""
        return (pos - self.home) * self.direction * self.angle_per_unit

    def angle_to_pos(self, angle_deg: float) -> int:
        """Angle in degrees → servo position, clamped to limits."""
        raw = self.home + angle_deg / (self.direction * self.angle_per_unit)
        return int(round(max(self.pos_min, min(self.pos_max, raw))))


# ── Robot model builder ──────────────────────────────────────────────────────


def _build_robot(config: dict) -> rtb.DHRobot:
    """Construct a DHRobot (Modified DH) from xarm_config.json.

    Shoulder has a -90° offset so that q=0 (servo home) corresponds to
    the arm pointing straight up.  The tool length is combined with the
    wrist link as a₄ so the end-effector stays on the forearm axis.
    """
    L = config["links"]
    servos = config["servos"]
    a_tool = L["L_wrist"] + L["L_tool"]  # tool extends along forearm

    def qlim(sid: int) -> list[float]:
        s = servos[str(sid)]
        lo = min(s["angle_min_deg"], s["angle_max_deg"])
        hi = max(s["angle_min_deg"], s["angle_max_deg"])
        return [np.radians(lo), np.radians(hi)]

    return rtb.DHRobot(
        [
            rtb.RevoluteMDH(alpha=0,     a=0,              d=L["L_base"],  qlim=qlim(6)),
            rtb.RevoluteMDH(alpha=-pi/2, a=0,              d=0,            qlim=qlim(5), offset=-pi/2),
            rtb.RevoluteMDH(alpha=0,     a=L["L_shoulder"], d=0,           qlim=qlim(4)),
            rtb.RevoluteMDH(alpha=0,     a=L["L_elbow"],   d=0,            qlim=qlim(3)),
            rtb.RevoluteMDH(alpha=pi/2,  a=a_tool,         d=0,            qlim=qlim(2)),
        ],
        name="xArm S1",
    )


# ── Controller ───────────────────────────────────────────────────────────────


class XArmController:
    """FK, IK, and trajectory planning for the xArm S1."""

    def __init__(self, config: dict):
        self.joints: dict[int, JointCalib] = {
            int(sid): JointCalib(data) for sid, data in config["servos"].items()
        }
        self.robot = _build_robot(config)
        self.gripper = config.get("gripper", {})

    # ── Servo ↔ angle conversion ──────────────────────────────────────

    def positions_to_angles(self, positions: dict[int, int]) -> list[float]:
        """Servo positions → [q1..q5] degrees."""
        return [
            self.joints[sid].pos_to_angle(
                positions.get(sid, self.joints[sid].home)
            )
            for sid in JOINT_IDS
        ]

    def angles_to_positions(self, angles_deg: list[float]) -> dict[int, int]:
        """[q1..q5] degrees → {servo_id: position}."""
        return {
            sid: self.joints[sid].angle_to_pos(a)
            for sid, a in zip(JOINT_IDS, angles_deg)
        }

    # ── Forward kinematics ────────────────────────────────────────────

    def fk(self, angles_deg: list[float]) -> SE3:
        """[q1..q5] degrees → SE3 end-effector pose (translations in mm)."""
        return self.robot.fkine(np.radians(angles_deg))

    def fk_position(self, angles_deg: list[float]) -> np.ndarray:
        """[q1..q5] degrees → [x, y, z] in mm."""
        return self.fk(angles_deg).t

    # ── Inverse kinematics ────────────────────────────────────────────

    def _geometric_seed(self, target_xyz: list[float]) -> list[float]:
        """Compute a geometric initial guess for IK (radians).

        q1 = atan2(y, x)      (base points toward target)
        q5 = 0                 (wrist rotation doesn't affect position)
        q2, q3, q4 estimated from planar 2-R reach + wrist compensation.

        q2 is measured from vertical (q2=0 = arm straight up) thanks to
        the -90° shoulder offset in the DH model.
        """
        x, y, z = target_xyz
        links = self.robot
        L_base = links[0].d      # 70
        L_sh = links[2].a        # 100  (shoulder)
        L_el = links[3].a        # 96   (elbow)
        L_tool = links[4].a      # 168  (wrist + tool, along forearm)

        q1 = atan2(y, x)

        # Planar reach from shoulder joint to target
        r = sqrt(x * x + y * y)  # horizontal distance
        v = z - L_base            # vertical distance above shoulder

        # Subtract wrist+tool contribution along the reach direction
        reach_angle = atan2(r, v)
        r_wr = r - L_tool * np.sin(reach_angle)
        v_wr = v - L_tool * np.cos(reach_angle)

        # 2-R IK for shoulder + elbow (in vertical reference frame)
        d2 = r_wr * r_wr + v_wr * v_wr
        cos_q3 = (d2 - L_sh * L_sh - L_el * L_el) / (2 * L_sh * L_el)
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        q3 = -abs(atan2(sqrt(max(0, 1 - cos_q3 * cos_q3)), cos_q3))

        # q2 = tilt from vertical (atan2(horizontal, vertical))
        q2 = atan2(r_wr, v_wr) - atan2(
            L_el * np.sin(q3), L_sh + L_el * np.cos(q3)
        )
        q4 = reach_angle - (q2 + q3)  # point tool toward target
        q5 = 0.0

        return [q1, q2, q3, q4, q5]

    def _solve_ik(
        self, T: SE3, q0: np.ndarray, mask: list[int],
    ) -> tuple[np.ndarray, bool]:
        sol = self.robot.ikine_LM(
            T, q0=q0, mask=mask,
            joint_limits=True, ilimit=500, slimit=50,
        )
        return sol.q, sol.success

    def ik(
        self,
        target_xyz: list[float],
        q0_deg: list[float] | None = None,
    ) -> list[float] | None:
        """
        Position-only IK (5-DOF arm, 3-DOF target).

        Uses a geometric initial guess and tries multiple seeds to find
        the most natural arm configuration (minimal wrist rotation,
        elbow-up preferred).

        Parameters
        ----------
        target_xyz : [x, y, z] in mm
        q0_deg     : initial guess in degrees (optional, overrides seed)

        Returns
        -------
        [q1..q5] in degrees, or None if IK failed.
        """
        T = SE3.Trans(*target_xyz)
        mask = [1, 1, 1, 0, 0, 0]
        geo = self._geometric_seed(target_xyz)
        q1 = geo[0]

        # Build diverse seeds covering the solution space
        seeds: list[np.ndarray] = []
        if q0_deg is not None:
            user = list(np.radians(q0_deg))
            user[4] = 0.0
            user[0] = q1
            seeds.append(np.array(user))
        seeds.append(np.array(geo))
        # Canonical arm configurations (all with correct q1 and q5=0)
        canonical = [
            [q1, 0.52, -1.05, 0.52, 0],   # elbow-up moderate
            [q1, 1.05, -1.57, 0.52, 0],    # elbow-up deep
            [q1, 0.35, 0.70, -1.05, 0],    # elbow-down
            [q1, 0, 0, 0, 0],              # straight arm
        ]
        seeds.extend(np.array(s) for s in canonical)

        best_q = None
        best_score = float("inf")
        for s in seeds:
            q, ok = self._solve_ik(T, s, mask)
            if not ok:
                continue
            # Score: penalize wrist rotation heavily, prefer small joints
            score = abs(q[4]) * 50 + sum(abs(qi) for qi in q)
            if score < best_score:
                best_score = score
                best_q = q

        return np.degrees(best_q).tolist() if best_q is not None else None

    def ik_pose(
        self,
        pose: SE3,
        q0_deg: list[float] | None = None,
        mask: list[int] | None = None,
    ) -> list[float] | None:
        """
        Full-pose IK (best effort for 5-DOF arm).

        Default mask [1,1,1,1,1,0] controls position + pitch + roll,
        leaving one orientation DOF free.
        """
        if mask is None:
            mask = [1, 1, 1, 1, 1, 0]
        q0 = np.radians(q0_deg) if q0_deg is not None else None
        sol = self.robot.ikine_LM(
            pose, q0=q0, mask=mask,
            joint_limits=True,
            ilimit=500,
            slimit=50,
        )
        return np.degrees(sol.q).tolist() if sol.success else None

    # ── Trajectory planning ───────────────────────────────────────────

    # Max servo velocity in degrees/second (xArm S1 hobby servos).
    SERVO_MAX_VEL_DEG_S = 75.0

    @staticmethod
    def _auto_steps(
        q_start_deg: list[float],
        q_end_deg: list[float],
        deg_per_step: float = 3.0,
        min_steps: int = 10,
        max_steps: int = 200,
    ) -> int:
        """Compute step count so the largest joint moves ~deg_per_step per step.

        Ensures enough resolution for smooth motion without wasting
        waypoints on tiny moves.
        """
        delta = max(abs(a - b) for a, b in zip(q_start_deg, q_end_deg))
        steps = int(round(delta / deg_per_step))
        return max(min_steps, min(max_steps, steps))

    def plan_joint(
        self,
        q_start_deg: list[float],
        q_end_deg: list[float],
        steps: int = 0,
        dt: float = 0.075,
        tacc: float = 0.3,
    ) -> np.ndarray:
        """
        Joint-space trajectory with trapezoidal velocity blending.

        Uses mstraj for bounded acceleration / jerk instead of a
        quintic polynomial.  The trajectory ramps up, cruises, and
        ramps down so servo motors are never asked for instantaneous
        velocity changes.

        Parameters
        ----------
        steps : ignored when 0 (step count determined by mstraj from
                dt and velocity limits).  When > 0 falls back to
                quintic jtraj for backwards compatibility.
        dt    : timestep in seconds (only used by mstraj path).
        tacc  : acceleration time in seconds for trapezoidal blend.

        Returns (n, 5) array of joint angles in degrees.
        """
        if steps > 0:
            # Legacy: explicit step count → quintic polynomial
            traj = rtb.jtraj(
                np.radians(q_start_deg), np.radians(q_end_deg), steps,
            )
            return np.degrees(traj.q)

        q0 = np.array(q_start_deg)
        q1 = np.array(q_end_deg)
        via = np.array([q0, q1])
        qdmax = np.full(5, self.SERVO_MAX_VEL_DEG_S)
        traj = rtb.mstraj(via, dt=dt, tacc=tacc, qdmax=qdmax)
        return traj.q

    @staticmethod
    def _smooth_trajectory(traj_deg: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply a moving-average low-pass filter per joint.

        Pins the first and last waypoints so start/end positions are
        exact.  Only the interior points are smoothed to remove
        step-to-step IK jitter.
        """
        if len(traj_deg) <= window:
            return traj_deg
        kernel = np.ones(window) / window
        smoothed = np.empty_like(traj_deg)
        for j in range(traj_deg.shape[1]):
            col = traj_deg[:, j]
            filt = np.convolve(col, kernel, mode="same")
            smoothed[:, j] = filt
        # Pin endpoints
        half = window // 2
        smoothed[:half] = traj_deg[:half]
        smoothed[-half:] = traj_deg[-half:]
        return smoothed

    def plan_line(
        self,
        start_xyz: list[float],
        end_xyz: list[float],
        steps: int = 0,
        q0_deg: list[float] | None = None,
        smooth: bool = True,
    ) -> np.ndarray | None:
        """
        Straight-line Cartesian trajectory between two points.

        Solves IK at each step, seeded from the previous solution.
        A moving-average filter is applied afterwards to suppress
        step-to-step IK jitter (disable with smooth=False).

        Parameters
        ----------
        steps  : number of waypoints.  0 = auto-compute from Cartesian
                 distance (~2 mm per step).
        smooth : apply low-pass filter to the joint-space result.

        Returns (steps, 5) array in degrees, or None if any IK fails.
        """
        if steps <= 0:
            dist = sqrt(sum((a - b) ** 2 for a, b in zip(start_xyz, end_xyz)))
            steps = max(10, min(200, int(round(dist / 2.0))))
        poses = rtb.ctraj(SE3.Trans(*start_xyz), SE3.Trans(*end_xyz), steps)
        mask = [1, 1, 1, 0, 0, 0]
        if q0_deg is not None:
            q_prev = np.radians(q0_deg)
            q_prev[4] = 0.0  # fix wrist rotation
        else:
            q_prev = np.array(self._geometric_seed(start_xyz))
        result = []
        for T in poses:
            q, ok = self._solve_ik(T, q_prev, mask)
            if not ok:
                return None
            result.append(q)
            q_prev = q
        traj = np.degrees(np.array(result))
        if smooth:
            traj = self._smooth_trajectory(traj)
        return traj

    def plan_waypoints(
        self,
        waypoints_xyz: list[list[float]],
        dt: float = 0.05,
        tacc: float = 0.5,
        qdmax_deg_s: float = 60.0,
    ) -> np.ndarray | None:
        """
        Multi-segment trajectory through xyz waypoints via mstraj.

        Uses trapezoidal velocity blending.  qdmax_deg_s is the max
        joint velocity in degrees/second for all joints.
        Returns (n, 5) array in degrees, or None if any IK fails.
        """
        via_rad = []
        q_prev = None
        mask = [1, 1, 1, 0, 0, 0]
        for xyz in waypoints_xyz:
            seed = np.array(self._geometric_seed(xyz)) if q_prev is None else q_prev
            q, ok = self._solve_ik(SE3.Trans(*xyz), seed, mask)
            if not ok:
                return None
            via_rad.append(q)
            q_prev = q
        traj = rtb.mstraj(
            np.array(via_rad),
            dt=dt,
            tacc=tacc,
            qdmax=np.full(5, np.radians(qdmax_deg_s)),
        )
        return np.degrees(traj.q)

    # ── Physical arm execution ────────────────────────────────────────

    def _enforce_velocity(
        self, trajectory_deg: np.ndarray, dt: float,
    ) -> float:
        """Check max joint velocity and stretch dt if it exceeds servo limits.

        Returns the (possibly increased) dt so no joint exceeds
        SERVO_MAX_VEL_DEG_S.
        """
        if len(trajectory_deg) < 2:
            return dt
        diffs = np.abs(np.diff(trajectory_deg, axis=0))  # (n-1, 5)
        max_step_deg = float(np.max(diffs))
        max_vel = max_step_deg / dt  # deg/s
        if max_vel > self.SERVO_MAX_VEL_DEG_S:
            new_dt = max_step_deg / self.SERVO_MAX_VEL_DEG_S
            print(f"  [velocity limit] Peak {max_vel:.0f}°/s exceeds "
                  f"{self.SERVO_MAX_VEL_DEG_S:.0f}°/s — stretching dt "
                  f"{dt*1000:.0f}ms → {new_dt*1000:.0f}ms")
            return new_dt
        return dt

    def move_to(
        self,
        target_deg: list[float],
        arm,
        duration_ms: int = 3000,
    ):
        """Move to a joint configuration with a single servo command.

        Lets the servo's built-in interpolation handle smoothing —
        identical to how home.py works.

        Parameters
        ----------
        target_deg  : [q1..q5] target joint angles in degrees
        arm         : XArmHID instance (open)
        duration_ms : time for the move in milliseconds (default 3000,
                      same as home.py).
        """
        positions = self.angles_to_positions(target_deg)
        targets = list(positions.items())
        arm.move_servos(targets, duration_ms=duration_ms)
        # Wait for the move to complete
        time.sleep(duration_ms / 1000 + 0.5)

    # ── Gripper ────────────────────────────────────────────────────

    def open_gripper(self, arm, duration_ms: int = 1000):
        """Open the gripper using config from xarm_config.json."""
        sid = self.gripper.get("servo_id", 1)
        pos = self.gripper.get("open_position", 200)
        arm.move_servos([(sid, pos)], duration_ms=duration_ms)
        time.sleep(duration_ms / 1000 + 0.3)

    def close_gripper(self, arm, duration_ms: int = 1000):
        """Close the gripper using config from xarm_config.json."""
        sid = self.gripper.get("servo_id", 1)
        pos = self.gripper.get("closed_position", 600)
        arm.move_servos([(sid, pos)], duration_ms=duration_ms)
        time.sleep(duration_ms / 1000 + 0.3)

    def execute(
        self,
        trajectory_deg: np.ndarray,
        arm,
        dt: float = 0.05,
        overlap: float = 1.8,
    ):
        """
        Send a trajectory to the physical arm.

        Parameters
        ----------
        trajectory_deg : (n, 5) array of joint angles in degrees
        arm            : XArmHID instance (open)
        dt             : seconds between waypoint commands
        overlap        : servo duration multiplier relative to dt.
                         Values > 1.0 let consecutive commands overlap,
                         giving the servo more time to interpolate and
                         producing smoother motion.  Typical range 1.5–2.2.

        Joint velocities are checked before execution.  If any joint
        exceeds SERVO_MAX_VEL_DEG_S the timestep is stretched so the
        servos can keep up, preventing the arm from falling behind the
        trajectory.
        """
        dt = self._enforce_velocity(trajectory_deg, dt)
        duration_ms = max(20, int(dt * 1000 * overlap))
        t0 = time.monotonic()
        for i, row in enumerate(trajectory_deg):
            positions = self.angles_to_positions(row.tolist())
            targets = list(positions.items())
            arm.move_servos(targets, duration_ms=duration_ms)
            next_time = t0 + (i + 1) * dt
            sleep_remaining = next_time - time.monotonic()
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

    # ── Display helpers ───────────────────────────────────────────────

    def print_state(self, angles_deg: list[float]):
        """Print joint angles, end-effector position, and servo positions."""
        pos = self.fk_position(angles_deg)
        print("  Joint angles:")
        for sid, angle in zip(JOINT_IDS, angles_deg):
            print(f"    {self.joints[sid].name:15s}: {angle:+8.2f}°")
        print("  End-effector (mm):")
        print(f"    x = {pos[0]:+8.2f}    y = {pos[1]:+8.2f}    z = {pos[2]:+8.2f}")
        positions = self.angles_to_positions(angles_deg)
        print("  Servo positions:")
        for sid in JOINT_IDS:
            print(f"    {self.joints[sid].name:15s}: {positions[sid]}")

    def print_info(self):
        """Print robot DH model and joint limits."""
        print(self.robot)
        print()
        limits = self.robot.qlim
        print("  Joint limits (from calibration):")
        for i, sid in enumerate(JOINT_IDS):
            lo_d = np.degrees(limits[0, i])
            hi_d = np.degrees(limits[1, i])
            print(f"    {self.joints[sid].name:15s}: [{lo_d:+.1f}°, {hi_d:+.1f}°]")


# ── Config loading ───────────────────────────────────────────────────────────


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _demo(ctrl: XArmController):
    print()
    ctrl.print_info()

    print()
    print("  [FK] Home — all joints at 0°:")
    ctrl.print_state([0.0] * 5)

    print()
    print("  [FK] Elbow-up — shoulder=45°, elbow=-90°, wrist=45°:")
    ctrl.print_state([0.0, 45.0, -90.0, 45.0, 0.0])

    # IK
    target = [150.0, 0.0, 100.0]
    print()
    print(f"  [IK] Target: x={target[0]}, y={target[1]}, z={target[2]} mm")
    q0 = [0.0, 30.0, -60.0, 30.0, 0.0]
    result = ctrl.ik(target, q0_deg=q0)
    if result:
        ctrl.print_state(result)
        actual = ctrl.fk_position(result)
        err = np.linalg.norm(np.array(target) - actual)
        print(f"  Position error: {err:.3f} mm")
    else:
        print("  IK failed — target may be unreachable.")

    # Trajectory
    if result:
        print()
        print("  [Trajectory] Joint-space: home → IK solution (20 steps)")
        traj = ctrl.plan_joint([0.0] * 5, result, steps=20)
        print(f"    {'Step':>4s}  {'x':>8s}  {'y':>8s}  {'z':>8s}")
        for i, row in enumerate(traj):
            p = ctrl.fk_position(row.tolist())
            print(f"    {i:4d}  {p[0]:+8.1f}  {p[1]:+8.1f}  {p[2]:+8.1f}")


def _cmd_fk(ctrl: XArmController, args: list[str]):
    if len(args) != 5:
        print("Usage: python xarm_controller.py fk q1 q2 q3 q4 q5  (degrees)")
        sys.exit(1)
    ctrl.print_state([float(a) for a in args])


def _cmd_ik(ctrl: XArmController, args: list[str]):
    if len(args) != 3:
        print("Usage: python xarm_controller.py ik x y z  (mm)")
        sys.exit(1)
    target = [float(a) for a in args]
    print(f"  IK target: x={target[0]}, y={target[1]}, z={target[2]} mm")
    result = ctrl.ik(target)
    if result:
        ctrl.print_state(result)
        err = np.linalg.norm(np.array(target) - ctrl.fk_position(result))
        print(f"  Position error: {err:.3f} mm")
    else:
        print("  IK failed — target may be unreachable.")
        sys.exit(1)


def _cmd_goto(ctrl: XArmController, args: list[str]):
    if len(args) != 3:
        print("Usage: python xarm_controller.py goto x y z  (mm)")
        sys.exit(1)
    target = [float(a) for a in args]
    print(f"  Target: x={target[0]}, y={target[1]}, z={target[2]} mm")
    result = ctrl.ik(target)
    if not result:
        print("  IK failed — target may be unreachable.")
        sys.exit(1)
    ctrl.print_state(result)

    from xarm_hid import XArmHID

    with XArmHID() as arm:
        print("  Moving to target...")
        ctrl.move_to(result, arm)
        print("  Done.")


def _cmd_movej(ctrl: XArmController, args: list[str]):
    if len(args) != 5:
        print("Usage: python xarm_controller.py movej q1 q2 q3 q4 q5  (degrees)")
        sys.exit(1)
    target_deg = [float(a) for a in args]
    print(f"  Target joint angles: {[f'{a:+.1f}' for a in target_deg]}")
    ctrl.print_state(target_deg)

    from xarm_hid import XArmHID

    with XArmHID() as arm:
        print("  Moving to target...")
        ctrl.move_to(target_deg, arm)
        print("  Done.")


def _cmd_line(ctrl: XArmController, args: list[str]):
    if len(args) != 6:
        print("Usage: python xarm_controller.py line x1 y1 z1 x2 y2 z2  (mm)")
        sys.exit(1)
    start = [float(a) for a in args[0:3]]
    end = [float(a) for a in args[3:6]]
    print(f"  Line: ({start[0]}, {start[1]}, {start[2]})"
          f" → ({end[0]}, {end[1]}, {end[2]}) mm")

    q_start = ctrl.ik(start)
    if not q_start:
        print("  IK failed at start point.")
        sys.exit(1)

    traj = ctrl.plan_line(start, end, q0_deg=q_start)
    if traj is None:
        print("  Trajectory planning failed — path may be unreachable.")
        sys.exit(1)

    print(f"  Planned {len(traj)} steps:")
    print(f"    {'Step':>4s}  {'x':>8s}  {'y':>8s}  {'z':>8s}")
    for i in range(0, len(traj), max(1, len(traj) // 10)):
        p = ctrl.fk_position(traj[i].tolist())
        print(f"    {i:4d}  {p[0]:+8.1f}  {p[1]:+8.1f}  {p[2]:+8.1f}")

    from xarm_hid import XArmHID

    with XArmHID() as arm:
        # Approach start point with single smooth command
        print("  Moving to start...")
        ctrl.move_to(traj[0].tolist(), arm)
        # Execute line
        print("  Executing line trajectory...")
        ctrl.execute(traj, arm, dt=0.04)
        print("  Done.")


def _cmd_grip(ctrl: XArmController, args: list[str]):
    if len(args) != 1 or args[0] not in ("open", "close"):
        print("Usage: python xarm_controller.py grip open|close")
        sys.exit(1)

    from xarm_hid import XArmHID

    with XArmHID() as arm:
        if args[0] == "open":
            print("  Opening gripper...")
            ctrl.open_gripper(arm)
        else:
            print("  Closing gripper...")
            ctrl.close_gripper(arm)
        print("  Done.")


def main():
    try:
        config = load_config()
    except FileNotFoundError:
        print(f"ERROR: {CONFIG_PATH} not found. Run calibrate.py first.")
        sys.exit(1)

    ctrl = XArmController(config)

    print("=" * 56)
    print("  xArm S1 — Controller (roboticstoolbox)")
    print("=" * 56)

    args = sys.argv[1:]
    if not args:
        _demo(ctrl)
        return

    cmd = args[0].lower()
    commands = {
        "fk": _cmd_fk,
        "ik": _cmd_ik,
        "goto": _cmd_goto,
        "movej": _cmd_movej,
        "line": _cmd_line,
        "grip": _cmd_grip,
    }
    if cmd in commands:
        commands[cmd](ctrl, args[1:])
    else:
        print(f"Unknown command: {cmd!r}")
        print("Commands: fk, ik, goto, movej, line, grip")
        sys.exit(1)


if __name__ == "__main__":
    main()

