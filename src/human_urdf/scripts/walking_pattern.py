"""
Walking Pattern Generator — generates CoM and foot trajectories
using 3D Linear Inverted Pendulum Model (LIPM) with ZMP preview control.
"""
import numpy as np
from scipy.linalg import toeplitz
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Footstep:
    pose: np.ndarray        # [x, y, z, roll, pitch, yaw] in world frame
    support_foot: str       # "right" or "left"
    duration: float         # step duration in seconds
    is_double_support: bool = False


@dataclass
class WalkParams:
    com_height: float = 0.8         # CoM height (m) — from your URDF base_footprint→base_link
    step_length: float = 0.15       # forward step length
    step_width: float = 0.12        # lateral foot spacing
    step_height: float = 0.04       # foot clearance
    step_duration: float = 0.8      # single support duration
    double_support_duration: float = 0.15
    swing_ratio: float = 0.7        # how much of step is swing phase
    n_preview_steps: int = 16       # MPC preview horizon
    dt: float = 0.01                # control timestep
    ft: float = 0.005               # trajectory interpolation timestep


class WalkingPatternGenerator:
    """
    3D LIPM-based walking pattern generator with ZMP preview control.

    Generates:
      - CoM trajectory (x, y, z)
      - ZMP reference trajectory
      - Foot trajectories (swing + stance phases)
    """

    def __init__(self, params: WalkParams):
        self.p = params
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.p.com_height)
        self._reset()

    def _reset(self):
        self.com_traj: List[np.ndarray] = []
        self.zmp_traj: List[np.ndarray] = []
        self.left_foot_traj: List[np.ndarray] = []
        self.right_foot_traj: List[np.ndarray] = []
        self.com_vel_traj: List[np.ndarray] = []
        self.timestamps: List[float] = []

    # ---------- Preview Control (CoM from ZMP) ----------

    def _build_preview_gain(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build preview control gains for 1D LIPM."""
        dt = self.p.dt
        A = np.array([[1.0, dt, dt**2/2],
                       [0.0, 1.0, dt],
                       [0.0, 0.0, 1.0]])
        b = np.array([dt**3/6, dt**2/2, dt]).reshape(3, 1)
        c = np.array([1.0, 0.0, -self.p.com_height / self.g]).reshape(1, 3)

        # LQR weights
        Qe = 1.0
        Qx = np.diag([0.0, 0.0, 0.0])
        R = 1e-6

        # Solve discrete algebraic Riccati equation
        P = Qx.copy()
        for _ in range(500):
            K = np.linalg.inv(R + b.T @ P @ b) @ (b.T @ P @ A)
            P_new = Qx + c.T @ Qe @ c + (A - b @ K).T @ P @ (A - b @ K)
            if np.max(np.abs(P_new - P)) < 1e-10:
                break
            P = P_new

        K = np.linalg.inv(R + b.T @ P @ b) @ (b.T @ P @ A)
        Ac = A - b @ K
        Gi = np.linalg.inv(R + b.T @ P @ b) @ b.T

        # Preview gains
        N = self.p.n_preview_steps
        f = np.zeros(N)
        X = -Ac.T @ P @ np.array([1.0, 0.0, 0.0]).reshape(3, 1) * Qe
        for i in range(N):
            f[i] = (Gi @ X).item()
            X = Ac.T @ X

        return K.flatten(), f, Ac

    def _preview_control_1d(self, zmp_ref: np.ndarray) -> np.ndarray:
        """Compute 1D CoM trajectory from ZMP reference using preview control."""
        K, f, Ac = self._build_preview_gain()
        N_t = len(zmp_ref)
        N = self.p.n_preview_steps

        x = np.array([0.0, 0.0, 0.0])  # [com_pos, com_vel, com_acc]
        com_out = np.zeros(N_t)

        for k in range(N_t):
            preview = zmp_ref[k:k+N]
            n_preview = len(preview)
            u = -K @ x
            for j in range(n_preview):
                u += f[j] * preview[j]
            for j in range(n_preview, N):
                u += f[j] * preview[-1]  # constant tail

            com_out[k] = x[0]
            x = Ac @ x + np.array([0.0, 0.0, 0.0])  # simplified; adding input
            # Full state update with input
            b_vec = np.array([self.p.dt**3/6, self.p.dt**2/2, self.p.dt])
            x = Ac @ x + b_vec * u

        return com_out

    # ---------- Footstep Planning ----------

    def plan_footsteps(self,
                       n_steps: int,
                       start_left: np.ndarray,
                       start_right: np.ndarray,
                       com_start: np.ndarray) -> List[Footstep]:
        """
        Generate a sequence of footsteps.

        Args:
            n_steps: number of steps
            start_left: initial left foot pose [x,y,z,roll,pitch,yaw]
            start_right: initial right foot pose
            com_start: initial CoM position [x,y,z]
        """
        footsteps = []
        step_len = self.p.step_length
        step_w = self.p.step_width

        # Full step cycle
        for i in range(n_steps):
            if i == 0:
                # Double support
                dp = Footstep(
                    pose=com_start.copy(),
                    support_foot="double",
                    duration=self.p.double_support_duration,
                    is_double_support=True
                )
                dp.pose[2] = self.p.com_height
                footsteps.append(dp)

            support = "left" if i % 2 == 0 else "right"
            swing = "right" if support == "left" else "left"

            if swing == "right":
                # Right foot steps forward
                pose = start_right.copy()
                pose[0] += (i // 2 + 1) * step_len
                pose[1] = -step_w / 2
            else:
                pose = start_left.copy()
                pose[0] += (i // 2 + 1) * step_len
                pose[1] = step_w / 2

            footsteps.append(Footstep(
                pose=pose,
                support_foot=support,
                duration=self.p.step_duration
            ))

        return footsteps

    # ---------- ZMP Reference ----------

    def generate_zmp_reference(self, footsteps: List[Footstep]) -> np.ndarray:
        """Generate ZMP reference trajectory from footsteps."""
        total_time = sum(f.duration for f in footsteps)
        n = int(total_time / self.p.dt)
        zmp = np.zeros((n, 3))

        t = 0.0
        idx = 0
        for fs in footsteps:
            n_fs = int(fs.duration / self.p.dt)
            for j in range(n_fs):
                if idx < n:
                    if fs.is_double_support:
                        # ZMP at midpoint during double support
                        pass  # keep previous
                    else:
                        zmp[idx, 0] = fs.pose[0]
                        zmp[idx, 1] = fs.pose[1]
                        zmp[idx, 2] = 0.0
                    idx += 1

        return zmp

    # ---------- Swing Foot Trajectory ----------

    def _swing_foot_trajectory(self,
                                start: np.ndarray,
                                end: np.ndarray,
                                duration: float,
                                clearance: float) -> np.ndarray:
        """Generate a smooth swing foot trajectory using a cycloid."""
        n = int(duration / self.p.ft)
        traj = np.zeros((n, 6))
        t = np.linspace(0, 1, n)

        # Linear interpolation for x, y, yaw
        for d in range(6):
            traj[:, d] = start[d] + (end[d] - start[d]) * t

        # Cycloid for z (lift, then land)
        traj[:, 2] = start[2] + clearance * np.sin(np.pi * t)
        traj[-1, 2] = end[2]  # ensure landing at target height

        return traj

    # ---------- Full Pattern Generation ----------

    def generate(self,
                 n_steps: int,
                 left_foot_start: np.ndarray,
                 right_foot_start: np.ndarray,
                 com_start: np.ndarray) -> dict:
        """
        Generate full walking pattern.

        Returns dict with keys:
          com_position, com_velocity, left_foot, right_foot, zmp, timestamps
        """
        self._reset()

        # Plan footsteps
        footsteps = self.plan_footsteps(n_steps, left_foot_start, right_foot_start, com_start)

        # ZMP reference
        zmp_ref = self.generate_zmp_reference(footsteps)

        # CoM trajectories (x and y separately)
        com_x = self._preview_control_1d(zmp_ref[:, 0])
        com_y = self._preview_control_1d(zmp_ref[:, 1])
        com_z = np.full_like(com_x, self.p.com_height)

        com_pos = np.column_stack([com_x, com_y, com_z])

        # Compute velocity by differentiation
        com_vel = np.zeros_like(com_pos)
        com_vel[1:] = (com_pos[1:] - com_pos[:-1]) / self.p.dt

        # Foot trajectories
        lf_start = left_foot_start.copy()
        rf_start = right_foot_start.copy()
        lf_traj = []
        rf_traj = []
        timestamps = []

        t = 0.0
        for fs in footsteps:
            if fs.is_double_support:
                # Keep feet in place during double support
                n_ds = int(fs.duration / self.p.ft)
                for _ in range(n_ds):
                    lf_traj.append(lf_start.copy())
                    rf_traj.append(rf_start.copy())
                    timestamps.append(t)
                    t += self.p.ft
            else:
                # Swing and stance
                n_step = int(fs.duration / self.p.ft)
                if fs.support_foot == "left":
                    # Right foot swings
                    rf_end = fs.pose.copy()
                    swing = self._swing_foot_trajectory(rf_start, rf_end, fs.duration, self.p.step_height)
                    for j in range(n_step):
                        lf_traj.append(lf_start.copy())
                        rf_traj.append(swing[j].copy())
                        timestamps.append(t)
                        t += self.p.ft
                    rf_start = rf_end.copy()
                else:
                    # Left foot swings
                    lf_end = fs.pose.copy()
                    swing = self._swing_foot_trajectory(lf_start, lf_end, fs.duration, self.p.step_height)
                    for j in range(n_step):
                        lf_traj.append(swing[j].copy())
                        rf_traj.append(rf_start.copy())
                        timestamps.append(t)
                        t += self.p.ft
                    lf_start = lf_end.copy()

        # Interpolate CoM to match foot trajectory timestamps
        n_out = len(timestamps)
        com_pos_out = np.zeros((n_out, 3))
        com_vel_out = np.zeros((n_out, 3))
        for i in range(n_out):
            idx = min(int(timestamps[i] / self.p.dt), len(com_pos) - 1)
            com_pos_out[i] = com_pos[idx]
            com_vel_out[i] = com_vel[idx]

        return {
            "com_position": com_pos_out,
            "com_velocity": com_vel_out,
            "left_foot": np.array(lf_traj),
            "right_foot": np.array(rf_traj),
            "timestamps": np.array(timestamps)
        }
