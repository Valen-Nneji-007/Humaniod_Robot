"""
Hierarchical Whole-Body Controller (WBC) using Quadratic Programming.

Formulation:
  min   ||A_i * q_ddot - b_i||²   (soft tasks in priority order)
  s.t.  M * q_ddot + h = J_c^T * F_c   (constrained dynamics)
        J_c * q_ddot + J_dot_c * q_dot = 0   (contact constraint)
        tau_min <= tau <= tau_max
"""
import numpy as np
from scipy.sparse import csc_matrix
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import osqp


@dataclass
class WBCTask:
    """A task for the WBC: J * q_ddot = desired_accel."""
    jacobian: np.ndarray       # task Jacobian [m × n]
    desired_accel: np.ndarray  # desired task-space acceleration [m]
    weight: float = 1.0        # task weight
    name: str = "task"


@dataclass
class WBCConfig:
    n_joints: int = 8           # actuated joints
    dt: float = 0.01
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))

    # Task weights (highest priority → lowest)
    w_contact: float = 1e4      # contact constraint (hard)
    w_com: float = 100.0        # CoM tracking
    w_foot_swing: float = 80.0  # swing foot tracking
    w_foot_stance: float = 50.0 # stance foot (should not move)
    w_torso_orientation: float = 30.0
    w_posture: float = 1.0      # joint posture regularization
    w_arm: float = 10.0         # arm swinging (for balance)

    # Joint limits
    q_min: Optional[np.ndarray] = None
    q_max: Optional[np.ndarray] = None
    tau_max: float = 10.0

    # Regularization
    lambda_reg: float = 0.01


class WholeBodyController:
    """
    Hierarchical Whole-Body Controller.

    Solves a cascade of QPs enforcing tasks in strict priority order
    using the null-space projection method.
    """

    def __init__(self, config: WBCConfig):
        self.cfg = config
        self.n = config.n_joints
        self.g = config.gravity

        # Robot model references (set by update())
        self.M: Optional[np.ndarray] = None      # mass matrix [n×n]
        self.h: Optional[np.ndarray] = None      # bias forces [n]
        self.q: Optional[np.ndarray] = None      # current joint positions
        self.q_dot: Optional[np.ndarray] = None  # current joint velocities
        self.q_ddot: Optional[np.ndarray] = None # solution

        # Previous solution for smoothing
        self.q_ddot_prev: Optional[np.ndarray] = None

    # ---------- Robot Model Interface ----------

    def update_robot_state(self,
                           q: np.ndarray,
                           q_dot: np.ndarray,
                           M: np.ndarray,
                           h: np.ndarray):
        """Update robot state from URDF/dynamics engine."""
        self.q = q.copy()
        self.q_dot = q_dot.copy()
        self.M = M.copy()
        self.h = h.copy()

    # ---------- Kinematics Helpers (simplified — use Pinocchio/KDL in practice) ----------

    def _compute_com_jacobian(self) -> np.ndarray:
        """
        Compute CoM Jacobian.
        In practice, use Pinocchio: pinocchio.jacobianCenterOfMass(model, data, q)
        """
        # Placeholder — replace with actual kinematics library
        # For the URDF structure: base_link has the CoM
        J_com = np.zeros((3, self.n))
        # Approximate: CoM is largely affected by leg joints
        # waistR: index 0, waistL: index 2, kneeR: index 1, kneeL: index 3
        # These indices depend on your joint ordering
        return J_com

    def _compute_foot_jacobian(self, foot: str) -> np.ndarray:
        """Compute foot Jacobian (end of knee link)."""
        J = np.zeros((6, self.n))
        # Placeholder — compute via Pinocchio or KDL
        # foot position = knee_link origin (no ankle, so foot is at knee_link)
        return J

    def _compute_torso_jacobian(self) -> np.ndarray:
        """Jacobian of base_link orientation."""
        J = np.zeros((3, self.n))
        return J

    def _compute_contact_jacobian(self, support_foot: str) -> np.ndarray:
        """Contact Jacobian for stance foot (linear part only)."""
        J_foot = self._compute_foot_jacobian(support_foot)
        return J_foot[:3, :]  # position only for contact

    # ---------- QP Solver ----------

    def _solve_qp(self,
                  A: np.ndarray,
                  b: np.ndarray,
                  A_eq: Optional[np.ndarray] = None,
                  b_eq: Optional[np.ndarray] = None,
                  A_ineq: Optional[np.ndarray] = None,
                  l_ineq: Optional[np.ndarray] = None,
                  u_ineq: Optional[np.ndarray] = None,
                  N_prev: Optional[np.ndarray] = None,
                  x_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve: min ||A*x - b||²  s.t. constraints and null-space projection.

        If N_prev and x_prev are given, solves for x = x_prev + N_prev * v,
        i.e., searches in the null space of higher-priority tasks.
        """
        if N_prev is not None and x_prev is not None:
            # Reduce variables: x = x_prev + N_prev * v
            n_v = N_prev.shape[1]
            A_red = A @ N_prev
            b_red = b - A @ x_prev
        else:
            n_v = self.n
            A_red = A
            b_red = b
            N_prev = np.eye(self.n)
            x_prev = np.zeros(self.n)

        # Build OSQP problem
        P = A_red.T @ A_red + self.cfg.lambda_reg * np.eye(n_v)
        q_vec = -A_red.T @ b_red

        # Convert to sparse
        P_sparse = csc_matrix(P)

        # Constraints
        constraints = []
        if A_eq is not None and b_eq is not None:
            A_eq_red = A_eq @ N_prev
            constraints.append(("eq", A_eq_red, b_eq - A_eq @ x_prev))
        if A_ineq is not None:
            A_ineq_red = A_ineq @ N_prev
            constraints.append(("ineq", A_ineq_red, l_ineq, u_ineq))

        if constraints:
            A_c_list = []
            l_list = []
            u_list = []
            for c_type, *args in constraints:
                if c_type == "eq":
                    A_c_list.append(args[0])
                    l_list.append(args[1])
                    u_list.append(args[1])
                elif c_type == "ineq":
                    A_c_list.append(args[0])
                    l_list.append(args[1])
                    u_list.append(args[2])

            A_c = np.vstack(A_c_list)
            l_c = np.concatenate(l_list)
            u_c = np.concatenate(u_list)

            prob = osqp.OSQP()
            prob.setup(P=P_sparse, q=q_vec,
                       A=csc_matrix(A_c), l=l_c, u=u_c,
                       verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                       max_iter=2000, polish=True)
        else:
            prob = osqp.OSQP()
            prob.setup(P=P_sparse, q=q_vec,
                       verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                       max_iter=2000, polish=True)

        result = prob.solve()
        if result.info.status != 'solved':
            # Fallback: use damped least squares
            v = np.linalg.solve(P + 10 * np.eye(n_v), -q_vec)
        else:
            v = result.x

        return x_prev + N_prev @ v

    # ---------- Task Construction ----------

    def _compute_desired_com_accel(self,
                                    com_pos: np.ndarray,
                                    com_vel: np.ndarray,
                                    com_des: np.ndarray,
                                    com_vel_des: np.ndarray) -> np.ndarray:
        """PD control law for CoM tracking."""
        kp = 100.0  # position gain
        kd = 20.0   # velocity damping
        return kp * (com_des - com_pos) + kd * (com_vel_des - com_vel)

    def _compute_desired_foot_accel(self,
                                     foot_pos: np.ndarray,
                                     foot_vel: np.ndarray,
                                     foot_des: np.ndarray,
                                     foot_vel_des: np.ndarray,
                                     is_stance: bool) -> np.ndarray:
        """PD control for foot tracking."""
        if is_stance:
            kp = 500.0
            kd = 50.0
        else:
            kp = 300.0
            kd = 30.0
        return kp * (foot_des - foot_pos) + kd * (foot_vel_des - foot_vel)

    # ---------- Main Control Step ----------

    def compute(self,
                com_des: np.ndarray,
                com_vel_des: np.ndarray,
                left_foot_des: np.ndarray,
                right_foot_des: np.ndarray,
                left_foot_vel_des: np.ndarray,
                right_foot_vel_des: np.ndarray,
                support_foot: str = "double",
                torso_orientation_des: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint accelerations and torques for one control step.

        Returns:
            q_ddot: joint accelerations [n]
            tau: joint torques [n]
        """
        if self.M is None or self.h is None:
            raise RuntimeError("Call update_robot_state() before compute()")

        # Current kinematics (use actual kinematics library in practice)
        J_com = self._compute_com_jacobian()
        com_pos = np.zeros(3)  # replace with actual forward kinematics
        com_vel = J_com @ self.q_dot

        J_foot_L = self._compute_foot_jacobian("left")
        J_foot_R = self._compute_foot_jacobian("right")

        J_torso = self._compute_torso_jacobian()

        # ---- Desired task-space accelerations ----
        x_ddot_com = self._compute_desired_com_accel(com_pos, com_vel, com_des, com_vel_des)

        # For simplicity, foot 3D tracking
        lf_pos = np.zeros(3)  # replace with FK
        rf_pos = np.zeros(3)
        lf_vel = J_foot_L[:3] @ self.q_dot
        rf_vel = J_foot_R[:3] @ self.q_dot

        is_l_stance = support_foot in ("left", "double")
        is_r_stance = support_foot in ("right", "double")

        x_ddot_lf = self._compute_desired_foot_accel(
            lf_pos, lf_vel, left_foot_des[:3], left_foot_vel_des[:3], is_l_stance)
        x_ddot_rf = self._compute_desired_foot_accel(
            rf_pos, rf_vel, right_foot_des[:3], right_foot_vel_des[:3], is_r_stance)

        # ---- Priority 1: Contact constraints (hard) ----
        A_eq = None
        b_eq = None
        contact_jacs = []
        if is_l_stance:
            J_c_L = self._compute_contact_jacobian("left")
            contact_jacs.append(J_c_L)
        if is_r_stance:
            J_c_R = self._compute_contact_jacobian("right")
            contact_jacs.append(J_c_R)

        if contact_jacs:
            J_c = np.vstack(contact_jacs)
            # J_c * q_ddot + J_dot_c * q_dot = 0  (no acceleration at contact)
            J_dot_qdot = np.zeros(J_c.shape[0])  # compute properly in practice
            A_eq = J_c
            b_eq = -J_dot_qdot

        # ---- Priority 2: CoM tracking (soft) ----
        N1 = np.eye(self.n)
        x1 = np.zeros(self.n)

        task_com = WBCTask(
            jacobian=J_com,
            desired_accel=x_ddot_com,
            weight=self.cfg.w_com,
            name="com"
        )
        A1 = np.sqrt(task_com.weight) * task_com.jacobian
        b1 = np.sqrt(task_com.weight) * task_com.desired_accel
        x1 = self._solve_qp(A1, b1, A_eq=A_eq, b_eq=b_eq)
        N1 = self._null_space(A1 @ N1)  # project null space

        # ---- Priority 3: Stance foot tracking ----
        stance_tasks = []
        if is_l_stance:
            stance_tasks.append(("left", J_foot_L[:3], x_ddot_lf, self.cfg.w_foot_stance))
        if is_r_stance:
            stance_tasks.append(("right", J_foot_R[:3], x_ddot_rf, self.cfg.w_foot_stance))

        x2 = x1.copy()
        N2 = N1.copy()
        for name, J, x_ddot, w in stance_tasks:
            A_i = np.sqrt(w) * J
            b_i = np.sqrt(w) * x_ddot
            x2 = self._solve_qp(A_i, b_i, A_eq=A_eq, b_eq=b_eq,
                               N_prev=N2, x_prev=x2)
            N2 = self._null_space(np.vstack([A1, np.sqrt(w) * J]) @ N1)

        # ---- Priority 4: Swing foot tracking ----
        x3 = x2.copy()
        N3 = N2.copy()
        swing_tasks = []
        if not is_l_stance:
            swing_tasks.append(("left_swing", J_foot_L[:3], x_ddot_lf, self.cfg.w_foot_swing))
        if not is_r_stance:
            swing_tasks.append(("right_swing", J_foot_R[:3], x_ddot_rf, self.cfg.w_foot_swing))

        for name, J, x_ddot, w in swing_tasks:
            A_i = np.sqrt(w) * J
            b_i = np.sqrt(w) * x_ddot
            x3 = self._solve_qp(A_i, b_i, A_eq=A_eq, b_eq=b_eq,
                               N_prev=N3, x_prev=x3)

        # ---- Priority 5: Posture regularization ----
        q_des = np.zeros(self.n)  # desired posture (neutral)
        A_posture = np.sqrt(self.cfg.w_posture) * np.eye(self.n)
        b_posture = np.sqrt(self.cfg.w_posture) * (
            -self.cfg.w_posture * self.q - 2 * np.sqrt(self.cfg.w_posture) * self.q_dot
        )

        q_ddot = self._solve_qp(A_posture, b_posture,
                                A_eq=A_eq, b_eq=b_eq,
                                N_prev=np.eye(self.n), x_prev=x3)

        # ---- Joint torque from inverse dynamics ----
        tau = self.M @ q_ddot + self.h

        # Clamp torques
        tau = np.clip(tau, -self.cfg.tau_max, self.cfg.tau_max)

        self.q_ddot = q_ddot
        self.q_ddot_prev = q_ddot

        return q_ddot, tau

    @staticmethod
    def _null_space(J: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Compute orthonormal basis for null space of J."""
        if J.size == 0 or J.shape[0] == 0:
            return np.eye(J.shape[1])
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        rank = np.sum(s > tol)
        return Vt[rank:, :].T
