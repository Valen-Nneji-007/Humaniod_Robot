"""
gait_generator.py
==================

Pure-math joint-space gait generator for your humanoid's legs, using
ONLY the 4 DOF you actually have: joint_waistR, joint_kneeR,
joint_waistL, joint_kneeL (hip pitch + knee pitch per leg, no ankle,
no hip roll/yaw). No ROS dependency in this file on purpose, so it can
be unit-tested / plotted / tuned without Gazebo running.

--------------------------------------------------------------------
WHY THIS SHAPE OF GAIT (read this before tuning)
--------------------------------------------------------------------
Your robot cannot actively shift its center of mass sideways (no hip
roll, no ankle roll -- verified directly from your URDF's joint axes,
every leg/arm/neck joint rotates about the same local Y axis). That
means real alternating single-support walking -- where one foot is
fully off the ground and the whole body balances on the other -- isn't
something software can safely produce here: it would require exactly
the lateral degree of freedom the hardware doesn't have. This is a
physical limitation, not a control-algorithm one.

So this generator deliberately produces a "double-support-dominant"
shuffle/waddle gait instead:
    - each leg spends most of its cycle (`duty_factor`, default 0.8)
      in a slow STANCE sweep: hip pitches from front-of-stance to
      back-of-stance while (in principle) bearing weight -- this is
      what actually pushes the body forward over a planted foot
    - the remaining fraction of the cycle is a quick SWING: the hip
      snaps back to front-of-stance for the next step, with a knee
      "lift" added at the middle of the swing for ground clearance
    - the two legs are phase-offset by half a cycle (contralateral
      alternation, like normal walking), but with duty_factor > 0.5
      there is always overlap -- both feet are near/at the ground for
      most of the cycle, minimizing how long the robot ever spends
      relying on a single foot for lateral support
    - step size, speed and knee lift are all kept small/slow by
      default -- this is a cautious "can it walk at all" gait, not a
      fast dynamic one, because there's no IMU yet to close a balance
      loop, so everything here is open-loop and needs to stay well
      within whatever passive stability your foot/shin contact
      geometry actually has

IMPORTANT SIGN CONVENTION (verified against your actual URDF numbers,
not assumed): joint_waistR/joint_kneeR and joint_waistL/joint_kneeL
are mirrored joints. A *positive* angle swings the RIGHT leg forward,
but a *positive* angle swings the LEFT leg BACKWARD (confirmed via
forward kinematics on your exact joint origins/axes). This module
handles that by defining "forward" once and applying a per-leg sign
internally -- you should never need to flip a sign yourself.

WHAT YOU'LL LIKELY NEED TO TUNE after watching it in Gazebo:
    - KNEE_NEUTRAL: how much knee bend to hold in stance (affects
      standing height/stability -- more bend = lower CoM = more
      stable, but less like a "normal" stride)
    - HIP_SWING_AMP / step size: how far each hip swings fore/aft
    - CYCLE_PERIOD: slower = more cautious/stable, faster = more
      likely to tip given the lack of active lateral balance
    - duty_factor: higher = more conservative (more time on both feet)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Joint limits, taken directly from your URDF <limit> tags
# ---------------------------------------------------------------------------
LIMITS = {
    "joint_waistR": (-1.5708, 1.5708),
    "joint_kneeR": (-1.5708, 0.75),
    "joint_waistL": (-1.5708, 1.5708),
    "joint_kneeL": (-0.2, 1.5708),
}

JOINT_NAMES = ["joint_waistR", "joint_kneeR", "joint_waistL", "joint_kneeL"]

# Per-leg sign so a positive `forward_angle` always means "forward" in the
# physical world, regardless of the URDF's own mirrored sign convention.
# Verified by forward-kinematics on the real joint origins/axes (see the
# FK check run alongside this script) -- do not flip without re-verifying.
FORWARD_SIGN = {"R": +1.0, "L": -1.0}


class GaitParams:
    def __init__(
        self,
        cycle_period=9.0,       # seconds per full stride cycle -- deliberately very slow
        duty_factor=0.88,       # fraction of each leg's own cycle spent in "stance"
        hip_swing_amp=0.14,     # rad, +/- swing of the hip -- small, cautious step
        knee_neutral=0.10,      # rad, baseline knee bend held through stance (small crouch)
        knee_lift_R=0.20,       # rad, extra knee flex added at mid-swing (right leg)
        knee_lift_L=0.08,       # rad, extra knee flex added at mid-swing (left leg)
        # NOTE knee_lift_L is deliberately smaller than knee_lift_R: joint_kneeL's
        # own <limit> only allows -0.2 rad in the "forward/lift" direction (see
        # module docstring), so the left knee physically cannot lift the foot as
        # much as the right knee can. This is a hardware asymmetry, not a bug.
        #
        # These defaults were tightened after a first real test on hardware/Gazebo
        # showed the robot toppling almost immediately -- see walk_gait_node.py's
        # settle+ramp startup sequence for the other half of that fix (the fall
        # was very likely triggered by an abrupt 0.1s jump straight to full swing
        # amplitude from a standing start, not just gait speed/size on its own,
        # but slowing everything down further gives the best remaining margin
        # given there's no hip-roll/ankle to actively recover balance with).
    ):
        self.cycle_period = cycle_period
        self.duty_factor = duty_factor
        self.hip_swing_amp = hip_swing_amp
        self.knee_neutral = knee_neutral
        self.knee_lift = {"R": knee_lift_R, "L": knee_lift_L}
        self._validate()

    def _validate(self):
        # Clip amplitude choices so they can never exceed the URDF's own limits,
        # regardless of what values are passed in above.
        for leg, hip_j, knee_j in (("R", "joint_waistR", "joint_kneeR"), ("L", "joint_waistL", "joint_kneeL")):
            lo, hi = LIMITS[hip_j]
            max_amp = min(abs(lo), abs(hi))
            if self.hip_swing_amp > max_amp:
                raise ValueError(f"hip_swing_amp {self.hip_swing_amp} exceeds {hip_j} limit ({lo},{hi})")
            lo, hi = LIMITS[knee_j]
            fwd = FORWARD_SIGN[leg]
            lift_bound = hi if fwd > 0 else -lo  # room available in the "forward/lift" direction
            total = self.knee_neutral + self.knee_lift[leg]
            if total > lift_bound + 1e-6:
                raise ValueError(
                    f"{knee_j}: knee_neutral+knee_lift ({total:.3f}) exceeds available "
                    f"limit in the lift direction ({lift_bound:.3f}). Reduce knee_lift_{leg} "
                    f"or knee_neutral."
                )
            back_bound = -lo if fwd > 0 else hi  # room available in the opposite direction
            if self.knee_neutral > back_bound + 1e-6:
                raise ValueError(f"{knee_j}: knee_neutral ({self.knee_neutral:.3f}) exceeds limit.")


def _smoothstep(x):
    """Smooth 0->1 ease (3x^2 - 2x^3), x clipped to [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def leg_angles(phase, leg, params: GaitParams):
    """
    phase: this leg's own gait phase in [0, 1) (0 = start of stance,
        duty_factor = start of swing, 1 = back to start of stance)
    leg: "R" or "L"
    Returns (hip_angle, knee_angle) in radians, already signed correctly
    for this leg and clipped to its URDF limits as a final safety net.
    """
    fwd = FORWARD_SIGN[leg]
    df = params.duty_factor

    if phase < df:
        # --- stance: slow sweep from front-of-stance to back-of-stance ---
        s = phase / df  # 0..1 across the stance portion
        forward_angle = params.hip_swing_amp * (1 - 2 * s)  # +amp -> -amp, linear
        knee = params.knee_neutral
    else:
        # --- swing: quick reset from back-of-stance to front-of-stance,
        #     with a knee "lift" bump at mid-swing for ground clearance ---
        s = (phase - df) / (1 - df)  # 0..1 across the swing portion
        ease = _smoothstep(s)
        forward_angle = params.hip_swing_amp * (-1 + 2 * ease)  # -amp -> +amp
        lift = np.sin(np.pi * s) * params.knee_lift[leg]  # 0 -> max -> 0
        knee = params.knee_neutral + lift

    hip = fwd * forward_angle
    knee = fwd * knee  # same per-leg mirroring applies to the knee (see module docstring)

    hip_lo, hip_hi = LIMITS[f"joint_waist{leg}"]
    knee_lo, knee_hi = LIMITS[f"joint_knee{leg}"]
    hip = float(np.clip(hip, hip_lo, hip_hi))
    knee = float(np.clip(knee, knee_lo, knee_hi))
    return hip, knee


def sample_cycle(params: GaitParams, n_samples=60):
    """Sample one full cycle_period at n_samples points.

    Returns (t, angles) where angles is a dict joint_name -> array[n_samples].
    Right leg leads (phase 0); left leg is phase-offset by half a cycle.
    """
    t = np.linspace(0, params.cycle_period, n_samples, endpoint=False)
    phase_R = (t / params.cycle_period) % 1.0
    phase_L = ((t / params.cycle_period) + 0.5) % 1.0

    angles = {name: np.zeros(n_samples) for name in JOINT_NAMES}
    for i in range(n_samples):
        hipR, kneeR = leg_angles(phase_R[i], "R", params)
        hipL, kneeL = leg_angles(phase_L[i], "L", params)
        angles["joint_waistR"][i] = hipR
        angles["joint_kneeR"][i] = kneeR
        angles["joint_waistL"][i] = hipL
        angles["joint_kneeL"][i] = kneeL
    return t, angles


def cycle_start_pose(params: GaitParams):
    """The (hip, knee) pose for both legs at phase=0 (start of the repeating
    cycle) -- i.e. where a smooth startup ramp should be ramping *to*."""
    hipR, kneeR = leg_angles(0.0, "R", params)
    hipL, kneeL = leg_angles(0.5, "L", params)  # left leg starts half a cycle in
    return {
        "joint_waistR": hipR, "joint_kneeR": kneeR,
        "joint_waistL": hipL, "joint_kneeL": kneeL,
    }


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = GaitParams()
    t, angles = sample_cycle(params, n_samples=200)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, angles["joint_waistR"], label="joint_waistR (hip)")
    axes[0].plot(t, angles["joint_kneeR"], label="joint_kneeR (knee)")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_title("Right leg")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, angles["joint_waistL"], label="joint_waistL (hip)")
    axes[1].plot(t, angles["joint_kneeL"], label="joint_kneeL (knee)")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_title("Left leg")
    axes[1].set_xlabel("time (s) -- one full cycle")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("gait_preview.png", dpi=140)
    print("wrote gait_preview.png")
    print("max |hip| R/L:", np.max(np.abs(angles["joint_waistR"])), np.max(np.abs(angles["joint_waistL"])))
    print("knee range R:", angles["joint_kneeR"].min(), angles["joint_kneeR"].max())
    print("knee range L:", angles["joint_kneeL"].min(), angles["joint_kneeL"].max())