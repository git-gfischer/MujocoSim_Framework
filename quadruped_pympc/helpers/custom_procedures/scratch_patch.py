"""
Runtime patch for SwingTrajectoryController to use a scratch-on-ground trajectory
for one leg instead of the normal swing (lift and place).

Three-phase cycle:
  1) Swing forward: foot lifts, moves forward, touches down.
  2) Slide back a certain amount: foot in contact, straight line back by back_amount.
  3) Slide to initial: foot in contact, straight line from current position to initial.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# Default scratch:
#   - forward distance (m) the foot swings in front of its initial position
#   - backward distance (m) the foot rubs past the initial position
#   - total period (s) for the whole 3-phase motion
#   - step height (m) for swing phases
SCRATCH_AMPLITUDE_DEFAULT = 0.15
SCRATCH_BACK_AMOUNT_DEFAULT = 0.10
SCRATCH_PERIOD_DEFAULT = 4.0
SCRATCH_STEP_HEIGHT_DEFAULT = 0.03

# Phase boundaries: [0, p1]=swing, [p1, p2]=slide back amount, [p2, 1]=slide to initial
SCRATCH_P1 = 1.0 / 3.0   # end of swing 
SCRATCH_P2 = 2.0 / 3.0   # end of slide-back amount


def _scratch_trajectory(
    swing_time: float,
    center: np.ndarray,
    amplitude: float,
    period: float,
    step_height: float,
    back_amount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase 1 [0, p1]: Swing forward — lift, move from initial to A = center + amplitude.
    Phase 2 [p1, p2]: Rub back in contact — slide on ground from A past initial to B = center - back.
    Phase 3 [p2, 1]: Swing back — lift from B and swing back to the initial position.
    """
    if period <= 0:
        period = 1e-6
    back = max(back_amount, 0.0)
    center_flat = np.asarray(center).reshape(3)
    cx, cy, ground_z = center_flat[0], center_flat[1], float(center_flat[2])
    t_cycle = swing_time % period
    phase = t_cycle / period  # in [0, 1)
    p1, p2 = SCRATCH_P1, SCRATCH_P2
    one_over_T = 1.0 / period

    if phase < p1:
        # --- Phase 1: Swing forward (initial -> A) ---
        # A = cx + amplitude
        u = phase / p1  # u in [0, 1]
        # x: linear from initial to A
        x = cx + amplitude * u
        vx = amplitude * one_over_T / p1
        ax = 0.0
        # z: smooth bump with zero vertical velocity at start and end
        z = ground_z + step_height * (math.sin(math.pi * u) ** 2)
        dz_du = step_height * math.pi * math.sin(2.0 * math.pi * u)
        d2z_du2 = step_height * 2.0 * (math.pi ** 2) * math.cos(2.0 * math.pi * u)
        vz = dz_du * (one_over_T / p1)
        az = d2z_du2 * (one_over_T / p1) ** 2
    elif phase < p2:
        # --- Phase 2: Rub back in contact (A -> B), straight line at ground ---
        # A = cx + amplitude, B = cx - back
        u = (phase - p1) / (p2 - p1)  # u in [0, 1]
        x_start = cx + amplitude
        x_end = cx - back
        x = x_start + (x_end - x_start) * u
        z = ground_z
        vx = (x_end - x_start) * one_over_T / (p2 - p1)
        ax = 0.0
        vz = 0.0
        az = 0.0
    else:
        # --- Phase 3: Swing back (B -> initial) ---
        # B = cx - back -> cx
        u = (phase - p2) / (1.0 - p2)  # u in [0, 1]
        x_start = cx - back
        x_end = cx
        x = x_start + (x_end - x_start) * u
        vx = (x_end - x_start) * one_over_T / (1.0 - p2)
        ax = 0.0
        # z: swing bump with zero vertical velocity at start and end
        z = ground_z + step_height * (math.sin(math.pi * u) ** 2)
        dz_du = step_height * math.pi * math.sin(2.0 * math.pi * u)
        d2z_du2 = step_height * 2.0 * (math.pi ** 2) * math.cos(2.0 * math.pi * u)
        vz = dz_du * (one_over_T / (1.0 - p2))
        az = d2z_du2 * (one_over_T / (1.0 - p2)) ** 2

    des_pos = np.array([x, cy, z])
    des_vel = np.array([vx, 0.0, vz])
    des_acc = np.array([ax, 0.0, az])
    return des_pos, des_vel, des_acc


def install_scratch_patch(stc: Any) -> None:
    """
    Patch a SwingTrajectoryController to support scratch mode for one leg.
    Idempotent: safe to call multiple times.
    """
    if getattr(stc, "_scratch_patch_installed", False):
        return

    stc.scratch_leg_id = None
    stc.scratch_center = None  # fixed initial foot position (set once when scratch starts)
    stc.scratch_amplitude = SCRATCH_AMPLITUDE_DEFAULT
    stc.scratch_back_amount = SCRATCH_BACK_AMOUNT_DEFAULT
    stc.scratch_period = SCRATCH_PERIOD_DEFAULT
    stc.scratch_step_height = SCRATCH_STEP_HEIGHT_DEFAULT

    _original_compute = stc.compute_swing_control_cartesian_space

    def _patched_compute_swing_control_cartesian_space(
        leg_id,
        q_dot,
        J,
        J_dot,
        lift_off,
        touch_down,
        foot_pos,
        foot_vel,
        passive_force,
        h,
        mass_matrix,
        early_stance_hitmoments,
        early_stance_hitpoints,
    ):
        if stc.scratch_leg_id is not None and leg_id == stc.scratch_leg_id:
            # Fix center once at first use: use actual foot position so we stay on the ground
            if stc.scratch_center is None:
                stc.scratch_center = np.asarray(foot_pos, dtype=np.float64).reshape(3).copy()
            center = stc.scratch_center
            # For scratch, swing_time can grow beyond swing_period; use modulo for phase
            t_scratch = stc.swing_time[leg_id] % max(stc.scratch_period, 1e-6)
            des_foot_pos, des_foot_vel, des_foot_acc = _scratch_trajectory(
                t_scratch,
                center,
                stc.scratch_amplitude,
                stc.scratch_period,
                getattr(stc, "scratch_step_height", SCRATCH_STEP_HEIGHT_DEFAULT),
                getattr(stc, "scratch_back_amount", SCRATCH_BACK_AMOUNT_DEFAULT),
            )
            des_foot_pos = des_foot_pos.reshape((3,))
            des_foot_vel = des_foot_vel.reshape((3,))
            foot_pos_arr = np.asarray(foot_pos).reshape(3)
            foot_vel_arr = np.asarray(foot_vel).reshape(3)
            err_pos = des_foot_pos - foot_pos_arr
            err_vel = des_foot_vel - foot_vel_arr

            # Phase within scratch cycle
            phase = (t_scratch % stc.scratch_period) / max(stc.scratch_period, 1e-6)

            # Trajectory-based Cartesian impedance for all phases, with softer Z in rub-back
            kp_arr = np.array(stc.position_gain_fb, dtype=float).ravel()
            kd_arr = np.array(stc.velocity_gain_fb, dtype=float).ravel()
            if kp_arr.size == 1:
                kp = np.repeat(kp_arr[0], 3)
            else:
                kp = kp_arr[:3].copy()
            if kd_arr.size == 1:
                kd = np.repeat(kd_arr[0], 3)
            else:
                kd = kd_arr[:3].copy()
            if SCRATCH_P1 <= phase < SCRATCH_P2:
                # In rub-back phase, reduce Z gains to be more compliant vertically.
                kp[2] *= 0.3
                kd[2] *= 0.3

            acceleration = des_foot_acc + kp * err_pos + kd * err_vel
            tau_swing = J.T @ (kp * err_pos + kd * err_vel)
            if stc.use_feedback_linearization:
                tau_swing += mass_matrix @ np.linalg.pinv(J) @ (
                    acceleration - J_dot @ q_dot
                ) + h
            return tau_swing, des_foot_pos, des_foot_vel

        return _original_compute(
            leg_id=leg_id,
            q_dot=q_dot,
            J=J,
            J_dot=J_dot,
            lift_off=lift_off,
            touch_down=touch_down,
            foot_pos=foot_pos,
            foot_vel=foot_vel,
            passive_force=passive_force,
            h=h,
            mass_matrix=mass_matrix,
            early_stance_hitmoments=early_stance_hitmoments,
            early_stance_hitpoints=early_stance_hitpoints,
        )

    stc.compute_swing_control_cartesian_space = _patched_compute_swing_control_cartesian_space

    # Let the scratch leg use a continuous phase that is NOT reset on stance,
    # so all three phases (swing, rub back, return) are executed.
    _original_update_swing_time = stc.update_swing_time

    def _patched_update_swing_time(current_contact, legs_order, dt):
        for leg_id, leg_name in enumerate(legs_order):
            if stc.scratch_leg_id is not None and leg_id == stc.scratch_leg_id:
                # Continuous time for scratch leg: ignore contact pattern, just accumulate
                stc.swing_time[leg_id] += dt
            else:
                # Default behaviour for all other legs
                if current_contact[leg_id] == 0:
                    if stc.swing_time[leg_id] < stc.swing_period:
                        stc.swing_time[leg_id] += dt
                else:
                    stc.swing_time[leg_id] = 0

    stc.update_swing_time = _patched_update_swing_time
    stc._scratch_patch_installed = True


def clear_scratch(stc: Any) -> None:
    """Turn off scratch mode (do not remove the patch). Next scratch will capture a new initial position."""
    stc.scratch_leg_id = None
    stc.scratch_center = None
