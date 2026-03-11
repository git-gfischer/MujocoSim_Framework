"""
Runtime patch for SwingTrajectoryController to randomize single-leg trot footholds.

When enabled for one leg, the swing touchdown position for that leg is jittered
inside a circle around the nominal touchdown point, so the foot does not always
land at exactly the same place.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# Default radius for random touchdown (meters)
TROT_RANDOM_RADIUS_DEFAULT = 0.1


def _sample_offset(radius: float) -> np.ndarray:
    """Sample a 2D offset uniformly inside a circle of given radius."""
    # Polar sampling: r ~ sqrt(U) * R, theta ~ U * 2π
    u = np.random.rand(2)
    r = radius * np.sqrt(u[0])
    theta = 2.0 * np.pi * u[1]
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.array([dx, dy], dtype=np.float64)


def install_trot_random_patch(stc: Any) -> None:
    """
    Patch a SwingTrajectoryController to add random touchdown for trot swings.
    Idempotent: safe to call multiple times.
    """
    if getattr(stc, "_trot_random_patch_installed", False):
        return

    stc.random_trot_leg_id = None
    stc.random_trot_radius = TROT_RANDOM_RADIUS_DEFAULT

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
        # Only randomize touchdown for the configured trot leg
        if getattr(stc, "random_trot_leg_id", None) is not None and leg_id == stc.random_trot_leg_id:
            td = np.asarray(touch_down, dtype=np.float64).reshape(3).copy()
            radius = float(getattr(stc, "random_trot_radius", TROT_RANDOM_RADIUS_DEFAULT))
            if radius > 0.0:
                offset_xy = _sample_offset(radius)
                td[0] += offset_xy[0]
                td[1] += offset_xy[1]
            touch_down = td

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
    stc._trot_random_patch_installed = True


def clear_trot_random(stc: Any) -> None:
    """Disable random touchdown for trot (but keep the patch installed)."""
    stc.random_trot_leg_id = None

