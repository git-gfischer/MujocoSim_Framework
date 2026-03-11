"""
Runtime patch for PeriodicGaitGenerator to add single-leg swing support.

This module adds single_leg_swing_leg_id and related methods to a gait generator
instance without modifying the original periodic_gait_generator.py source.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from quadruped_pympc.helpers.quadruped_utils import GaitType


def install_gait_patch(pgg: Any) -> None:
    """
    Patch a PeriodicGaitGenerator instance to support single-leg swing mode.
    Idempotent: safe to call multiple times on the same instance.
    """
    if getattr(pgg, "_custom_procedures_patched", False):
        return

    pgg.single_leg_swing_leg_id = None
    pgg._original_run = pgg.run
    pgg.run = _patched_run.__get__(pgg, type(pgg))
    _original_set_full_stance = pgg.set_full_stance
    _original_restore_previous_gait = pgg.restore_previous_gait

    def _set_full_stance():
        pgg.single_leg_swing_leg_id = None
        _original_set_full_stance()

    def _restore_previous_gait():
        pgg.single_leg_swing_leg_id = None
        _original_restore_previous_gait()

    def set_single_leg_swing(leg_id: int):
        assert 0 <= leg_id < 4
        if pgg.gait_type == GaitType.FULL_STANCE.value:
            pgg.gait_type = copy.deepcopy(pgg.previous_gait_type)
            pgg.reset()
        pgg.single_leg_swing_leg_id = leg_id
        pgg._phase_signal[leg_id] = 0.0
        pgg._init[leg_id] = False

    def clear_single_leg_swing():
        pgg.single_leg_swing_leg_id = None
        pgg.reset()

    pgg.set_full_stance = _set_full_stance
    pgg.restore_previous_gait = _restore_previous_gait
    pgg.set_single_leg_swing = set_single_leg_swing
    pgg.clear_single_leg_swing = clear_single_leg_swing
    pgg._custom_procedures_patched = True


def _patched_run(self: Any, dt: float, new_step_freq: float) -> np.ndarray:
    """Run with single-leg swing: only the selected leg has swing phase; others stay in contact."""
    if self.single_leg_swing_leg_id is None:
        return self._original_run(dt, new_step_freq)
    contact = np.zeros(self.n_contact)
    for leg in range(self.n_contact):
        if leg != self.single_leg_swing_leg_id:
            contact[leg] = 1
            continue
        self._phase_signal[leg] += dt * new_step_freq
        self._phase_signal[leg] = self._phase_signal[leg] % 1.0
        if self._init[leg]:
            if self._phase_signal[leg] <= self.phase_offset[leg]:
                contact[leg] = 1
            else:
                self._init[leg] = False
                contact[leg] = 1
                self._phase_signal[leg] = 0
        else:
            contact[leg] = 1 if self._phase_signal[leg] < self.duty_factor else 0
    return contact
