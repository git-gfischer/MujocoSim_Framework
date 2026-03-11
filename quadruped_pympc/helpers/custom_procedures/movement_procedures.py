"""
Movement procedures for the quadruped: static hold, single-leg trot, sniff (head down),
scratch (one leg scratches the floor).

These procedures configure the existing whole-body controller (gait + reference
velocity and optionally reference pose). Single-leg swing is enabled via a runtime patch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quadruped_pympc.helpers.custom_procedures.gait_patch import install_gait_patch
from quadruped_pympc.helpers.custom_procedures.scratch_patch import (
    clear_scratch,
    install_scratch_patch,
)
from quadruped_pympc.helpers.custom_procedures.trot_random_patch import (
    clear_trot_random,
    install_trot_random_patch,
)

if TYPE_CHECKING:
    from quadruped_pympc.interfaces.wb_interface import WBInterface


LEGS_ORDER = ("FL", "FR", "RL", "RR")
LEG_NAME_TO_ID = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}

# Sniff procedure: pitch (rad) = nose down, height_ratio = ref_z multiplier for base height
SNIFF_PITCH_RAD = 0.6  # ~40 deg nose down
SNIFF_HEIGHT_RATIO = 0.8  # base a bit lower than nominal stance


class MovementProcedures:
    """
    Named movement procedures: static hold, single-leg trot, sniff (head down like a dog).
    """

    def __init__(self, wb_interface: WBInterface):
        self.wb_interface = wb_interface
        self._current_procedure: str | None = None
        self._single_leg_name: str | None = None
        self._ref_override: dict | None = None

    @property
    def current_procedure(self) -> str | None:
        return self._current_procedure

    def static_hold(self) -> None:
        """Make the robot static: all legs in contact, zero reference velocity."""
        self.wb_interface.pgg.set_full_stance()
        self._current_procedure = "static_hold"
        self._single_leg_name = None
        self._ref_override = None

    def single_leg_trot(self, leg_name: str) -> None:
        """Only the given leg performs a trot-like swing; the other three stay in contact."""
        if leg_name not in LEG_NAME_TO_ID:
            raise ValueError(f"leg_name must be one of {list(LEG_NAME_TO_ID.keys())}, got {leg_name!r}")
        install_gait_patch(self.wb_interface.pgg)
        leg_id = LEG_NAME_TO_ID[leg_name]
        self.wb_interface.pgg.set_single_leg_swing(leg_id)
        # Enable random touchdown positions for this trot leg so it does not always land
        # at exactly the same point.
        install_trot_random_patch(self.wb_interface.stc)
        self.wb_interface.stc.random_trot_leg_id = leg_id
        self._current_procedure = "single_leg_trot"
        self._single_leg_name = leg_name
        self._ref_override = None  # no pose override for single-leg trot

    def scratch(self, leg_name: str) -> None:
        """One leg scratches the floor (back-and-forth on the ground); the other three stay in stance."""
        if leg_name not in LEG_NAME_TO_ID:
            raise ValueError(f"leg_name must be one of {list(LEG_NAME_TO_ID.keys())}, got {leg_name!r}")
        install_gait_patch(self.wb_interface.pgg)
        leg_id = LEG_NAME_TO_ID[leg_name]
        self.wb_interface.pgg.set_single_leg_swing(leg_id)
        install_scratch_patch(self.wb_interface.stc)
        self.wb_interface.stc.scratch_leg_id = leg_id
        self.wb_interface.stc.scratch_center = None  # capture initial position on first control step
        self._current_procedure = "scratch"
        self._single_leg_name = leg_name
        self._ref_override = None

    def sniff(self, pitch_rad: float | None = None, height_ratio: float | None = None) -> None:
        """Robot holds still with head down (nose toward ground), like a dog sniffing."""
        from quadruped_pympc import config as cfg

        self.wb_interface.pgg.set_full_stance()
        self._current_procedure = "sniff"
        self._single_leg_name = None
        ref_z = cfg.simulation_params["ref_z"]
        pitch = pitch_rad if pitch_rad is not None else SNIFF_PITCH_RAD
        ratio = height_ratio if height_ratio is not None else SNIFF_HEIGHT_RATIO
        self._ref_override = {
            "ref_orientation": np.array([0.0, float(pitch), 0.0]),
            "ref_position": np.array([0.0, 0.0, float(ref_z) * ratio]),
        }

    def clear_procedure(self) -> None:
        """Clear the current procedure and restore the default gait."""
        pgg = self.wb_interface.pgg
        if getattr(self.wb_interface.stc, "_scratch_patch_installed", False):
            clear_scratch(self.wb_interface.stc)
        if getattr(self.wb_interface.stc, "_trot_random_patch_installed", False):
            clear_trot_random(self.wb_interface.stc)
        if getattr(pgg, "_custom_procedures_patched", False) and getattr(pgg, "single_leg_swing_leg_id", None) is not None:
            pgg.clear_single_leg_swing()
        else:
            pgg.restore_previous_gait()
        self._current_procedure = None
        self._single_leg_name = None
        self._ref_override = None

    def get_ref_override(self) -> dict | None:
        """Return ref_override dict for the wrapper (ref_orientation, ref_position) when procedure needs it."""
        return self._ref_override

    def get_ref_velocity_for_procedure(self) -> tuple[bool, tuple[float, float, float], tuple[float, float, float]]:
        """Return (override, (vx, vy, vz), (wx, wy, wz)); override=True means use zero ref vel."""
        zero_ang = (0.0, 0.0, 0.0)
        if self._current_procedure == "static_hold":
            return True, (0.0, 0.0, 0.0), zero_ang
        if self._current_procedure == "single_leg_trot":
            return True, (0.0, 0.0, 0.0), zero_ang
        if self._current_procedure == "scratch":
            return True, (0.0, 0.0, 0.0), zero_ang
        if self._current_procedure == "sniff":
            return True, (0.0, 0.0, 0.0), zero_ang
        return False, (0.0, 0.0, 0.0), zero_ang
