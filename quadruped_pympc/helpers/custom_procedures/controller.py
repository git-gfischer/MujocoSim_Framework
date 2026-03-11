"""
Simulation controller: selects movement procedures and handles keyboard input.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quadruped_pympc.helpers.custom_procedures.movement_procedures import MovementProcedures


# Letter keys to avoid conflicting with MuJoCo viewer number keys (1-4 etc.)
SINGLE_LEG_KEYS = {"f": "FL", "g": "FR", "h": "RL", "j": "RR"}  # F/G = front, H/J = rear
SCRATCH_KEYS = {"r": "FL", "t": "FR", "y": "RL", "u": "RR"}  # R/T/Y/U = scratch that leg
STATIC_HOLD_KEYS = ("s", "0")  # S = stance
SNIFF_KEY = "n"  # N = head down (sniff)
CLEAR_KEYS = ("c", "5", "e", "escape")


class SimulationController:
    """
    Controller that switches between normal walking and movement procedures
    (static hold, single-leg trot) via keyboard.

    Keys: S = static hold | F/G/H/J = single-leg trot | R/T/Y/U = scratch | N = sniff | C = clear.
    """

    def __init__(self, movement_procedures: MovementProcedures):
        self.movement_procedures = movement_procedures

    def handle_key(self, key: str) -> None:
        """Handle a key press (e.g. from terminal). Key is a single character string."""
        if not key:
            return
        key = key.strip().lower()
        if key in STATIC_HOLD_KEYS:
            self.movement_procedures.static_hold()
            print("[Controller] Procedure: static_hold (robot static)")
        elif key in SINGLE_LEG_KEYS:
            leg = SINGLE_LEG_KEYS[key]
            self.movement_procedures.single_leg_trot(leg)
            print(f"[Controller] Procedure: single_leg_trot ({leg})")
        elif key in SCRATCH_KEYS:
            leg = SCRATCH_KEYS[key]
            self.movement_procedures.scratch(leg)
            print(f"[Controller] Procedure: scratch ({leg})")
        elif key == SNIFF_KEY:
            self.movement_procedures.sniff()
            print("[Controller] Procedure: sniff (head down)")
        elif key in CLEAR_KEYS:
            self.movement_procedures.clear_procedure()
            print("[Controller] Procedure cleared (normal gait)")

    def key_callback(self, keycode: int, keyname: str | None, action: int, mods: int, viewer: Any) -> None:
        if action != 1:
            return
        key = keyname or self._keycode_to_char(keycode)
        if key:
            self.handle_key(key)

    @staticmethod
    def _keycode_to_char(keycode: int) -> str | None:
        if keycode == 256:
            return "Escape"
        if 48 <= keycode <= 57:  # 0-9
            return chr(keycode)
        if 65 <= keycode <= 90:  # A-Z (uppercase)
            return chr(keycode).lower()
        if 97 <= keycode <= 122:  # a-z (lowercase)
            return chr(keycode)
        return None

    def apply_ref_velocity_override(self, ref_base_lin_vel, ref_base_ang_vel):
        """If a procedure is active, return zero ref velocity; else return given ref velocities."""
        override, (vx, vy, vz), (wx, wy, wz) = self.movement_procedures.get_ref_velocity_for_procedure()
        if override:
            import numpy as np
            ref_lin = np.array([vx, vy, vz], dtype=np.float64)
            ref_ang = np.array([wx, wy, wz], dtype=np.float64)
            try:
                if np.asarray(ref_base_ang_vel).size != 3:
                    ref_ang = np.zeros_like(ref_base_ang_vel)
            except Exception:
                pass
            return ref_lin, ref_ang
        return ref_base_lin_vel, ref_base_ang_vel

    @staticmethod
    def print_help() -> None:
        print("Controller keys:  S = stance | F/G/H/J = single-leg trot | R/T/Y/U = scratch | N = sniff | C = clear")
