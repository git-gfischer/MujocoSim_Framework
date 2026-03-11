"""
Custom movement procedures and simulation controller (static hold, single-leg trot).
"""
from quadruped_pympc.helpers.custom_procedures.movement_procedures import MovementProcedures
from quadruped_pympc.helpers.custom_procedures.controller import SimulationController
from quadruped_pympc.helpers.custom_procedures.gait_patch import install_gait_patch
from quadruped_pympc.helpers.custom_procedures.keyboard_interface import (
    run_interface,
    send_key,
    PROCEDURE_KEY_FILE,
)

__all__ = [
    "MovementProcedures",
    "SimulationController",
    "install_gait_patch",
    "run_interface",
    "send_key",
    "PROCEDURE_KEY_FILE",
]
