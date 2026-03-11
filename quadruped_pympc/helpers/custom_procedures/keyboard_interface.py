"""
Terminal interface to send procedure keys to the procedure-controller simulation.

Run this in a separate terminal (same working directory as the simulation).
Press a key + Enter to trigger the corresponding procedure; the simulation
reads procedure_key.txt each step and executes the command.
"""
from __future__ import annotations

import pathlib
import sys

# Must match simulation_procedure_controller.PROCEDURE_KEY_FILE
PROCEDURE_KEY_FILE = pathlib.Path.cwd() / "procedure_key.txt"

# Key bindings (match controller.py)
LEGEND = """
  Procedure keys (type one character + Enter):
  ┌─────────────────────────────────────────────────────────┐
  │  S  or  0   →  Static hold (stance)                     │
  │  F         →  Single-leg trot: FL (front left)          │
  │  G         →  Single-leg trot: FR (front right)         │
  │  H         →  Single-leg trot: RL (rear left)           │
  │  J         →  Single-leg trot: RR (rear right)          │
  │  R         →  Scratch floor: FL                          │
  │  T         →  Scratch floor: FR                         │
  │  Y         →  Scratch floor: RL                         │
  │  U         →  Scratch floor: RR                         │
  │  N         →  Sniff (head down, like a dog)             │
  │  C , 5, E  →  Clear procedure (normal gait)             │
  │  Q         →  Quit this interface                        │
  └─────────────────────────────────────────────────────────┘
"""

VALID_KEYS = frozenset("s0fgjhjrtyuc5enq")  # r/t/y/u = scratch, n = sniff


def send_key(key: str) -> bool:
    """Write a single key to the procedure file. Returns True if written."""
    if not key or key.lower() not in VALID_KEYS:
        return False
    try:
        PROCEDURE_KEY_FILE.write_text(key.strip().lower())
        return True
    except OSError:
        return False


def run_interface() -> None:
    """Run the terminal menu loop: read key, write to file, repeat until quit."""
    print("Procedure keyboard interface")
    print("Run the simulation in another terminal from the same directory.")
    print(f"Commands are written to: {PROCEDURE_KEY_FILE}")
    print(LEGEND)

    while True:
        try:
            line = input("Key [S/F/G/H/J/R/T/Y/U/N/C/Q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        key = line[0]
        if key == "q":
            print("Quit.")
            break
        if key not in VALID_KEYS:
            print("  Unknown key. Use S F G H J R T Y U N C 0 5 E or Q to quit.")
            continue
        if send_key(key):
            action = {
                "s": "Static hold",
                "0": "Static hold",
                "f": "Single-leg trot FL",
                "g": "Single-leg trot FR",
                "h": "Single-leg trot RL",
                "j": "Single-leg trot RR",
                "r": "Scratch FL",
                "t": "Scratch FR",
                "y": "Scratch RL",
                "u": "Scratch RR",
                "n": "Sniff (head down)",
                "c": "Clear procedure",
                "5": "Clear procedure",
                "e": "Clear procedure",
            }.get(key, key)
            print(f"  → Sent '{key}' ({action})")
        else:
            print("  → Failed to write (check path and permissions).", file=sys.stderr)


if __name__ == "__main__":
    run_interface()
