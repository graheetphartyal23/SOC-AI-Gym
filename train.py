#!/usr/bin/env python3
"""
Placeholder training entrypoint.

Wire your preferred RL stack (e.g. CleanRL, RLlib) to the HTTP API or import
`SOCEnvironment` from `server.environment` for in-process rollouts.
"""

from __future__ import annotations

from server.constants import MAX_STEPS


def main() -> None:
    print(
        "train.py: implement your RL training loop here.\n"
        "Tip: from server.environment import SOCEnvironment\n"
        f"     # MAX_STEPS = {MAX_STEPS} (forced episode end + limit penalty if not solved).\n"
        "     env = SOCEnvironment(); env.reset('easy'); obs, r, d, info = env.step(...)"
    )


if __name__ == "__main__":
    main()
