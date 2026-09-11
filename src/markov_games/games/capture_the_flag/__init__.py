"""2v2 Capture the Flag on a small symmetric grid.

Rules: ``docs/capture-the-flag.md``. The modules map one-to-one onto that
document: :mod:`config` (§1), :mod:`state` (§2), :mod:`rules` (§3-§7),
:mod:`observations` (§8), plus the :mod:`env` wrapper and the :mod:`render`
renderer.
"""

from .config import AGENT_NAMES, BLUE, RED, CtfConfig
from .env import CaptureTheFlagEnv
from .rules import Action, TransitionEvents, transition
from .state import CtfState, initial_state

__all__ = [
    "AGENT_NAMES",
    "BLUE",
    "RED",
    "Action",
    "CaptureTheFlagEnv",
    "CtfConfig",
    "CtfState",
    "TransitionEvents",
    "initial_state",
    "transition",
]
