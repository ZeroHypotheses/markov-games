"""markov-games — small Markov games for studying multi-agent RL.

The games live in :mod:`markov_games.games`; the environment for each is a
PettingZoo ``ParallelEnv``. See ``AGENTS.md`` for how this repo is organised and
``docs/capture-the-flag.md`` for the rules of the first game.
"""

from .games.capture_the_flag import Action, CaptureTheFlagEnv, CtfConfig, CtfState

__version__ = "0.1.0"

__all__ = ["Action", "CaptureTheFlagEnv", "CtfConfig", "CtfState", "__version__"]
