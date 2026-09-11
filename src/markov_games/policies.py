"""Non-learning policies. The floor that every baseline has to beat.

A policy here is a callable ``(observations) -> actions`` over the agents it was
constructed with. Learning agents will not share this interface — they need the
rest of the transition — but random play needs nothing else, and keeping it this
small makes the rollout loop readable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .games.capture_the_flag.rules import NUM_ACTIONS


class RandomPolicy:
    """Uniformly random actions, seeded and therefore reproducible."""

    def __init__(self, agents: Iterable[str], seed: int | None = None) -> None:
        self.agents = list(agents)
        self._rng = np.random.default_rng(seed)

    def __call__(self, observations: Mapping[str, np.ndarray]) -> dict[str, int]:
        del observations  # a random policy is, pointedly, not looking
        return {agent: int(self._rng.integers(NUM_ACTIONS)) for agent in self.agents}
