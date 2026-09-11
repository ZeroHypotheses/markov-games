"""Shared helpers for building exact game states.

Tests construct the position they want to examine directly and call
``transition`` once. Driving the environment for fifty steps to reach an
interesting cell makes a test that fails for reasons unrelated to its name.
"""

from __future__ import annotations

import pytest

from markov_games.games.capture_the_flag.config import CtfConfig, Position
from markov_games.games.capture_the_flag.state import CtfState


@pytest.fixture
def config() -> CtfConfig:
    """The default 7x5 board."""
    return CtfConfig()


def make_state(
    positions: tuple[Position, Position, Position, Position],
    carried_by: tuple[int | None, int | None] = (None, None),
    scores: tuple[int, int] = (0, 0),
    step: int = 0,
) -> CtfState:
    """A state with everything named, so a test reads as the scenario it is."""
    return CtfState(positions=positions, carried_by=carried_by, scores=scores, step=step)
